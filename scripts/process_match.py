"""
Run the full data pipeline on a single match video.

Usage:
    python -m scripts.process_match \
        --video path/to/match.mp4 \
        --mcp   path/to/match_charting.csv \
        --out   data/processed/match_id/

Long matches checkpoint automatically: a checkpoint.json is written to --out
after every completed point (a clean state-machine boundary — see
BoundaryDetector, whose state resets to DEAD there) and re-running the same
command resumes from it. Pass --restart to ignore an existing checkpoint and
start over.
"""

import argparse
import json
from collections import deque
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

# HSV ranges (OpenCV: H 0-180, S 0-255, V 0-255) for each surface type.
# Used by the frame gate to detect broadcast wide-angle court view.
_COURT_HSV = {
    "hard":  (np.array([ 95,  40,  60], np.uint8), np.array([135, 220, 210], np.uint8)),
    "clay":  (np.array([  5,  80,  80], np.uint8), np.array([ 25, 255, 210], np.uint8)),
    "grass": (np.array([ 35,  40,  60], np.uint8), np.array([ 80, 220, 210], np.uint8)),
}
# Frames to stay active after court color disappears (handles brief close-ups mid-rally).
_GATE_KEEPALIVE = 90  # ~3 sec at 30 fps


def _court_visible(frame: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    """True when court surface color spans the middle third of the frame horizontally."""
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    mid = mask[30:60]  # middle vertical third
    row_coverage = np.mean(mid > 0, axis=1)
    return bool(np.mean(row_coverage > 0.35) > 0.5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process a single tennis match through the data pipeline.")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--mcp",   required=True, type=Path)
    p.add_argument("--out",   required=True, type=Path)
    p.add_argument("--config", default="configs/pipeline.yaml", type=Path)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (for testing)")
    p.add_argument("--surface", default="hard", choices=["hard", "clay", "grass"],
                   help="Court surface type for frame gate color filter")
    p.add_argument("--restart", action="store_true",
                   help="Ignore any existing checkpoint.json in --out and start over")
    return p.parse_args()


# --- Checkpointing ------------------------------------------------------
# Only ball/player track results and point boundaries need to survive a
# restart: homography and the boundary detector are safe to reconstruct
# fresh at a completed-point boundary (BoundaryDetector resets to its
# default DEAD state there; homography re-estimates from scratch on the
# next frame regardless of history).

def _ser_ball(b) -> dict:
    return {
        "frame_idx": b.frame_idx,
        "position_px": b.position_px.tolist() if b.position_px is not None else None,
        "position_m": b.position_m.tolist() if b.position_m is not None else None,
        "confidence": b.confidence,
        "is_bounce": b.is_bounce,
        "bounce_zone": b.bounce_zone,
        "trajectory_segment": b.trajectory_segment,
        "interpolated": b.interpolated,
    }


def _deser_ball(d: dict):
    from src.pipeline.ball.tracker import BallTrackResult
    return BallTrackResult(
        frame_idx=d["frame_idx"],
        position_px=np.array(d["position_px"]) if d["position_px"] is not None else None,
        position_m=np.array(d["position_m"]) if d["position_m"] is not None else None,
        confidence=d["confidence"],
        is_bounce=d["is_bounce"],
        bounce_zone=d["bounce_zone"],
        trajectory_segment=d["trajectory_segment"],
        interpolated=d.get("interpolated", False),
    )


def _ser_player_state(s) -> dict | None:
    if s is None:
        return None
    return {
        "track_id": s.track_id,
        "bbox_px": s.bbox_px.tolist(),
        "position_m": s.position_m.tolist(),
        "velocity_ms": s.velocity_ms.tolist(),
        "confidence": s.confidence,
    }


def _deser_player_state(d: dict | None):
    if d is None:
        return None
    from src.pipeline.tracking.player_tracker import PlayerState
    return PlayerState(
        track_id=d["track_id"],
        bbox_px=np.array(d["bbox_px"]),
        position_m=np.array(d["position_m"]),
        velocity_ms=np.array(d["velocity_ms"]),
        confidence=d["confidence"],
    )


def _ser_player(p) -> dict:
    return {
        "frame_idx": p.frame_idx,
        "near_player": _ser_player_state(p.near_player),
        "far_player": _ser_player_state(p.far_player),
    }


def _deser_player(d: dict):
    from src.pipeline.tracking.player_tracker import PlayerTrackResult
    return PlayerTrackResult(
        frame_idx=d["frame_idx"],
        near_player=_deser_player_state(d["near_player"]),
        far_player=_deser_player_state(d["far_player"]),
    )


def _write_checkpoint(path: Path, *, next_frame, skipped, gate_keepalive,
                       point_counter, ball_tracks, player_tracks, point_segments,
                       homography) -> None:
    data = {
        "next_frame": next_frame,
        "skipped": skipped,
        "gate_keepalive": gate_keepalive,
        "point_counter": point_counter,
        "ball_tracks": [_ser_ball(b) for b in ball_tracks],
        "player_tracks": [_ser_player(p) for p in player_tracks],
        "point_segments": [asdict(s) for s in point_segments],
        # Carry the re-estimation schedule forward so a fresh estimator
        # doesn't force an off-cycle re-estimate right at the resume frame
        # (that shift was measured to perturb projected positions by
        # sub-mm — enough to flip a shot's zone right at a boundary).
        "homography_last_frame": homography._last_frame,
        "homography_last_H": (
            homography._last_H.tolist() if homography._last_H is not None else None
        ),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)  # atomic — never leaves a half-written checkpoint


def _load_checkpoint(path: Path) -> dict:
    from src.pipeline.events.boundary_detector import PointSegment
    raw = json.loads(path.read_text())
    return {
        "next_frame": raw["next_frame"],
        "skipped": raw["skipped"],
        "gate_keepalive": raw["gate_keepalive"],
        "point_counter": raw["point_counter"],
        "ball_tracks": [_deser_ball(b) for b in raw["ball_tracks"]],
        "player_tracks": [_deser_player(p) for p in raw["player_tracks"]],
        "point_segments": [PointSegment(**s) for s in raw["point_segments"]],
        "homography_last_frame": raw["homography_last_frame"],
        "homography_last_H": (
            np.array(raw["homography_last_H"]) if raw["homography_last_H"] is not None else None
        ),
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.out / "checkpoint.json"
    if args.restart and checkpoint_path.exists():
        checkpoint_path.unlink()
    checkpoint = _load_checkpoint(checkpoint_path) if checkpoint_path.exists() else None

    cfg = yaml.safe_load(args.config.read_text())["pipeline"]

    from src.pipeline.homography.detector import CourtHomographyEstimator
    from src.pipeline.tracking.player_tracker import PlayerTracker
    from src.pipeline.ball.tracker import BallTracker
    from src.pipeline.events.boundary_detector import BoundaryDetector
    from src.pipeline.alignment.mcp_aligner import MCPAligner

    hcfg = cfg["homography"]
    homography = CourtHomographyEstimator(
        device=args.device,
        reestimate_interval=hcfg["reestimate_interval"],
    )

    tcfg = cfg["tracking"]
    players = PlayerTracker(
        detection_threshold=tcfg["detection_threshold"],
        iou_threshold=tcfg["iou_threshold"],
        fps=cfg["fps"],
        device=args.device,
    )

    bcfg = cfg["ball"]
    ball = BallTracker(
        weights_path=bcfg["tracknet_weights"],
        device=args.device,
        detection_threshold=bcfg["detection_threshold"],
        bounce_velocity_threshold=bcfg["bounce_velocity_threshold"],
    )

    boundary = BoundaryDetector(fps=cfg["fps"])
    if checkpoint is not None:
        boundary._point_counter = checkpoint["point_counter"]
        homography._last_frame = checkpoint["homography_last_frame"]
        homography._last_H = checkpoint["homography_last_H"]

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or cfg["fps"]
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)
    print(f"Processing {args.video.name}: {total} frames @ {fps:.1f} fps", flush=True)

    from src.pipeline.ball.tracker import BallTrackResult
    from src.pipeline.tracking.player_tracker import PlayerTrackResult

    def _empty_ball(idx: int) -> BallTrackResult:
        return BallTrackResult(
            frame_idx=idx, position_px=None, position_m=None,
            confidence=0.0, is_bounce=False, bounce_zone=None, trajectory_segment=0,
        )

    def _empty_players(idx: int) -> PlayerTrackResult:
        return PlayerTrackResult(frame_idx=idx, near_player=None, far_player=None)

    hsv_lo, hsv_hi = _COURT_HSV[args.surface]

    if checkpoint is not None:
        gate_keepalive = checkpoint["gate_keepalive"]
        skipped = checkpoint["skipped"]
        ball_tracks = checkpoint["ball_tracks"]
        player_tracks = checkpoint["player_tracks"]
        point_segments = checkpoint["point_segments"]
        frame_idx = checkpoint["next_frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        print(f"Resuming from checkpoint at frame {frame_idx} "
              f"({len(point_segments)} points already detected)", flush=True)
    else:
        gate_keepalive = 0
        skipped = 0
        ball_tracks = []
        player_tracks = []
        point_segments = []
        frame_idx = 0

    frame_buf: deque = deque(maxlen=3)

    while args.max_frames is None or frame_idx < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        frame_buf.append(frame)

        h_result = homography.process_frame(frame, frame_idx)
        H = h_result.H

        # Frame gate: skip expensive tracking on non-court frames (replays, close-ups, crowd)
        if _court_visible(frame, hsv_lo, hsv_hi):
            gate_keepalive = _GATE_KEEPALIVE
        elif gate_keepalive > 0:
            gate_keepalive -= 1
        # Also deactivate immediately when homography re-estimated and found nothing
        if not h_result.camera_angle_valid and gate_keepalive == _GATE_KEEPALIVE:
            gate_keepalive = 0
        is_active = gate_keepalive > 0

        if is_active:
            p_result = players.process_frame(frame, frame_idx, H)
            if len(frame_buf) == 3:
                b_result = ball.process_frame(tuple(frame_buf), frame_idx, H)
            else:
                b_result = _empty_ball(frame_idx)
        else:
            p_result = _empty_players(frame_idx)
            b_result = _empty_ball(frame_idx)
            skipped += 1

        player_tracks.append(p_result)
        ball_tracks.append(b_result)

        completed = boundary.process_frame(frame_idx, b_result, p_result)
        if completed is not None:
            point_segments.append(completed)
            _write_checkpoint(
                checkpoint_path,
                next_frame=frame_idx + 1,
                skipped=skipped,
                gate_keepalive=gate_keepalive,
                point_counter=boundary._point_counter,
                ball_tracks=ball_tracks,
                player_tracks=player_tracks,
                point_segments=point_segments,
                homography=homography,
            )

        if frame_idx % 500 == 0:
            pct_skipped = 100 * skipped / max(frame_idx, 1)
            print(f"  frame {frame_idx}/{total}  points={len(point_segments)}  skipped={pct_skipped:.0f}%", flush=True)
        frame_idx += 1

    cap.release()

    # flush any open point
    if boundary._state.name != "DEAD":
        seg = boundary._close_point()
        if seg is not None:
            point_segments.append(seg)

    print(f"Detected {len(point_segments)} points. Aligning to MCP...")

    aligner = MCPAligner(fps=fps)
    records = aligner.align(
        mcp_path=args.mcp,
        point_boundaries=point_segments,
        ball_tracks=ball_tracks,
        player_tracks=player_tracks,
    )

    out_path = args.out / "shot_records.json"
    MCPAligner.save(records, out_path)

    summary = {
        "video": str(args.video),
        "mcp": str(args.mcp),
        "fps": fps,
        "total_frames": frame_idx,
        "frames_skipped": skipped,
        "skip_pct": round(100 * skipped / max(frame_idx, 1), 1),
        "points_detected": len(point_segments),
        "shots_aligned": len(records),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    checkpoint_path.unlink(missing_ok=True)

    print(f"Done. {len(records)} shots → {out_path}")


if __name__ == "__main__":
    main()
