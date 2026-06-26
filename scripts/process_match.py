"""
Run the full data pipeline on a single match video.

Usage:
    python -m scripts.process_match \
        --video path/to/match.mp4 \
        --mcp   path/to/match_charting.csv \
        --out   data/processed/match_id/
"""

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process a single tennis match through the data pipeline.")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--mcp",   required=True, type=Path)
    p.add_argument("--out",   required=True, type=Path)
    p.add_argument("--config", default="configs/pipeline.yaml", type=Path)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

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

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or cfg["fps"]
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {args.video.name}: {total} frames @ {fps:.1f} fps")

    ball_tracks = []
    player_tracks = []
    point_segments = []

    frame_buf: deque = deque(maxlen=3)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_buf.append(frame)

        h_result = homography.process_frame(frame, frame_idx)
        H = h_result.H if h_result.H is not None else None

        p_result = players.process_frame(frame, frame_idx, H)
        player_tracks.append(p_result)

        if len(frame_buf) == 3:
            b_result = ball.process_frame(tuple(frame_buf), frame_idx, H)
        else:
            from src.pipeline.ball.tracker import BallTrackResult
            b_result = BallTrackResult(
                frame_idx=frame_idx,
                position_px=None,
                position_m=None,
                confidence=0.0,
                is_bounce=False,
                bounce_zone=None,
                trajectory_segment=0,
            )
        ball_tracks.append(b_result)

        completed = boundary.process_frame(frame_idx, b_result, p_result)
        if completed is not None:
            point_segments.append(completed)

        if frame_idx % 500 == 0:
            print(f"  frame {frame_idx}/{total}  points={len(point_segments)}")
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
        "points_detected": len(point_segments),
        "shots_aligned": len(records),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Done. {len(records)} shots → {out_path}")


if __name__ == "__main__":
    main()
