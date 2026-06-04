"""Verify pipeline steps 2–4 on a video clip.

Runs homography → player tracking → ball tracking → point boundary detection
on a short clip and writes an annotated video showing:
  - Player bounding boxes (green = near, blue = far) with court coordinates
  - Ball position with trajectory trail and bounce markers
  - State banner (DEAD / LIVE / COOLDOWN) + detected point count
  - Green/red border flash on point start/end
  - Court mini-map (top-down, bottom-right corner)

Usage:
    # Download a clip first (same method as verify_keypoint_channels.py):
    yt-dlp -o match.mp4 --download-sections "*00:01:00-00:02:30" <URL>
    ffmpeg -i match.mp4 -t 60 -c copy clip.mp4

    python scripts/verify_pipeline.py \\
        --video clip.mp4 \\
        --output annotated.mp4 \\
        --duration 60 \\
        --device cpu

Performance note: CPU processes ~0.5–1 fps (player tracking is the bottleneck).
Use --device mps (Apple Silicon) or --device cuda for real-time speed.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Colours (BGR) ────────────────────────────────────────────────────────────
_C_NEAR      = (50, 220, 50)     # green
_C_FAR       = (220, 100, 50)    # blue
_C_BALL      = (0, 230, 230)     # yellow
_C_INTERP    = (50, 150, 230)    # orange  (gap-filled ball position)
_C_BOUNCE    = (50, 50, 230)     # red
_C_STATE = {"LIVE": (50, 220, 50), "COOLDOWN": (50, 220, 220), "DEAD": (120, 120, 120)}

_TRAIL_LEN  = 20   # frames of ball trail to keep
_DIAG_W     = 140
_DIAG_H     = 240
_BORDER_TTL = 4    # frames to show point-start/end border flash


# ── Mini-map helpers ─────────────────────────────────────────────────────────

def _m2d(cx: float, cy: float) -> tuple[int, int]:
    """Court meters → mini-map pixel coordinates (far end at top)."""
    pad = 8
    px = int((cx + 5.485) / 10.97 * (_DIAG_W - 2 * pad) + pad)
    py = int((1.0 - cy / 23.77) * (_DIAG_H - 2 * pad) + pad)
    return (px, py)


def _draw_mini_map(
    near_m: np.ndarray | None,
    far_m: np.ndarray | None,
    ball_trail: list[np.ndarray | None],
    ball_m: np.ndarray | None,
) -> np.ndarray:
    d = np.full((_DIAG_H, _DIAG_W, 3), 25, dtype=np.uint8)
    W = (200, 200, 200)
    G = (130, 130, 130)

    # Court lines
    cv2.rectangle(d, _m2d(-5.485, 23.77), _m2d(5.485, 0.0), W, 1)
    cv2.line(d, _m2d(-4.115, 0), _m2d(-4.115, 23.77), G, 1)
    cv2.line(d, _m2d(4.115, 0),  _m2d(4.115, 23.77),  G, 1)
    cv2.line(d, _m2d(-5.485, 11.885), _m2d(5.485, 11.885), W, 1)  # net
    cv2.line(d, _m2d(-4.115, 6.40),  _m2d(4.115, 6.40),  G, 1)
    cv2.line(d, _m2d(-4.115, 17.37), _m2d(4.115, 17.37), G, 1)
    cv2.line(d, _m2d(0, 6.40),       _m2d(0, 17.37),     G, 1)

    # Ball trail (fades toward old)
    n = len(ball_trail)
    for i, pos in enumerate(ball_trail):
        if pos is None:
            continue
        alpha = (i + 1) / max(n, 1)
        c = tuple(int(v * alpha) for v in _C_BALL)
        cv2.circle(d, _m2d(float(pos[0]), float(pos[1])), 2, c, -1)

    # Players
    if near_m is not None:
        cv2.circle(d, _m2d(float(near_m[0]), float(near_m[1])), 5, _C_NEAR, -1)
    if far_m is not None:
        cv2.circle(d, _m2d(float(far_m[0]), float(far_m[1])), 5, _C_FAR,  -1)

    # Ball
    if ball_m is not None:
        cv2.circle(d, _m2d(float(ball_m[0]), float(ball_m[1])), 4, _C_BALL, -1)

    return d


# ── Frame annotation ─────────────────────────────────────────────────────────

def _annotate(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    H: np.ndarray | None,
    players,
    ball,
    state_label: str,
    point_count: int,
    ball_trail: list,
    border_color: tuple | None,
) -> np.ndarray:
    out = frame.copy()
    fh, fw = out.shape[:2]

    # ── Player boxes ──────────────────────────────────────────────────────
    if players is not None:
        for player, color, side in [
            (players.near_player, _C_NEAR, "NEAR"),
            (players.far_player,  _C_FAR,  "FAR"),
        ]:
            if player is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in player.bbox_px]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            spd = float(np.linalg.norm(player.velocity_ms))
            label = (
                f"{side} ({player.position_m[0]:+.1f},{player.position_m[1]:.1f})m"
                f"  {spd:.1f}m/s"
            )
            cv2.putText(out, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # ── Ball ──────────────────────────────────────────────────────────────
    if ball is not None and ball.position_px is not None:
        bx, by = int(ball.position_px[0]), int(ball.position_px[1])
        bc = _C_INTERP if ball.interpolated else _C_BALL
        cv2.circle(out, (bx, by), 7, bc, 2)
        if ball.is_bounce and ball.bounce_zone:
            cv2.drawMarker(out, (bx, by), _C_BOUNCE, cv2.MARKER_STAR, 20, 2)
            cv2.putText(out, ball.bounce_zone, (bx + 10, by - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _C_BOUNCE, 1, cv2.LINE_AA)

    # ── State banner (top strip) ──────────────────────────────────────────
    sc = _C_STATE.get(state_label, _C_STATE["DEAD"])
    hom_ok = H is not None
    banner = (
        f"  {state_label}   pts={point_count}   "
        f"frame={frame_idx}  t={frame_idx / fps:.1f}s   "
        f"hom={'OK' if hom_ok else 'MISS'}"
    )
    cv2.rectangle(out, (0, 0), (fw, 30), (0, 0, 0), -1)
    cv2.putText(out, banner, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, sc, 2, cv2.LINE_AA)

    # ── Border flash on point start / end ─────────────────────────────────
    if border_color is not None:
        cv2.rectangle(out, (3, 3), (fw - 3, fh - 3), border_color, 5)

    # ── Court mini-map (bottom-right) ─────────────────────────────────────
    near_m  = players.near_player.position_m if (players and players.near_player) else None
    far_m   = players.far_player.position_m  if (players and players.far_player)  else None
    ball_m  = ball.position_m if (ball and ball.position_m is not None) else None
    diag    = _draw_mini_map(near_m, far_m, ball_trail, ball_m)

    dy, dx = diag.shape[:2]
    px, py = fw - dx - 8, fh - dy - 8
    if px > 0 and py > 0:
        roi = out[py:py + dy, px:px + dx]
        out[py:py + dy, px:px + dx] = cv2.addWeighted(roi, 0.2, diag, 0.8, 0)

    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify pipeline steps 2–4 (player tracking, ball tracking, boundaries)"
    )
    parser.add_argument("--video",    required=True, help="Input video path")
    parser.add_argument("--output",   default="annotated.mp4", help="Output annotated video")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Seconds to process (default: 30)")
    parser.add_argument("--device",   default="cpu",
                        help="Inference device: cpu / cuda / mps")
    parser.add_argument("--homography-weights", default=None,
                        help="Path to homography .pth (omit to auto-download)")
    parser.add_argument("--ball-weights", default=None,
                        help="Path to ball tracker .pth (omit to auto-download)")
    parser.add_argument("--skip-player-tracker", action="store_true",
                        help="Skip YOLO+BoT-SORT (useful if boxmot not installed)")
    args = parser.parse_args()

    # ── Imports ───────────────────────────────────────────────────────────
    print("Loading pipeline components…")
    from src.pipeline.homography.detector import CourtHomographyEstimator
    from src.pipeline.tracking.player_tracker import PlayerTracker, PlayerTrackResult
    from src.pipeline.ball.tracker import BallTracker
    from src.pipeline.events.boundary_detector import BoundaryDetector

    hom_est = CourtHomographyEstimator(
        weights_path=args.homography_weights,
        device=args.device,
        reestimate_interval=30,
    )
    print("  ✓ Homography estimator")

    player_tracker = None
    if not args.skip_player_tracker:
        try:
            player_tracker = PlayerTracker(device=args.device)
            print("  ✓ Player tracker (YOLO26m + BoT-SORT)")
        except Exception as e:
            print(f"  ✗ Player tracker skipped: {e}")
            print("    Re-run with --skip-player-tracker to suppress this warning.")

    ball_tracker = BallTracker(weights_path=args.ball_weights, device=args.device)
    print("  ✓ Ball tracker (TrackNet)")

    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {args.video}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_fr    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_fr      = int(args.duration * fps)
    n_frames    = min(total_fr, max_fr)
    vw          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo: {vw}×{vh} @ {fps:.1f} fps  |  processing {n_frames} frames ({args.duration:.0f}s)")

    boundary = BoundaryDetector(fps=fps)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (vw, vh),
    )

    # ── Processing loop ───────────────────────────────────────────────────
    frame_buf:  deque[np.ndarray]       = deque(maxlen=3)
    ball_trail: deque[np.ndarray | None]= deque(maxlen=_TRAIL_LEN)
    segments:   list                    = []
    border_color: tuple | None          = None
    border_ttl: int                     = 0
    H_current:  np.ndarray | None       = None

    for frame_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        # Homography
        hom = hom_est.process_frame(frame, frame_idx)
        H_current = hom.H

        # Player tracking
        if player_tracker is not None:
            player_result = player_tracker.process_frame(frame, frame_idx, H_current)
        else:
            player_result = PlayerTrackResult(
                frame_idx=frame_idx, near_player=None, far_player=None
            )

        # Ball tracking (needs 3-frame buffer)
        frame_buf.append(frame.copy())
        if len(frame_buf) == 3:
            ball_result = ball_tracker.process_frame(
                (frame_buf[0], frame_buf[1], frame_buf[2]), frame_idx, H_current
            )
        else:
            ball_result = None

        ball_trail.append(
            ball_result.position_m.copy()
            if ball_result and ball_result.position_m is not None
            else None
        )

        # Boundary detection
        completed = boundary.process_frame(frame_idx, ball_result, player_result)
        if completed is not None:
            segments.append(completed)
            dur = completed.end_sec - completed.start_sec
            print(
                f"  ● Point {completed.point_id}: "
                f"{completed.start_sec:.1f}s – {completed.end_sec:.1f}s  ({dur:.1f}s)"
            )
            border_color = _C_STATE["LIVE"]
            border_ttl   = _BORDER_TTL

        # Decay border flash
        if border_ttl > 0:
            border_ttl -= 1
        else:
            border_color = None

        state_label = boundary._state.name

        annotated = _annotate(
            frame, frame_idx, fps, H_current,
            player_result, ball_result,
            state_label, len(segments),
            list(ball_trail), border_color,
        )
        writer.write(annotated)

        if frame_idx % 90 == 0:
            pct = frame_idx / n_frames * 100
            print(f"  {frame_idx:>5}/{n_frames}  ({pct:4.0f}%)  state={state_label:<9}  pts={len(segments)}")

    # Flush any open point at end of clip
    final = boundary.flush()
    if final:
        segments.append(final)

    cap.release()
    writer.release()

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"Annotated video → {out_path}")
    print(f"Detected {len(segments)} point(s) in {args.duration:.0f}s clip:")
    for seg in segments:
        dur = seg.end_sec - seg.start_sec
        print(f"  Point {seg.point_id:>2}: {seg.start_sec:6.1f}s – {seg.end_sec:6.1f}s  ({dur:.1f}s)")
    if not segments:
        print("  (none — try a clip that includes live play, not just warm-up or replays)")


if __name__ == "__main__":
    main()
