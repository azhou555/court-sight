"""Detect ball-racket contact frames from ball tracking trajectory data.

Each time BallTracker increments trajectory_segment, a court bounce has
occurred and the ball is now on a new arc.  The first frame of each arc
is used as the approximate contact frame for the shot that launched it.

This approach gives one contact event per shot with ~frame accuracy —
sufficient for aligning shot sequences to MCP labels.  Exact contact
timing (sub-frame) is not needed for alignment.
"""

from __future__ import annotations

from ..ball.tracker import BallTrackResult


def detect_contacts(
    ball_tracks: list[BallTrackResult],
    start_frame: int,
    end_frame: int,
) -> list[int]:
    """Return approximate contact frame indices within [start_frame, end_frame].

    A contact is detected at:
      - The first frame the ball is visible (serve contact).
      - The first frame of each subsequent trajectory segment (racket contact
        after a court bounce).

    Interpolated frames (from gap filling) are excluded to avoid phantom
    contacts during brief occlusions.

    Args:
        ball_tracks: Full per-frame BallTrackResult list for the match.
        start_frame: Inclusive start of the point window.
        end_frame:   Inclusive end of the point window.

    Returns:
        Sorted list of frame indices, one per detected contact event.
    """
    in_window = [
        r for r in ball_tracks
        if start_frame <= r.frame_idx <= end_frame
        and r.position_px is not None
        and not r.interpolated
    ]

    if not in_window:
        return []

    contacts: list[int] = []
    prev_segment: int | None = None

    for r in in_window:
        if prev_segment is None or r.trajectory_segment != prev_segment:
            contacts.append(r.frame_idx)
            prev_segment = r.trajectory_segment

    return contacts
