"""Align Match Charting Project labels to CV-extracted positional data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class ShotRecord:
    # Match Charting Project fields
    match_id: str
    point_id: str
    shot_in_rally: int
    shot_type_mcp: str
    direction_mcp: str
    depth_mcp: str
    point_outcome: int              # 1 = striker won, 0 = striker lost

    # CV pipeline fields
    contact_frame: int
    striker_role: str               # "near" or "far"
    striker_position_m: list        # [x, y]
    striker_velocity_ms: list       # [vx, vy]
    striker_pose: list              # (17, 2) keypoints in court meters
    striker_pose_confidence: float
    opponent_position_m: list       # [x, y]
    opponent_velocity_ms: list      # [vx, vy]
    ball_landing_zone: Optional[str]
    ball_contact_position_m: Optional[list]

    # Context
    rally_context: list = field(default_factory=list)
    alignment_confidence: float = 0.0
    pose_quality: float = 0.0


class MCPAligner:
    """
    Aligns Match Charting Project CSV data to CV-extracted frame-level data.

    Alignment proceeds in three steps:
    1. Point boundary detection (from video) + MCP score sequence matching
    2. Per-point shot sequence alignment
    3. Quality filtering and confidence scoring
    """

    def __init__(
        self,
        fps: float = 30.0,
        max_dtw_distance: float = 5.0,
    ):
        self.fps = fps
        self.max_dtw_distance = max_dtw_distance

    def align(
        self,
        mcp_path: str,
        point_boundaries: list[dict],
        ball_tracks: list,
        player_tracks: list,
        pose_tracks: list,
    ) -> list[ShotRecord]:
        mcp_df = pd.read_csv(mcp_path)
        records: list[ShotRecord] = []

        for point_boundary in point_boundaries:
            mcp_point = self._match_point(point_boundary, mcp_df)
            if mcp_point is None:
                continue
            shot_records = self._align_shots(
                mcp_point,
                point_boundary,
                ball_tracks,
                player_tracks,
                pose_tracks,
            )
            records.extend(shot_records)

        return records

    def _match_point(
        self, boundary: dict, mcp_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        # TODO: match video point to MCP point via score sequence DTW
        raise NotImplementedError

    def _align_shots(
        self,
        mcp_point: pd.DataFrame,
        boundary: dict,
        ball_tracks: list,
        player_tracks: list,
        pose_tracks: list,
    ) -> list[ShotRecord]:
        # TODO: align individual shots within a matched point
        raise NotImplementedError

    @staticmethod
    def _compute_alignment_confidence(
        mcp_shot_count: int,
        cv_shot_count: int,
        pose_quality: float,
        ball_confidence: float,
    ) -> float:
        count_agreement = 1.0 - abs(mcp_shot_count - cv_shot_count) / max(mcp_shot_count, 1)
        return float(count_agreement * 0.4 + pose_quality * 0.4 + ball_confidence * 0.2)
