"""
Run the full data pipeline on a single match video.

Usage:
    python scripts/process_match.py \
        --video path/to/match.mp4 \
        --mcp   path/to/match_charting.csv \
        --out   data/processed/match_id/
"""

import argparse
from pathlib import Path


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

    # TODO: wire up pipeline components
    # 1. CourtHomographyEstimator
    # 2. PlayerTracker
    # 3. PoseExtractor
    # 4. BallTracker
    # 5. MCPAligner
    # 6. Serialize ShotRecords to args.out / shot_records.jsonl

    print(f"Processing {args.video} → {args.out}")
    raise NotImplementedError("Pipeline not yet wired. Implement per PLAN.md §Phase 1.")


if __name__ == "__main__":
    main()
