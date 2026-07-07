#!/usr/bin/env bash
# Usage: scripts/run_match.sh <match_id> [surface]
set -euo pipefail

match_id=$1
surface=${2:-hard}

conda activate court-sight
export OMP_NUM_THREADS=1

python -m scripts.process_match \
  --video "data/external/matches/$match_id/video.mp4" \
  --mcp   "data/external/matches/$match_id/mcp_points.csv" \
  --out   "data/processed/$match_id/" \
  --device mps \
  --surface "$surface"

python -m scripts.train_models \
  --data data/processed/ \
  --model all \
  --out checkpoints/
