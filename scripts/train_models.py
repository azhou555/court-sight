"""
Train all models in development order.

Usage:
    python scripts/train_models.py \
        --data data/processed/ \
        --model shot_classifier \
        --config configs/training.yaml
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True, type=Path)
    p.add_argument("--model",  required=True,
                   choices=["shot_classifier", "execution_prob", "win_prob", "all"])
    p.add_argument("--config", default="configs/training.yaml", type=Path)
    p.add_argument("--out",    default="weights/", type=Path)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # TODO: load shot_records from args.data
    # TODO: dispatch to appropriate training module
    raise NotImplementedError("Training not yet implemented.")


if __name__ == "__main__":
    main()
