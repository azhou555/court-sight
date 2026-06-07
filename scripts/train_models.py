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
                   choices=["shot_classifier", "execution_prob",
                            "neutral_position", "win_prob", "all"])
    p.add_argument("--config", default="configs/training.yaml", type=Path)
    p.add_argument("--out",    default="weights/", type=Path)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.model in ("neutral_position", "all"):
        from src.models.neutral_position.train import train as train_neutral_position
        record_paths = sorted(args.data.glob("*.json"))
        if not record_paths:
            raise FileNotFoundError(f"No ShotRecord JSON files found in {args.data}")
        train_neutral_position(
            shot_record_paths=record_paths,
            output_dir=args.out / "neutral_position",
        )

    if args.model in ("shot_classifier", "execution_prob", "win_prob"):
        raise NotImplementedError(
            f"Dispatch for '{args.model}' not yet wired into this script. "
            f"Use the module's own train entrypoint."
        )


if __name__ == "__main__":
    main()
