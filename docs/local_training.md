# Running the pipeline locally

Colab kept dropping (kernel restarts wiping `/tmp`, Drive/rclone flakiness,
and eventually a ToS ban). Everything needed to process a match and train
models already runs locally — no Drive, no rclone, no Colab.

## One-time setup

```bash
conda create -y -n court-sight python=3.11
conda activate court-sight
pip install -r requirements.txt
brew install libomp   # needed by xgboost/lightgbm on macOS
```

No editable install needed (`pip install -e .` fails — `pyproject.toml`'s
`build-backend` is misconfigured). Just run everything as `python -m
scripts.<name>` from the repo root so `src/` resolves.

`mmpose`/`mmcv`/`mmengine` were removed from `requirements.txt` — pose
estimation runs on `yolo26m-pose.pt` (ultralytics) instead, see
`src/pipeline/pose/extractor.py`.

## Required env var

```bash
export OMP_NUM_THREADS=1
```

torch and xgboost each bundle their own OpenMP runtime. On Apple Silicon,
whichever one runs a parallel op *second* segfaults, regardless of import
order. Pinning to one thread avoids both libraries touching their thread
pools. This is set automatically in `tests/conftest.py` for the test suite,
but scripts run outside pytest need it exported manually.

## Per-match data layout

Video + MCP charting CSV live under `data/external/matches/<match_id>/`,
e.g. the already-present:

```
data/external/matches/cincinnati_2023_final/
  video.mp4
  mcp_points.csv
```

Model weights are already local: `weights/tracknet_v2.pth`, `yolo26m.pt`.

## Run a match

```bash
conda activate court-sight
export OMP_NUM_THREADS=1

python -m scripts.process_match \
  --video data/external/matches/<match_id>/video.mp4 \
  --mcp   data/external/matches/<match_id>/mcp_points.csv \
  --out   data/processed/<match_id>/ \
  --device mps \
  --surface hard   # hard | clay | grass
```

`--device mps` uses the Apple GPU. Drop `--max-frames` for a full run (a
~15 min clip at 25fps is ~22,500 frames); add `--max-frames 300` for a quick
smoke test.

Output: `data/processed/<match_id>/shot_records.json` + `summary.json`.

## Train models

```bash
python -m scripts.train_models \
  --data data/processed/<match_id>/ \
  --model all \
  --out checkpoints/
```

`--model` accepts `neutral_position`, `win_prob`, or `all` (`shot_classifier`
and `execution_prob` aren't wired into this script yet — see
`NotImplementedError` in `scripts/train_models.py`). `--device` is accepted
but unused by either trainer, safe to ignore.

Saves to `checkpoints/<model>/` — `.pkl`/`.pt` weights + `metrics.json`.
Feed multiple matches' `data/processed/*/` in as more become available;
`train_models.py` globs all `*.json` under `--data`.

## Verified working end-to-end

Ran on `cincinnati_2023_final` (full clip, no `--max-frames`):
- `process_match.py` → 55 shot records
- `train_models.py --model all` → `checkpoints/neutral_position/` and
  `checkpoints/win_prob/` populated

## Bugs fixed along the way

- **`tests/conftest.py`**: previously imported `xgboost` first with a
  comment claiming that avoided a torch/xgboost segfault. That ordering no
  longer holds with current package versions — replaced with
  `OMP_NUM_THREADS=1` set before either import, which fixes it regardless
  of order. All 118 tests pass.
- **`scripts/train_models.py`** docstring said to run it as
  `python scripts/train_models.py`, which breaks its `from src...` imports.
  Fixed to `python -m scripts.train_models`.
