# Win Probability Model (2C) — Training Pipeline Design

**Status:** approved (design phase)
**Date:** 2026-06-08
**Step:** 9 of the development order (`PLAN.md`)
**Spec for:** the response/reward side of the EV framework —
`EV_adjusted = P(make) × P(win | make) − P(miss) − displacement_penalty`

---

## 1. Goal & Scope

Deliver a **unified supervised training pipeline** that trains the existing
`RallyTransformerEncoder` on `ShotRecord` rally sequences to predict
`P(win point | rally state)`.

A single code path covers both regimes the architecture doc (`07_win_probability.md`)
describes, via a **positions-off + mask-flag switch**:

- **Pretrain regime** — categorical tokens only (shot type, direction, landing
  zone); positional features zeroed and flagged absent.
- **Fine-tune regime** — full feature set including positions and velocities.

This mirrors step 8's delivery (the trainable Stage-1 pipeline on the available
record format), and is **test-driven on synthetic records**: neither the raw
Match Charting CSVs nor CV-processed match records exist locally yet
(`data/external/` is empty).

### Explicitly deferred (YAGNI — blocked on real data)

- A separate MCP-only pretrain corpus loader.
- The two-corpus checkpoint-transfer orchestration (pretrain → fine-tune).

The unified path *supports* both regimes; orchestrating a real two-stage run
waits until data is sourced.

---

## 2. Model & Tokenizer Repairs (`win_prob/model.py`)

The existing stub is a sketch with four correctness bugs that must be fixed
before any training can work.

### 2.1 Learnable embeddings inside the module

**Bug:** `RallyTokenizer` owns the `nn.Embedding`s, runs them under
`torch.no_grad()`, and lives outside `RallyTransformerEncoder`. They would never
be optimized or saved — categorical embeddings stay at random init forever.

**Fix:** Move embedding *into* the `nn.Module`. The tokenizer produces integer
index tensors + raw float features; `forward()` performs the embedding lookups
and concatenation, so embeddings are part of the gradient graph and the
checkpoint.

### 2.2 MCP-native vocabularies

**Bug:** `SHOT_TYPES.index(shot["shot_type_mcp"])` assumes canonical names
(`"forehand_groundstroke"`) but records store raw MCP codes (`'f'`, `'b'`,
`'serve'`). Same for `direction_mcp` (`'1'/'2'/'3'`) vs the 6-name `DIRECTIONS`.
Both raise `ValueError` on the first real shot.

**Fix (decision: MCP-native):** Build embedding vocabularies over the MCP-native
codes actually stored in `ShotRecord`, each with a reserved `<pad>` (index 0) and
`<unk>` slot:

- **Shot type:** the MCP alphabet `f b r s v z l o j k y p q` + `serve`.
- **Direction:** `1 2 3` (rally) and `4 5 6` (serve wide/body/T), plus `''`.
- **Landing zone:** the 9-zone enum
  (`T_deep … W_short`).

Rationale: mapping to the 6-way `DIRECTIONS` enum is strictly lossy
(inside-in/out/body have no standard MCP source), and the canonical-enum path
crashes on unseen codes. MCP-native is lossless and crash-proof; the vocab not
being human-pretty is irrelevant for a learned embedding.

### 2.3 Positions mask flag (token 42 → 43)

**Bug:** the unified switch needs the model to distinguish "position genuinely
`(0,0)`" from "position withheld (pretrain mode)." No flag exists.

**Fix:** add a single `positions_known` flag dim. In the positions-off regime,
zero the 8 positional dims + `court_opening` and set the flag to 0. One flag
(not per-feature): positions, velocities, and `court_opening` are all CV-derived
and present-or-absent as a unit.

Updated token layout (43 dims, before linear projection to 128):

| segment              | dims | source                         |
|----------------------|------|--------------------------------|
| shot_type embedding  | 16   | learned (MCP-native vocab)     |
| direction embedding  | 8    | learned (MCP-native vocab)     |
| landing_zone embed   | 8    | learned (9-zone vocab)         |
| striker position     | 2    | court meters                   |
| striker velocity     | 2    | m/s                            |
| opponent position    | 2    | court meters                   |
| opponent velocity    | 2    | m/s                            |
| court_opening        | 1    | opponent distance from center  |
| rally_depth          | 1    | shot index / MAX_RALLY_LEN     |
| **positions_known**  | 1    | **new mask flag**              |

### 2.4 Left-padding for batched sequences

**Bug:** `forward()` reads `x[:, -1, :]` as the current shot, but batched
variable-length sequences require padding; with right-padding the last token is
padding.

**Fix:** left-pad sequences and pass `src_key_padding_mask`. `x[:, -1, :]`
remains the true current shot; padding tokens are masked out of attention.

### 2.5 Persistence consequence

`save()`/`load()` must round-trip the encoder `state_dict` (now includes
embeddings — automatic), the calibrator (currently dropped on save), and the
vocab maps (so inference tokenizes identically).

---

## 3. Training Pipeline (`win_prob/train.py`)

### 3.1 Example construction

Each shot in a point becomes one training example:

- **Sequence:** the rally up to and including that shot, left-truncated to
  `MAX_RALLY_LEN = 20`.
- **Label:** that shot's `point_outcome` (1 = that shot's striker won the point).

A k-shot point yields k examples with labels alternating by striker. This is the
spec's "evaluate at each rally position."

### 3.2 Train/val split — grouped by point

Split on `_point_key` (the `point_id` prefix, reusing step 8's helper) so all
shots from one point land on the same side. Splitting by shot would leak
near-identical prefixes across the boundary.

### 3.3 Known optimism (documented, not silently inherited)

The current shot's own `ball_landing_zone` is in its token, but landing happens
*after* contact — at true inference time it isn't yet known. This matches the
spec's token design, so it is kept, but recorded as a known optimism in
`07_win_probability.md`. The label `point_outcome` is the target (fine); no token
encodes the terminal winner/error flag, so there is no hard leak.

### 3.4 Training loop (per `07_win_probability.md`)

- Loss: binary cross-entropy.
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-2.
- Scheduler: cosine annealing.
- Gradient clipping: max norm 1.0.
- Batch size: 64 points, left-padded with `padding_mask`.
- Early stopping on validation log-loss.

### 3.5 Calibration & metrics

- Fit **isotonic regression** on validation predictions; persist in checkpoint.
- Report **Brier, log-loss, AUC-ROC, ECE**, plus the doc's per-rally-position
  breakdown (accuracy by shot index).

### 3.6 Persistence

Single checkpoint dict: encoder `state_dict`, calibrator, vocab maps. A
`metrics.json` is written alongside, matching step 8.

### 3.7 Wiring

Add `win_prob` to `scripts/train_models.py` choices and dispatch, matching the
step-8 pattern.

---

## 4. Testing

Synthetic `ShotRecord` generator (reuse the step-8 shape), covering:

- Vocab handles unseen codes (maps to `<unk>`, no crash).
- Positions-off regime zeros positional dims and sets the flag to 0.
- Grouped split keeps all shots of a point on one side.
- Left-padding + `padding_mask` correctness (current shot is the last real token).
- End-to-end train → save → load → predict round-trip (predictions match).
- Leak guard: `point_outcome` never enters any token.

OpenMP note: torch is imported here; `tests/conftest.py` already imports xgboost
first to avoid the macOS OpenMP segfault — no change needed.

---

## 5. Files Touched

| File                                   | Change                                  |
|----------------------------------------|-----------------------------------------|
| `src/models/win_prob/model.py`         | Repairs 2.1–2.5; token 42→43            |
| `src/models/win_prob/train.py`         | **New** — full training pipeline        |
| `scripts/train_models.py`              | Add `win_prob` choice + dispatch        |
| `tests/models/test_win_prob_training.py` | **New** — test suite (§4)             |
| `docs/architecture/07_win_probability.md` | Note the §3.3 landing-zone optimism  |
