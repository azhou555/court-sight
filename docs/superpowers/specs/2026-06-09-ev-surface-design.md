# EV Surface + Shot Quality Scoring (Model 2D) — Design

**Status:** approved (design phase)
**Date:** 2026-06-09
**Step:** 10 of the development order (`PLAN.md`)
**Spec for:** the combiner that turns 2B + 2C + 2E into an Expected-Value surface
and a two-dimensional shot-quality score.

---

## 1. Goal & Scope

Rework `src/models/ev_surface/scorer.py` from a **pre-API sketch** (committed at
`b05a408`, never updated, never used outside its pure helpers) into a correct,
tested **combiner**. `EVScorer` owns no training — it orchestrates the three
trained models' inference interfaces into:

- a per-zone **EV surface** (9 zones), and
- a **two-dimensional quality score**: `zone_score` (target selection, relative)
  + `displacement_penalty` (positioning, absolute).

### In scope
- Rebuild `EVScorer.compute_ev_surface` and `score_shot` against the **real**
  model APIs.
- Add `predict_zone_neutrals` — per-zone 2E neutral; **all 9** attached to output
  for the step-11 feedback layer (does NOT enter EV math).
- **Fix the win-prob `positions_known` mismatch** at its source (per-shot flag in
  the tokenizer — see §5).
- A real test suite for the combiner using lightweight fake models.

### Explicitly deferred
- **Aggregate pattern analysis** (`group_shots_by`, systematic-weakness flags) →
  application layer (step 11+); no consumer exists yet.
- **Heatmap visualization** → step 11 (feedback layer).

### Overarching caveat
All three models are **untrained** (no real data — see `docs/PROGRESS_LEDGER.md`).
The EV numbers this step produces are structurally correct but **placeholder**
until a dataset exists. This step builds and validates the *wiring and math*, not
meaningful EV values.

---

## 2. 2E Integration Decision

The win-prob token (step 9) has **no `next_neutral` slot**, so the original doc's
vision of feeding `next_neutral` into 2C is not realizable without re-opening and
retraining 2C (out of scope). Decided integration:

1. **2E → displacement penalty (the EV-affecting path):** 2E's `predict_neutral`
   supplies the *neutral position* that `displacement_penalty` measures against.
   This already flows through the `RecoveryState` the caller builds for the actual
   shot. The penalty is the 2E contribution to the absolute EV.
2. **2E → per-zone output attach (non-EV):** `EVScorer` additionally calls
   `predict_neutral` per zone and attaches the predicted `reachable_neutral_m` to
   the output, for the step-11 feedback layer. This does NOT enter the EV ranking.

EV math is therefore `P(make) × (P(win)+1) − 1 − dp`, with 2E entering only via
`dp` and the output attachment.

---

## 3. Architecture

`EVScorer` is a **pure combiner**: it receives pre-built feature objects and a
`RecoveryState`, and calls the models' inference methods. It does NOT run CV or
re-derive the actual-shot neutral itself — the caller (a future inference driver)
does that. This keeps dependencies explicit and makes the combiner testable
against fakes.

### Constructor
```python
EVScorer(safety_model, win_prob_model, neutral_model)
```
All three are duck-typed by their inference methods (`predict`, `predict`,
`predict_neutral` respectively) — any object satisfying the interface works,
which is what enables fake-model testing.

### Verified API touchpoints (against current code)
- 2B: `safety_model.predict(ShotGeometryFeatures) -> {"p_make", "p_miss", ...}`
- 2C: `win_prob_model.predict(rally: list[dict]) -> {"p_win_point", ...}`
- 2E: `neutral_model.predict_neutral(features: ResponseDistributionFeatures,
  current_pos, current_vel, t_recovery, rng=None) -> {"reachable_neutral_m", ...}`
- penalty: `execution_prob/displacement.py::displacement_penalty(contact_pos_m,
  neutral_pos_m, movement_vector, shot_direction, ball_pos_at_contact, shot_type)`

> Implementation note: the rebuild uses `dataclasses.replace` on
> `ShotGeometryFeatures` and `ResponseDistributionFeatures`. The plan MUST verify
> the exact field names against the current dataclasses before writing
> `replace(...)` — the old sketch may reference drifted fields.

---

## 4. Components & Data Flow (per shot)

1. `dp = displacement_penalty(recovery_state)` — once; constant across zones.
2. For each of the 9 `COURT_ZONES`:
   - **2B:** `_build_zone_features(base_safety_features, zone_center)` →
     `safety_model.predict` → `p_make`.
   - **2C:** `win_prob_model.predict(rally_context + [{"ball_landing_zone": zone}])`
     → `p_win`.
   - **EV:** `compute_ev(p_make, p_win) - dp` where
     `compute_ev = p_make*(p_win+1) - 1`.
3. `predict_zone_neutrals`: for each zone,
   `replace(base_response_features, your_landing_x=zone_center[0],
   your_landing_y=zone_center[1])` → `neutral_model.predict_neutral(features,
   striker_pos, striker_vel, t_recovery)` → store `reachable_neutral_m`.
4. `score_shot`: rank zones (dp cancels in `ev_loss`), compute `zone_score`,
   select tiered feedback, attach all 9 `next_neutrals` plus the actual/best
   convenience pointers.

### Method boundaries (each independently testable)
| Method | Returns | Responsibility |
|---|---|---|
| `compute_ev_surface(...)` | `dict[str, float]` | EV ranking surface only |
| `predict_zone_neutrals(...)` | `dict[str, np.ndarray]` | per-zone reachable neutral (output attach) |
| `score_shot(...)` | `dict` | assemble surface + neutrals + scores + feedback |

### Quality score (unchanged formula from the doc)
```
ev_actual_base = ev_surface[actual] + dp
ev_best_base   = max over zones of (ev_surface[z] + dp)
ev_worst_base  = min over zones of (ev_surface[z] + dp)
ev_loss   = ev_best_base - ev_actual_base          # dp cancels
ev_range  = ev_best_base - ev_worst_base
zone_score = 1 - ev_loss/ev_range  (0.5 if ev_range == 0)
```

### Output schema (per shot)
Existing `score_shot` keys (`shot_id`, `actual_zone`, `ev_actual`,
`ev_actual_base`, `ev_best_base`, `ev_loss`, `best_zone`, `zone_score`,
`displacement_penalty`, `p_make_actual`, `feedback_zone`, `feedback_position`,
`ev_surface`) PLUS:
- `next_neutrals`: **all 9** zones → `reachable_neutral_m` as `list[float]`,
  parallel to `ev_surface` (from `predict_zone_neutrals`).
- `next_neutral_actual`: convenience pointer = `next_neutrals[actual_zone]`.
- `next_neutral_best`: convenience pointer = `next_neutrals[best_zone]`.

---

## 5. Fix: per-shot `positions_known` flag (win-prob tokenizer)

The per-zone 2C call appends `{"ball_landing_zone": zone}` — a sparse dict with no
positions. Previously `RallyTokenizer.encode_shot` set `positions_known=1.0` for
every shot in the default path (zeroing missing positions but still flagging them
"known"), a train/inference mismatch (ledger trap #2).

**Fix (in `src/models/win_prob/model.py`):** make `positions_known` a **per-shot**
determination.
- `positions_off=True` (pretrain regime): unchanged — ALL shots get zeroed
  positions and flag 0.0.
- `positions_off=False` (default): per shot, detect whether position data is
  present. A shot **has positions** iff `striker_position_m` is present, non-None,
  and length ≥ 2. If present → real positions + flag 1.0; if absent → zeroed
  positions + flag 0.0.

This correctly handles the scorer's mixed rally: real prior shots are flagged
known, the appended hypothetical zone is flagged unknown. It does not change
behavior for any shot that carries positions, so existing step-9 tests still pass;
a new unit test covers the sparse-shot → flag 0.0 case.

Downstream bookkeeping: mark ledger trap #2 resolved and update the
`winprob-predict-positions-known-flag` memory.

---

## 6. Testing

Test the combiner logic in isolation with lightweight **fake models** (duck-typed):
- `FakeSafety.predict(features) -> {"p_make": <derived from features>}`
- `FakeWinProb.predict(rally) -> {"p_win_point": <derived from rally>}`
- `FakeNeutral.predict_neutral(...) -> {"reachable_neutral_m": np.array([...])}`

Cases:
- `compute_ev` math (already covered) + `compute_ev_surface` produces 9 finite EVs.
- Best-zone selection picks the max-EV zone; `dp` cancels in `ev_loss`
  (vary `dp`, assert `ev_loss`/`zone_score` unchanged).
- `zone_score` in [0,1]; `ev_range == 0` → 0.5 fallback.
- `predict_zone_neutrals` returns a neutral for all 9 zones; `score_shot` attaches
  `next_neutrals` (9) plus `next_neutral_actual` / `next_neutral_best` pointers.
- Feedback-tier boundaries (`_get_tier`) and template selection per tier.
- `_build_zone_features` sets the expected margin fields for a known zone center.

Plus, in `tests/models/test_win_prob_model.py`: a shot dict with no
`striker_position_m` yields `positions_known == 0.0` and zeroed positional floats
in the default (`positions_off=False`) path, while a shot with positions yields
`1.0` (guards the §5 fix).

OpenMP note: importing the real models pulls torch; `tests/conftest.py` already
imports xgboost first. Tests use fakes, but module import still loads the real
classes — no conftest change needed.

---

## 7. Files Touched

| File | Change |
|---|---|
| `src/models/win_prob/model.py` | Per-shot `positions_known` detection in `RallyTokenizer.encode_shot` (§5) |
| `tests/models/test_win_prob_model.py` | Add sparse-shot → `positions_known==0.0` test |
| `src/models/ev_surface/scorer.py` | Rework `EVScorer`; add `predict_zone_neutrals`; new output keys (all 9 neutrals); `base_response_features` param |
| `tests/models/test_ev_scorer.py` | Add combiner tests with fake models (keep existing pure-helper tests) |
| `docs/architecture/08_ev_surface.md` | Correct the 2E-integration description (neutral→penalty + output attach, not next_neutral→2C) |
| `docs/PROGRESS_LEDGER.md` | Finalize step-10 section; mark trap #2 resolved |
| memory `winprob-predict-positions-known-flag` | Update to reflect the fix |
