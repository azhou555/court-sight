# EV Surface (2D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the pre-API `EVScorer` sketch into a correct, tested combiner that turns 2B + 2C + 2E into an EV surface + a two-dimensional shot-quality score, and fix the win-prob `positions_known` mismatch at its source.

**Architecture:** `EVScorer` is a pure combiner over the three models' inference interfaces. EV math is `P(make) × (P(win)+1) − 1 − dp`; 2E enters only via the displacement penalty (the neutral it measures against) and via per-zone neutrals attached to output (not the EV math). The win-prob tokenizer gains per-shot `positions_known` detection so sparse hypothetical shots are correctly flagged. Combiner logic is tested against lightweight fake models, isolated from the untrained heavy models.

**Tech Stack:** NumPy, Python dataclasses (`replace`), pytest. No training; models are duck-typed by their `predict`/`predict_neutral` methods.

**Spec:** `docs/superpowers/specs/2026-06-09-ev-surface-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/models/win_prob/model.py` | **Modify** `RallyTokenizer.encode_shot`: per-shot `positions_known` detection. |
| `tests/models/test_win_prob_model.py` | **Add** sparse-shot → `positions_known==0.0` test. |
| `src/models/ev_surface/scorer.py` | **Rework** `EVScorer`: `compute_ev_surface(dp param)`, new `predict_zone_neutrals`, `score_shot` (adds `base_response_features` param + all-9 `next_neutrals` output). Module-level helpers/constants/dataclass unchanged. |
| `tests/models/test_ev_scorer.py` | **Add** combiner tests with fake models (keep existing pure-helper tests). |
| `docs/architecture/08_ev_surface.md` | **Modify** 2E-integration description. |
| `docs/PROGRESS_LEDGER.md` | **Modify** finalize step-10 section; mark trap #2 resolved. |
| memory `winprob-predict-positions-known-flag` | **Modify** to reflect the fix. |

**Conventions:** `tests/conftest.py` already imports xgboost before torch (OpenMP guard) — do not touch. Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Branch `claude/tennis-shot-feedback-Tclpt`; commit directly. `python` = project interpreter (3.13, torch + sklearn + xgboost).

---

## Task 1: Per-shot `positions_known` in the win-prob tokenizer

**Files:**
- Test: `tests/models/test_win_prob_model.py` (append)
- Modify: `src/models/win_prob/model.py` (the `else` branch of `RallyTokenizer.encode_shot`)

- [ ] **Step 1: Append the failing test**

Append to `tests/models/test_win_prob_model.py` (the `_shot` helper already exists in this file):

```python
def test_tokenizer_missing_positions_flags_unknown():
    """A shot dict with no positions → positions_known=0.0 + zeroed positions,
    even in the default (positions_off=False) path. Guards the EV-scorer's
    sparse hypothetical-zone shots."""
    tok = RallyTokenizer()
    sparse = {"ball_landing_zone": "C_deep"}            # no striker_position_m
    _, _, _, floats = tok.encode_shot(sparse)           # positions_off defaults False
    assert floats[10] == 0.0                            # positions_known flag
    assert all(v == 0.0 for v in floats[:9])            # positional dims zeroed
    # a shot WITH positions still flags known
    _, _, _, f2 = tok.encode_shot(_shot())
    assert f2[10] == 1.0
    assert f2[:9] != [0.0] * 9
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/models/test_win_prob_model.py::test_tokenizer_missing_positions_flags_unknown -q`
Expected: FAIL — `assert 1.0 == 0.0` (current code flags every default-path shot known).

- [ ] **Step 3: Implement per-shot detection**

In `src/models/win_prob/model.py`, replace the `else` branch of `encode_shot`:

```python
            if positions_off:
                positional = [0.0] * 9
                known = 0.0
            else:
                sp = shot.get("striker_position_m") or [0.0, 0.0]
                sv = shot.get("striker_velocity_ms") or [0.0, 0.0]
                op = shot.get("opponent_position_m") or [0.0, 0.0]
                ov = shot.get("opponent_velocity_ms") or [0.0, 0.0]
                co = float(shot.get("court_opening", 0.0))
                positional = [float(sp[0]), float(sp[1]), float(sv[0]), float(sv[1]),
                              float(op[0]), float(op[1]), float(ov[0]), float(ov[1]), co]
                known = 1.0
```

with:

```python
            sp_raw = shot.get("striker_position_m")
            has_positions = sp_raw is not None and len(sp_raw) >= 2
            if positions_off or not has_positions:
                # pretrain regime (all shots) OR a shot that carries no position
                # data (e.g. a hypothetical zone appended by the EV scorer):
                # zero the positional features and flag them unknown.
                positional = [0.0] * 9
                known = 0.0
            else:
                sv = shot.get("striker_velocity_ms") or [0.0, 0.0]
                op = shot.get("opponent_position_m") or [0.0, 0.0]
                ov = shot.get("opponent_velocity_ms") or [0.0, 0.0]
                co = float(shot.get("court_opening", 0.0))
                positional = [float(sp_raw[0]), float(sp_raw[1]), float(sv[0]), float(sv[1]),
                              float(op[0]), float(op[1]), float(ov[0]), float(ov[1]), co]
                known = 1.0
```

- [ ] **Step 4: Run the full win-prob model + training suites**

Run: `python -m pytest tests/models/test_win_prob_model.py tests/models/test_win_prob_training.py -q`
Expected: PASS (the new test + all existing — real shots still flag `1.0`, so nothing regresses).

- [ ] **Step 5: Commit**

```bash
git add src/models/win_prob/model.py tests/models/test_win_prob_model.py
git commit -m "$(cat <<'EOF'
Fix win_prob positions_known: per-shot detection for sparse shots

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `EVScorer.compute_ev_surface` (dp param) + `predict_zone_neutrals`

**Files:**
- Modify: `src/models/ev_surface/scorer.py`
- Test: `tests/models/test_ev_scorer.py` (append fakes + tests)

- [ ] **Step 1: Append fake models + tests**

Append to `tests/models/test_ev_scorer.py`:

```python
import numpy as np

from src.models.ev_surface.scorer import EVScorer, RecoveryState
from src.models.execution_prob.model import ShotGeometryFeatures
from src.models.neutral_position.model import ResponseDistributionFeatures


class _FakeSafety:
    def __init__(self, p_make=0.8):
        self.p = p_make
    def predict(self, features):
        return {"p_make": self.p, "p_miss": 1.0 - self.p}


class _FakeWinProb:
    """p_win keyed off the last (hypothetical) shot's landing zone."""
    def __init__(self, zone_pwin):
        self.m = zone_pwin
    def predict(self, rally):
        zone = rally[-1].get("ball_landing_zone")
        return {"p_win_point": self.m.get(zone, 0.5), "rally_context_used": len(rally)}


class _FakeNeutral:
    """reachable neutral biased by your_landing_x so zones differ."""
    def predict_neutral(self, features, current_pos, current_vel, t_recovery, rng=None):
        return {"reachable_neutral_m": np.array([features.your_landing_x * 0.5, 2.0])}


def _safety_features():
    return ShotGeometryFeatures(
        lateral_margin_m=1.0, depth_margin_m=2.0, net_clearance_m=0.3,
        shot_direction_deg=0.0, landing_x=0.0, landing_y=10.0,
        shot_type="forehand_groundstroke", striker_y_m=2.0, striker_x_m=0.0,
        incoming_depth_m=18.0,
    )


def _response_features():
    return ResponseDistributionFeatures(
        your_landing_x=0.0, your_landing_y=10.0, ball_height_at_bounce=0.0,
        opponent_pos_x=0.0, opponent_pos_y=18.0, opponent_vel_x=0.0, opponent_vel_y=0.0,
        opponent_displacement_from_neutral=1.0, your_shot_type="forehand_groundstroke",
        rally_depth=4, prev_shot_direction_1=0.0, prev_shot_direction_2=0.0,
    )


def _uniform_pwin(value=0.5):
    return {z: value for z in COURT_ZONES}


def test_compute_ev_surface_returns_nine_finite_evs():
    scorer = EVScorer(_FakeSafety(0.8), _FakeWinProb(_uniform_pwin(0.5)), _FakeNeutral())
    surface = scorer.compute_ev_surface(_safety_features(), [{"ball_landing_zone": "C_deep"}], dp=0.0)
    assert len(surface) == 9
    assert all(np.isfinite(v) for v in surface.values())


def test_compute_ev_surface_dp_is_constant_offset():
    """dp shifts every zone equally and never changes the argmax."""
    zone_pwin = _uniform_pwin(0.3); zone_pwin["W_deep"] = 0.95
    scorer = EVScorer(_FakeSafety(0.8), _FakeWinProb(zone_pwin), _FakeNeutral())
    feats, rally = _safety_features(), [{"ball_landing_zone": "C_deep"}]
    s0 = scorer.compute_ev_surface(feats, rally, dp=0.0)
    s2 = scorer.compute_ev_surface(feats, rally, dp=0.2)
    for z in COURT_ZONES:
        assert abs((s0[z] - 0.2) - s2[z]) < 1e-6
    assert max(s0, key=s0.get) == max(s2, key=s2.get) == "W_deep"


def test_predict_zone_neutrals_one_per_zone():
    scorer = EVScorer(_FakeSafety(), _FakeWinProb(_uniform_pwin()), _FakeNeutral())
    neutrals = scorer.predict_zone_neutrals(
        _response_features(), striker_pos=np.array([0.0, 2.0]),
        striker_vel=np.array([0.0, 0.0]), t_recovery=0.8,
    )
    assert set(neutrals) == set(COURT_ZONES)
    # neutral x tracks the zone center x (FakeNeutral returns your_landing_x * 0.5)
    from src.models.ev_surface.scorer import ZONE_CENTERS
    for z in COURT_ZONES:
        assert abs(neutrals[z][0] - ZONE_CENTERS[z][0] * 0.5) < 1e-6
```

(`COURT_ZONES` is already imported at the top of this test file.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/models/test_ev_scorer.py -k "surface or neutrals" -q`
Expected: FAIL — `compute_ev_surface` has the old signature (takes `recovery_state`, no `dp`) and `predict_zone_neutrals` does not exist yet.

- [ ] **Step 3: Hoist the `replace` import and rework the two methods**

In `src/models/ev_surface/scorer.py`, add `from dataclasses import replace` to the top imports (next to the existing `from dataclasses import dataclass`):

```python
from dataclasses import dataclass, replace
```

and remove the local `from dataclasses import replace` line inside `_build_zone_features`.

Replace the existing `compute_ev_surface` method with this (new signature: takes a precomputed `dp`, drops the unused recovery/striker args):

```python
    def compute_ev_surface(
        self,
        base_safety_features: ShotGeometryFeatures,
        rally_context: list[dict],
        dp: float = 0.0,
    ) -> dict[str, float]:
        """EV (adjusted) per zone. dp is a constant offset applied uniformly —
        it does not change which zone is best."""
        ev_surface = {}
        for zone in COURT_ZONES:
            zone_center = ZONE_CENTERS[zone]

            # 2B: geometric margin for this hypothetical target zone
            zone_features = _build_zone_features(base_safety_features, zone_center)
            p_make = self.safety_model.predict(zone_features)["p_make"]

            # 2C: win probability given this zone was the landing target
            rally_with_zone = rally_context + [{"ball_landing_zone": zone}]
            p_win = self.win_prob_model.predict(rally_with_zone)["p_win_point"]

            ev_surface[zone] = round(compute_ev(p_make, p_win) - dp, 4)
        return ev_surface

    def predict_zone_neutrals(
        self,
        base_response_features: "ResponseDistributionFeatures",
        striker_pos: np.ndarray,
        striker_vel: np.ndarray,
        t_recovery: float,
    ) -> dict[str, np.ndarray]:
        """Per-zone reachable neutral (2E). Output-only — does NOT enter EV math.
        For each zone, the player's landing is set to that zone's center."""
        neutrals = {}
        for zone in COURT_ZONES:
            zone_center = ZONE_CENTERS[zone]
            feats = replace(
                base_response_features,
                your_landing_x=float(zone_center[0]),
                your_landing_y=float(zone_center[1]),
            )
            result = self.neutral_model.predict_neutral(
                feats, striker_pos, striker_vel, t_recovery,
            )
            neutrals[zone] = result["reachable_neutral_m"]
        return neutrals
```

Add the import for the type hint at the top of `scorer.py` (it is already imported for the constructor — confirm `from src.models.neutral_position.model import NeutralPositionModel` exists; add `ResponseDistributionFeatures` to that import):

```python
from src.models.neutral_position.model import NeutralPositionModel, ResponseDistributionFeatures
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/models/test_ev_scorer.py -k "surface or neutrals" -q`
Expected: PASS.

> NOTE: `score_shot` still references the old `compute_ev_surface` signature at this point and will be reworked in Task 3. Do not run the whole `test_ev_scorer.py` file yet if it imports/exercises `score_shot` — it doesn't (the existing tests only touch `compute_ev`, `_get_tier`, `COURT_ZONES`). The `-k` filter keeps this task's run scoped.

- [ ] **Step 5: Commit**

```bash
git add src/models/ev_surface/scorer.py tests/models/test_ev_scorer.py
git commit -m "$(cat <<'EOF'
Rework EVScorer.compute_ev_surface (dp param) + add predict_zone_neutrals

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `EVScorer.score_shot` rework (base_response_features + all-9 neutrals)

**Files:**
- Modify: `src/models/ev_surface/scorer.py` (`score_shot` method)
- Test: `tests/models/test_ev_scorer.py` (append)

- [ ] **Step 1: Append the failing test**

Append to `tests/models/test_ev_scorer.py`:

```python
def _recovery_state():
    return RecoveryState(
        contact_pos_m=np.array([1.0, 3.0]),
        neutral_pos_m=np.array([0.0, 2.0]),
        movement_vector=np.array([0.5, 0.0]),
        shot_direction=np.array([0.0, 1.0]),
        ball_pos_at_contact=np.array([1.0, 3.0]),
        shot_type="forehand_groundstroke",
        recovery_displacement_m=1.4,
    )


def _score(actual_zone, zone_pwin):
    scorer = EVScorer(_FakeSafety(0.8), _FakeWinProb(zone_pwin), _FakeNeutral())
    return scorer.score_shot(
        actual_zone=actual_zone,
        base_safety_features=_safety_features(),
        base_response_features=_response_features(),
        rally_context=[{"ball_landing_zone": "C_deep", "shot_type_mcp": "f"}],
        recovery_state=_recovery_state(),
        striker_pos=np.array([0.0, 2.0]),
        striker_vel=np.array([0.0, 0.0]),
        t_recovery=0.8,
        shot_id="s1",
    )


def test_score_shot_picks_best_zone_and_attaches_all_neutrals():
    zone_pwin = _uniform_pwin(0.3); zone_pwin["W_deep"] = 0.95
    out = _score("T_short", zone_pwin)
    assert out["best_zone"] == "W_deep"
    assert out["shot_id"] == "s1"
    assert len(out["next_neutrals"]) == 9
    assert out["next_neutral_best"] == out["next_neutrals"]["W_deep"]
    assert out["next_neutral_actual"] == out["next_neutrals"]["T_short"]
    assert isinstance(out["next_neutrals"]["W_deep"], list) and len(out["next_neutrals"]["W_deep"]) == 2
    assert 0.0 <= out["zone_score"] <= 1.0


def test_score_shot_uniform_surface_zone_score_half():
    """All zones equal EV → ev_range 0 → zone_score 0.5 fallback."""
    out = _score("C_mid", _uniform_pwin(0.5))
    assert out["zone_score"] == 0.5


def test_score_shot_optimal_zone_scores_one():
    zone_pwin = _uniform_pwin(0.3); zone_pwin["W_deep"] = 0.95
    out = _score("W_deep", zone_pwin)        # actual == best
    assert out["ev_loss"] == 0.0
    assert out["zone_score"] == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/models/test_ev_scorer.py -k "score_shot" -q`
Expected: FAIL — `score_shot` has the old signature (no `base_response_features`) and old internals (calls `compute_ev_surface` with removed args; no `next_neutrals`).

- [ ] **Step 3: Replace `score_shot`**

In `src/models/ev_surface/scorer.py`, replace the entire `score_shot` method with:

```python
    def score_shot(
        self,
        actual_zone: str,
        base_safety_features: ShotGeometryFeatures,
        base_response_features: ResponseDistributionFeatures,
        rally_context: list[dict],
        recovery_state: RecoveryState,
        striker_pos: np.ndarray,
        striker_vel: np.ndarray,
        t_recovery: float,
        shot_id: Optional[str] = None,
    ) -> dict:
        dp = displacement_penalty(
            contact_pos_m=recovery_state.contact_pos_m,
            neutral_pos_m=recovery_state.neutral_pos_m,
            movement_vector=recovery_state.movement_vector,
            shot_direction=recovery_state.shot_direction,
            ball_pos_at_contact=recovery_state.ball_pos_at_contact,
            shot_type=recovery_state.shot_type,
        )

        ev_surface = self.compute_ev_surface(base_safety_features, rally_context, dp)
        neutrals = self.predict_zone_neutrals(
            base_response_features, striker_pos, striker_vel, t_recovery,
        )
        next_neutrals = {
            z: [float(v[0]), float(v[1])] for z, v in neutrals.items()
        }

        # Base EV values (before displacement penalty); dp cancels in ev_loss.
        ev_actual_base = ev_surface[actual_zone] + dp
        ev_best_zone = max(COURT_ZONES, key=lambda z: ev_surface[z])
        ev_best_base = ev_surface[ev_best_zone] + dp
        ev_worst_base = min(ev_surface[z] + dp for z in COURT_ZONES)

        ev_loss = ev_best_base - ev_actual_base
        ev_range = ev_best_base - ev_worst_base
        zone_score = 1.0 - (ev_loss / ev_range) if ev_range > 0 else 0.5

        tier = _get_tier(zone_score)
        p_make = self.safety_model.predict(base_safety_features)["p_make"]

        return {
            "shot_id": shot_id,
            "actual_zone": actual_zone,
            "ev_actual": round(ev_surface[actual_zone], 3),
            "ev_actual_base": round(ev_actual_base, 3),
            "ev_best_base": round(ev_best_base, 3),
            "ev_loss": round(ev_loss, 3),
            "best_zone": ev_best_zone,
            "zone_score": round(zone_score, 3),
            "displacement_penalty": round(dp, 3),
            "p_make_actual": round(p_make, 3),
            "feedback_zone": _zone_feedback(
                tier, actual_zone, ev_best_zone, ev_loss,
                ev_best_base, ev_surface[actual_zone], p_make,
            ),
            "feedback_position": _position_feedback(
                dp, recovery_state.recovery_displacement_m,
            ),
            "ev_surface": {z: round(v, 3) for z, v in ev_surface.items()},
            "next_neutrals": next_neutrals,
            "next_neutral_actual": next_neutrals[actual_zone],
            "next_neutral_best": next_neutrals[ev_best_zone],
        }
```

- [ ] **Step 4: Run the whole EV scorer suite**

Run: `python -m pytest tests/models/test_ev_scorer.py -q`
Expected: PASS (existing pure-helper tests + Task 2 + Task 3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/models/ev_surface/scorer.py tests/models/test_ev_scorer.py
git commit -m "$(cat <<'EOF'
Rework EVScorer.score_shot: base_response_features + all-9 next_neutrals

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Docs, ledger, memory + full-suite green

**Files:**
- Modify: `docs/architecture/08_ev_surface.md`
- Modify: `docs/PROGRESS_LEDGER.md`
- Modify: memory `winprob-predict-positions-known-flag.md`

- [ ] **Step 1: Correct the 2E-integration description in the architecture doc**

In `docs/architecture/08_ev_surface.md`, the `compute_ev_surface` pseudocode (around lines 86–115) feeds a `next_neutral` into the win-prob input. Replace the `# 2E:` / `# 2C:` block inside that pseudocode loop:

```python
        # 2E: neutral position the player would target after hitting this zone
        # feeds into the opponent's likely response distribution
        next_neutral = neutral_model.predict_neutral(
            your_landing=zone_center,
            your_position=state.striker_position,
        )

        # 2C: win probability given this zone was hit and the resulting neutral
        win = win_prob_model.predict(state.rally_context + [{
            "ball_landing_zone": zone,
            "next_neutral": next_neutral,
        }])
        p_win = win["p_win_point"]
```

with:

```python
        # 2C: win probability given this zone was the landing target.
        # NOTE: the win-prob token has no next_neutral slot, so 2E does NOT
        # feed 2C. 2E enters the EV only via the displacement penalty (below).
        win = win_prob_model.predict(state.rally_context + [{"ball_landing_zone": zone}])
        p_win = win["p_win_point"]

        # 2E (output-only): the neutral the player would recover to after this
        # zone, attached to the per-shot output for the feedback layer. Computed
        # via predict_zone_neutrals; not part of the EV computation.
```

- [ ] **Step 2: Finalize the ledger step-10 section and mark trap #2 resolved**

In `docs/PROGRESS_LEDGER.md`, replace the step-10 block:

```markdown
### Step 10 — EV Surface (2D) — IN PROGRESS
- Deferring **aggregate pattern analysis** (`group_shots_by`) to the application
  layer (step 11+) — it has no consumer until the feedback UI exists.
- Heatmap visualization belongs to step 11 (feedback layer), not here.
- (Design in progress; this section will be finalized when the spec lands.)
```

with:

```markdown
### Step 10 — EV Surface (2D) — DONE
- `EVScorer` reworked into a tested combiner: `compute_ev_surface`,
  `predict_zone_neutrals`, `score_shot`. EV = 2B×(2C+1)−1−dp.
- 2E integration: neutral feeds the displacement penalty; all 9 per-zone
  neutrals are attached to per-shot output (not the EV math).
- Untrained (no real data) — EV values are placeholder scaffolding.
- **Deferred:** aggregate pattern analysis → application layer (step 11+);
  heatmap visualization → step 11.
```

and replace ledger trap #2:

```markdown
2. **WinProb sparse-dict `positions_known` flag** — `WinProbModel.predict` sets
   `positions_known=1.0` even when a shot dict has only `ball_landing_zone` (no
   positions), which differs from the `positions_off` training path. The EV scorer
   appends `{"ball_landing_zone": zone}` hypotheticals — this train/inference
   mismatch must be addressed when real training data is assembled.
```

with:

```markdown
2. **WinProb `positions_known` flag — RESOLVED (step 10).** `RallyTokenizer.encode_shot`
   now sets `positions_known` per-shot: a shot lacking `striker_position_m` gets
   flag 0.0 + zeroed positions even in the default path. The EV scorer's appended
   `{"ball_landing_zone": zone}` hypotheticals are correctly flagged unknown.
```

- [ ] **Step 3: Update the memory to reflect the fix**

Overwrite `/Users/azhou/.claude/projects/-Users-azhou-Coding-court-sight/memory/winprob-predict-positions-known-flag.md` body so it documents the resolution (keep the frontmatter; update `description` and body):

```markdown
---
name: winprob-predict-positions-known-flag
description: WinProbModel positions_known is now per-shot — sparse dicts (only ball_landing_zone) correctly flag 0.0 (resolved step 10)
metadata:
  node_type: memory
  type: project
---

`src/models/win_prob/model.py` `RallyTokenizer.encode_shot` sets `positions_known`
**per shot**: in the default (`positions_off=False`) path, a shot that lacks
`striker_position_m` (e.g. the EV scorer's appended `{"ball_landing_zone": zone}`
hypothetical) gets zeroed positional floats AND flag `0.0`; a shot that carries
positions gets real floats + flag `1.0`. `positions_off=True` still forces all
shots unknown (pretrain regime).

This resolves the earlier train/inference mismatch where every default-path shot
was flagged `1.0` regardless of whether it carried positions. Relevant to
[[step-10-ev-surface]] (the consumer) and the EV framework. See also
[[shotrecord-contact-vs-landing]].
```

- [ ] **Step 4: Run the entire test suite**

Run: `python -m pytest -q`
Expected: all tests PASS, 0 failures (prior suite + the new win-prob and EV-scorer tests).

- [ ] **Step 5: Commit**

```bash
git add docs/architecture/08_ev_surface.md docs/PROGRESS_LEDGER.md
git commit -m "$(cat <<'EOF'
Update EV surface docs + ledger; mark positions_known trap resolved

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

(The memory file lives outside the repo and is not committed.)

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** Task 1 → §5 (positions_known fix). Task 2 → §3/§4 `compute_ev_surface` + `predict_zone_neutrals`. Task 3 → §4 `score_shot` + all-9 `next_neutrals` output schema. Task 4 → §1/§5/§7 docs, ledger, memory. §6 testing is covered across Tasks 1–3.
- **Signature consistency:** `compute_ev_surface(base_safety_features, rally_context, dp)` is defined in Task 2 and called with those args in Task 3's `score_shot`. `predict_zone_neutrals(base_response_features, striker_pos, striker_vel, t_recovery)` likewise. `score_shot` gains `base_response_features` (Task 3) — any future caller must pass it.
- **No real caller yet:** `EVScorer` is still not wired into an inference entry point (no `ShotRecord → EVScorer` driver exists — there's no data). That driver belongs to a later step; not in scope here. Flagging so its absence is a conscious deferral, not an omission.
- **`displacement_penalty` is exercised for real** in Task 3's tests (via `_recovery_state()`), not faked — if its signature or numerics differ from expectation, that surfaces here. The three *models* are faked; the penalty function is real.
```
