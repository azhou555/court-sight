# Progress Ledger — Deferred Work & Cross-Step Integration

A living record of (1) what each step left undone, (2) how the steps connect, and
(3) known integration traps. Update this whenever a step defers work or a wiring
point changes. Last updated: 2026-06-09 (during step 10 brainstorming).

---

## ⚠️ Overarching blocker: no real data yet

`data/external/` is empty (only `.gitkeep`). Neither the raw Match Charting CSVs
nor any CV-processed match records exist locally. **Consequence:** every model
below is architecturally complete and unit-tested on *synthetic* data, but
**none is trained on real data**. All downstream numeric outputs (p_make, p_win,
EV surfaces, quality scores) are structurally correct but **not yet meaningful** —
they are scaffolding awaiting a real dataset. Sourcing/processing data is the
critical path that unblocks genuine evaluation of steps 7–10.

---

## Per-step deferred work

### Steps 1–6 (data pipeline + shot classifier) — status to re-verify
Components are scaffolded; an earlier note recorded "core CV inference
unimplemented" for parts of the pipeline. Before relying on end-to-end data flow,
verify: homography stability, player/ball tracking inference, point-boundary
detection, and the MCP alignment producing real `ShotRecord`s. (Not audited
during steps 7–10; flagged here so it isn't assumed done.)

### Step 7 — Shot Margin Safety (2B)
- Trained only on synthetic data (no real make/miss labels yet).

### Step 8 — Neutral Position (2E)
- **Stage-1 only** (response distribution + geometric median). Delivered.
- `ball_height_at_bounce` is a **placeholder (0.0)** — no ball-arc extraction exists.
- `opponent_displacement_from_neutral` uses a **geometric-center proxy**, not the
  true neutral (circular dependency: the model predicts neutral).
- Serve regime: `SERVE_NEUTRAL_DEFAULTS` lookup exists but is not yet wired into
  any inference path.
- Untrained on real data.

### Step 9 — Win Probability (2C)
- **Deferred:** separate MCP-only pretrain corpus loader; two-corpus
  pretrain→fine-tune checkpoint transfer (both blocked on data).
- **Deferred:** strictly-causal variant that masks the current shot's own
  `ball_landing_zone` (kept as documented "optimism").
- **Not implemented:** per-rally-position accuracy metric (optional polish).
- Untrained on real data.

### Step 10 — EV Surface (2D) — DONE
- `EVScorer` reworked into a tested combiner: `compute_ev_surface`,
  `predict_zone_neutrals`, `score_shot`. EV = 2B×(2C+1)−1−dp.
- 2E integration: neutral feeds the displacement penalty; all 9 per-zone
  neutrals are attached to per-shot output (not the EV math).
- Untrained (no real data) — EV values are placeholder scaffolding.
- **No `ShotRecord → EVScorer` inference driver yet** (no data) — belongs to a
  later wiring/inference step.
- **Deferred:** aggregate pattern analysis → application layer (step 11+);
  heatmap visualization → step 11.

### Step 11 — Feedback Visualization — not started
- Will own: heatmap viz, shot-level overlay, and the deferred aggregate pattern
  analysis from step 10.

---

## Cross-step integration map (what feeds what)

```
CV pipeline (1–4) ──┐
                    ├─► MCP alignment (5) ─► ShotRecord(s) ──┐
Match Charting ─────┘                                        │
                                                             ▼
                            ┌──────────────── ShotRecord feeds everything ───────────────┐
                            │                                                             │
   2B ShotGeometryFeatures ─┤                          2C rally_context (list[ShotRecord])│
   2E ResponseDistribution ─┘                          2E ResponseDistributionFeatures    │
                            │                                                             │
                            ▼                                                             ▼
        EV_adjusted(zone) = P(make│zone) × (P(win│zone)+1) − 1 − displacement_penalty   (Model 2D)
                              [2B]            [2C]                      [2E penalty side]
```

**Concrete API touchpoints (verified against current code):**
- 2B: `ShotSafetyModel.predict(ShotGeometryFeatures) -> {"p_make", "p_miss", ...}`
- 2C: `WinProbModel.predict(rally: list[dict]) -> {"p_win_point", "rally_context_used"}`
- 2E penalty: `execution_prob/displacement.py::displacement_penalty(contact, neutral, movement, shot_direction, ball_pos, shot_type)` — needs a **neutral position**.
- 2E neutral: `NeutralPositionModel.predict_neutral(features: ResponseDistributionFeatures, current_pos, current_vel, t_recovery, rng) -> {"reachable_neutral_m", "theoretical_neutral_m", "response_mu", ...}` — supplies that neutral position.

**2E's real role in 2D (decided in step-10 brainstorming):** 2E defines the
*neutral position* the displacement penalty measures against — NOT a `next_neutral`
fed into 2C. The win-prob token has no `next_neutral` slot. Additionally, 2D will
call `predict_neutral` per zone and **attach** the predicted neutral to per-shot
output for the step-11 feedback layer, without it entering the EV math.

---

## Known integration traps (also captured in agent memory)

1. **ShotRecord contact vs landing** — `ball_contact_position_m` = where a shot was
   *struck*; `ball_landing_zone` = where it *lands*. Don't conflate when pairing
   consecutive shots (bit the 2E label; relevant to 2C/2D).
2. **WinProb `positions_known` flag — RESOLVED (step 10).** `RallyTokenizer.encode_shot`
   now sets `positions_known` per-shot: a shot lacking `striker_position_m` gets
   flag 0.0 + zeroed positions even in the default path. The EV scorer's appended
   `{"ball_landing_zone": zone}` hypotheticals are correctly flagged unknown.

---

## How to use this file
- When a step defers work, add it under that step.
- When a new wiring point is created or changes, update the integration map.
- When a deferred item is completed, strike it (don't delete — keeps history).
