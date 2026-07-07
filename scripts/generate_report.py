"""
Generate a match feedback report from scored_shots.json.

Produces:
  - report.json   structured feedback (per-shot + aggregate patterns)
  - printed text summary to stdout

Usage:
    python -m scripts.generate_report <scored_shots.json> [--out report.json]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


# --- Aggregate helpers -------------------------------------------------------

def _group_by(shots: list[dict], key_fn) -> dict:
    groups = defaultdict(list)
    for s in shots:
        groups[key_fn(s)].append(s)
    return dict(groups)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _shot_type(rec: dict) -> str:
    from src.models.execution_prob.train import _MCP_TO_SHOT_TYPE
    return _MCP_TO_SHOT_TYPE.get(rec.get("shot_type_mcp", ""), "unknown")


def _scoring(rec: dict) -> dict:
    return rec.get("scoring") or {}


# --- Pattern analysis --------------------------------------------------------

def _zone_tendency(shots: list[dict]) -> dict:
    """Which zones the player targeted and how that compared to optimal."""
    by_zone = _group_by(shots, lambda s: _scoring(s).get("actual_zone", "unknown"))
    return {
        zone: {
            "count": len(group),
            "avg_zone_score": round(_avg([_scoring(s).get("zone_score", 0.0) for s in group]), 3),
            "avg_ev_loss": round(_avg([_scoring(s).get("ev_loss", 0.0) for s in group]), 3),
            "pct_best_choice": round(
                sum(1 for s in group if _scoring(s).get("actual_zone") == _scoring(s).get("best_zone"))
                / len(group), 3
            ),
        }
        for zone, group in sorted(by_zone.items())
    }


def _shot_type_breakdown(shots: list[dict]) -> dict:
    by_type = _group_by(shots, _shot_type)
    return {
        stype: {
            "count": len(group),
            "avg_zone_score": round(_avg([_scoring(s).get("zone_score", 0.0) for s in group]), 3),
            "avg_displacement_penalty": round(_avg([_scoring(s).get("displacement_penalty", 0.0) for s in group]), 3),
            "avg_p_make": round(_avg([_scoring(s).get("p_make_actual", 0.0) for s in group]), 3),
        }
        for stype, group in sorted(by_type.items())
    }


def _directional_tendency(shots: list[dict]) -> dict:
    """How often the player hits each direction vs. what zone had highest EV."""
    by_best = _group_by(shots, lambda s: _scoring(s).get("best_zone", "unknown"))
    chosen_counts: dict[str, int] = defaultdict(int)
    for s in shots:
        chosen_counts[_scoring(s).get("actual_zone", "unknown")] += 1
    optimal_counts = {z: len(g) for z, g in by_best.items()}
    return {
        "most_chosen": sorted(chosen_counts.items(), key=lambda x: -x[1])[:3],
        "most_optimal": sorted(optimal_counts.items(), key=lambda x: -x[1])[:3],
    }


def _pressure_performance(shots: list[dict], dp_threshold: float = 0.20) -> dict:
    """Zone selection quality when player is out of position vs. in position."""
    in_pos = [s for s in shots if _scoring(s).get("displacement_penalty", 0.0) <= dp_threshold]
    out_pos = [s for s in shots if _scoring(s).get("displacement_penalty", 0.0) > dp_threshold]
    return {
        "in_position": {
            "count": len(in_pos),
            "avg_zone_score": round(_avg([_scoring(s).get("zone_score", 0.0) for s in in_pos]), 3),
        },
        "under_pressure": {
            "count": len(out_pos),
            "avg_zone_score": round(_avg([_scoring(s).get("zone_score", 0.0) for s in out_pos]), 3),
        },
    }


# --- Report assembly ---------------------------------------------------------

def build_report(shots: list[dict]) -> dict:
    scored = [s for s in shots if s.get("scoring")]
    estimated = [s for s in scored if s.get("zone_estimated")]

    zone_scores = [_scoring(s).get("zone_score", 0.0) for s in scored]
    dp_values = [_scoring(s).get("displacement_penalty", 0.0) for s in scored]
    ev_losses = [_scoring(s).get("ev_loss", 0.0) for s in scored]

    by_zone_score = sorted(scored, key=lambda s: _scoring(s).get("zone_score", 0.0))

    return {
        "summary": {
            "shots_scored": len(scored),
            "shots_skipped": len(shots) - len(scored),
            "pct_zone_estimated": round(len(estimated) / len(scored), 3) if scored else 0.0,
            "avg_zone_score": round(_avg(zone_scores), 3),
            "avg_displacement_penalty": round(_avg(dp_values), 3),
            "avg_ev_loss": round(_avg(ev_losses), 3),
        },
        "patterns": {
            "zone_tendency": _zone_tendency(scored),
            "shot_type_breakdown": _shot_type_breakdown(scored),
            "directional_tendency": _directional_tendency(scored),
            "pressure_performance": _pressure_performance(scored),
        },
        "highlights": {
            "worst_shots": [_shot_summary(s) for s in by_zone_score[:5]],
            "best_shots": [_shot_summary(s) for s in by_zone_score[-5:][::-1]],
        },
        "per_shot": scored,
    }


def _shot_summary(rec: dict) -> dict:
    sc = _scoring(rec)
    return {
        "shot_id": rec.get("point_id"),
        "shot_type": _shot_type(rec),
        "actual_zone": sc.get("actual_zone"),
        "best_zone": sc.get("best_zone"),
        "zone_score": sc.get("zone_score"),
        "ev_loss": sc.get("ev_loss"),
        "displacement_penalty": sc.get("displacement_penalty"),
        "feedback_zone": sc.get("feedback_zone"),
        "feedback_position": sc.get("feedback_position"),
    }


# --- Text summary ------------------------------------------------------------

def _grade(avg_zone_score: float) -> str:
    if avg_zone_score >= 0.75: return "A"
    if avg_zone_score >= 0.60: return "B"
    if avg_zone_score >= 0.45: return "C"
    return "D"


def print_summary(report: dict) -> None:
    s = report["summary"]
    p = report["patterns"]

    print("\n=== Match Shot Quality Report ===\n")
    print(f"Shots scored:    {s['shots_scored']}  (skipped: {s['shots_skipped']}, "
          f"{s['pct_zone_estimated']:.0%} zone estimated)")
    print(f"Overall grade:   {_grade(s['avg_zone_score'])}  "
          f"(avg zone score {s['avg_zone_score']:.2f})")
    print(f"Avg EV loss:     {s['avg_ev_loss']:.3f}  "
          f"(avg displacement penalty: {s['avg_displacement_penalty']:.3f})")

    print("\n--- Shot type breakdown ---")
    for stype, stats in p["shot_type_breakdown"].items():
        print(f"  {stype:<30}  n={stats['count']:>3}  "
              f"zone={stats['avg_zone_score']:.2f}  "
              f"dp={stats['avg_displacement_penalty']:.2f}  "
              f"p_make={stats['avg_p_make']:.2f}")

    pp = p["pressure_performance"]
    print(f"\n--- Zone selection: in position vs. under pressure ---")
    print(f"  In position    (n={pp['in_position']['count']:>3}):  "
          f"avg zone score {pp['in_position']['avg_zone_score']:.2f}")
    print(f"  Under pressure (n={pp['under_pressure']['count']:>3}):  "
          f"avg zone score {pp['under_pressure']['avg_zone_score']:.2f}")

    dt = p["directional_tendency"]
    print(f"\n--- Directional tendency ---")
    print(f"  Most chosen:   {', '.join(f'{z}({n})' for z, n in dt['most_chosen'])}")
    print(f"  Most optimal:  {', '.join(f'{z}({n})' for z, n in dt['most_optimal'])}")

    print("\n--- Worst shots (by zone score) ---")
    for shot in report["highlights"]["worst_shots"]:
        print(f"  [{shot['shot_id']}] {shot['shot_type']}  "
              f"→ {shot['actual_zone']} (score={shot['zone_score']:.2f}, "
              f"best={shot['best_zone']}, ev_loss={shot['ev_loss']:.3f})")
        print(f"    {shot['feedback_zone']}")
        print(f"    {shot['feedback_position']}")

    print()


# --- Entry point -------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate match feedback report from scored shots.")
    ap.add_argument("scored", type=Path, help="scored_shots.json from score_shots.py")
    ap.add_argument("--out", type=Path, default=None, help="Output report.json path")
    args = ap.parse_args()

    shots = json.loads(args.scored.read_text())
    report = build_report(shots)
    print_summary(report)

    out_path = args.out or args.scored.parent / "report.json"
    # Exclude per_shot from the written JSON to keep it readable; link back to scored file
    report_slim = {k: v for k, v in report.items() if k != "per_shot"}
    report_slim["source"] = str(args.scored)
    out_path.write_text(json.dumps(report_slim, indent=2))
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
