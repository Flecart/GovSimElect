#!/usr/bin/env python3
"""Summarize a fishing experiment run from log_env.json.

This script computes the GovSim metrics from Section 2.4 of
https://arxiv.org/html/2404.16698v2 and prints a concise run summary.

Cross-run helpers used by ``aggregate_fishing_experiments.py``:

- Bootstrap percentile CI for pooled survival rate ``q``, resampling **runs** with
  replacement (primary for reporting when runs are the exchangeable units).
- Bootstrap percentile CIs for the mean of per-run metrics (survival months, R,
  efficiency, equality, over-usage).
- Optional reference-only Wilson / IID-trial SE helpers for ``q`` (assumptions often
  violated; see aggregate JSON notes).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np


LIMIT_PATTERN = re.compile(r"Detected negotiated resource limit:\s*([0-9]+(?:\.[0-9]+)?)")


def full_capacity_sustainable_total_extraction(capacity: float, r_t: float) -> float:
    """Max lake-wide catch in one round from a full pool so one regeneration returns to capacity.

    One round: harvest from stock ``capacity``, then multiply by ``r_t``; to land back on
    ``capacity`` requires ``(capacity - catch) * r_t = capacity`` i.e.
    ``catch = capacity - capacity / r_t = capacity * (1 - 1/r_t)``.

    For ``capacity = 100``: ``r_t = 1.5`` → ``100/3``; ``r_t = 2`` → ``50``; ``r_t = 2.5`` → ``60``.
    """
    if capacity <= 0 or r_t <= 0:
        return 0.0
    return max(0.0, capacity - capacity / r_t)


def approximate_se_pooled_survival_q(q: float, *, k: int, max_rounds: int) -> float | None:
    """Reference-only SE for q under an IID agent-month model (optimistic if violated).

    Uses SE(q) ≈ sqrt(q(1-q) / (K * max_rounds)), i.e. independent Bernoulli trials
    per agent-month. Within-run months are usually correlated; this understates SE
    when stated as uncertainty for q — prefer ``bootstrap_ci_pooled_survival_q``.
    """
    trials = k * max_rounds
    if trials <= 0 or q < 0 or q > 1:
        return None
    return math.sqrt(max(0.0, q * (1.0 - q) / float(trials)))


def wilson_score_interval_for_binomial(
    successes: int, trials: int, *, alpha: float = 0.05
) -> tuple[float | None, float | None]:
    """Wilson score CI for binomial proportion (successes out of trials)."""
    if trials <= 0 or successes < 0 or successes > trials:
        return None, None
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z2 = z * z
    phat = successes / trials
    denom = 1.0 + z2 / trials
    center = (phat + z2 / (2.0 * trials)) / denom
    inner = phat * (1.0 - phat) / trials + z2 / (4.0 * trials * trials)
    half = z * math.sqrt(max(0.0, inner)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def bootstrap_ci_pooled_survival_q(
    survival_rounds_per_run: Sequence[float],
    *,
    max_rounds: int,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap percentile CI for pooled q = sum_k n_k / (K * max_rounds).

    Resamples **runs** with replacement (each replicate uses K draws, same K as
    observed). For each replicate, q* = sum(n*) / (K * max_rounds).
    """
    arr = np.asarray(list(survival_rounds_per_run), dtype=float)
    k = int(arr.size)
    n_boot = int(n_boot)
    if k <= 0 or max_rounds <= 0:
        return {
            "q": None,
            "ci_low": None,
            "ci_high": None,
            "alpha": alpha,
            "n_bootstrap": n_boot,
            "rng_seed": rng_seed,
            "n_runs": k,
            "max_rounds": max_rounds,
        }
    denom = float(k * max_rounds)
    q_obs = float(arr.sum() / denom)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    q_boot = arr[idx].sum(axis=1) / denom
    pct_lo = 100.0 * alpha / 2.0
    pct_hi = 100.0 * (1.0 - alpha / 2.0)
    lo, hi = np.percentile(q_boot, [pct_lo, pct_hi])
    return {
        "q": q_obs,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "alpha": alpha,
        "n_bootstrap": n_boot,
        "rng_seed": rng_seed,
        "n_runs": k,
        "max_rounds": max_rounds,
        "procedure": "Resample runs with replacement; q* = sum(survival_rounds_n) / (K*max_rounds).",
    }


def bootstrap_ci_mean(
    values: Sequence[float],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap percentile CI for the mean (resample runs with replacement)."""
    arr = np.asarray(list(values), dtype=float)
    n_boot = int(n_boot)
    if arr.size == 0:
        return {
            "mean": None,
            "ci_low": None,
            "ci_high": None,
            "alpha": alpha,
            "n_bootstrap": n_boot,
            "rng_seed": rng_seed,
            "n_values": 0,
        }
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    pct_lo = 100.0 * alpha / 2.0
    pct_hi = 100.0 * (1.0 - alpha / 2.0)
    lo, hi = np.percentile(means, [pct_lo, pct_hi])
    return {
        "mean": float(arr.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "alpha": alpha,
        "n_bootstrap": n_boot,
        "rng_seed": rng_seed,
        "n_values": int(arr.size),
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def gini(values: list[float]) -> float:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return 0.0
    clean.sort()
    n = len(clean)
    total = sum(clean)
    if total <= 0:
        return 0.0
    num = 0.0
    for i, x in enumerate(clean, start=1):
        num += (2 * i - n - 1) * x
    return num / (n * total)


def parse_log(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def summarize(
    records: list[dict[str, Any]],
    *,
    capacity: float,
    expected_regen: float,
    collapse_threshold: float,
    verbose: bool = False,
) -> dict[str, Any]:
    harvesting_rows = [
        r
        for r in records
        if r.get("action") == "harvesting"
        and isinstance(r.get("agent_id"), str)
        and str(r.get("agent_id")).startswith("persona_")
    ]
    regen_rows = [r for r in records if r.get("action") == "regen"]
    utterance_rows = [r for r in records if r.get("action") == "utterance"]

    harvest_by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in harvesting_rows:
        round_id = int(row.get("round", -1))
        harvest_by_round[round_id].append(row)

    per_agent_gain: dict[str, float] = defaultdict(float)
    per_round_stock: dict[int, float] = {}
    per_round_total_catch: dict[int, float] = {}
    per_round_limit_hits: dict[int, float] = {}
    regen_by_harvest_round: dict[int, float] = {}
    over_usage_count = 0

    # Regen rows are logged at round (harvest_round + 1).
    for row in regen_rows:
        regen_round = int(row.get("round", -1))
        realized = _safe_float(row.get("realized_r_t"))
        harvest_round = regen_round - 1
        if harvest_round >= 0 and realized is not None and realized > 0:
            regen_by_harvest_round[harvest_round] = realized

    for round_id, rows in sorted(harvest_by_round.items()):
        catches = []
        for row in rows:
            agent_id = str(row["agent_id"])
            c = _safe_float(row.get("resource_collected")) or 0.0
            catches.append(c)
            per_agent_gain[agent_id] += c

        total_catch = sum(catches)
        per_round_total_catch[round_id] = total_catch

        stock_before = _safe_float(rows[0].get("resource_in_pool_before_harvesting"))
        if stock_before is not None:
            per_round_stock[round_id] = stock_before
            # Use realized regen for this specific harvest round when available.
            r_t = regen_by_harvest_round.get(round_id, expected_regen)
            threshold = max(0.0, stock_before - stock_before / r_t) if r_t > 0 else 0.0
            if total_catch > threshold:
                over_usage_count += len(catches)
            if verbose:
                cap_bench = full_capacity_sustainable_total_extraction(capacity, r_t)
                print(
                    f"Round {round_id}: stock_before = {stock_before}, "
                    f"r_t = {r_t}, threshold = {threshold}, "
                    f"full_pool_benchmark = {cap_bench}, catches = {catches}"
                )

        # If framework extracted a numeric resource_limit in this round,
        # keep the latest detected one.
        for msg in utterance_rows:
            if int(msg.get("round", -1)) != round_id:
                continue
            if str(msg.get("agent_id")) != "framework":
                continue
            utterance = str(msg.get("utterance") or "")
            m = LIMIT_PATTERN.search(utterance)
            if m:
                per_round_limit_hits[round_id] = float(m.group(1))

    # Survival time m = max{t | h(t) > C}
    # We approximate h(t) as stock before harvesting each round.
    survived_rounds = [r for r, stock in per_round_stock.items() if stock > collapse_threshold]
    survival_time = (max(survived_rounds) + 1) if survived_rounds else 0

    total_gain = sum(per_agent_gain.values())
    rounds_played = (max(per_round_total_catch.keys()) + 1) if per_round_total_catch else 0

    # Efficiency baseline: full-pool sustainable yield per round, H - H/r_t (only H and r_t).
    # Stock-based f_t = h - h/r is still recorded below for diagnostics / over-usage threshold.
    per_round_sustainable_capacity: dict[int, float] = {}
    per_round_full_pool_yield: dict[int, float] = {}
    max_possible_full_pool = 0.0
    for r, stock_before in per_round_stock.items():
        r_t = regen_by_harvest_round.get(r, expected_regen)
        g_t = full_capacity_sustainable_total_extraction(capacity, r_t)
        per_round_full_pool_yield[r] = g_t
        max_possible_full_pool += g_t
        if r_t <= 0:
            f_t = 0.0
        else:
            f_t = max(0.0, stock_before - stock_before / r_t)
        per_round_sustainable_capacity[r] = f_t

    if max_possible_full_pool > 0:
        efficiency = 1.0 - max(0.0, max_possible_full_pool - total_gain) / max_possible_full_pool
    else:
        efficiency = 0.0

    # e = 1 - Gini(total agent gains)
    equality = 1.0 - gini(list(per_agent_gain.values()))

    num_agents = len(per_agent_gain)

    # o = (# agent-harvests in rounds where aggregate catch exceeds stock-based MSY) / (|I| * m)
    denom_ou = num_agents * max(survival_time, 1)
    over_usage = (over_usage_count / denom_ou) if denom_ou > 0 else 0.0

    # Optional: total gain vs sum of full-pool benchmarks over survived months (not paper o).
    over_usage_denominator = 0.0
    for t in range(survival_time):
        r_t = regen_by_harvest_round.get(t, expected_regen)
        over_usage_denominator += full_capacity_sustainable_total_extraction(capacity, r_t)
    gain_over_full_pool_survival_baseline = (
        (total_gain / over_usage_denominator) if over_usage_denominator > 0 else 0.0
    )

    # Consistency: how often actual average catch exceeded negotiated limit.
    negotiated_limit_violations = 0
    rounds_with_detected_limit = 0
    for r, lim in per_round_limit_hits.items():
        rows = harvest_by_round.get(r, [])
        if not rows:
            continue
        rounds_with_detected_limit += 1
        for row in rows:
            c = _safe_float(row.get("resource_collected")) or 0.0
            if c > lim + 1e-9:
                negotiated_limit_violations += 1

    action_counts = Counter(str(r.get("action")) for r in records)
    regen_factors = [
        _safe_float(r.get("realized_r_t"))
        for r in regen_rows
        if _safe_float(r.get("realized_r_t")) is not None
    ]

    return {
        "num_records": len(records),
        "num_agents": num_agents,
        "rounds_played": rounds_played,
        "actions": dict(action_counts),
        "paper_metrics": {
            "survival_time_m": survival_time,
            "total_gain_R": total_gain,
            "efficiency_u": efficiency,
            "equality_e": equality,
            "over_usage_o": over_usage,
            "gain_R_over_full_pool_survival_baseline": gain_over_full_pool_survival_baseline,
        },
        "agent_total_gain": dict(sorted(per_agent_gain.items())),
        "round_stock_before_harvest": dict(sorted(per_round_stock.items())),
        "round_total_extraction": dict(sorted(per_round_total_catch.items())),
        "round_detected_resource_limit": dict(sorted(per_round_limit_hits.items())),
        "round_realized_regen_for_harvest": dict(sorted(regen_by_harvest_round.items())),
        "round_sustainable_capacity_f_t": dict(
            sorted(per_round_sustainable_capacity.items())
        ),
        "round_full_pool_regen_yield_H_minus_H_over_r": dict(
            sorted(per_round_full_pool_yield.items())
        ),
        "over_usage_capacity_regen_denominator": over_usage_denominator,
        "consistency_checks": {
            "rounds_with_detected_limit_and_harvest": rounds_with_detected_limit,
            "negotiated_limit_violations": negotiated_limit_violations,
        },
        "regen_summary": {
            "num_regen_events": len(regen_rows),
            "realized_r_t_values": regen_factors,
            "realized_r_t_min": min(regen_factors) if regen_factors else None,
            "realized_r_t_max": max(regen_factors) if regen_factors else None,
        },
        "assumptions": {
            "capacity": capacity,
            "expected_regen_fallback_for_missing_r_t": expected_regen,
            "over_usage_o_formula": (
                "o = (# harvesting rows in rounds where sum_i catch_i > h(t)-h(t)/r_t) "
                "/ (|I| * max(m,1)); m = survival_time_m"
            ),
            "gain_R_over_full_pool_survival_baseline_note": (
                "total_gain_R / sum_{t=0}^{m-1} (H - H/r_t) over survived months only; "
                "not the same as over_usage_o"
            ),
            "collapse_threshold_C": collapse_threshold,
            "efficiency_formula": (
                "u = 1 - max(0, sum_t g_t - total_gain_R) / sum_t g_t, "
                "g_t = H - H/r_t at carrying capacity H (regen-only), same t as harvest rounds "
                "with logged stock"
            ),
        },
    }


def print_human_summary(report: dict[str, Any], source: Path) -> None:
    m = report["paper_metrics"]
    print(f"# Experiment summary for {source}")
    print()
    print("## Core run facts")
    print(f"- records: {report['num_records']}")
    print(f"- agents: {report['num_agents']}")
    print(f"- rounds played: {report['rounds_played']}")
    print(f"- actions: {report['actions']}")
    print()
    print("## GovSim metrics (paper Section 2.4)")
    print(f"- survival_time_m: {m['survival_time_m']}")
    print(f"- total_gain_R: {m['total_gain_R']:.2f}")
    print(f"- efficiency_u: {m['efficiency_u']:.4f}")
    print(f"- equality_e: {m['equality_e']:.4f}")
    print(f"- over_usage_o: {m['over_usage_o']:.4f}")
    print(
        "- gain_R_over_full_pool_survival_baseline: "
        f"{m.get('gain_R_over_full_pool_survival_baseline', 0.0):.4f}"
    )
    print()
    print("## Agent gains")
    for agent, gain in report["agent_total_gain"].items():
        print(f"- {agent}: {gain:.2f}")
    print()
    print("## Consistency checks")
    cc = report["consistency_checks"]
    print(
        "- detected-limit rounds with harvest data: "
        f"{cc['rounds_with_detected_limit_and_harvest']}"
    )
    print(f"- per-agent negotiated-limit violations: {cc['negotiated_limit_violations']}")
    print()
    print("## Regeneration")
    rs = report["regen_summary"]
    print(f"- regen events: {rs['num_regen_events']}")
    print(f"- realized_r_t min/max: {rs['realized_r_t_min']} / {rs['realized_r_t_max']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a fishing experiment from log_env.json"
    )
    parser.add_argument("log_path", type=Path, help="Path to log_env.json")
    parser.add_argument(
        "--capacity",
        type=float,
        default=100.0,
        help="Lake carrying capacity (default: 100)",
    )
    parser.add_argument(
        "--expected-regen",
        type=float,
        default=2.0,
        help="Expected regen factor used for sustainability thresholds (default: 2.0)",
    )
    parser.add_argument(
        "--collapse-threshold",
        type=float,
        default=0.0,
        help="Collapse threshold C for survival metric (default: 0.0)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save the full summary JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-round stock / r_t / threshold / catches while summarizing",
    )
    args = parser.parse_args()

    records = parse_log(args.log_path)
    report = summarize(
        records,
        capacity=args.capacity,
        expected_regen=args.expected_regen,
        collapse_threshold=args.collapse_threshold,
        verbose=args.verbose,
    )
    print_human_summary(report, args.log_path)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print()
        print(f"Wrote JSON summary to: {args.json_out}")


if __name__ == "__main__":
    main()
