#!/usr/bin/env python3
"""Discover fishing experiment run folders, cache per-run metrics JSON, aggregate stats.

Each run directory is expected to contain ``log_env.json``. This script can write
``fishing_experiment_metrics.json`` (full ``summarize_fishing_log.summarize`` output
plus derived fields), then aggregate across runs.

Survival reporting (per user spec):
  - Per run: ``survival_rounds_n`` equals paper metric ``survival_time_m`` (rounds
    with stock above collapse threshold, as computed by summarize_fishing_log).
  - Aggregate reporting emphasizes bootstrap percentile CIs (resample runs with
    replacement) for mean survival months m, pooled survival rate q, paper metrics
    R/u/e/o mean across runs. Sample std across runs remains in JSON for appendix use.
  - Legacy fields remain for compatibility:
    ``survival_rounds_n / max_rounds`` and pooled ``sum(n)/(K*max_rounds)``.
  - Reference-only Wilson / IID-se formulas for q live under
    ``reference_only_intervals_under_strong_or_violated_assumptions`` (do not treat as
    primary uncertainty when runs are correlated).

Bootstrap helpers (including ``bootstrap_ci_pooled_survival_q``) live in
``summarize_fishing_log.py``.

Regeneration factor R:
  - Per run: mean of realized ``r_t`` over regen events (round-averaged in log space).
  - Across runs: mean and (sample) standard deviation of those per-run means.

Usage:
  python scripts/aggregate_fishing_experiments.py /path/to/parent \\
      --output /path/to/aggregate.json

  python scripts/aggregate_fishing_experiments.py /path/to/parent --recompute
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
METRICS_BASENAME = "fishing_experiment_metrics.json"
LOG_BASENAME = "log_env.json"


def _load_summarize_module():
    script_path = SCRIPTS_DIR / "summarize_fishing_log.py"
    spec = importlib.util.spec_from_file_location("govsim_summarize_fishing_log", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load summarizer from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_run_dirs(root: Path) -> list[Path]:
    """If root contains log_env.json, treat root as a single run; else use subdirs that have it."""
    log_here = root / LOG_BASENAME
    if log_here.is_file():
        return [root.resolve()]
    runs: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / LOG_BASENAME).is_file():
            runs.append(child.resolve())
    return runs


def _safe_mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _safe_stdev(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(clean) < 2:
        return 0.0 if len(clean) == 1 else None
    return float(statistics.stdev(clean))


def build_experiment_metrics(
    mod: Any,
    run_dir: Path,
    *,
    capacity: float,
    expected_regen: float,
    collapse_threshold: float,
    max_rounds: int,
    verbose: bool,
) -> dict[str, Any]:
    log_path = run_dir / LOG_BASENAME
    records = mod.parse_log(log_path)
    report = mod.summarize(
        records,
        capacity=capacity,
        expected_regen=expected_regen,
        collapse_threshold=collapse_threshold,
        verbose=verbose,
    )
    r_vals = report.get("regen_summary", {}).get("realized_r_t_values") or []
    numeric_r = [float(x) for x in r_vals if x is not None]
    mean_r = float(sum(numeric_r) / len(numeric_r)) if numeric_r else None

    n_survive = int(report.get("paper_metrics", {}).get("survival_time_m", 0))
    denom = float(max_rounds) if max_rounds > 0 else 1.0
    survival_rate = float(n_survive) / denom

    derived = {
        "max_rounds": max_rounds,
        "survival_rounds_n": n_survive,
        "survival_rate_n_over_max_rounds": survival_rate,
        "mean_realized_r_t_over_rounds": mean_r,
        "num_regen_samples_for_r": len(numeric_r),
    }
    return {
        "run_dir": str(run_dir),
        "log_env": str(log_path),
        "derived": derived,
        "summarize_report": report,
    }


def metrics_needs_recompute(metrics_path: Path, log_path: Path) -> bool:
    if not metrics_path.is_file():
        return True
    try:
        return os.path.getmtime(metrics_path) < os.path.getmtime(log_path)
    except OSError:
        return True


def ensure_metrics_for_run(
    mod: Any,
    run_dir: Path,
    *,
    capacity: float,
    expected_regen: float,
    collapse_threshold: float,
    max_rounds: int,
    recompute: bool,
    verbose: bool,
) -> dict[str, Any]:
    log_path = run_dir / LOG_BASENAME
    metrics_path = run_dir / METRICS_BASENAME
    if not recompute and not metrics_needs_recompute(metrics_path, log_path):
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    payload = build_experiment_metrics(
        mod,
        run_dir,
        capacity=capacity,
        expected_regen=expected_regen,
        collapse_threshold=collapse_threshold,
        max_rounds=max_rounds,
        verbose=verbose,
    )
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return payload


@dataclass
class RunAggRow:
    run_dir: str
    survival_rounds_n: int
    survival_rate: float
    mean_realized_r_t: float | None
    total_gain_R: float
    efficiency_u: float
    equality_e: float
    over_usage_o: float


def collect_rows(payloads: Iterable[dict[str, Any]], max_rounds: int) -> list[RunAggRow]:
    rows: list[RunAggRow] = []
    for p in payloads:
        d = p.get("derived") or {}
        pm = (p.get("summarize_report") or {}).get("paper_metrics") or {}
        n = int(d.get("survival_rounds_n", pm.get("survival_time_m", 0)))
        rate = float(d.get("survival_rate_n_over_max_rounds", n / max(1, max_rounds)))
        mean_r = d.get("mean_realized_r_t_over_rounds")
        mean_r_f = float(mean_r) if mean_r is not None else None
        rows.append(
            RunAggRow(
                run_dir=str(p.get("run_dir", "")),
                survival_rounds_n=n,
                survival_rate=rate,
                mean_realized_r_t=mean_r_f,
                total_gain_R=float(pm.get("total_gain_R", 0.0)),
                efficiency_u=float(pm.get("efficiency_u", 0.0)),
                equality_e=float(pm.get("equality_e", 0.0)),
                over_usage_o=float(pm.get("over_usage_o", 0.0)),
            )
        )
    return rows


def aggregate_payload(
    mod: Any,
    rows: list[RunAggRow],
    *,
    max_rounds: int,
    n_boot: int = 10_000,
    bootstrap_seed: int = 0,
    bootstrap_alpha: float = 0.05,
    wilson_alpha: float = 0.05,
    decomposition_cutoff_month: int | None = None,
) -> dict[str, Any]:
    k = len(rows)
    sum_n = sum(r.survival_rounds_n for r in rows)
    trials = k * max_rounds
    pooled_survival = float(sum_n) / float(trials) if k > 0 and max_rounds > 0 else None
    if k > 0 and max_rounds > 0 and trials > 0:
        q_point = float(pooled_survival) if pooled_survival is not None else 0.0
        se_q_iid_reference = mod.approximate_se_pooled_survival_q(
            q_point, k=k, max_rounds=max_rounds
        )
        w_lo, w_hi = mod.wilson_score_interval_for_binomial(
            sum_n, trials, alpha=wilson_alpha
        )
    else:
        se_q_iid_reference = None
        w_lo, w_hi = None, None

    rates = [r.survival_rate for r in rows]
    ns = [float(r.survival_rounds_n) for r in rows]
    mean_ns = _safe_mean(ns)
    r_means = [r.mean_realized_r_t for r in rows if r.mean_realized_r_t is not None]

    gains = [r.total_gain_R for r in rows]
    boot_m = mod.bootstrap_ci_mean(
        ns,
        n_boot=n_boot,
        alpha=bootstrap_alpha,
        rng_seed=bootstrap_seed,
    )
    boot_r = mod.bootstrap_ci_mean(
        gains,
        n_boot=n_boot,
        alpha=bootstrap_alpha,
        rng_seed=bootstrap_seed + 1,
    )
    boot_q = mod.bootstrap_ci_pooled_survival_q(
        ns,
        max_rounds=max_rounds,
        n_boot=n_boot,
        alpha=bootstrap_alpha,
        rng_seed=bootstrap_seed + 2,
    )
    us = [r.efficiency_u for r in rows]
    es = [r.equality_e for r in rows]
    os = [r.over_usage_o for r in rows]
    boot_u = mod.bootstrap_ci_mean(
        us,
        n_boot=n_boot,
        alpha=bootstrap_alpha,
        rng_seed=bootstrap_seed + 3,
    )
    boot_e = mod.bootstrap_ci_mean(
        es,
        n_boot=n_boot,
        alpha=bootstrap_alpha,
        rng_seed=bootstrap_seed + 4,
    )
    boot_o = mod.bootstrap_ci_mean(
        os,
        n_boot=n_boot,
        alpha=bootstrap_alpha,
        rng_seed=bootstrap_seed + 5,
    )

    at_full = sum(1 for r in rows if r.survival_rounds_n >= max_rounds)
    below_full = k - at_full
    decomp: dict[str, Any] = {
        "runs_reaching_full_horizon_survival_n_ge_max_rounds": at_full,
        "runs_below_full_horizon_survival_n_lt_max_rounds": below_full,
        "per_run_survival_rounds_n": [r.survival_rounds_n for r in rows],
    }
    if decomposition_cutoff_month is not None:
        cu = int(decomposition_cutoff_month)
        decomp["early_collapse_before_month_cutoff"] = {
            "cutoff_month": cu,
            "interpretation": (
                "survival_rounds_n strictly less than cutoff counts as collapsed "
                "before completing that month (paper survival_time_m is months survived)."
            ),
            "runs_collapsed_strictly_before_cutoff_survival_n_lt_cutoff": sum(
                1 for r in rows if r.survival_rounds_n < cu
            ),
            "runs_not_collapsed_before_cutoff_survival_n_ge_cutoff": sum(
                1 for r in rows if r.survival_rounds_n >= cu
            ),
        }

    return {
        "num_runs": k,
        "max_rounds": max_rounds,
        "pooled_survival_rate_sum_n_over_k_max_rounds": pooled_survival,
        "pooled_survival_numerator_sum_survival_rounds_n": sum_n,
        "pooled_survival_denominator_k_times_max_rounds": trials,
        "pooled_survival_q_statistics": {
            "q": pooled_survival,
            "bootstrap_ci_q_resample_runs": boot_q,
            "q_equals_mean_survival_months_over_max_rounds": (
                (float(mean_ns) / float(max_rounds))
                if mean_ns is not None and max_rounds > 0
                else None
            ),
            "reference_only_intervals_under_strong_or_violated_assumptions": {
                "wilson_on_pooled_binomial_K_times_max_rounds_trials": {
                    "alpha": wilson_alpha,
                    "ci_low": w_lo,
                    "ci_high": w_hi,
                    "note": (
                        "Wilson treats sum(n) successes as binomial over K*max_rounds "
                        "trials; independence of agent-months is typically violated."
                    ),
                },
                "optimistic_se_if_agent_months_were_iid": {
                    "value": se_q_iid_reference,
                    "note": (
                        "sqrt(q(1-q)/(K*max_rounds)) is a lower bound on uncertainty "
                        "when months within a run are correlated (e.g. early death vs "
                        "full horizon). Prefer bootstrap_ci_q_resample_runs for primary "
                        "inference."
                    ),
                },
            },
        },
        "survival_months_m": {
            "mean": _safe_mean(ns),
            "std": _safe_stdev(ns),
            "values": [r.survival_rounds_n for r in rows],
            "bootstrap_ci_mean_across_runs": boot_m,
        },
        "survival_rounds_n": {
            "mean": _safe_mean(ns),
            "std": _safe_stdev(ns),
            "values": [r.survival_rounds_n for r in rows],
        },
        "survival_rate_per_run_n_over_max_rounds": {
            "mean": _safe_mean(rates),
            "std": _safe_stdev(rates),
            "values": rates,
        },
        "mean_realized_r_t_per_run": {
            "mean": _safe_mean([float(x) for x in r_means]),
            "std": _safe_stdev([float(x) for x in r_means]),
            "values": r_means,
        },
        "survival_outcome_decomposition": decomp,
        "bootstrap_settings": {
            "n_bootstrap": n_boot,
            "alpha": bootstrap_alpha,
            "rng_seed_base": bootstrap_seed,
        },
        "paper_metrics_across_runs": {
            "total_gain_R": {
                "mean": _safe_mean(gains),
                "std": _safe_stdev(gains),
                "bootstrap_ci_mean_across_runs": boot_r,
            },
            "efficiency_u": {
                "mean": _safe_mean(us),
                "std": _safe_stdev(us),
                "bootstrap_ci_mean_across_runs": boot_u,
            },
            "equality_e": {
                "mean": _safe_mean(es),
                "std": _safe_stdev(es),
                "bootstrap_ci_mean_across_runs": boot_e,
            },
            "over_usage_o": {
                "mean": _safe_mean(os),
                "std": _safe_stdev(os),
                "bootstrap_ci_mean_across_runs": boot_o,
            },
        },
        "runs": [
            {
                "run_dir": r.run_dir,
                "survival_rounds_n": r.survival_rounds_n,
                "survival_rate_n_over_max_rounds": r.survival_rate,
                "mean_realized_r_t_over_rounds": r.mean_realized_r_t,
                "total_gain_R": r.total_gain_R,
                "efficiency_u": r.efficiency_u,
                "equality_e": r.equality_e,
                "over_usage_o": r.over_usage_o,
            }
            for r in rows
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate fishing experiment metrics from log_env.json under a parent directory."
    )
    parser.add_argument(
        "parent_dir",
        type=Path,
        help="Directory containing experiment subfolders (each with log_env.json), "
        "or a single run directory with log_env.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write aggregate JSON to this path (default: print to stdout only)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help=f"Always rebuild {METRICS_BASENAME} from log_env.json",
    )
    parser.add_argument("--capacity", type=float, default=100.0)
    parser.add_argument("--expected-regen", type=float, default=2.0)
    parser.add_argument("--collapse-threshold", type=float, default=0.0)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=12,
        help="Denominator for survival rate and pooled survival (default: 12)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Per-round debug prints from summarize (very noisy)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Resamples for bootstrap CIs on mean survival months and total gain R",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--bootstrap-alpha",
        type=float,
        default=0.05,
        help="Alpha for bootstrap percentile intervals (default: 0.05 → 95%% CI)",
    )
    parser.add_argument(
        "--wilson-alpha",
        type=float,
        default=0.05,
        help="Alpha for Wilson score interval on pooled q (default: 0.05)",
    )
    parser.add_argument(
        "--decomposition-cutoff-month",
        type=int,
        default=None,
        metavar="M",
        help=(
            "Optional survival month cutoff for decomposition: count runs with "
            "survival_rounds_n < M vs ≥ M (e.g. 8 for collapsed before month 8)"
        ),
    )
    args = parser.parse_args(argv)

    root = args.parent_dir.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    runs = discover_run_dirs(root)
    if not runs:
        print(
            f"No {LOG_BASENAME} found under {root} (expected file here or in subdirs).",
            file=sys.stderr,
        )
        return 1

    try:
        mod = _load_summarize_module()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1

    payloads: list[dict[str, Any]] = []
    # runs = sorted(runs, key=lambda x: x.name)
    for run_dir in runs:
        try:
            p = ensure_metrics_for_run(
                mod,
                run_dir,
                capacity=args.capacity,
                expected_regen=args.expected_regen,
                collapse_threshold=args.collapse_threshold,
                max_rounds=args.max_rounds,
                recompute=args.recompute,
                verbose=args.verbose,
            )
            payloads.append(p)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"Skip {run_dir}: {e}", file=sys.stderr)

    if not payloads:
        print("No runs could be processed.", file=sys.stderr)
        return 1

    rows = collect_rows(payloads, args.max_rounds)
    agg = aggregate_payload(
        mod,
        rows,
        max_rounds=args.max_rounds,
        n_boot=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_alpha=args.bootstrap_alpha,
        wilson_alpha=args.wilson_alpha,
        decomposition_cutoff_month=args.decomposition_cutoff_month,
    )
    agg["parent_dir"] = str(root)
    agg["metrics_basename"] = METRICS_BASENAME

    # text = json.dumps(agg, indent=2) + "\n"
    # if args.output is not None:
    #     args.output.parent.mkdir(parents=True, exist_ok=True)
    #     args.output.write_text(text, encoding="utf-8")
    #     print(f"Wrote aggregate ({len(payloads)} run(s)) to {args.output}", file=sys.stderr)
    # else:
    #     print(text, end="")

    # Human-readable summary on stderr (bootstrap CIs primary; std only in JSON)
    sm = agg.get("survival_months_m") or {}
    pq = agg.get("pooled_survival_q_statistics") or {}

    print("\n--- Summary ---", file=sys.stderr)
    print(f"Directory: {root}", file=sys.stderr)

    m_mean = sm.get("mean")
    bsm = sm.get("bootstrap_ci_mean_across_runs") or {}
    if m_mean is not None and bsm.get("ci_low") is not None and bsm.get("ci_high") is not None:
        alpha_b = float(bsm.get("alpha", 0.05))
        print(
            f"Mean survival months m: {m_mean:.2f} "
            f"({(1.0 - alpha_b) * 100:.0f}% bootstrap CI for mean: "
            f"[{bsm['ci_low']:.2f}, {bsm['ci_high']:.2f}])",
            file=sys.stderr,
        )
    elif m_mean is not None:
        print(f"Mean survival months m: {m_mean:.2f}", file=sys.stderr)
    else:
        print("Mean survival months m: n/a", file=sys.stderr)

    bq = pq.get("bootstrap_ci_q_resample_runs") or {}
    q_pt = bq.get("q")
    if q_pt is not None and bq.get("ci_low") is not None and bq.get("ci_high") is not None:
        alpha_q = float(bq.get("alpha", 0.05))
        print(
            f"Pooled survival rate q: {q_pt:.4f} "
            f"({(1.0 - alpha_q) * 100:.0f}% bootstrap CI, resample runs: "
            f"[{bq['ci_low']:.4f}, {bq['ci_high']:.4f}])",
            file=sys.stderr,
        )
    else:
        print(f"Pooled survival rate q: {agg.get('pooled_survival_rate_sum_n_over_k_max_rounds')}", file=sys.stderr)

    pm = agg.get("paper_metrics_across_runs") or {}
    for key, label in (
        ("total_gain_R", "Total gain R"),
        ("efficiency_u", "Efficiency u"),
        ("equality_e", "Equality e"),
        ("over_usage_o", "Over-usage o"),
    ):
        block = pm.get(key) or {}
        mean_v = block.get("mean")
        boot = block.get("bootstrap_ci_mean_across_runs") or {}
        if mean_v is None:
            print(f"{label}: n/a", file=sys.stderr)
            continue
        if boot.get("ci_low") is not None and boot.get("ci_high") is not None:
            alpha_x = float(boot.get("alpha", 0.05))
            print(
                f"{label}: {mean_v:.4f} "
                f"({(1.0 - alpha_x) * 100:.0f}% bootstrap CI for mean: "
                f"[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}])",
                file=sys.stderr,
            )
        else:
            print(f"{label}: {mean_v:.4f}", file=sys.stderr)

    dd = agg.get("survival_outcome_decomposition") or {}
    print(
        "Survival decomposition: "
        f"full horizon (n ≥ max_rounds)={dd.get('runs_reaching_full_horizon_survival_n_ge_max_rounds', '?')}, "
        f"below horizon={dd.get('runs_below_full_horizon_survival_n_lt_max_rounds', '?')}",
        file=sys.stderr,
    )
    early = dd.get("early_collapse_before_month_cutoff")
    if isinstance(early, dict):
        print(
            f"  Before month {early.get('cutoff_month')}: "
            f"collapsed (n < cutoff)={early.get('runs_collapsed_strictly_before_cutoff_survival_n_lt_cutoff')}, "
            f"not collapsed (n ≥ cutoff)={early.get('runs_not_collapsed_before_cutoff_survival_n_ge_cutoff')}",
            file=sys.stderr,
        )

    bs = agg.get("bootstrap_settings") or {}
    if bs:
        print(
            f"Bootstrap params: n={bs.get('n_bootstrap')}, alpha={bs.get('alpha')}, "
            f"seed_base={bs.get('rng_seed_base')}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
