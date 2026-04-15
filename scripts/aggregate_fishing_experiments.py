#!/usr/bin/env python3
"""Discover fishing experiment run folders, cache per-run metrics JSON, aggregate stats.

Each run directory is expected to contain ``log_env.json``. This script can write
``fishing_experiment_metrics.json`` (full ``summarize_fishing_log.summarize`` output
plus derived fields), then aggregate across runs.

Survival reporting (per user spec):
  - Per run: ``survival_rounds_n`` equals paper metric ``survival_time_m`` (rounds
    with stock above collapse threshold, as computed by summarize_fishing_log).
  - Aggregate reporting includes direct month-based survival statistics:
    average survival months and sample std across runs.
  - Legacy rate fields remain available for compatibility:
    ``survival_rounds_n / max_rounds`` and pooled ``sum(n)/(K*max_rounds)``.

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


def aggregate_payload(rows: list[RunAggRow], *, max_rounds: int) -> dict[str, Any]:
    k = len(rows)
    sum_n = sum(r.survival_rounds_n for r in rows)
    pooled_survival = float(sum_n) / float(k * max_rounds) if k > 0 and max_rounds > 0 else None

    rates = [r.survival_rate for r in rows]
    ns = [float(r.survival_rounds_n) for r in rows]
    r_means = [r.mean_realized_r_t for r in rows if r.mean_realized_r_t is not None]

    return {
        "num_runs": k,
        "max_rounds": max_rounds,
        "pooled_survival_rate_sum_n_over_k_max_rounds": pooled_survival,
        "pooled_survival_numerator_sum_survival_rounds_n": sum_n,
        "pooled_survival_denominator_k_times_max_rounds": k * max_rounds,
        "survival_months_m": {
            "mean": _safe_mean(ns),
            "std": _safe_stdev(ns),
            "values": [r.survival_rounds_n for r in rows],
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
        "paper_metrics_across_runs": {
            "total_gain_R": {
                "mean": _safe_mean([r.total_gain_R for r in rows]),
                "std": _safe_stdev([r.total_gain_R for r in rows]),
            },
            "efficiency_u": {
                "mean": _safe_mean([r.efficiency_u for r in rows]),
                "std": _safe_stdev([r.efficiency_u for r in rows]),
            },
            "equality_e": {
                "mean": _safe_mean([r.equality_e for r in rows]),
                "std": _safe_stdev([r.equality_e for r in rows]),
            },
            "over_usage_o": {
                "mean": _safe_mean([r.over_usage_o for r in rows]),
                "std": _safe_stdev([r.over_usage_o for r in rows]),
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
    agg = aggregate_payload(rows, max_rounds=args.max_rounds)
    agg["parent_dir"] = str(root)
    agg["metrics_basename"] = METRICS_BASENAME

    text = json.dumps(agg, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote aggregate ({len(payloads)} run(s)) to {args.output}", file=sys.stderr)
    else:
        print(text, end="")

    # Short human-readable lines on stderr when output file is set (or always as supplement)
    p = agg.get("pooled_survival_rate_sum_n_over_k_max_rounds")
    sm = agg.get("survival_months_m", {})
    print("\n--- Summary ---", file=sys.stderr)
    # print the directory name
    print(f"Directory: {root}", file=sys.stderr)
    print(
        f"Average survival months (m) mean ± std: {sm.get('mean')} ± {sm.get('std')}",
        file=sys.stderr,
    )
    print(
        f"Pooled survival rate sum(n)/(K*max_rounds): {p}",
        file=sys.stderr,
    )
    pm = agg.get("paper_metrics_across_runs") or {}
    for key, label in (
        ("total_gain_R", "Total gain R"),
        ("efficiency_u", "Efficiency u"),
        ("equality_e", "Equality e"),
        ("over_usage_o", "Over-usage o"),
    ):
        block = pm.get(key) or {}
        print(
            f"{label} mean ± std: {block.get('mean')} ± {block.get('std')}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
