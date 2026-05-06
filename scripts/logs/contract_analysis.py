#!/usr/bin/env python3
"""Analyze contract renegotiation rate and violation rate across fishing runs.

Two arguments for the paper:

1. **Renegotiation gap**: when a contract (resource limit) exists, it is rarely
   changed — even when the agreed limit is *inefficient* (i.e., permits more
   total fishing than the sustainable cap f_t = max(0, h - h/r_t)).

2. **Violation rate**: even when agents know the negotiated cap, individual
   agents frequently exceed it at harvest time.

Usage
-----
# Analyze a single run directory (containing log_env.json):
    python scripts/logs/contract_analysis.py simulation/results/sto/gemma-4-31b-it-p1-*/2-nl-p1/2-nl-p1

# Analyze all nl runs across all sto experiments:
    python scripts/logs/contract_analysis.py "simulation/results/sto/*/2-nl-*/2-nl-*"

# Analyze a specific log file directly:
    python scripts/logs/contract_analysis.py path/to/log_env.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic models for log_env.json records
# ---------------------------------------------------------------------------

class _Base(BaseModel):
    agent_id: Optional[str] = None
    round: int
    action: str

    model_config = {"extra": "allow"}


class HarvestingRecord(_Base):
    action: Literal["harvesting"]
    resource_in_pool_before_harvesting: Optional[float] = None
    resource_in_pool_after_harvesting: Optional[float] = None
    concurrent_harvesting: Optional[bool] = None
    resource_collected: Optional[float] = None
    wanted_resource: Optional[float] = None
    commands: Optional[list[str]] = None
    contract_reward_adjustment: Optional[float] = None
    html_interactions: Optional[Any] = None


class RegenRecord(_Base):
    action: Literal["regen"]
    stock_before_extraction: Optional[float] = None
    total_extraction: Optional[float] = None
    realized_r_t: Optional[float] = None
    regen_mode: Optional[str] = None
    # endogenous_hysteresis extras
    regime: Optional[str] = None
    m_t: Optional[float] = None
    p_shift: Optional[float] = None
    p_recover: Optional[float] = None
    transitioned: Optional[bool] = None


class ConversationResourceLimitRecord(_Base):
    action: Literal["conversation_resource_limit"]
    resource_limit: Optional[float] = None
    html_interactions: Optional[str] = None


class UtteranceRecord(_Base):
    action: Literal["utterance"]
    utterance: Optional[str] = None
    html_interactions: Optional[Any] = None


class ConversationSummaryRecord(_Base):
    action: Literal["conversation_summary"]
    html_interactions: Optional[Any] = None


# Discriminated union — Pydantic v2 uses a plain Union with Annotated literal
LogRecord = Annotated[
    Union[
        HarvestingRecord,
        RegenRecord,
        ConversationResourceLimitRecord,
        UtteranceRecord,
        ConversationSummaryRecord,
    ],
    Field(discriminator="action"),
]


class LogFile(BaseModel):
    records: list[LogRecord]

    @classmethod
    def from_path(cls, path: Path) -> "LogFile":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Expected JSON list in {path}")
        return cls(records=raw)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def analyze_run(records: list[Any], *, expected_regen: float = 2.0) -> dict[str, Any]:
    """Compute renegotiation and violation metrics for a single run.

    Returns a dict with keys:
        rounds_with_contract (int)
        renegotiation_rate (float)        — among consecutive contract rounds
        inefficient_rounds (int)          — contract rounds where L*n > f_t
        inefficient_renegotiation_rate    — among inefficient-contract rounds
        violation_rate (float)            — agent-rounds where catch > limit
        violation_count (int)
        violation_denominator (int)
        per_round (list[dict])            — per-round breakdown
    """
    harvest_rows: dict[int, list[HarvestingRecord]] = defaultdict(list)
    regen_rows: dict[int, RegenRecord] = {}            # keyed by harvest_round (regen_round-1)
    limit_rows: dict[int, float] = {}                  # round -> negotiated limit (may be None)

    for r in records:
        if isinstance(r, HarvestingRecord):
            harvest_rows[r.round].append(r)
        elif isinstance(r, RegenRecord):
            harvest_round = r.round - 1
            if harvest_round >= 0:
                regen_rows[harvest_round] = r
        elif isinstance(r, ConversationResourceLimitRecord):
            # Only store if a numeric limit was detected (not N/A)
            if r.resource_limit is not None:
                limit_rows[r.round] = 3 #r.resource_limit

    # Determine number of agents from first harvest round
    n_agents = max((len(v) for v in harvest_rows.values()), default=1)

    # Sorted contract-present rounds
    contract_rounds = sorted(limit_rows.keys())

    per_round: list[dict[str, Any]] = []

    violation_count = 0
    violation_denominator = 0

    # Renegotiation tracking
    renegotiations = 0                  # consecutive pairs where limit changed
    consecutive_pairs = 0               # total consecutive contract-round pairs

    inefficient_rounds = 0             # contract rounds where L*n > f_t
    inefficient_renegotiations = 0     # of those, where limit tightened next round

    for i, rnd in enumerate(contract_rounds):
        limit = limit_rows[rnd]
        rows = harvest_rows.get(rnd, [])

        # Compute sustainable cap f_t for this round
        regen = regen_rows.get(rnd)
        stock_before: Optional[float] = None
        if rows:
            stock_before = _safe_float(rows[0].resource_in_pool_before_harvesting)
        r_t = _safe_float(regen.realized_r_t) if regen else expected_regen
        if r_t and r_t > 0 and stock_before is not None:
            f_t = max(0.0, stock_before - stock_before / r_t)
        else:
            raise ValueError(f"Invalid regen: {regen}")

        total_authorized = limit * n_agents if limit is not None else None
        is_inefficient = (
            f_t is not None
            and total_authorized is not None
            and total_authorized < f_t + 1e-9
        )

        # Violations in this round
        round_violations = 0
        for row in rows:
            c = _safe_float(row.wanted_resource) or 0.0
            violation_denominator += 1
            if c > limit + 1e-9:
                violation_count += 1
                round_violations += 1

        if is_inefficient:
            inefficient_rounds += 1
        # Renegotiation: compare this round's limit to the next contract round
        next_limit: Optional[float] = None
        renegotiated: Optional[bool] = None
        if i + 1 < len(contract_rounds):
            next_rnd = contract_rounds[i + 1]
            next_limit = limit_rows[next_rnd]
            consecutive_pairs += 1
            changed = abs(next_limit - limit) > 1e-9
            renegotiated = changed
            if changed:
                renegotiations += 1
            if is_inefficient:
                # Did agents tighten or stay the same the next round?
                if changed and next_limit <= limit:
                    inefficient_renegotiations += 1

        per_round.append({
            "round": rnd,
            "limit": limit,
            "n_agents": n_agents,
            "stock_before": stock_before,
            "r_t": r_t,
            "f_t": f_t,
            "total_authorized": total_authorized,
            "is_inefficient": is_inefficient,
            "round_violations": round_violations,
            "next_round_limit": next_limit,
            "renegotiated": renegotiated,
        })

    renegotiation_rate = renegotiations / consecutive_pairs if consecutive_pairs > 0 else None
    violation_rate = violation_count / violation_denominator if violation_denominator > 0 else None
    inefficient_renegotiation_rate = (
        inefficient_renegotiations / inefficient_rounds if inefficient_rounds > 0 else None
    )

    return {
        "rounds_with_contract": len(contract_rounds),
        "n_agents": n_agents,
        "consecutive_pairs": consecutive_pairs,
        "renegotiations": renegotiations,
        "renegotiation_rate": renegotiation_rate,
        "inefficient_rounds": inefficient_rounds,
        "inefficient_renegotiations": inefficient_renegotiations,
        "inefficient_renegotiation_rate": inefficient_renegotiation_rate,
        "violation_count": violation_count,
        "violation_denominator": violation_denominator,
        "violation_rate": violation_rate,
        "per_round": per_round,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool counts across runs and compute aggregate rates."""

    def _pool(key_num: str, key_den: str) -> Optional[float]:
        num = sum(r[key_num] for r in run_results)
        den = sum(r[key_den] for r in run_results)
        return num / den if den > 0 else None

    renegotiation_rate = _pool("renegotiations", "consecutive_pairs")
    inefficient_renegotiation_rate = _pool("inefficient_renegotiations", "inefficient_rounds")
    violation_rate = _pool("violation_count", "violation_denominator")

    return {
        "n_runs": len(run_results),
        "total_consecutive_pairs": sum(r["consecutive_pairs"] for r in run_results),
        "total_renegotiations": sum(r["renegotiations"] for r in run_results),
        "renegotiation_rate": renegotiation_rate,
        "total_inefficient_rounds": sum(r["inefficient_rounds"] for r in run_results),
        "total_inefficient_renegotiations": sum(
            r["inefficient_renegotiations"] for r in run_results
        ),
        "inefficient_renegotiation_rate": inefficient_renegotiation_rate,
        "total_violation_count": sum(r["violation_count"] for r in run_results),
        "total_violation_denominator": sum(r["violation_denominator"] for r in run_results),
        "violation_rate": violation_rate,
        "runs_with_any_contract": sum(
            1 for r in run_results if r["rounds_with_contract"] > 0
        ),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _pct(v: Optional[float], decimals: int = 1) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def print_run_summary(result: dict[str, Any], source: str, *, verbose: bool = False) -> None:
    print(f"\n{'─'*60}")
    print(f"  {source}")
    print(f"{'─'*60}")
    print(f"  Contract rounds         : {result['rounds_with_contract']}")
    print(f"  Renegotiation rate      : {_pct(result['renegotiation_rate'])} "
          f"({result['renegotiations']}/{result['consecutive_pairs']} consecutive pairs changed)")
    print(f"  Inefficient rounds      : {result['inefficient_rounds']}  "
          f"(agreed limit × n_agents > f_t)")
    print(f"  Ineff. renegot. rate    : {_pct(result['inefficient_renegotiation_rate'])} "
          f"({result['inefficient_renegotiations']}/{result['inefficient_rounds']} inefficient rounds tightened)")
    print(f"  Violation rate          : {_pct(result['violation_rate'])} "
          f"({result['violation_count']}/{result['violation_denominator']} agent-rounds exceeded cap)")

    if verbose and result["per_round"]:
        print()
        print(f"  {'Rnd':>4}  {'Limit':>6}  {'f_t':>7}  {'L×n':>7}  {'Ineff':>6}  "
              f"{'Viols':>5}  {'Reneg?':>6}  {'Next L':>6}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*6}")
        for row in result["per_round"]:
            ft = f"{row['f_t']:.1f}" if row["f_t"] is not None else "  N/A "
            ln = f"{row['total_authorized']:.1f}" if row["total_authorized"] is not None else "  N/A "
            ineff = "YES" if row["is_inefficient"] else "no"
            reneg = "YES" if row["renegotiated"] else ("no" if row["renegotiated"] is False else "—")
            next_l = f"{row['next_round_limit']:.1f}" if row["next_round_limit"] is not None else "—"
            print(f"  {row['round']:>4}  {row['limit']:>6.1f}  {ft:>7}  {ln:>7}  "
                  f"{ineff:>6}  {row['round_violations']:>5}  {reneg:>6}  {next_l:>6}")


def print_aggregate_summary(agg: dict[str, Any]) -> None:
    print(f"\n{'═'*60}")
    print("  AGGREGATE RESULTS")
    print(f"{'═'*60}")
    print(f"  Runs analysed           : {agg['n_runs']}  "
          f"({agg['runs_with_any_contract']} had at least one contract round)")
    print()
    print("  ── Renegotiation ──")
    print(f"  Rate (all rounds)       : {_pct(agg['renegotiation_rate'])} "
          f"({agg['total_renegotiations']}/{agg['total_consecutive_pairs']})")
    print(f"  Rate (inefficient only) : {_pct(agg['inefficient_renegotiation_rate'])} "
          f"({agg['total_inefficient_renegotiations']}/{agg['total_inefficient_rounds']})")
    print()
    print("  ── Violations ──")
    print(f"  Rate                    : {_pct(agg['violation_rate'])} "
          f"({agg['total_violation_count']}/{agg['total_violation_denominator']} agent-rounds)")
    print(f"{'═'*60}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def resolve_log_paths(patterns: list[str]) -> list[Path]:
    """Expand glob patterns to concrete log_env.json paths."""
    paths: list[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.is_file() and p.name == "log_env.json":
            paths.append(p)
        elif p.is_dir():
            log = p / "log_env.json"
            if log.exists():
                paths.append(log)
            else:
                paths.extend(sorted(p.rglob("log_env.json")))
        else:
            # Treat as glob
            for m in sorted(glob.glob(pat, recursive=True)):
                mp = Path(m)
                if mp.is_file() and mp.name == "log_env.json":
                    paths.append(mp)
                elif mp.is_dir():
                    log = mp / "log_env.json"
                    if log.exists():
                        paths.append(log)
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze contract renegotiation and violation rates in fishing logs."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "Path(s) to log_env.json files, run directories, or glob patterns. "
            "Directories are searched recursively for log_env.json."
        ),
    )
    parser.add_argument(
        "--expected-regen",
        type=float,
        default=2.0,
        help="Fallback regen factor when realized r_t is missing (default: 2.0)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-round breakdown for each run",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Save aggregate + per-run results as JSON",
    )
    args = parser.parse_args()

    log_paths = resolve_log_paths(args.paths)
    if not log_paths:
        print("ERROR: No log_env.json files found for the given paths.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(log_paths)} log file(s).")

    run_results: list[dict[str, Any]] = []
    run_labels: list[str] = []

    for lp in log_paths:
        try:
            log_file = LogFile.from_path(lp)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: could not parse {lp}: {exc}", file=sys.stderr)
            continue

        result = analyze_run(log_file.records, expected_regen=args.expected_regen)
        run_results.append(result)

        label = str(lp.relative_to(Path.cwd())) if lp.is_relative_to(Path.cwd()) else str(lp)
        run_labels.append(label)
        print_run_summary(result, label, verbose=args.verbose)

    if not run_results:
        print("No runs could be parsed.", file=sys.stderr)
        sys.exit(1)

    agg = aggregate(run_results)
    print_aggregate_summary(agg)

    if args.json_out is not None:
        out_data = {
            "aggregate": agg,
            "runs": [
                {"source": lbl, **{k: v for k, v in res.items() if k != "per_round"}}
                for lbl, res in zip(run_labels, run_results)
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"\nWrote JSON results to: {args.json_out}")


if __name__ == "__main__":
    main()
