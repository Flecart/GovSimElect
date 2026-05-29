#!/usr/bin/env python3
"""Aggregate code_ddl-vs-no_contract metrics across models into a markdown table.

Scans simulation/results/{dry_gpt4o,sweep/<tag>}/{noc,ddl} and prints a summary.
Usage: python scripts/ddl_aggregate.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ddl_metrics import metrics  # noqa: E402

# (display label, results glob for noc, results glob for ddl)
MODELS = [
    ("gpt-4o",        "simulation/results/dry_gpt4o/noc",   "simulation/results/dry_gpt4o/ddl"),
    ("gpt-5.4",       "simulation/results/sweep/gpt54/noc", "simulation/results/sweep/gpt54/ddl"),
    ("gpt-5.4-mini",  "simulation/results/sweep/gpt54mini/noc", "simulation/results/sweep/gpt54mini/ddl"),
    ("gemma-4-31b-it","simulation/results/sweep/gemma4/noc","simulation/results/sweep/gemma4/ddl"),
    ("grok-4.20",     "simulation/results/sweep/grok420/noc","simulation/results/sweep/grok420/ddl"),
]


def _find(root):
    hits = glob.glob(os.path.join(root, "**", "log_env.json"), recursive=True)
    return os.path.dirname(hits[0]) if hits else None


def _row(label, noc_root, ddl_root):
    out = {"model": label}
    for cond, root in (("noc", noc_root), ("ddl", ddl_root)):
        d = _find(root)
        out[cond] = metrics(d) if d else None
    return out


def _fmt(m):
    if not m:
        return "—"
    surv = f"{m['survived_rounds']}/{m['rounds_logged']}"
    col = "collapse" if m["collapsed"] else "survives"
    return f"{col} {surv}, gain={m['total_gain']:.0f}, net={m['net_reward']:.0f}"


if __name__ == "__main__":
    rows = [_row(*m) for m in MODELS]
    print("| model | no_contract | code_ddl (rule by law) |")
    print("|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {_fmt(r['noc'])} | {_fmt(r['ddl'])} |")
    print("\nDetail:")
    for r in rows:
        for cond in ("noc", "ddl"):
            if r[cond]:
                print(f"  {r['model']:14s} {cond}: pool={r[cond]['pool_trajectory']} "
                      f"fines={r[cond]['total_fines']}")
