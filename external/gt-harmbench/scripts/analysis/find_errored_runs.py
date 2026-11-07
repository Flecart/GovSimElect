"""Find contracting runs whose samples contain failed model calls (any phase).

A failed decision call is parsed as the first listed action, and failed negotiation/coding calls
silently prevent agreements, so such runs bias results. With --quarantine, errored (and
incomplete) .eval files are moved to logs/_quarantine/ so the resume script re-runs them.

    uv run python scripts/analysis/find_errored_runs.py "logs/eval-20261106-*/*/*/*.eval" [--quarantine]
"""

import glob
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import click


def _errors_in_sample(sample: dict) -> Counter:
    errs: Counter = Counter()
    meta = sample.get("metadata", {})
    for out in (meta.get("decision_outputs") or {}).values():
        if "error during generation" in (out or ""):
            errs["decision"] += 1
    neg = meta.get("negotiation_result") or {}
    for event in neg.get("conversations", []) or []:
        tel = event.get("telemetry") if isinstance(event, dict) else None
        if isinstance(tel, dict) and tel.get("error"):
            errs[tel.get("phase") or "negotiation"] += 1
    if sample.get("error"):
        errs["sample"] += 1
    return errs


def scan(path: str) -> tuple[str, Counter]:
    try:
        z = zipfile.ZipFile(path)
    except Exception:
        return "unreadable", Counter()
    names = z.namelist()
    if "header.json" not in names:
        return "incomplete", Counter()
    if json.loads(z.read("header.json")).get("status") != "success":
        return "not-success", Counter()
    total: Counter = Counter()
    for n in names:
        if n.startswith("samples/"):
            total += _errors_in_sample(json.loads(z.read(n)))
    return ("errored" if total else "clean"), total


@click.command()
@click.argument("patterns", nargs=-1, required=True)
@click.option("--quarantine", is_flag=True, help="Move errored/incomplete logs to logs/_quarantine/.")
def main(patterns: tuple[str, ...], quarantine: bool) -> None:
    files = sorted({f for p in patterns for f in glob.glob(p)})
    status_counts: Counter = Counter()
    for f in files:
        status, errs = scan(f)
        status_counts[status] += 1
        if status == "clean":
            continue
        short = f.split("logs/", 1)[-1]
        print(f"{status:11s} {dict(errs)!s:40s} {short}")
        if quarantine:
            dest = Path("logs/_quarantine") / short
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(f, dest)
    print("\nsummary:", dict(status_counts), "(moved to logs/_quarantine)" if quarantine else "")


if __name__ == "__main__":
    sys.exit(main())
