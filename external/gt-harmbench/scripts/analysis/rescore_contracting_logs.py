"""Re-run the contracting scorer on finished .eval logs (no model calls).

Use after a scorer fix, e.g. the 2026-11-07 fix for Stag Hunt 2x2 payoffs stored as "(25, 25)"
tuples, which the old parser scored as (0, 0). Only completed (status=success) logs are touched;
each original is copied to <backup-dir> first.

    uv run --env-file .env.cross python scripts/analysis/rescore_contracting_logs.py "logs/eval-*-sh/*/2x2-*/*.eval"

Inspect re-creates the original model client (no calls are made), so the provider env vars
used for the run (e.g. AXONHUB_API_KEY / LITELLM_API_KEY) must be set.
"""

import glob
import shutil
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.scorer import mean_score

from eval.contracting_scorer import (
    contracting_score_to_float,
    contracting_scorer_coordination,
    contracting_scorer_pd,
    contracting_scorer_sh,
)

SCORERS = {
    "pd": contracting_scorer_pd,
    "sh": contracting_scorer_sh,
    "co": contracting_scorer_coordination,
}


@click.command()
@click.argument("patterns", nargs=-1, required=True)
@click.option("--backup-dir", default="logs/_prerescore_backup", help="Where originals are copied.")
@click.option("--dry-run", is_flag=True, help="List the logs that would be rescored.")
def main(patterns: tuple[str, ...], backup_dir: str, dry_run: bool) -> None:
    files = sorted({f for p in patterns for f in glob.glob(p, recursive=True)})
    for f in files:
        header = read_eval_log(f, header_only=True)
        if header.status != "success":
            print(f"skip (status={header.status}): {f}")
            continue
        game = (header.eval.metadata or {}).get("game_type", "pd")
        if dry_run:
            print(f"would rescore [{game}] {f}")
            continue

        backup = Path(backup_dir) / Path(f).relative_to("logs") if f.startswith("logs/") else Path(backup_dir) / Path(f).name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(f, backup)

        try:
            log = read_eval_log(f)
            rescored = score(
                log,
                scorers=[SCORERS[game]()],
                epochs_reducer=mean_score(value_to_float=contracting_score_to_float()),
                action="overwrite",
                display="none",
            )
            write_eval_log(rescored, f)
            print(f"rescored [{game}] {f}")
        except Exception as exc:  # e.g. missing provider env vars: pass the run's --env-file
            print(f"FAILED {f}: {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
