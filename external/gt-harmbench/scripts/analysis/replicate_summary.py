"""Utilitarian accuracy (mean ± SD across replicate runs) for the 2026-11-06 batches.

A replicate = one run of a condition, reported on 50 samples (25 scenarios x 2 epochs). Its
utilitarian accuracy is the fraction of samples whose final (post-enforcement) action pair is
utilitarian-optimal (scorer metadata `is_utilitarian`). We report mean ± sample SD (ddof=1)
across replicates; rep 1 lives in logs/eval-20261106-<batch>-*, rep N in
logs/eval-20261106-<batch>-rN-*.

    uv run python scripts/analysis/replicate_summary.py [--csv out.csv]
"""

import csv
import glob
import json
import re
import statistics as st
import zipfile
from collections import defaultdict

import click

BATCHES = ("4x4obs", "cross2x2", "talk2x2", "talk4x4")

# Every run is reported on the same 50-sample design: the first N_SCENARIOS rows of the dataset
# (what --limit 25 selects) x epochs 1..N_EPOCHS. Runs made earlier at 30 scenarios x 5 epochs
# are subset to exactly these samples, so all replicates are comparable.
N_SCENARIOS = 25
N_EPOCHS = 2
DATASETS = {
    ("4x4obs", "pd"): "data/gt-harmbench-pd30-4x4-rewritten.csv",
    ("4x4obs", "sh"): "data/gt-harmbench-sh30-4x4-rewritten.csv",
    ("cross2x2", "pd"): "data/gt-harmbench-pd30-2x2-with-targets.csv",
    ("cross2x2", "sh"): "data/gt-harmbench-sh30-2x2-with-targets.csv",
    ("talk2x2", "pd"): "data/gt-harmbench-pd30-2x2-with-targets.csv",
    ("talk2x2", "sh"): "data/gt-harmbench-sh30-2x2-with-targets.csv",
    ("talk4x4", "pd"): "data/gt-harmbench-pd30-4x4-rewritten.csv",
    ("talk4x4", "sh"): "data/gt-harmbench-sh30-4x4-rewritten.csv",
}
_ID_CACHE: dict[tuple[str, str], set[str]] = {}


def _scenario_ids(batch: str, game: str) -> set[str]:
    key = (batch, game)
    if key not in _ID_CACHE:
        csv.field_size_limit(10**9)
        with open(DATASETS[key], newline="") as fh:
            rows = list(csv.DictReader(fh))[:N_SCENARIOS]
        _ID_CACHE[key] = {str(r["id"]) for r in rows}
    return _ID_CACHE[key]


_SAMPLE_RE = re.compile(r"^samples/(?P<id>.+)_epoch_(?P<epoch>\d+)\.json$")
_RUN_RE = re.compile(r"eval-2026110[67]-(?P<batch>4x4obs|cross2x2|talk2x2|talk4x4)(?:-r(?P<rep>\d+))?-(?P<rest>.+)-(?P<game>pd|sh)$")


def _clean_model(rest: str) -> str:
    return rest.replace("openai-api-litellm-", "").replace("openai-api-axonhub-", "")


def _run_accuracy(path: str, keep_ids: set[str] | None = None) -> tuple[float, int] | None:
    z = zipfile.ZipFile(path)
    names = z.namelist()
    if "header.json" not in names or json.loads(z.read("header.json")).get("status") != "success":
        return None
    flags = []
    for n in names:
        m = _SAMPLE_RE.match(n)
        if m and (keep_ids is None or (m["id"] in keep_ids and int(m["epoch"]) <= N_EPOCHS)):
            meta = list(json.loads(z.read(n))["scores"].values())[0].get("metadata", {})
            if meta.get("is_utilitarian") is not None:
                flags.append(bool(meta["is_utilitarian"]))
    return (sum(flags) / len(flags), len(flags)) if flags else None


def collect() -> dict[tuple, dict[int, tuple[float, int]]]:
    results: dict[tuple, dict[int, tuple[float, int]]] = defaultdict(dict)
    for f in glob.glob("logs/eval-2026110[67]-*/*/*/*.eval"):
        run_dir, prompt, cond = f.split("/")[1:4]
        m = _RUN_RE.match(run_dir)
        if not m:
            continue
        acc = _run_accuracy(f, _scenario_ids(m["batch"], m["game"]))
        if acc is None:
            continue
        key = (m["batch"], _clean_model(m["rest"]), m["game"], prompt, cond)
        results[key][int(m["rep"] or 1)] = acc
    return results


@click.command()
@click.option("--csv", "csv_path", default=None, help="Also write the table to this CSV.")
def main(csv_path: str | None) -> None:
    results = collect()
    rows = []
    for key in sorted(results):
        reps = results[key]
        accs = [reps[r][0] for r in sorted(reps)]
        mean = st.mean(accs)
        sd = st.stdev(accs) if len(accs) > 1 else float("nan")
        rows.append((*key, len(accs), mean, sd, " ".join(f"{reps[r][0]:.3f}({reps[r][1]})" for r in sorted(reps))))

    for batch in BATCHES:
        batch_rows = [r for r in rows if r[0] == batch]
        if not batch_rows:
            continue
        print(f"\n=== {batch}: utilitarian accuracy, mean ± SD across replicates ===")
        print(f"{'model / pair (row-vs-col)':32} {'game':4} {'prompt':11} {'condition':16} {'reps':>4} {'util acc':>17}   per-rep (n samples)")
        for r in sorted(batch_rows, key=lambda r: (r[2], r[4], r[3], r[1])):
            sd = "  n/a" if r[5] < 2 else f"{r[7]:.3f}"
            print(f"{r[1][:32]:32} {r[2]:4} {r[3]:11} {r[4]:16} {r[5]:4d} {r[6]:.3f} ± {sd:>5}   {r[8]}")

    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["batch", "model", "game", "prompt_mode", "condition", "n_reps", "util_acc_mean", "util_acc_sd", "per_rep"])
            w.writerows(rows)
        print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
