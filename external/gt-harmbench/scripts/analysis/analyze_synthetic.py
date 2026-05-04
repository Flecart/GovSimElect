"""Report Nash / Utilitarian / Rawlsian accuracy (mean ± std) per game from a synthetic eval log.

Usage:
    uv run python -m scripts.analysis.analyze_synthetic <path/to/log.eval>
"""

from __future__ import annotations

import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pandas as pd

# Reuse the existing loading and metrics pipeline
from eval.analysis.core import (
    build_dataframe,
    compute_sample_metrics,
    load_log_samples_from_path,
)
from eval.analysis.parsing import parse_actions_from_answer, standardize_action


def _load_and_filter(log_path: str) -> pd.DataFrame:
    """Load a log, build the flat dataframe, and drop samples that can't be parsed."""
    with redirect_stdout(StringIO()):
        samples, model_name = load_log_samples_from_path(log_path, "eval")
        df = build_dataframe(samples)

    # build_dataframe converts game_type to a restricted pd.Categorical using the
    # main dataset's GAME_TYPE_ORDER. Names not in that list (e.g. "Battle of the
    # Sexes", "Stag Hunt", "No Conflict") silently become NaN. Recover the original
    # strings directly from the raw sample metadata (same order as df rows).
    df["game_type"] = [
        s.get("metadata", {}).get("formal_game", "unknown") for s in samples
    ]

    print(f"Log:   {Path(log_path).name}")
    print(f"Model: {model_name}")
    print(f"Samples loaded: {len(df)}")

    # Keep only samples where both actions can be standardized (same filter as heatmap)
    valid_mask = df.apply(
        lambda row: (
            standardize_action(
                parse_actions_from_answer(row["answer"], row["actions_row"], row["actions_column"])[0],
                row["actions_row"],
                is_row=True,
            ) is not None
            and standardize_action(
                parse_actions_from_answer(row["answer"], row["actions_row"], row["actions_column"])[1],
                row["actions_column"],
                is_row=False,
            ) is not None
        ),
        axis=1,
    )
    df_valid = df[valid_mask].copy()
    dropped = len(df) - len(df_valid)
    if dropped:
        print(f"Dropped {dropped} unparseable sample(s); using {len(df_valid)}.")
    return df_valid, model_name


def _per_sample_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Run compute_sample_metrics on every row, return a tidy DataFrame."""
    rows = []
    for _, row in df.iterrows():
        m = compute_sample_metrics(row)
        if m is not None:
            rows.append(m)
    return pd.DataFrame(rows)


def _report(metrics_df: pd.DataFrame) -> None:
    """Print mean ± std table for each game, plus an overall row."""
    cols = {
        "nash": "Nash",
        "utilitarian": "Utilitarian",
        "rawlsian": "Rawlsian",
    }

    grouped = metrics_df.groupby("game_type", sort=True)

    # Build summary table
    records = []
    for game, grp in grouped:
        row = {"Game": game, "N": len(grp)}
        for key, label in cols.items():
            mu = grp[key].mean()
            sd = grp[key].std(ddof=1) if len(grp) > 1 else float("nan")
            row[label] = f"{mu:.3f} ± {sd:.3f}"
        records.append(row)

    # Overall row
    overall = {"Game": "OVERALL", "N": len(metrics_df)}
    for key, label in cols.items():
        mu = metrics_df[key].mean()
        sd = metrics_df[key].std(ddof=1) if len(metrics_df) > 1 else float("nan")
        overall[label] = f"{mu:.3f} ± {sd:.3f}"
    records.append(overall)

    summary = pd.DataFrame(records).set_index("Game")
    print("\n" + "=" * 70)
    print("Accuracy  (mean ± std across samples,  scale 0–1)")
    print("=" * 70)
    print(summary.to_string())
    print("=" * 70)


def main(log_path: str) -> None:
    df_valid, _ = _load_and_filter(log_path)
    if df_valid.empty:
        print("No valid samples to analyse.", file=sys.stderr)
        sys.exit(1)

    metrics_df = _per_sample_metrics(df_valid)
    if metrics_df.empty:
        print("No metrics could be computed (missing rewards_matrix?).", file=sys.stderr)
        sys.exit(1)

    _report(metrics_df)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python -m scripts.analysis.analyze_synthetic <log.eval>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
