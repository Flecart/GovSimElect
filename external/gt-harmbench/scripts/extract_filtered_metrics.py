#!/usr/bin/env python3
"""Extract metrics from eval logs, filtered by game type.

This script processes evaluation logs and computes metrics for specific game types.
It uses the game-aware normalization from eval.plots.data.processor.

Usage:
    uv run python scripts/extract_filtered_metrics.py --directory logs --output results/filtered.csv
    uv run python scripts/extract_filtered_metrics.py --games "Prisoner's Dilemma" --games "Chicken"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import click
import pandas as pd

from eval.plots.data.loader import load_eval_log
from eval.plots.data.processor import (
    GAME_TYPE_ORDER,
    process_sample,
)


def extract_metrics_from_log(
    log_path: Path,
    game_filter: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Extract metrics from a single eval log file with optional game filtering.

    Args:
        log_path: Path to the .eval file
        game_filter: Set of game types to include. If None, include all games.

    Returns:
        Dictionary with log_id, model_name, and all metrics, or None on error
    """
    log_id = log_path.stem

    # Load the eval log
    try:
        eval_log, model_name = load_eval_log(str(log_path), header_only=False)
    except Exception as e:
        print(f"Error loading {log_path.name}: {e}", file=sys.stderr)
        return None

    if eval_log is None or not eval_log.samples:
        print(f"No samples found in {log_path.name}", file=sys.stderr)
        return None

    # Process samples with game-aware normalization
    records = []
    for sample in eval_log.samples:
        record = process_sample(sample, model_name)

        # Apply game filter
        if game_filter and record.game_type not in game_filter:
            continue

        if record.parse_success:
            records.append(record)

    if not records:
        print(f"No valid samples after filtering in {log_path.name}", file=sys.stderr)
        return None

    # Compute metrics
    n_samples = len(records)

    # Accuracy metrics (mean of correctness flags)
    nash_correct = sum(1 for r in records if r.nash_correct) / n_samples
    utilitarian_correct = sum(1 for r in records if r.utilitarian_correct) / n_samples
    rawlsian_correct = sum(1 for r in records if r.rawlsian_correct) / n_samples
    nash_social_correct = sum(1 for r in records if r.nash_social_correct) / n_samples

    # Efficiency metrics (mean of scores)
    utilitarian_scores = [r.utilitarian_score for r in records if r.utilitarian_score is not None]
    rawlsian_scores = [r.rawlsian_score for r in records if r.rawlsian_score is not None]
    nash_social_scores = [r.nash_social_score for r in records if r.nash_social_score is not None]

    utilitarian_efficiency = sum(utilitarian_scores) / len(utilitarian_scores) if utilitarian_scores else 0.0
    rawlsian_efficiency = sum(rawlsian_scores) / len(rawlsian_scores) if rawlsian_scores else 0.0
    nash_social_efficiency = sum(nash_social_scores) / len(nash_social_scores) if nash_social_scores else 0.0

    return {
        "log_id": log_id,
        "model_name": model_name,
        "n_samples": n_samples,
        "nash_accuracy": nash_correct,
        "utilitarian_accuracy": utilitarian_correct,
        "rawlsian_accuracy": rawlsian_correct,
        "nash_social_welfare_accuracy": nash_social_correct,
        "utilitarian_efficiency": utilitarian_efficiency,
        "rawlsian_efficiency": rawlsian_efficiency,
        "nash_social_welfare_efficiency": nash_social_efficiency,
    }


def compute_metrics_by_game(
    log_path: Path,
    game_filter: Optional[Set[str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Extract metrics from a log file, broken down by game type.

    Args:
        log_path: Path to the .eval file
        game_filter: Set of game types to include. If None, include all games.

    Returns:
        List of dictionaries, one per game type, or None on error
    """
    log_id = log_path.stem

    # Load the eval log
    try:
        eval_log, model_name = load_eval_log(str(log_path), header_only=False)
    except Exception as e:
        print(f"Error loading {log_path.name}: {e}", file=sys.stderr)
        return None

    if eval_log is None or not eval_log.samples:
        print(f"No samples found in {log_path.name}", file=sys.stderr)
        return None

    # Process samples and group by game type
    game_records: Dict[str, List] = {}
    for sample in eval_log.samples:
        record = process_sample(sample, model_name)

        # Apply game filter
        if game_filter and record.game_type not in game_filter:
            continue

        if record.parse_success:
            if record.game_type not in game_records:
                game_records[record.game_type] = []
            game_records[record.game_type].append(record)

    if not game_records:
        print(f"No valid samples after filtering in {log_path.name}", file=sys.stderr)
        return None

    # Compute metrics per game type
    results = []
    for game_type, records in game_records.items():
        n_samples = len(records)

        nash_correct = sum(1 for r in records if r.nash_correct) / n_samples
        utilitarian_correct = sum(1 for r in records if r.utilitarian_correct) / n_samples
        rawlsian_correct = sum(1 for r in records if r.rawlsian_correct) / n_samples
        nash_social_correct = sum(1 for r in records if r.nash_social_correct) / n_samples

        utilitarian_scores = [r.utilitarian_score for r in records if r.utilitarian_score is not None]
        rawlsian_scores = [r.rawlsian_score for r in records if r.rawlsian_score is not None]
        nash_social_scores = [r.nash_social_score for r in records if r.nash_social_score is not None]

        utilitarian_efficiency = sum(utilitarian_scores) / len(utilitarian_scores) if utilitarian_scores else 0.0
        rawlsian_efficiency = sum(rawlsian_scores) / len(rawlsian_scores) if rawlsian_scores else 0.0
        nash_social_efficiency = sum(nash_social_scores) / len(nash_social_scores) if nash_social_scores else 0.0

        results.append({
            "log_id": log_id,
            "model_name": model_name,
            "game_type": game_type,
            "n_samples": n_samples,
            "nash_accuracy": nash_correct,
            "utilitarian_accuracy": utilitarian_correct,
            "rawlsian_accuracy": rawlsian_correct,
            "nash_social_welfare_accuracy": nash_social_correct,
            "utilitarian_efficiency": utilitarian_efficiency,
            "rawlsian_efficiency": rawlsian_efficiency,
            "nash_social_welfare_efficiency": nash_social_efficiency,
        })

    return results


@click.command()
@click.option(
    "--directory",
    "-d",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default="logs",
    help="Directory containing .eval files",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="results/filtered_metrics.csv",
    help="Output CSV file path",
)
@click.option(
    "--games",
    "-g",
    multiple=True,
    default=["Prisoner's Dilemma", "Chicken"],
    help="Game types to include. Can be specified multiple times. Default: Prisoner's Dilemma, Chicken",
)
@click.option(
    "--by-game",
    is_flag=True,
    default=False,
    help="Output metrics broken down by game type (adds game_type column)",
)
@click.option(
    "--all-games",
    is_flag=True,
    default=False,
    help="Include all game types (ignores --games)",
)
def main(
    directory: Path,
    output: Path,
    games: tuple,
    by_game: bool,
    all_games: bool,
) -> None:
    """Extract metrics from eval logs with game type filtering.

    Uses game-aware normalization to ensure consistent action standardization:
    - For Prisoner's Dilemma, Chicken, Stag Hunt, No Conflict:
      Best utilitarian outcome is at UP-LEFT
    - For Coordination, Bach or Stravinski:
      Nash equilibria are on the main diagonal

    Examples:

        # Default: Prisoner's Dilemma and Chicken only
        uv run python scripts/extract_filtered_metrics.py

        # Custom games
        uv run python scripts/extract_filtered_metrics.py --games "Stag hunt" --games "Coordination"

        # All games
        uv run python scripts/extract_filtered_metrics.py --all-games

        # Breakdown by game type
        uv run python scripts/extract_filtered_metrics.py --by-game
    """
    # Determine game filter
    if all_games:
        game_filter = None
        print("Including all game types")
    else:
        game_filter = set(games)
        print(f"Filtering to game types: {', '.join(sorted(game_filter))}")

    # Find all .eval files
    eval_files = sorted(directory.glob("*.eval"))

    if not eval_files:
        print(f"No .eval files found in {directory}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(eval_files)} .eval files in {directory}")
    print("Processing files...\n")

    results: List[Dict[str, Any]] = []

    for eval_file in eval_files:
        print(f"Processing {eval_file.name}...")

        if by_game:
            metrics_list = compute_metrics_by_game(eval_file, game_filter)
            if metrics_list:
                results.extend(metrics_list)
                total_samples = sum(m["n_samples"] for m in metrics_list)
                print(f"  -> {len(metrics_list)} game types, {total_samples} samples")
            else:
                print(f"  -> Skipped (no valid samples)")
        else:
            metrics = extract_metrics_from_log(eval_file, game_filter)
            if metrics:
                results.append(metrics)
                print(f"  -> {metrics['n_samples']} samples")
            else:
                print(f"  -> Skipped (no valid samples)")

    if not results:
        print("\nNo valid results extracted.", file=sys.stderr)
        sys.exit(1)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns
    if by_game:
        column_order = [
            "log_id",
            "model_name",
            "game_type",
            "n_samples",
            "nash_accuracy",
            "utilitarian_accuracy",
            "rawlsian_accuracy",
            "nash_social_welfare_accuracy",
            "utilitarian_efficiency",
            "rawlsian_efficiency",
            "nash_social_welfare_efficiency",
        ]
    else:
        column_order = [
            "log_id",
            "model_name",
            "n_samples",
            "nash_accuracy",
            "utilitarian_accuracy",
            "rawlsian_accuracy",
            "nash_social_welfare_accuracy",
            "utilitarian_efficiency",
            "rawlsian_efficiency",
            "nash_social_welfare_efficiency",
        ]

    # Ensure all columns exist
    for col in column_order:
        if col not in df.columns:
            df[col] = 0.0

    df = df[column_order]

    # Sort by model name and optionally game type
    if by_game:
        # Sort game types by GAME_TYPE_ORDER
        game_order = {g: i for i, g in enumerate(GAME_TYPE_ORDER)}
        df["_game_order"] = df["game_type"].map(lambda x: game_order.get(x, 999))
        df = df.sort_values(["model_name", "_game_order"]).drop(columns=["_game_order"])
    else:
        df = df.sort_values("model_name")

    # Save to CSV
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\nExtracted metrics from {len(eval_files)} logs")
    print(f"Total rows: {len(df)}")
    print(f"Saved to: {output}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    numeric_cols = [
        "nash_accuracy",
        "utilitarian_accuracy",
        "rawlsian_accuracy",
        "nash_social_welfare_accuracy",
        "utilitarian_efficiency",
        "rawlsian_efficiency",
        "nash_social_welfare_efficiency",
    ]

    summary = df[numeric_cols].describe().round(3)
    print(summary)

    if by_game:
        print("\n" + "=" * 60)
        print("Mean by Game Type")
        print("=" * 60)
        game_summary = df.groupby("game_type")[numeric_cols].mean().round(3)
        print(game_summary)


if __name__ == "__main__":
    main()
