"""Extract accuracy and average scores from all eval logs in a directory."""

import json
import sys
import zipfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

import click
import pandas as pd

from .core import build_dataframe, compute_metrics, load_log_samples_from_path


def extract_metrics_from_log(log_path: Path) -> Dict[str, Any]:
    """Extract metrics from a single eval log file.
    
    Returns:
        Dictionary with log_id, model_name, and all metrics
    """
    log_id = log_path.stem  # Filename without .eval extension
    
    # Load samples (suppress verbose output)
    try:
        with redirect_stdout(StringIO()):
            samples, model_name = load_log_samples_from_path(str(log_path), "eval")
    except Exception as e:
        print(f"Error loading samples from {log_path.name}: {e}", file=sys.stderr)
        return None
    
    if not samples:
        print(f"No samples found in {log_path.name}", file=sys.stderr)
        return None
    
    # Build dataframe and compute metrics (suppress verbose output)
    try:
        with redirect_stdout(StringIO()):
            df = build_dataframe(samples)
            _, _, welfare_by_game, welfare_variance_by_game, overall_accuracy, _ = compute_metrics(df)
    except Exception as e:
        print(f"Error computing metrics from {log_path.name}: {e}", file=sys.stderr)
        return None
    
    # Extract accuracy metrics
    result = {
        "log_id": log_id,
        "model_name": model_name,
    }
    
    # Add accuracy metrics
    if len(overall_accuracy) > 0:
        result["nash_accuracy"] = overall_accuracy.get("nash", 0.0)
        result["utilitarian_accuracy"] = overall_accuracy.get("utilitarian", 0.0)
        result["rawlsian_accuracy"] = overall_accuracy.get("rawlsian", 0.0)
        result["nash_social_welfare_accuracy"] = overall_accuracy.get("nash_social_welfare", 0.0)
    else:
        result["nash_accuracy"] = 0.0
        result["utilitarian_accuracy"] = 0.0
        result["rawlsian_accuracy"] = 0.0
        result["nash_social_welfare_accuracy"] = 0.0
    
    # Add average welfare scores (mean across all game types)
    if not welfare_by_game.empty:
        overall_welfare = welfare_by_game.mean()
        result["utilitarian_efficiency"] = overall_welfare.get("utilitarian_efficiency", 0.0)
        result["rawlsian_efficiency"] = overall_welfare.get("rawlsian_efficiency", 0.0)
        result["nash_social_welfare_efficiency"] = overall_welfare.get("nash_social_welfare_efficiency", 0.0)
    else:
        result["utilitarian_efficiency"] = 0.0
        result["rawlsian_efficiency"] = 0.0
        result["nash_social_welfare_efficiency"] = 0.0

    # Add variance scores (mean across all game types)
    if not welfare_variance_by_game.empty:
        overall_variance = welfare_variance_by_game.mean()
        result["utilitarian_variance"] = overall_variance.get("utilitarian_variance", 0.0)
        result["rawlsian_variance"] = overall_variance.get("rawlsian_variance", 0.0)
        result["nash_social_welfare_variance"] = overall_variance.get("nash_social_welfare_variance", 0.0)
    else:
        result["utilitarian_variance"] = 0.0
        result["rawlsian_variance"] = 0.0
        result["nash_social_welfare_variance"] = 0.0

    return result


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
    default="metrics_summary.csv",
    help="Output CSV file path",
)
def main(directory: Path, output: Path) -> None:
    """Extract accuracy and average scores from all eval logs in a directory.
    
    Processes all .eval files in the specified directory and extracts:
    - Accuracy metrics: nash, utilitarian, rawlsian, nash_social_welfare
    - Average scores: utilitarian_efficiency, rawlsian_efficiency, nash_social_welfare_efficiency
    
    Outputs a CSV with columns: log_id, model_name, and all metrics.
    """
    # Find all .eval files
    eval_files = sorted(directory.glob("*.eval"))
    
    if not eval_files:
        print(f"No .eval files found in {directory}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(eval_files)} .eval files in {directory}")
    print("Processing files...")
    
    results: List[Dict[str, Any]] = []
    
    for eval_file in eval_files:
        print(f"Processing {eval_file.name}...")
        metrics = extract_metrics_from_log(eval_file)
        if metrics:
            results.append(metrics)
        else:
            print(f"  Skipped {eval_file.name} due to errors")
    
    if not results:
        print("No valid results extracted.", file=sys.stderr)
        sys.exit(1)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    
    # Reorder columns: log_id, model_name, then metrics
    column_order = [
        "log_id",
        "model_name",
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
    
    # Save to CSV
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    
    print(f"\nExtracted metrics from {len(results)} logs")
    print(f"Saved to: {output}")
    print(f"\nSummary:")
    print(df.describe())


if __name__ == "__main__":
    main()
