"""CLI entrypoint for the new plots module.

This CLI provides a clean interface for generating plots from .eval files,
using the intermediary data format for efficient processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import click
import pickle
import pandas as pd

from .data import (
    EvalRunData,
    MultiEvalData,
    filter_evaluation_logs,
    list_eval_logs,
    process_eval_log,
    process_multiple_logs,
)
from .renderers import (
    render_accuracy_heatmap,
    render_accuracy_heatmap_latex,
    render_accuracy_heatmap_latex_landscape,
    render_accuracy_plot,
    render_action_heatmap,
    render_action_probability_grid,
    render_welfare_plot,
)


def save_intermediary_data(
    data: EvalRunData | MultiEvalData,
    output_path: Path,
) -> Path:
    """Save intermediary data to CSV file.

    Args:
        data: Processed evaluation data
        output_path: Path to save the CSV file

    Returns:
        Path to saved file
    """
    if isinstance(data, EvalRunData):
        df = data.samples_df
    else:
        df = data.get_all_samples_df()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_single_log_analysis(
    run_data: EvalRunData,
    plot_kinds: Sequence[str],
    output_dir: str,
    prefix: str,
    save_data: bool,
) -> None:
    """Run analysis on a single evaluation log.

    Args:
        run_data: Processed evaluation run data
        plot_kinds: Which plots to generate
        output_dir: Directory to save plots
        prefix: Prefix for output filenames
        save_data: Whether to save intermediary data
    """
    print(f"\nProcessing model: {run_data.model_name}")
    valid_samples = sum(1 for s in run_data.samples if s.parse_success)
    print(f"  Valid samples: {valid_samples}/{run_data.num_samples}")

    if save_data:
        safe_name = run_data.model_name.replace("/", "-").replace(" ", "_")
        data_path = Path(output_dir) / f"{prefix}data_{safe_name}.csv" if prefix else Path(output_dir) / f"data_{safe_name}.csv"
        save_intermediary_data(run_data, data_path)
        print(f"  Saved intermediary data to: {data_path}")

    if "accuracy" in plot_kinds:
        acc_path = render_accuracy_plot(run_data, output_dir=output_dir, prefix=prefix)
        print(f"  Saved accuracy plot to: {acc_path}")

    if "welfare" in plot_kinds:
        welfare_path = render_welfare_plot(run_data, output_dir=output_dir, prefix=prefix)
        print(f"  Saved welfare plot to: {welfare_path}")

    if "heatmap" in plot_kinds:
        heatmap_path = render_action_heatmap(run_data, output_dir=output_dir, prefix=prefix)
        print(f"  Saved action heatmap to: {heatmap_path}")


def run_multi_log_analysis(
    multi_data: MultiEvalData,
    plot_kinds: Sequence[str],
    output_dir: str,
    prefix: str,
    save_data: bool,
    latex: bool = False,
    metric: str = "utilitarian_accuracy",
) -> None:
    """Run analysis on multiple evaluation logs.

    Args:
        multi_data: Data from multiple evaluation runs
        plot_kinds: Which plots to generate
        output_dir: Directory to save plots
        prefix: Prefix for output filenames
        save_data: Whether to save intermediary data
        latex: Output LaTeX table instead of PNG image
        metric: Which metric to use for the heatmap
    """
    print(f"\nProcessing {len(multi_data.runs)} evaluation runs")

    if save_data:
        data_path = Path(output_dir) / f"{prefix}data_all.csv" if prefix else Path(output_dir) / "data_all.csv"
        save_intermediary_data(multi_data, data_path)
        pickle_path = Path(output_dir) / f"{prefix}data_all.pkl" if prefix else Path(output_dir) / "data_all.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(multi_data, f)
        print(f"  Saved intermediary data to: {data_path} and {pickle_path}")

    if "welfare-heatmap" in plot_kinds or "accuracy-heatmap" in plot_kinds:
        if latex:
            # Output LaTeX table
            metric_suffix = metric.replace("_", "-")
            output_path = Path(output_dir) / f"{prefix}accuracy_heatmap_{metric_suffix}.tex" if prefix else Path(output_dir) / f"accuracy_heatmap_{metric_suffix}.tex"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            latex_code = render_accuracy_heatmap_latex_landscape(multi_data, metric=metric, output_path=str(output_path))
            print(f"  Saved LaTeX table to: {output_path}")
            print("\n" + "="*80)
            print("LaTeX CODE (copy below):")
            print("="*80 + "\n")
            print(latex_code)
            print("\n" + "="*80)
        else:
            heatmap_path = render_accuracy_heatmap(multi_data, output_dir=output_dir, prefix=prefix)
            print(f"  Saved accuracy heatmap to: {heatmap_path}")

    if "action-probability-grid" in plot_kinds:
        grid_path = render_action_probability_grid(multi_data, output_dir=output_dir, prefix=prefix)
        print(f"  Saved action probability grid to: {grid_path}")


@click.command()
@click.option(
    "--log-path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Path to a single .eval file. If not provided, lists available logs.",
)
@click.option(
    "--logs-dir",
    type=click.Path(path_type=Path, exists=True),
    default="logs",
    help="Directory containing .eval files (used when --log-path is not provided).",
)
@click.option(
    "--max-logs",
    type=int,
    default=100,
    help="Maximum number of logs to process when using --logs-dir.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="assets",
    help="Directory where plots will be saved.",
)
@click.option(
    "--plot",
    "plots",
    type=click.Choice(
        ["accuracy", "welfare", "heatmap", "welfare-heatmap", "accuracy-heatmap", "action-probability-grid", "all"],
        case_sensitive=False,
    ),
    multiple=True,
    help="Which plots to generate. Default: accuracy, welfare, heatmap.",
)
@click.option(
    "--prefix",
    type=str,
    default="",
    help="Prefix to add to output filenames.",
)
@click.option(
    "--save-data",
    is_flag=True,
    default=False,
    help="Save intermediary data to CSV file.",
)
@click.option(
    "--load-data",
    is_flag=True,
    default=False,
    help="Load intermediary data from CSV file.",
)
@click.option(
    "--list-logs",
    "list_only",
    is_flag=True,
    default=False,
    help="Only list available logs, don't generate plots.",
)
@click.option(
    "--multi",
    is_flag=True,
    default=False,
    help="Process all logs in --logs-dir for multi-model plots.",
)
@click.option(
    "--latex",
    is_flag=True,
    default=False,
    help="Output LaTeX table instead of PNG image (for accuracy-heatmap).",
)
@click.option(
    "--metric",
    type=click.Choice(
        ["utilitarian_accuracy", "nash_accuracy", "rawlsian_accuracy", "nash_social_accuracy"],
        case_sensitive=False,
    ),
    default="utilitarian_accuracy",
    help="Which accuracy metric to use for heatmap.",
)
def main(
    log_path: Optional[Path],
    logs_dir: Path,
    max_logs: int,
    output_dir: Path,
    plots: tuple[str, ...],
    prefix: str,
    save_data: bool,
    load_data: bool,
    list_only: bool,
    multi: bool,
    latex: bool,
    metric: str,
) -> None:
    """Generate plots from evaluation logs.

    This CLI uses a clean intermediary data format for efficient processing.
    Data is loaded once and can be saved for later use.

    Examples:

        # List available logs
        uv run python -m eval.plots.cli --list-logs

        # Generate default plots for a specific log
        uv run python -m eval.plots.cli --log-path logs/your-log.eval

        # Generate plots with intermediary data saved
        uv run python -m eval.plots.cli --log-path logs/your-log.eval --save-data

        # Generate multi-model heatmap from all logs
        uv run python -m eval.plots.cli --multi --plot accuracy-heatmap
    """
    # Set default plots if none specified
    if not plots or plots == ("all",):
        plots = ("accuracy", "welfare", "heatmap")
    elif "all" in plots:
        plots = ("accuracy", "welfare", "heatmap", "welfare-heatmap", "accuracy-heatmap", "action-probability-grid")

    # List logs mode
    if list_only:
        log_info = list_eval_logs(str(logs_dir), max_logs=max_logs)
        print(f"\nAvailable logs in {logs_dir}:")
        print(f"{'#':<4} {'Samples':<8} {'Type':<25} {'Model':<40} {'Filename'}")
        print("-" * 120)
        for i, entry in enumerate(log_info):
            log_type = entry.get("log_type", "unknown")
            model = entry.get("model_name") or "?"
            samples = entry.get("num_samples", 0)
            filename = entry.get("filename", "?")
            print(f"{i:<4} {samples:<8} {log_type:<25} {model:<40} {filename}")
        return

    # Multi-model mode
    if multi:
        print(f"Processing logs from: {logs_dir}")
        log_info = list_eval_logs(str(logs_dir), max_logs=max_logs)
        eval_logs = filter_evaluation_logs(log_info, log_type="evaluation", min_samples=1)

        if not eval_logs:
            print("No evaluation logs found.")
            return

        log_paths = [entry["path"] for entry in eval_logs]
        if load_data:
            pickle_path = Path(output_dir) / f"{prefix}data_all.pkl" if prefix else Path(output_dir) / "data_all.pkl"
            with open(pickle_path, "rb") as f:
                multi_data = pickle.load(f)
        else:
            multi_data = process_multiple_logs(log_paths, verbose=True)

        run_multi_log_analysis(
            multi_data,
            plot_kinds=plots,
            output_dir=str(output_dir),
            prefix=prefix,
            save_data=save_data,
            latex=latex,
            metric=metric,
        )
        return

    # Single log mode
    if log_path:
        run_data = process_eval_log(str(log_path))
        if run_data is None:
            print(f"Failed to process {log_path}")
            return

        run_single_log_analysis(
            run_data,
            plot_kinds=plots,
            output_dir=str(output_dir),
            prefix=prefix,
            save_data=save_data,
        )
    else:
        # Auto-select first available log
        log_info = list_eval_logs(str(logs_dir), max_logs=max_logs)
        eval_logs = filter_evaluation_logs(log_info, log_type="evaluation", min_samples=1)

        if not eval_logs:
            print("No evaluation logs found. Use --list-logs to see available logs.")
            return

        selected = eval_logs[0]
        print(f"Auto-selected: {selected['filename']}")

        run_data = process_eval_log(selected["path"])
        if run_data is None:
            print(f"Failed to process {selected['path']}")
            return

        run_single_log_analysis(
            run_data,
            plot_kinds=plots,
            output_dir=str(output_dir),
            prefix=prefix,
            save_data=save_data,
        )


if __name__ == "__main__":
    main()
