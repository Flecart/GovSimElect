"""CLI entrypoint for running log analysis."""

from __future__ import annotations

import click

from pathlib import Path

from .core import run_analysis


def run_cli(
    log_path: Path | None,
    log_type: str | None,
    max_logs: int,
    output_dir: Path,
    plots: tuple[str, ...] | None = None,
    prefix: str | None = None,
) -> None:
    """Recompute strategy metrics from logs and generate plots."""
    run_analysis(
        log_path=str(log_path) if log_path else None,
        log_type=log_type,
        max_logs=max_logs,
        output_dir=str(output_dir),
        plot_kinds=plots,
        prefix=prefix or "",
    )


@click.command()
@click.option(
    "--log-path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Path to the log file (.eval) or directory to analyze. If not provided, will list available logs and use the first one.",
)
@click.option(
    "--log-type",
    type=click.Choice(["eval", "dir"], case_sensitive=False),
    default=None,
    help="Type of log: 'eval' for compressed .eval file, 'dir' for directory. Auto-detected if not specified.",
)
@click.option(
    "--max-logs",
    type=int,
    default=15,
    help="Maximum number of recent logs to list and consider (when log-path is not provided).",
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
    type=click.Choice(["accuracy", "welfare", "heatmap", "action-probability-grid", "history", "epoch-accuracy", "epoch-scores", "math-accuracy", "math-scores",  "welfare-heatmap" , "due-diligence"], case_sensitive=False),
    multiple=True,
    help="Which plots to generate (can be given multiple times). Default: accuracy,welfare,heatmap. Note: math-accuracy and math-scores are deprecated, use epoch-accuracy and epoch-scores instead.",
)
@click.option(
    "--prefix",
    type=str,
    default=None,
    help="Prefix to add to output filenames (e.g., 'gamify-' will create 'gamify-accuracy_by_game_...').",
)
def main(log_path: Path | None, log_type: str | None, max_logs: int, output_dir: Path, plots: tuple[str, ...], prefix: str | None) -> None:
    """Click entrypoint that forwards to ``run_cli``."""
    if not plots:
        plots = ("accuracy", "welfare", "heatmap")
    
    # Auto-detect log type if not specified
    if log_path and not log_type:
        if log_path.is_dir():
            log_type = "dir"
        elif log_path.suffix == ".eval":
            log_type = "eval"
        else:
            click.echo(f"Error: Cannot determine log type for {log_path}. Please specify --log-type.", err=True)
            return
    
    # Warn about deprecated math plots
    if "math-accuracy" in plots or "math-scores" in plots:
        import warnings
        warnings.warn(
            "math-accuracy and math-scores plots are deprecated. Use epoch-accuracy and epoch-scores instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    
    run_cli(log_path=log_path, log_type=log_type, max_logs=max_logs, output_dir=output_dir, plots=plots, prefix=prefix)


if __name__ == "__main__":
    main()


