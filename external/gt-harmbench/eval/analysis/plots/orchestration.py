"""Orchestration functions for running analysis pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..core import (
    build_actions_df,
    build_dataframe,
    compute_metrics,
    load_log_samples_from_path,
)
from .due_diligence import plot_due_diligence_results
from .multi_log import (
    plot_action_probability_grid,
    plot_epoch_score_vs_accuracy,
    plot_epoch_score_vs_avg_scores,
    plot_historical_trends,
    plot_math_rank_vs_accuracy,
    plot_math_rank_vs_avg_scores,
    plot_welfare_heatmap,
)
from .single_log import (
    plot_action_heatmaps,
    plot_accuracy_by_game,
    plot_welfare_by_game,
)


def handle_due_diligence_plot(
    log_info: List[Dict[str, Any]],
    *,
    max_logs: int,
    output_dir: str,
    prefix: str,
) -> None:
    """Handle due diligence plotting (processes all logs)."""
    dd_path = plot_due_diligence_results(log_info, max_logs=max_logs, output_dir=output_dir, prefix=prefix)
    print(f"Saved due diligence plot to: {dd_path}")


def handle_multi_log_plots(
    log_info: List[Dict[str, Any]],
    plot_kinds: Sequence[str],
    *,
    max_logs: int,
    output_dir: str,
    prefix: str,
) -> None:
    """Handle plots that process all logs (history, epoch, math plots, welfare-heatmap, action-probability-grid)."""
    if "history" in plot_kinds:
        history_path = plot_historical_trends(log_info, max_logs=max_logs, output_dir=output_dir, prefix=prefix)
        print(f"Saved historical trend plot to: {history_path}")
    
    if "epoch-accuracy" in plot_kinds:
        epoch_acc_path = plot_epoch_score_vs_accuracy(log_info, max_logs=max_logs, output_dir=output_dir, prefix=prefix)
        print(f"Saved epoch score vs accuracy plot to: {epoch_acc_path}")
    
    if "epoch-scores" in plot_kinds:
        epoch_scores_path = plot_epoch_score_vs_avg_scores(log_info, max_logs=max_logs, output_dir=output_dir, prefix=prefix)
        print(f"Saved epoch score vs avg scores plot to: {epoch_scores_path}")
    
    if "welfare-heatmap" in plot_kinds:
        welfare_heatmap_path = plot_welfare_heatmap(log_info, max_logs=max_logs, output_dir=output_dir, prefix=prefix)
        print(f"Saved welfare heatmap to: {welfare_heatmap_path}")
    
    if "action-probability-grid" in plot_kinds:
        action_prob_path = plot_action_probability_grid(log_info, max_logs=max_logs, output_dir=output_dir, prefix=prefix)
        print(f"Saved action probability grid to: {action_prob_path}")
    
    # Deprecated: math plots
    if "math-accuracy" in plot_kinds:
        import warnings
        warnings.warn("math-accuracy plot is deprecated, use epoch-accuracy instead", DeprecationWarning, stacklevel=2)
        math_acc_path = plot_math_rank_vs_accuracy(log_info, max_logs=max_logs, output_dir=output_dir, prefix=prefix)
        print(f"Saved math rank vs accuracy plot to: {math_acc_path}")
    
    if "math-scores" in plot_kinds:
        import warnings
        warnings.warn("math-scores plot is deprecated, use epoch-scores instead", DeprecationWarning, stacklevel=2)
        math_scores_path = plot_math_rank_vs_avg_scores(log_info, max_logs=max_logs, output_dir=output_dir, prefix=prefix)
        print(f"Saved math rank vs avg scores plot to: {math_scores_path}")


def select_and_load_log(
    log_path: Optional[str],
    log_type: Optional[str],
    log_info: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], str, str, str]:
    """Select and load a log file for single-log analysis.
    
    Returns:
        Tuple of (samples, model_name, log_path, log_type)
    """
    if log_path:
        # Use provided path
        selected_log_path = log_path
        if not log_type:
            # Auto-detect
            path_obj = Path(log_path)
            if path_obj.is_dir():
                selected_log_type = "dir"
            elif path_obj.suffix == ".eval":
                selected_log_type = "eval"
            else:
                print(f"Error: Cannot determine log type for {log_path}. Please specify log_type.")
                return [], "", "", ""
        else:
            selected_log_type = log_type
        samples, model_name = load_log_samples_from_path(selected_log_path, selected_log_type)
        return samples, model_name, selected_log_path, selected_log_type
    else:
        # Fall back to index-based selection (for backward compatibility)
        # Find first log with samples that is NOT a due diligence log
        log_with_samples = [info for info in log_info if info["num_samples"] > 0]
        if not log_with_samples:
            print("No logs with samples found; exiting.")
            return [], "", "", ""
        
        # Try to find a non-due-diligence log
        selected = None
        for info in log_with_samples:
            # Quick check: skip if filename suggests due diligence
            if "game-classification" not in Path(info["path"]).name and "nash-equilibrium-detection" not in Path(info["path"]).name:
                selected = info
                break
        
        # If all logs appear to be due diligence, use the first one but warn
        if selected is None:
            selected = log_with_samples[0]
            print(f"\n[WARNING] Selected log appears to be a due diligence log. "
                  f"Use --plot due-diligence to analyze due diligence results.")
        
        selected_log_path = selected["path"]
        selected_log_type = selected["type"]
        print(
            f"\n[AUTO SELECTION] Using: {Path(selected_log_path).name} "
            f"({selected['num_samples']} samples, {selected_log_type})"
        )
        samples, model_name = load_log_samples_from_path(selected_log_path, selected_log_type)
        return samples, model_name, selected_log_path, selected_log_type


def process_single_log_analysis(
    samples: List[Dict[str, Any]],
    model_name: str,
    plot_kinds: Sequence[str],
    *,
    output_dir: str,
    prefix: str,
) -> None:
    """Process a single log: build dataframe, compute metrics, and generate plots."""
    df = build_dataframe(samples)
    _, accuracy_by_game, welfare_by_game, _, _, _ = compute_metrics(df)
    actions_df = build_actions_df(df)

    if "accuracy" in plot_kinds:
        acc_path = plot_accuracy_by_game(accuracy_by_game, model_name=model_name, output_dir=output_dir, prefix=prefix)
        print(f"\nSaved accuracy plot to: {acc_path}")

    if "welfare" in plot_kinds:
        welfare_path = plot_welfare_by_game(welfare_by_game, model_name=model_name, output_dir=output_dir, prefix=prefix)
        print(f"Saved welfare plot to: {welfare_path}")

    if "heatmap" in plot_kinds:
        heatmap_path = plot_action_heatmaps(actions_df, model_name=model_name, output_dir=output_dir, prefix=prefix)
        print(f"Saved action heatmaps to: {heatmap_path}")