"""Evaluation plots module.

This module provides a clean architecture for generating plots from .eval files:

1. **Data Layer** (eval/plots/data/):
   - loader.py: Load .eval files using inspect_ai
   - processor.py: Process samples into DataFrames with metrics
   - schemas.py: Data class definitions (SampleRecord, EvalRunData, MultiEvalData)

2. **Renderers** (eval/plots/renderers/):
   - accuracy.py: Accuracy by game type bar plots
   - welfare.py: Welfare scores by game type bar plots
   - heatmap.py: Action heatmaps and cross-model accuracy heatmaps

3. **CLI** (eval/plots/cli.py):
   - Command-line interface for generating plots

Usage:
    # Via CLI
    PYTHONPATH=. uv run python -m eval.plots.cli --plot accuracy --log-path logs/your-log.eval

    # Via Python
    from eval.plots.data import process_eval_log
    from eval.plots.renderers import render_accuracy_plot

    run_data = process_eval_log("logs/your-log.eval")
    render_accuracy_plot(run_data, output_dir="assets")
"""

from .data import (
    GAME_TYPE_ORDER,
    EvalRunData,
    MultiEvalData,
    SampleRecord,
    compute_accuracy_by_game,
    compute_overall_accuracy,
    compute_welfare_by_game,
    filter_evaluation_logs,
    list_eval_logs,
    load_eval_log,
    process_eval_log,
    process_multiple_logs,
)
from .renderers import (
    AccuracyByGameRenderer,
    AccuracyHeatmapRenderer,
    ActionHeatmapRenderer,
    WelfareByGameRenderer,
    render_accuracy_heatmap,
    render_accuracy_plot,
    render_action_heatmap,
    render_welfare_plot,
)

__all__ = [
    # Data layer
    "list_eval_logs",
    "load_eval_log",
    "filter_evaluation_logs",
    "process_eval_log",
    "process_multiple_logs",
    "compute_accuracy_by_game",
    "compute_welfare_by_game",
    "compute_overall_accuracy",
    "GAME_TYPE_ORDER",
    # Schemas
    "SampleRecord",
    "EvalRunData",
    "MultiEvalData",
    # Renderers
    "AccuracyByGameRenderer",
    "WelfareByGameRenderer",
    "ActionHeatmapRenderer",
    "AccuracyHeatmapRenderer",
    "render_accuracy_plot",
    "render_welfare_plot",
    "render_action_heatmap",
    "render_accuracy_heatmap",
]
