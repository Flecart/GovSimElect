"""Plot renderers for evaluation data.

This module provides renderer classes and convenience functions for
creating various types of plots from evaluation data.
"""

from .accuracy import AccuracyByGameRenderer, render_accuracy_plot
from .base import BasePlotRenderer, MultiLogRenderer, SingleLogRenderer, setup_plot_style
from .heatmap import (
    AccuracyHeatmapRenderer,
    ActionHeatmapRenderer,
    ActionProbabilityGridRenderer,
    collect_model_game_actions,
    compute_probability_matrix,
    render_accuracy_heatmap,
    render_action_heatmap,
    render_action_probability_grid,
    render_empty_cell,
    render_probability_cell,
)
from .latex import render_accuracy_heatmap_latex, render_accuracy_heatmap_latex_landscape
from .welfare import WelfareByGameRenderer, render_welfare_plot

__all__ = [
    # Base
    "BasePlotRenderer",
    "SingleLogRenderer",
    "MultiLogRenderer",
    "setup_plot_style",
    # Single-log renderers
    "AccuracyByGameRenderer",
    "WelfareByGameRenderer",
    "ActionHeatmapRenderer",
    "render_accuracy_plot",
    "render_welfare_plot",
    "render_action_heatmap",
    # Multi-log renderers
    "AccuracyHeatmapRenderer",
    "ActionProbabilityGridRenderer",
    "render_accuracy_heatmap",
    "render_action_probability_grid",
    # Action probability grid helper functions
    "collect_model_game_actions",
    "compute_probability_matrix",
    "render_probability_cell",
    "render_empty_cell",
    # LaTeX renderers
    "render_accuracy_heatmap_latex",
    "render_accuracy_heatmap_latex_landscape",
]
