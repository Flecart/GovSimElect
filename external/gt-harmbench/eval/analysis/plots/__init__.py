"""Plotting functions for analysis results."""

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

__all__ = [
    "plot_due_diligence_results",
    "plot_action_probability_grid",
    "plot_epoch_score_vs_accuracy",
    "plot_epoch_score_vs_avg_scores",
    "plot_historical_trends",
    "plot_math_rank_vs_accuracy",
    "plot_math_rank_vs_avg_scores",
    "plot_welfare_heatmap",
    "plot_action_heatmaps",
    "plot_accuracy_by_game",
    "plot_welfare_by_game",
]
