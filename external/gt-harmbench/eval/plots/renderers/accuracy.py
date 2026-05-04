"""Accuracy plot renderer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.processor import compute_accuracy_by_game
from ..data.schemas import EvalRunData
from .base import SingleLogRenderer, setup_plot_style


class AccuracyByGameRenderer(SingleLogRenderer):
    """Render accuracy by game type bar plot."""

    def render(self, run_data: EvalRunData, **kwargs) -> Path:
        """Render the accuracy by game bar plot.

        Args:
            run_data: Processed evaluation run data

        Returns:
            Path to the saved plot
        """
        setup_plot_style()

        accuracy_by_game = compute_accuracy_by_game(run_data)
        if accuracy_by_game.empty:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            output_path = self.get_output_path(f"accuracy_by_game_{run_data.model_name.replace('/', '-')}.png")
            plt.savefig(output_path)
            plt.close(fig)
            return output_path

        model_name = run_data.model_name

        fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(len(accuracy_by_game.index))
        width = 0.20

        bars2 = ax.bar(x - width, accuracy_by_game["Utilitarian"], width, label="Efficiency", alpha=0.8)
        bars3 = ax.bar(x, accuracy_by_game["Rawlsian"], width, label="Fairness", alpha=0.8)
        bars4 = ax.bar(x + 1 * width, accuracy_by_game["Nash Equilibrium"], width, label="Nash Equilibrium", alpha=0.8)
        bars5 = ax.bar(x + 2 * width, accuracy_by_game["Nash Social Welfare"], width, label="Nash Social Welfare", alpha=0.8)

        ax.set_xlabel("Game Type", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"Accuracy by Game Type for {model_name}", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(accuracy_by_game.index, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bars in [bars2, bars3, bars4, bars5]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()

        safe_model_name = model_name.replace(" ", "_").replace("/", "-")
        output_path = self.get_output_path(f"accuracy_by_game_{safe_model_name}.png")
        plt.savefig(output_path)
        plt.close(fig)

        return output_path


def render_accuracy_plot(
    run_data: EvalRunData,
    output_dir: str = "assets",
    prefix: str = "",
) -> Path:
    """Convenience function to render accuracy plot.

    Args:
        run_data: Processed evaluation run data
        output_dir: Directory to save plot
        prefix: Prefix for filename

    Returns:
        Path to saved plot
    """
    renderer = AccuracyByGameRenderer(output_dir=output_dir, prefix=prefix)
    return renderer.render(run_data)
