"""Welfare plot renderer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.processor import compute_welfare_by_game
from ..data.schemas import EvalRunData
from .base import SingleLogRenderer, setup_plot_style


class WelfareByGameRenderer(SingleLogRenderer):
    """Render welfare scores by game type bar plot."""

    def render(self, run_data: EvalRunData, **kwargs) -> Path:
        """Render the welfare by game bar plot.

        Args:
            run_data: Processed evaluation run data

        Returns:
            Path to the saved plot
        """
        setup_plot_style()

        welfare_by_game = compute_welfare_by_game(run_data)
        if welfare_by_game.empty:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            output_path = self.get_output_path(f"welfare_by_game_{run_data.model_name.replace('/', '-')}.png")
            plt.savefig(output_path)
            plt.close(fig)
            return output_path

        model_name = run_data.model_name

        fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(len(welfare_by_game.index))
        width = 0.25

        bars_u = ax.bar(
            x - width,
            welfare_by_game["utilitarian_efficiency"],
            width,
            label="utilitarian_efficiency",
            alpha=0.8,
        )
        bars_r = ax.bar(
            x,
            welfare_by_game["rawlsian_efficiency"],
            width,
            label="rawlsian_efficiency",
            alpha=0.8,
        )
        bars_n = ax.bar(
            x + width,
            welfare_by_game["nash_social_welfare_efficiency"],
            width,
            label="nash_social_welfare_efficiency",
            alpha=0.8,
        )

        ax.set_xlabel("Game Type", fontsize=12)
        ax.set_ylabel("Normalized Score", fontsize=12)
        ax.set_title(
            f"Average Welfare by Game Type for {model_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(welfare_by_game.index, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for bars in [bars_u, bars_r, bars_n]:
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
        output_path = self.get_output_path(f"welfare_by_game_{safe_model_name}.png")
        plt.savefig(output_path)
        plt.close(fig)

        return output_path


def render_welfare_plot(
    run_data: EvalRunData,
    output_dir: str = "assets",
    prefix: str = "",
) -> Path:
    """Convenience function to render welfare plot.

    Args:
        run_data: Processed evaluation run data
        output_dir: Directory to save plot
        prefix: Prefix for filename

    Returns:
        Path to saved plot
    """
    renderer = WelfareByGameRenderer(output_dir=output_dir, prefix=prefix)
    return renderer.render(run_data)
