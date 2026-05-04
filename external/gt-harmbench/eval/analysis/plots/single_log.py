"""Plotting functions for single log analysis."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..constants import GAME_TYPE_ORDER


def _create_outcome_matrix(game_df: pd.DataFrame) -> tuple[np.ndarray, List[str], List[str]]:
    """Create a matrix of action pair frequencies for a game type using standardized labels."""
    row_actions = ["UP", "DOWN"]
    col_actions = ["LEFT", "RIGHT"]

    matrix = np.zeros((len(row_actions), len(col_actions)))
    for _, row in game_df.iterrows():
        if row["std_row"] and row["std_col"]:
            try:
                r_idx = row_actions.index(row["std_row"])
                c_idx = col_actions.index(row["std_col"])
                matrix[r_idx, c_idx] += 1
            except (ValueError, IndexError):
                continue

    return matrix, row_actions, col_actions


def plot_accuracy_by_game(
    accuracy_by_game: pd.DataFrame,
    model_name: str,
    output_dir: str = "assets",
    prefix: str = "",
) -> Path:
    """Create and save the accuracy-by-game bar plot."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(accuracy_by_game.index))
    width = 0.20

    bars2 = ax.bar(x - width, accuracy_by_game["Utilitarian"], width, label="Efficiency", alpha=0.8)
    bars3 = ax.bar(x, accuracy_by_game["Rawlsian"], width, label="Fairness", alpha=0.8)
    bars4 = ax.bar(x + 1 * width, accuracy_by_game["Nash Equilibrium"], width, label="Nash Equilibrium", alpha=0.8)
    bars5 = ax.bar(x + 2 * width, accuracy_by_game["Nash Social Welfare"], width, label="Nash Social Welfare", alpha=0.8)

    ax.set_xlabel("Game Type", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Accuracy by Game Type standard for {model_name}", fontsize=14, fontweight="bold")
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
    prefix_str = f"{prefix}-" if prefix else ""
    file_path = output_path / f"{prefix_str}accuracy_by_game_{model_name.replace(' ', '_').replace('/', '-')}.png"
    plt.savefig(file_path)
    plt.close(fig)

    return file_path


def plot_welfare_by_game(
    welfare_by_game: pd.DataFrame,
    model_name: str,
    output_dir: str = "assets",
    prefix: str = "",
) -> Path:
    """Create and save the average welfare-by-game bar plot."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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
    ax.set_ylabel("Normalized score", fontsize=12)
    ax.set_title(
        f"Average welfare by Game Type for {model_name}",
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
    prefix_str = f"{prefix}-" if prefix else ""
    file_path = output_path / f"{prefix_str}welfare_by_game_{model_name.replace(' ', '_').replace('/', '-')}.png"
    plt.savefig(file_path)
    plt.close(fig)

    return file_path


def plot_action_heatmaps(
    actions_df: pd.DataFrame,
    model_name: str,
    output_dir: str = "assets",
    prefix: str = "",
) -> Path:
    """Create and save action outcome heatmaps per game type."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use ordered game types (filter to only those present in the data)
    game_types_ordered = [gt for gt in GAME_TYPE_ORDER if gt in actions_df["game_type"].values]
    num_games = len(game_types_ordered)

    ncols = 3
    nrows = (num_games + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten() if num_games > 1 else [axes]

    for idx, game_type in enumerate(game_types_ordered):
        ax = axes[idx]
        game_df = actions_df[actions_df["game_type"] == game_type]

        matrix, row_actions, col_actions = _create_outcome_matrix(game_df)

        if matrix is not None and matrix.sum() > 0:
            matrix_pct = (matrix / matrix.sum() * 100) if matrix.sum() > 0 else matrix

            im = ax.imshow(matrix_pct, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

            ax.set_xticks(np.arange(len(col_actions)))
            ax.set_yticks(np.arange(len(row_actions)))
            ax.set_xticklabels(col_actions, fontsize=12, fontweight="bold")
            ax.set_yticklabels(row_actions, fontsize=12, fontweight="bold")

            for i in range(len(row_actions)):
                for j in range(len(col_actions)):
                    count = int(matrix[i, j])
                    pct = matrix_pct[i, j]
                    if count > 0:
                        ax.text(
                            j,
                            i,
                            f"{count}\n({pct:.1f}%)",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=14,
                            fontweight="bold",
                        )
                    else:
                        ax.text(
                            j,
                            i,
                            "0",
                            ha="center",
                            va="center",
                            color="gray",
                            fontsize=10,
                            alpha=0.5,
                        )

            ax.set_title(f"{game_type}\n(n={len(game_df)})", fontweight="bold", fontsize=13)
            ax.set_xlabel("Column Player Action", fontsize=11, fontweight="bold")
            ax.set_ylabel("Row Player Action", fontsize=11, fontweight="bold")

            cbar = plt.colorbar(im, ax=ax, label="Percentage (%)")
            cbar.set_label("Percentage (%)", fontweight="bold")
        else:
            ax.text(
                0.5,
                0.5,
                f"{game_type}\n(No data)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    for idx in range(num_games, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        f"Action Outcome Distribution by Game Type (Standardized Labels)\n{model_name}",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()

    prefix_str = f"{prefix}-" if prefix else ""
    file_path = output_path / f"{prefix_str}action-{model_name.replace(' ', '_').replace('/', '-')}.png"
    plt.savefig(file_path)
    plt.close(fig)

    return file_path
