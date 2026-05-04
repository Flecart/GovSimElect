"""Shared utilities for plotting."""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

def render_action_probability_grid(
    model_game_matrices: Dict[Tuple[str, str], np.ndarray],
    model_game_counts: Dict[Tuple[str, str], int],
    models: List[str],
    games: List[str],
    suptitle: str,
    output_path: str
):
    fig = plt.figure(figsize=(len(games) * 2.5, len(models) * 2.5))
    gs = fig.add_gridspec(len(models), len(games), hspace=0.15, wspace=0.15)

    for i, model in enumerate(models):
        for j, game in enumerate(games):
            ax = fig.add_subplot(gs[i, j])
            
            key = (model, game)
            if key in model_game_matrices:
                matrix = model_game_matrices[key]
                total = model_game_counts[key]
                
                # Plot mini heatmap
                im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
                
                # Add probability labels
                for r in range(2):
                    for c in range(2):
                        text_color = "white" if matrix[r, c] > 0.5 else "black"
                        ax.text(c, r, f"{matrix[r, c]:.2f}",
                             ha="center", va="center",
                             color=text_color, fontsize=9, fontweight="bold")
                
                # Remove tick marks and grid lines
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(length=0)
                
                # Title for first row (game names)
                if i == 0:
                    ax.set_title(game, fontsize=10, fontweight="bold", pad=10)
                
                # Y-label for first column (model names)
                if j == 0:
                    ax.set_ylabel(model.split("/")[-1][:20], fontsize=9, fontweight="bold", rotation=90, labelpad=10)
                
                # Add sample count in corner
                ax.text(0.98, 0.02, f"n={total}", transform=ax.transAxes,
                       fontsize=7, ha="right", va="bottom", alpha=0.6)
            else:
                # No data for this combination
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                       transform=ax.transAxes, fontsize=9, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                
                if i == 0:
                    ax.set_title(game, fontsize=10, fontweight="bold", pad=10)
                if j == 0:
                    ax.set_ylabel(model.split("/")[-1][:20], fontsize=9, fontweight="bold", rotation=90, labelpad=10)

    plt.suptitle(
        suptitle,
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
