"""Heatmap plot renderers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..data.processor import GAME_TYPE_ORDER
from ..data.schemas import EvalRunData, MultiEvalData
from .base import MultiLogRenderer, SingleLogRenderer, setup_plot_style


def _create_outcome_matrix(game_df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
    """Create a matrix of action pair frequencies for a game type."""
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


class ActionHeatmapRenderer(SingleLogRenderer):
    """Render action outcome heatmaps per game type."""

    def render(self, run_data: EvalRunData, **kwargs) -> Path:
        """Render heatmaps showing action outcomes by game type.

        Args:
            run_data: Processed evaluation run data

        Returns:
            Path to the saved plot
        """
        setup_plot_style()

        df = run_data.get_valid_samples_df()
        if df.empty:
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            output_path = self.get_output_path(f"action_{run_data.model_name.replace('/', '-')}.png")
            plt.savefig(output_path)
            plt.close(fig)
            return output_path

        model_name = run_data.model_name

        # Get game types in order
        game_types_ordered = [gt for gt in GAME_TYPE_ORDER if gt in df["game_type"].values]
        num_games = len(game_types_ordered)

        if num_games == 0:
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            output_path = self.get_output_path(f"action_{model_name.replace('/', '-')}.png")
            plt.savefig(output_path)
            plt.close(fig)
            return output_path

        ncols = 3
        nrows = (num_games + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
        axes = axes.flatten() if num_games > 1 else [axes]

        for idx, game_type in enumerate(game_types_ordered):
            ax = axes[idx]
            game_df = df[df["game_type"] == game_type]

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

        # Hide unused subplots
        for idx in range(num_games, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"Action Outcome Distribution by Game Type\n{model_name}",
            fontsize=16,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()

        safe_model_name = model_name.replace(" ", "_").replace("/", "-")
        output_path = self.get_output_path(f"action_{safe_model_name}.png")
        plt.savefig(output_path)
        plt.close(fig)

        return output_path


# Model name mapping for display
MODEL_NAME_MAP = {
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5-nano": "GPT-5 Nano",
    "gpt-5.1": "GPT-5.1",
    "gpt-5.2": "GPT-5.2",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "claude-sonnet-4.5": "Sonnet 4.5",
    "claude-3.5-sonnet": "Claude 3.5",
}

GAME_TYPE_NAME_MAP = {
    "Prisoner's Dilemma": "PD",
    "Chicken": "Chicken",
    "Matching pennies": "MP",
    "Bach or Stravinski": "BoS",
    "Stag hunt": "Stag",
    "Coordination": "Coord",
    "No conflict": "NC",
}

EXCLUDED_GAME_TYPES = {"Matching pennies"}


def shorten_model_name(model_name: str) -> str:
    """Shorten model name for display."""
    import re
    if not model_name or model_name == "?":
        return model_name
    normalized = model_name.lower()
    normalized = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', normalized)
    if "/" in normalized:
        normalized = normalized.split("/")[-1]
    if normalized in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[normalized]
    for key, value in MODEL_NAME_MAP.items():
        if key.lower() in normalized or normalized in key.lower():
            return value
    return model_name


def shorten_game_type_name(game_type: str) -> str:
    """Shorten game type name for display."""
    return GAME_TYPE_NAME_MAP.get(game_type, game_type)


class AccuracyHeatmapRenderer(MultiLogRenderer):
    """Render accuracy heatmap across models and game types."""

    def render(
        self,
        multi_data: MultiEvalData,
        welfare_metric: str = "Utilitarian",
        **kwargs,
    ) -> Path:
        """Render accuracy heatmap.

        Args:
            multi_data: Data from multiple evaluation runs
            welfare_metric: Which accuracy metric to plot

        Returns:
            Path to saved plot
        """
        setup_plot_style()

        from ..data.processor import compute_accuracy_by_game

        rows = []
        for run in multi_data.runs:
            if not run.samples:
                continue

            model_name = shorten_model_name(run.model_name)
            accuracy_df = compute_accuracy_by_game(run)

            if accuracy_df.empty or welfare_metric not in accuracy_df.columns:
                continue

            for game_type, score in accuracy_df[welfare_metric].items():
                if game_type in EXCLUDED_GAME_TYPES:
                    continue
                short_game = shorten_game_type_name(game_type)
                rows.append({
                    "model_name": model_name,
                    "game_type": short_game,
                    "score": score,
                })

        if not rows:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            output_path = self.get_output_path("accuracy_heatmap_empty.png")
            plt.savefig(output_path)
            plt.close(fig)
            return output_path

        heatmap_df = pd.DataFrame(rows)
        heatmap_df = heatmap_df.groupby(["model_name", "game_type"], as_index=False)["score"].mean()
        pivot_df = heatmap_df.pivot(index="model_name", columns="game_type", values="score")

        # Reorder columns
        ordered_cols = []
        for full_name in GAME_TYPE_ORDER:
            if full_name in EXCLUDED_GAME_TYPES:
                continue
            short_name = shorten_game_type_name(full_name)
            if short_name in pivot_df.columns:
                ordered_cols.append(short_name)
        for col in pivot_df.columns:
            if col not in ordered_cols:
                ordered_cols.append(col)
        pivot_df = pivot_df[ordered_cols]

        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot_df) * 0.5)))

        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0.5,
            vmin=0,
            vmax=1,
            cbar_kws={"label": f"{welfare_metric} Accuracy"},
            ax=ax,
            linewidths=0.5,
            linecolor='gray',
        )

        ax.set_xlabel("Game Type", fontsize=12, fontweight="bold")
        ax.set_ylabel("Model", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Accuracy Scores Across Models and Game Types\n({welfare_metric})",
            fontsize=14,
            fontweight="bold",
        )

        plt.xticks(rotation=0, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        metric_str = welfare_metric.lower().replace(" ", "-")
        output_path = self.get_output_path(f"accuracy_heatmap_{metric_str}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return output_path


def render_action_heatmap(
    run_data: EvalRunData,
    output_dir: str = "assets",
    prefix: str = "",
) -> Path:
    """Convenience function to render action heatmap."""
    renderer = ActionHeatmapRenderer(output_dir=output_dir, prefix=prefix)
    return renderer.render(run_data)


def render_accuracy_heatmap(
    multi_data: MultiEvalData,
    output_dir: str = "assets",
    prefix: str = "",
    welfare_metric: str = "Utilitarian",
) -> Path:
    """Convenience function to render accuracy heatmap."""
    renderer = AccuracyHeatmapRenderer(output_dir=output_dir, prefix=prefix)
    return renderer.render(multi_data, welfare_metric=welfare_metric)


def collect_model_game_actions(
    multi_data: MultiEvalData,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Collect action pairs from all evaluation runs, grouped by (model, game_type).

    Args:
        multi_data: Data from multiple evaluation runs

    Returns:
        Dictionary mapping (model_name, game_type) to list of (std_row, std_col) action pairs
    """
    model_game_actions: dict[tuple[str, str], list[tuple[str, str]]] = {}

    for run in multi_data.runs:
        if not run.samples:
            continue

        model_name = run.model_name
        if not model_name or model_name == "?":
            continue

        df = run.get_valid_samples_df()
        if df.empty:
            continue

        # Group by game type and collect standardized actions
        for game_type in df["game_type"].unique():
            if pd.isna(game_type):
                continue
            game_df = df[df["game_type"] == game_type]
            key = (model_name, str(game_type))

            # Extract action pairs
            action_pairs = []
            for _, row in game_df.iterrows():
                if pd.notna(row["std_row"]) and pd.notna(row["std_col"]):
                    action_pairs.append((row["std_row"], row["std_col"]))

            if action_pairs:
                if key in model_game_actions:
                    model_game_actions[key].extend(action_pairs)
                else:
                    model_game_actions[key] = action_pairs

    return model_game_actions


def compute_probability_matrix(
    action_pairs: list[tuple[str, str]],
) -> tuple[np.ndarray, int]:
    """Compute a 2x2 probability matrix from action pairs.

    The matrix layout is:
        [0,0] = P(UP, LEFT)     [0,1] = P(UP, RIGHT)
        [1,0] = P(DOWN, LEFT)   [1,1] = P(DOWN, RIGHT)

    Args:
        action_pairs: List of (row_action, col_action) tuples where
                      row_action is "UP" or "DOWN" and col_action is "LEFT" or "RIGHT"

    Returns:
        Tuple of (probability_matrix, total_count) where probability_matrix is 2x2 numpy array
        with probabilities summing to 1.0, and total_count is the number of valid action pairs
    """
    matrix = np.zeros((2, 2))
    valid_count = 0

    for row_action, col_action in action_pairs:
        if row_action in ["UP", "DOWN"] and col_action in ["LEFT", "RIGHT"]:
            r_idx = 0 if row_action == "UP" else 1
            c_idx = 0 if col_action == "LEFT" else 1
            matrix[r_idx, c_idx] += 1
            valid_count += 1

    # Convert counts to probabilities
    if valid_count > 0:
        matrix = matrix / valid_count

    return matrix, valid_count


def render_probability_cell(
    ax: plt.Axes,
    matrix: np.ndarray,
    total: int,
    game: str,
    model: str,
    is_first_row: bool,
    is_first_col: bool,
) -> None:
    """Render a single probability matrix cell in the grid.

    Args:
        ax: Matplotlib axes to render on
        matrix: 2x2 probability matrix
        total: Total number of samples
        game: Game type name (for title)
        model: Model name (for y-label)
        is_first_row: If True, add game title
        is_first_col: If True, add model label
    """
    # Plot mini heatmap
    ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    # Add probability labels
    for r in range(2):
        for c in range(2):
            text_color = "white" if matrix[r, c] > 0.5 else "black"
            ax.text(
                c, r, f"{matrix[r, c]:.2f}",
                ha="center", va="center",
                color=text_color, fontsize=9, fontweight="bold"
            )

    # Remove tick marks and grid lines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)

    # Title for first row (game names)
    if is_first_row:
        ax.set_title(game, fontsize=10, fontweight="bold", pad=10)

    # Y-label for first column (model names)
    if is_first_col:
        ax.set_ylabel(
            model.split("/")[-1][:20],
            fontsize=9, fontweight="bold", rotation=90, labelpad=10
        )

    # Add sample count in corner
    ax.text(
        0.98, 0.02, f"n={total}", transform=ax.transAxes,
        fontsize=7, ha="right", va="bottom", alpha=0.6
    )


def render_empty_cell(
    ax: plt.Axes,
    game: str,
    model: str,
    is_first_row: bool,
    is_first_col: bool,
) -> None:
    """Render an empty cell (no data) in the grid.

    Args:
        ax: Matplotlib axes to render on
        game: Game type name (for title)
        model: Model name (for y-label)
        is_first_row: If True, add game title
        is_first_col: If True, add model label
    """
    ax.text(
        0.5, 0.5, "No data", ha="center", va="center",
        transform=ax.transAxes, fontsize=9, alpha=0.5
    )
    ax.set_xticks([])
    ax.set_yticks([])

    if is_first_row:
        ax.set_title(game, fontsize=10, fontweight="bold", pad=10)
    if is_first_col:
        ax.set_ylabel(
            model.split("/")[-1][:20],
            fontsize=9, fontweight="bold", rotation=90, labelpad=10
        )


class ActionProbabilityGridRenderer(MultiLogRenderer):
    """Render a grid showing 2x2 action probability matrices for each model-game combination."""

    def render(self, multi_data: MultiEvalData, **kwargs) -> Path:
        """Render action probability grid.

        Creates a grid where each cell shows a 2x2 probability matrix
        for a specific model-game combination.

        Args:
            multi_data: Data from multiple evaluation runs

        Returns:
            Path to saved plot
        """
        setup_plot_style()

        # Collect action data for each model-game combination
        model_game_actions = collect_model_game_actions(multi_data)

        if not model_game_actions:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No valid action data found", ha="center", va="center")
            output_path = self.get_output_path("action_probability_grid_empty.png")
            plt.savefig(output_path)
            plt.close(fig)
            return output_path

        # Extract unique models and games
        models = sorted(set(k[0] for k in model_game_actions.keys()))
        games = [g for g in GAME_TYPE_ORDER if any(k[1] == g for k in model_game_actions.keys())]

        if not models or not games:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No valid models or games found", ha="center", va="center")
            output_path = self.get_output_path("action_probability_grid_empty.png")
            plt.savefig(output_path)
            plt.close(fig)
            return output_path

        # Create figure with subplots
        fig = plt.figure(figsize=(len(games) * 2.5, len(models) * 2.5))
        gs = fig.add_gridspec(len(models), len(games), hspace=0.15, wspace=0.15)

        for i, model in enumerate(models):
            for j, game in enumerate(games):
                ax = fig.add_subplot(gs[i, j])
                key = (model, game)

                if key in model_game_actions:
                    action_pairs = model_game_actions[key]
                    matrix, total = compute_probability_matrix(action_pairs)
                    render_probability_cell(
                        ax, matrix, total, game, model,
                        is_first_row=(i == 0),
                        is_first_col=(j == 0),
                    )
                else:
                    render_empty_cell(
                        ax, game, model,
                        is_first_row=(i == 0),
                        is_first_col=(j == 0),
                    )

        plt.suptitle(
            "Action Probability Distribution: Models × Game Types\n(U=UP, D=DOWN, L=LEFT, R=RIGHT)",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )

        output_path = self.get_output_path("action_probability_grid.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return output_path


def render_action_probability_grid(
    multi_data: MultiEvalData,
    output_dir: str = "assets",
    prefix: str = "",
) -> Path:
    """Convenience function to render action probability grid."""
    renderer = ActionProbabilityGridRenderer(output_dir=output_dir, prefix=prefix)
    return renderer.render(multi_data)
