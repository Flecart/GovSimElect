"""Plotting functions for multi-log comparison analysis."""

from __future__ import annotations

import datetime
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Use specific font for speed
import matplotlib.pyplot as plt


plt.rcParams['figure.dpi'] = 400

import numpy as np
import pandas as pd
import seaborn as sns

from ..constants import GAME_TYPE_ORDER
from ..core import build_actions_df, build_dataframe, compute_metrics
from ..epoch import get_epoch_scores
from ..leaderboard import get_math_rankings
from ..utils import (
    EXCLUDED_GAME_TYPES,
    is_game_type_excluded,
    shorten_game_type_name,
    shorten_model_name,
)


def plot_action_probability_grid(
    log_info: List[Dict[str, Any]],
    *,
    max_logs: int = 15,
    output_dir: str = "assets",
    prefix: str = "",
) -> Path:
    """Create a grid showing 2x2 action probability matrices for each model-game combination.
    
    Args:
        log_info: List of log metadata
        max_logs: Maximum number of logs to process
        output_dir: Directory to save the plot
        prefix: Prefix for output filename
    
    Returns:
        Path to saved plot
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect action data for each model-game combination
    model_game_actions: Dict[tuple[str, str], List[tuple[str, str]]] = {}

    for entry in log_info[:max_logs]:
        if entry.get("type") != "eval" or entry.get("num_samples", 0) == 0:
            continue

        model_name = entry.get("model_name", "") or "Unknown"
        if model_name == "?":
            continue

        # Load samples for this log
        samples: List[Dict[str, Any]] = []
        with zipfile.ZipFile(entry["path"], "r") as z:
            sample_files = sorted(
                f for f in z.namelist() if f.startswith("samples/") and f.endswith(".json")
            )
            for sample_file in sample_files:
                with z.open(sample_file) as f:
                    sample = json.load(f)
                samples.append(sample)

        if not samples:
            continue

        df = build_dataframe(samples)
        actions_df = build_actions_df(df)

        # Group by game type and collect standardized actions
        for game_type in actions_df["game_type"].unique():
            if pd.isna(game_type):
                continue
            game_actions = actions_df[actions_df["game_type"] == game_type]
            key = (model_name, str(game_type))
            
            # Extract action pairs
            action_pairs = []
            for _, row in game_actions.iterrows():
                if row["std_row"] and row["std_col"]:
                    action_pairs.append((row["std_row"], row["std_col"]))
            
            if action_pairs:
                if key in model_game_actions:
                    model_game_actions[key].extend(action_pairs)
                else:
                    model_game_actions[key] = action_pairs

    if not model_game_actions:
        print("No valid action data found for action probability grid.")
        prefix_str = f"{prefix}-" if prefix else ""
        dummy = output_path / f"{prefix_str}action_probability_grid_empty.png"
        return dummy

    # Extract unique models and games
    models = sorted(set(k[0] for k in model_game_actions.keys()))
    games = [g for g in GAME_TYPE_ORDER if any(k[1] == g for k in model_game_actions.keys())]

    if not models or not games:
        print("No valid models or games found for action probability grid.")
        prefix_str = f"{prefix}-" if prefix else ""
        dummy = output_path / f"{prefix_str}action_probability_grid_empty.png"
        return dummy

    # Prepare matrices and counts for shared renderer
    model_game_matrices = {}
    model_game_counts = {}
    
    for i, model in enumerate(models):
        for j, game in enumerate(games):
            key = (model, game)
            if key in model_game_actions:
                action_pairs = model_game_actions[key]
                
                # Compute 2x2 probability matrix
                matrix = np.zeros((2, 2))
                total = len(action_pairs)
                
                for row_action, col_action in action_pairs:
                    if row_action in ["UP", "DOWN"] and col_action in ["LEFT", "RIGHT"]:
                        r_idx = 0 if row_action == "UP" else 1
                        c_idx = 0 if col_action == "LEFT" else 1
                        matrix[r_idx, c_idx] += 1
                
                # Convert to probabilities
                if total > 0:
                    matrix = matrix / total
                    
                model_game_matrices[key] = matrix
                model_game_counts[key] = total
                
    from .shared import render_action_probability_grid
    
    suptitle = "Action Probability Distribution: Models × Game Types\n(U=UP, D=DOWN, L=LEFT, R=RIGHT)"
    
    prefix_str = f"{prefix}-" if prefix else ""
    file_path = output_path / f"{prefix_str}action_probability_grid.png"
    
    render_action_probability_grid(model_game_matrices, model_game_counts, models, games, suptitle, str(file_path))

    return file_path


def plot_welfare_heatmap(
    log_info: List[Dict[str, Any]],
    *,
    max_logs: int = 15,
    output_dir: str = "assets",
    prefix: str = "",
    welfare_metric: str = "Utilitarian",
) -> Path:
    """Create a heatmap of accuracy scores across models and game types.
    
    Args:
        log_info: List of log metadata
        max_logs: Maximum number of logs to process
        output_dir: Directory to save the plot
        prefix: Prefix for output filename
        welfare_metric: Accuracy metric to plot. Options: "Utilitarian", "Rawlsian", 
                        "Nash Equilibrium", "Nash Social Welfare"
    
    Returns:
        Path to saved heatmap
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for entry in log_info[:max_logs]:
        if entry.get("type") != "eval" or entry.get("num_samples", 0) == 0:
            continue

        model_name = entry.get("model_name", "") or "Unknown"
        if model_name == "?":
            continue

        # Shorten model name
        short_model_name = shorten_model_name(model_name)

        # Load samples for this log
        samples: List[Dict[str, Any]] = []
        with zipfile.ZipFile(entry["path"], "r") as z:
            sample_files = sorted(
                f for f in z.namelist() if f.startswith("samples/") and f.endswith(".json")
            )
            for sample_file in sample_files:
                with z.open(sample_file) as f:
                    sample = json.load(f)
                samples.append(sample)

        if not samples:
            continue

        df = build_dataframe(samples)
        try:
            _, accuracy_by_game, _, _, _, _ = compute_metrics(df)
            if accuracy_by_game.empty or welfare_metric not in accuracy_by_game.columns:
                continue
        except (KeyError, ValueError, IndexError, AttributeError) as e:
            print(f"Warning: Failed to compute metrics for {model_name}: {e}", file=sys.stderr)
            continue

        # Add row for each game type (excluding matching pennies and other excluded types)
        for game_type, score in accuracy_by_game[welfare_metric].items():
            # Skip excluded game types
            if is_game_type_excluded(game_type):
                continue
            
            # Shorten game type name
            short_game_type = shorten_game_type_name(game_type)
            
            rows.append({
                "model_name": short_model_name,
                "game_type": short_game_type,
                "score": score,
            })

    if not rows:
        print("No valid models found for accuracy heatmap.")
        prefix_str = f"{prefix}-" if prefix else ""
        dummy = output_path / f"{prefix_str}accuracy_heatmap_empty.png"
        return dummy

    # Create pivot table
    heatmap_df = pd.DataFrame(rows)
    
    # Aggregate duplicates (same model, same game type) by taking the mean
    # This handles cases where multiple logs exist for the same model
    heatmap_df = heatmap_df.groupby(["model_name", "game_type"], as_index=False)["score"].mean()
    
    pivot_df = heatmap_df.pivot(index="model_name", columns="game_type", values="score")
    
    # Reorder columns according to GAME_TYPE_ORDER (using shortened names)
    # Create ordered list of short names, excluding matching pennies
    ordered_short_cols = []
    for full_name in GAME_TYPE_ORDER:
        if full_name in EXCLUDED_GAME_TYPES:
            continue
        short_name = shorten_game_type_name(full_name)
        if short_name in pivot_df.columns:
            ordered_short_cols.append(short_name)
    
    # Add any remaining columns that weren't in GAME_TYPE_ORDER
    for col in pivot_df.columns:
        if col not in ordered_short_cols:
            ordered_short_cols.append(col)
    
    pivot_df = pivot_df[ordered_short_cols]

    # Create heatmap
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
    
    # Add horizontal annotation indicating adversarial (left) vs cooperative (right) games
    # Position it below the x-axis labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Calculate position: centered horizontally, below the x-axis
    x_center = (xlim[0] + xlim[1]) / 2
    y_bottom = ylim[0] - (ylim[1] - ylim[0]) * 0.15  # Position below the plot
    
    ax.text(
        x_center,
        y_bottom,
        "← More Adversarial                    More Cooperative →",
        ha="center",
        va="top",
        fontsize=10,
        style="italic",
        color="gray",
    )
    
    plt.tight_layout()

    prefix_str = f"{prefix}-" if prefix else ""
    metric_str = welfare_metric.lower().replace(" ", "-")
    file_path = output_path / f"{prefix_str}accuracy_heatmap_{metric_str}.png"
    plt.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return file_path


def plot_historical_trends(
    log_info: List[Dict[str, Any]],
    *,
    max_logs: int = 15,
    output_dir: str = "assets",
    prefix: str = "",
) -> Path:
    """Plot historical trends of overall efficiency, fairness, and Nash over model releases.

    We map known frontier model names to approximate release dates (1st of month inferred)
    and use overall accuracies:
    - utilitarian → efficiency
    - rawlsian → fairness
    - nash → nash accuracy
    """
    # Map substrings in model IDs to canonical names and release metadata
    model_meta = {
        "claude-3.5-sonnet": {
            "name": "Claude-3.5-Sonnet",
            "date": datetime.date(2024, 6, 1),  # day inferred
            "marker": "v",
            "color": "deeppink",
        },
        "gpt-4o-mini": {
            "name": "GPT-4o-mini",
            "date": datetime.date(2024, 7, 1),
            "marker": "s",
            "color": "green",
        },
        "gpt-4o": {
            "name": "GPT-4o",
            "date": datetime.date(2024, 5, 13),  # May 13, 2024
            "marker": "o",
            "color": "darkgreen",
        },
        "llama-3.1-405b": {
            "name": "Llama-3.1-405B",
            "date": datetime.date(2024, 7, 1),
            "marker": "D",
            "color": "blue",
        },
        "deepseek-r1": {
            "name": "DeepSeek-R1",
            "date": datetime.date(2025, 1, 1),
            "marker": "P",
            "color": "gold",
        },
        "grok-3": {
            "name": "Grok-3",
            "date": datetime.date(2025, 2, 1),
            "marker": "x",
            "color": "teal",
        },
        "claude-3.7-sonnet": {
            "name": "Claude-3.7-Sonnet",
            "date": datetime.date(2025, 2, 1),
            "marker": "o",
            "color": "orange",
        },
        "gemini-2.5-pro": {
            "name": "Gemini-2.5-Pro",
            "date": datetime.date(2025, 3, 1),
            "marker": "p",
            "color": "darkgreen",
        },
        "o4-mini": {
            "name": "o4-mini",
            "date": datetime.date(2025, 4, 1),
            "marker": "v",
            "color": "purple",
        },
        "o3-": {
            "name": "o3",
            "date": datetime.date(2025, 4, 1),
            "marker": "h",
            "color": "navy",
        },
        "llama-3.3-70b": {
            "name": "Llama-3.3-70B (inferred)",
            "date": datetime.date(2025, 12, 1),  # inferred: late 2025
            "marker": "*",
            "color": "crimson",
        },
        "gemini-3-flash": {
            "name": "Gemini-3-Flash (inferred)",
            "date": datetime.date(2025, 12, 1),  # inferred: late 2025
            "marker": "X",
            "color": "lime",
        },
        "claude-sonnet-4.5": {
            "name": "Claude-Sonnet-4.5 (inferred)",
            "date": datetime.date(2025, 11, 1),  # inferred: late 2025
            "marker": "d",
            "color": "coral",
        },
        "qwen3-30b": {
            "name": "Qwen3-30B (inferred)",
            "date": datetime.date(2025, 11, 1),  # inferred: late 2025
            "marker": "+",
            "color": "indigo",
        },
        "llama-3.2-3b": {
            "name": "Llama-3.2-3B (inferred)",
            "date": datetime.date(2024, 9, 1),  # inferred: mid-late 2024
            "marker": "^",
            "color": "skyblue",
        },
        "gpt-5-mini": {
            "name": "GPT-5-Mini",
            "date": datetime.date(2025, 8, 7),  # from model name
            "marker": "s",
            "color": "forestgreen",
        },
        "gpt-5-nano": {
            "name": "GPT-5-Nano",
            "date": datetime.date(2025, 8, 7),  # from model name
            "marker": "s",
            "color": "lightgreen",
        },
    }

    rows: List[Dict[str, Any]] = []

    for entry in log_info[:max_logs]:
        if entry.get("type") != "eval" or entry.get("num_samples", 0) == 0:
            continue

        model_name = entry.get("model_name", "") or ""
        key = None
        lower_name = model_name.lower()
        for pattern in model_meta:
            if pattern in lower_name:
                key = pattern
                break
        if key is None:
            continue

        # Load samples for this log
        samples: List[Dict[str, Any]] = []
        with zipfile.ZipFile(entry["path"], "r") as z:
            sample_files = sorted(
                f for f in z.namelist() if f.startswith("samples/") and f.endswith(".json")
            )
            for sample_file in sample_files:
                with z.open(sample_file) as f:
                    sample = json.load(f)
                samples.append(sample)

        if not samples:
            continue

        df = build_dataframe(samples)
        try:
            _, _, _, _, overall_accuracy, _ = compute_metrics(df)
            # Check if we got valid metrics (overall_accuracy is a Series)
            required_keys = ["utilitarian", "rawlsian", "nash"]
            if len(overall_accuracy) == 0 or any(k not in overall_accuracy.index for k in required_keys):
                print(f"Warning: No valid metrics for {model_name} ({entry['path']})", file=sys.stderr)
                continue

        except (KeyError, ValueError, IndexError, AttributeError) as e:
            print(f"Warning: Failed to compute metrics for {model_name} ({entry['path']}): {e}", file=sys.stderr)
            continue

        rows.append(
            {
                "model_key": key,
                "model_name": model_name,
                "utilitarian": overall_accuracy["utilitarian"],
                "rawlsian": overall_accuracy["rawlsian"],
                "nash": overall_accuracy["nash"],
            }
        )

    if not rows:
        print("No matching models found for historical trend plot.")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        dummy = output_path / "history_empty.png"
        return dummy

    hist_df = pd.DataFrame(rows)
    # Attach release metadata
    hist_df["date"] = hist_df["model_key"].map(lambda k: model_meta[k]["date"])
    hist_df["marker"] = hist_df["model_key"].map(lambda k: model_meta[k]["marker"])
    hist_df["color"] = hist_df["model_key"].map(lambda k: model_meta[k]["color"])
    hist_df["label"] = hist_df["model_key"].map(lambda k: model_meta[k]["name"])

    hist_df = hist_df.sort_values("date")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Colors per metric
    metric_colors = {
        "utilitarian": "tab:blue",
        "rawlsian": "tab:orange",
        "nash": "tab:green",
    }

    # Scatter points: color encodes metric, marker encodes model
    for _, row in hist_df.iterrows():
        for metric in ["utilitarian"]: #, "rawlsian", "nash"):
            ax.scatter(
                row["date"],
                row[metric],
                color=metric_colors[metric],
                marker=row["marker"],
                edgecolor=row["color"],
                linewidths=1.0,
            )
        # Annotate once per model (near utilitarian point)
        ax.annotate(
            row["label"],
            (row["date"], row["utilitarian"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )

    # Create dummy handles for legend by metric
    metric_handles = []
    for metric, color in metric_colors.items():
        (h,) = ax.plot(
            [],
            [],
            linestyle="",
            marker="o",
            color=color,
            label=f"{metric} accuracy",
        )
        metric_handles.append(h)

    ax.legend(handles=metric_handles, title="Metrics", loc="upper left")
    ax.set_xlabel("Model release date", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        "Historical trend of efficiency, fairness, and Nash accuracy\n"
        "(release day inferred as 1st of month when unknown)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    fig.autofmt_xdate()
    prefix_str = f"{prefix}-" if prefix else ""
    file_path = output_path / f"{prefix_str}historical_trends.png"
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close(fig)

    return file_path


def plot_math_rank_vs_accuracy(
    log_info: List[Dict[str, Any]],
    *,
    max_logs: int = 15,
    output_dir: str = "assets",
    prefix: str = "",
) -> Path:
    """Plot math rank (from LM Arena) vs accuracy metrics.
    
    X-axis: Math rank (lower is better)
    Y-axis: Accuracy metrics (utilitarian, rawlsian, nash)
    """
    cache_file = Path(output_dir) / "leaderboard_cache.csv"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rows: List[Dict[str, Any]] = []
    
    # Collect model names and metrics
    model_names = []
    for entry in log_info[:max_logs]:
        if entry.get("type") != "eval" or entry.get("num_samples", 0) == 0:
            continue
        model_name = entry.get("model_name", "") or ""
        if model_name and model_name != "?":
            model_names.append(model_name)
    
    # Get math rankings
    math_rankings = get_math_rankings(model_names, cache_file=cache_file)
    
    # Process each log to get metrics
    for entry in log_info[:max_logs]:
        if entry.get("type") != "eval" or entry.get("num_samples", 0) == 0:
            continue
        
        model_name = entry.get("model_name", "") or ""
        if model_name not in math_rankings:
            continue
        
        # Load samples
        samples: List[Dict[str, Any]] = []
        with zipfile.ZipFile(entry["path"], "r") as z:
            sample_files = sorted(
                f for f in z.namelist() if f.startswith("samples/") and f.endswith(".json")
            )
            for sample_file in sample_files:
                with z.open(sample_file) as f:
                    sample = json.load(f)
                samples.append(sample)
        
        if not samples:
            continue
        
        df = build_dataframe(samples)
        try:
            _, _, _, _, overall_accuracy, _ = compute_metrics(df)
            required_keys = ["utilitarian", "rawlsian", "nash"]
            if len(overall_accuracy) == 0 or any(k not in overall_accuracy.index for k in required_keys):
                continue
        except Exception:
            continue
        
        rows.append({
            "model_name": model_name,
            "math_rank": math_rankings[model_name]["math_rank"],
            "math_score": math_rankings[model_name]["math_score"],
            "utilitarian": overall_accuracy["utilitarian"],
            "rawlsian": overall_accuracy["rawlsian"],
            "nash": overall_accuracy["nash"],
        })
    
    if not rows:
        print("No models with math rankings found for math rank vs accuracy plot.")
        prefix_str = f"{prefix}-" if prefix else ""
        dummy = output_path / f"{prefix_str}math_rank_vs_accuracy_empty.png"
        return dummy
    
    plot_df = pd.DataFrame(rows)
    plot_df = plot_df.sort_values("math_rank")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each metric
    ax.scatter(plot_df["math_rank"], plot_df["utilitarian"], label="Efficiency (utilitarian)", alpha=0.7, s=100)
    ax.scatter(plot_df["math_rank"], plot_df["rawlsian"], label="Fairness (rawlsian)", alpha=0.7, s=100)
    ax.scatter(plot_df["math_rank"], plot_df["nash"], label="Nash accuracy", alpha=0.7, s=100)
    
    # Annotate model names
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["model_name"].split("/")[-1],  # Short name
            (row["math_rank"], row["utilitarian"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
            alpha=0.7,
        )
    
    ax.set_xlabel("Math Rank (LM Arena, lower is better)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Math Rank vs Game-Theoretic Accuracy Metrics", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.invert_xaxis()  # Lower rank (better) on the right
    
    plt.tight_layout()
    prefix_str = f"{prefix}-" if prefix else ""
    file_path = output_path / f"{prefix_str}math_rank_vs_accuracy.png"
    plt.savefig(file_path)
    plt.close(fig)
    
    return file_path


def plot_math_rank_vs_avg_scores(
    log_info: List[Dict[str, Any]],
    *,
    max_logs: int = 15,
    output_dir: str = "assets",
    prefix: str = "",
) -> Path:
    """Plot math rank (from LM Arena) vs average welfare scores.
    
    X-axis: Math rank (lower is better)
    Y-axis: Average welfare scores (utilitarian_efficiency, rawlsian_efficiency, nash_social_welfare_efficiency)
    """
    cache_file = Path(output_dir) / "leaderboard_cache.csv"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rows: List[Dict[str, Any]] = []
    
    # Collect model names
    model_names = []
    for entry in log_info[:max_logs]:
        if entry.get("type") != "eval" or entry.get("num_samples", 0) == 0:
            continue
        model_name = entry.get("model_name", "") or ""
        if model_name and model_name != "?":
            model_names.append(model_name)
    
    # Get math rankings
    math_rankings = get_math_rankings(model_names, cache_file=cache_file)
    
    # Process each log to get metrics
    for entry in log_info[:max_logs]:
        if entry.get("type") != "eval" or entry.get("num_samples", 0) == 0:
            continue
        
        model_name = entry.get("model_name", "") or ""
        if model_name not in math_rankings:
            continue
        
        # Load samples
        samples: List[Dict[str, Any]] = []
        with zipfile.ZipFile(entry["path"], "r") as z:
            sample_files = sorted(
                f for f in z.namelist() if f.startswith("samples/") and f.endswith(".json")
            )
            for sample_file in sample_files:
                with z.open(sample_file) as f:
                    sample = json.load(f)
                samples.append(sample)
        
        if not samples:
            continue
        
        df = build_dataframe(samples)
        try:
            _, _, welfare_by_game, _, _, _ = compute_metrics(df)
            # Get overall welfare scores (mean across all game types)
            if welfare_by_game.empty:
                continue
            overall_welfare = welfare_by_game.mean()
            required_keys = ["utilitarian_efficiency", "rawlsian_efficiency", "nash_social_welfare_efficiency"]
            if any(k not in overall_welfare.index for k in required_keys):
                continue
        except Exception:
            continue
        
        rows.append({
            "model_name": model_name,
            "math_rank": math_rankings[model_name]["math_rank"],
            "math_score": math_rankings[model_name]["math_score"],
            "utilitarian_efficiency": overall_welfare["utilitarian_efficiency"],
            "rawlsian_efficiency": overall_welfare["rawlsian_efficiency"],
            "nash_social_welfare_efficiency": overall_welfare["nash_social_welfare_efficiency"],
        })
    
    if not rows:
        print("No models with math rankings found for math rank vs avg scores plot.")
        prefix_str = f"{prefix}-" if prefix else ""
        dummy = output_path / f"{prefix_str}math_rank_vs_avg_scores_empty.png"
        return dummy
    
    plot_df = pd.DataFrame(rows)
    plot_df = plot_df.sort_values("math_rank")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each metric
    ax.scatter(plot_df["math_rank"], plot_df["utilitarian_efficiency"], label="utilitarian_efficiency", alpha=0.7, s=100)
    ax.scatter(plot_df["math_rank"], plot_df["rawlsian_efficiency"], label="rawlsian_efficiency", alpha=0.7, s=100)
    ax.scatter(plot_df["math_rank"], plot_df["nash_social_welfare_efficiency"], label="nash_social_welfare_efficiency", alpha=0.7, s=100)
    
    # Annotate model names
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["model_name"].split("/")[-1],  # Short name
            (row["math_rank"], row["utilitarian_efficiency"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
            alpha=0.7,
        )
    
    ax.set_xlabel("Math Rank (LM Arena, lower is better)", fontsize=12)
    ax.set_ylabel("Normalized Average Score", fontsize=12)
    ax.set_title("Math Rank vs Average Welfare Scores", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.invert_xaxis()  # Lower rank (better) on the right
    
    plt.tight_layout()
    prefix_str = f"{prefix}-" if prefix else ""
    file_path = output_path / f"{prefix_str}math_rank_vs_avg_scores.png"
    plt.savefig(file_path)
    plt.close(fig)
    
    return file_path


def plot_epoch_score_vs_accuracy(
    log_info: List[Dict[str, Any]],
    *,
    max_logs: int = 15,
    output_dir: str = "assets",
    prefix: str = "",
) -> Path:
    """Plot Epoch benchmark score vs accuracy metrics.
    
    X-axis: Best score from Epoch AI benchmarks (higher is better)
    Y-axis: Accuracy metrics (utilitarian, rawlsian, nash)
    """
    cache_file = Path(output_dir) / "epoch_cache.csv"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rows: List[Dict[str, Any]] = []
    
    # Collect model names and metrics
    model_names = []
    for entry in log_info[:max_logs]:
        if entry.get("type") != "eval" or entry.get("num_samples", 0) == 0:
            continue
        model_name = entry.get("model_name", "") or ""
        if model_name and model_name != "?":
            model_names.append(model_name)
    
    # Get Epoch scores
    epoch_scores = get_epoch_scores(model_names, cache_file=cache_file)
    
    # Process each log to get metrics
    for entry in log_info[:max_logs]:
        if entry.get("type") != "eval" or entry.get("num_samples", 0) == 0:
            continue
        
        model_name = entry.get("model_name", "") or ""
        if model_name not in epoch_scores:
            continue
        
        # Load samples
        samples: List[Dict[str, Any]] = []
        with zipfile.ZipFile(entry["path"], "r") as z:
            sample_files = sorted(
                f for f in z.namelist() if f.startswith("samples/") and f.endswith(".json")
            )
            for sample_file in sample_files:
                with z.open(sample_file) as f:
                    sample = json.load(f)
                samples.append(sample)
        
        if not samples:
            continue
        
        df = build_dataframe(samples)
        try:
            _, _, _, _, overall_accuracy, _ = compute_metrics(df)
            required_keys = ["utilitarian", "rawlsian", "nash"]
            if len(overall_accuracy) == 0 or any(k not in overall_accuracy.index for k in required_keys):
                continue
        except Exception:
            continue
        
        rows.append({
            "model_name": model_name,
            "epoch_score": epoch_scores[model_name]["best_score"],
            "utilitarian": overall_accuracy["utilitarian"],
            "rawlsian": overall_accuracy["rawlsian"],
            "nash": overall_accuracy["nash"],
        })
    
    if not rows:
        print("No models with Epoch scores found for epoch score vs accuracy plot.")
        prefix_str = f"{prefix}-" if prefix else ""
        dummy = output_path / f"{prefix_str}epoch_score_vs_accuracy_empty.png"
        return dummy
    
    plot_df = pd.DataFrame(rows)
    plot_df = plot_df.sort_values("epoch_score")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each metric
    ax.scatter(plot_df["epoch_score"], plot_df["utilitarian"], label="Efficiency (utilitarian)", alpha=0.7, s=100)
    ax.scatter(plot_df["epoch_score"], plot_df["rawlsian"], label="Fairness (rawlsian)", alpha=0.7, s=100)
    ax.scatter(plot_df["epoch_score"], plot_df["nash"], label="Nash accuracy", alpha=0.7, s=100)
    
    # Annotate model names
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["model_name"].split("/")[-1],  # Short name
            (row["epoch_score"], row["utilitarian"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
            alpha=0.7,
        )
    
    ax.set_xlabel("Epoch AI Best Score (higher is better)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Epoch AI Benchmark Score vs Game-Theoretic Accuracy Metrics", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    prefix_str = f"{prefix}-" if prefix else ""
    file_path = output_path / f"{prefix_str}epoch_score_vs_accuracy.png"
    plt.savefig(file_path)
    plt.close(fig)
    
    return file_path


def plot_epoch_score_vs_avg_scores(
    log_info: List[Dict[str, Any]],
    *,
    max_logs: int = 15,
    output_dir: str = "assets",
    prefix: str = "",
) -> Path:
    """Plot Epoch benchmark score vs average welfare scores.
    
    X-axis: Best score from Epoch AI benchmarks (higher is better)
    Y-axis: Average welfare scores (utilitarian_efficiency, rawlsian_efficiency, nash_social_welfare_efficiency)
    """
    cache_file = Path(output_dir) / "epoch_cache.csv"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rows: List[Dict[str, Any]] = []
    
    # Collect model names
    model_names = []
    for entry in log_info[:max_logs]:
        if entry.get("type") != "eval" or entry.get("num_samples", 0) == 0:
            continue
        model_name = entry.get("model_name", "") or ""
        if model_name and model_name != "?":
            model_names.append(model_name)
    
    # Get Epoch scores
    epoch_scores = get_epoch_scores(model_names, cache_file=cache_file)
    
    # Process each log to get metrics
    for entry in log_info[:max_logs]:
        if entry.get("type") != "eval" or entry.get("num_samples", 0) == 0:
            continue
        
        model_name = entry.get("model_name", "") or ""
        if model_name not in epoch_scores:
            continue
        
        # Load samples
        samples: List[Dict[str, Any]] = []
        with zipfile.ZipFile(entry["path"], "r") as z:
            sample_files = sorted(
                f for f in z.namelist() if f.startswith("samples/") and f.endswith(".json")
            )
            for sample_file in sample_files:
                with z.open(sample_file) as f:
                    sample = json.load(f)
                samples.append(sample)
        
        if not samples:
            continue
        
        df = build_dataframe(samples)
        try:
            _, _, welfare_by_game, _, _, _ = compute_metrics(df)
            # Get overall welfare scores (mean across all game types)
            if welfare_by_game.empty:
                continue
            overall_welfare = welfare_by_game.mean()
            required_keys = ["utilitarian_efficiency", "rawlsian_efficiency", "nash_social_welfare_efficiency"]
            if any(k not in overall_welfare.index for k in required_keys):
                continue
        except Exception:
            continue
        
        rows.append({
            "model_name": model_name,
            "epoch_score": epoch_scores[model_name]["best_score"],
            "utilitarian_efficiency": overall_welfare["utilitarian_efficiency"],
            "rawlsian_efficiency": overall_welfare["rawlsian_efficiency"],
            "nash_social_welfare_efficiency": overall_welfare["nash_social_welfare_efficiency"],
        })
    
    if not rows:
        print("No models with Epoch scores found for epoch score vs avg scores plot.")
        prefix_str = f"{prefix}-" if prefix else ""
        dummy = output_path / f"{prefix_str}epoch_score_vs_avg_scores_empty.png"
        return dummy
    
    plot_df = pd.DataFrame(rows)
    plot_df = plot_df.sort_values("epoch_score")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each metric
    ax.scatter(plot_df["epoch_score"], plot_df["utilitarian_efficiency"], label="utilitarian_efficiency", alpha=0.7, s=100)
    ax.scatter(plot_df["epoch_score"], plot_df["rawlsian_efficiency"], label="rawlsian_efficiency", alpha=0.7, s=100)
    ax.scatter(plot_df["epoch_score"], plot_df["nash_social_welfare_efficiency"], label="nash_social_welfare_efficiency", alpha=0.7, s=100)
    
    # Annotate model names
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["model_name"].split("/")[-1],  # Short name
            (row["epoch_score"], row["utilitarian_efficiency"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
            alpha=0.7,
        )
    
    ax.set_xlabel("Epoch AI Best Score (higher is better)", fontsize=12)
    ax.set_ylabel("Normalized Average Score", fontsize=12)
    ax.set_title("Epoch AI Benchmark Score vs Average Welfare Scores", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    prefix_str = f"{prefix}-" if prefix else ""
    file_path = output_path / f"{prefix_str}epoch_score_vs_avg_scores.png"
    plt.savefig(file_path)
    plt.close(fig)
    
    return file_path
