"""
Mechanism Design Visualization for GTHarmBench
Single publication-quality stacked bar figure showing mechanism effectiveness
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path
import json
import zipfile

# =============================================================================
# STYLE CONFIGURATION (matching teammate's style)
# =============================================================================

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (14, 8),
    "font.family": "sans-serif",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})

# Colors
COLOR_NASH_BASE = "steelblue"
COLOR_UTIL_BASE = "#ff8c00"  # Deep orange
COLOR_IMPROVE = "#2ecc71"    # Green
COLOR_DECREASE = "#e74c3c"   # Red
COLOR_NEUTRAL = "#95a5a6"    # Gray

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_DISPLAY = {
    "gpt-5-mini": "GPT-5 Mini",
    "deepseek-v3.2": "DeepSeek V3.2",
    "claude-4.5-sonnet": "Claude 4.5",
    "llama-3.2-3b": "Llama 3.2 3B",
    "grok-4.1-fast": "Grok 4.1",
    "qwen-3-30b": "Qwen3 30B",
    "qwen-3-8b": "Qwen3 8B",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3-pro-preview": "Gemini 3 Pro",
}

MECHANISM_DISPLAY = {
    "message": "Messages",
    "contracts": "Contracts",
    "payments": "Payments",
    "penalties": "Penalties",
    "mediator": "Mediator",
}

VARIANT_DISPLAY = {
    "v1_initial": "Initial",
    "v2_formal": "Formal",
    "v3_credibility": "Credibility",
    "v4_moral": "Moral",
}

# =============================================================================
# DATA LOADING FROM LOGS
# =============================================================================

# Mapping from mechanism file prefix to display name
MECHANISM_FILE_MAP = {
    "message": "Messages",
    "contracts": "Contracts",
    "contract": "Contracts",  # Handle typo in some filenames
    "payments": "Payments",
    "penalties": "Penalties",
    "mediator": "Mediator",
}

# Variants in eval files
VARIANTS = ["v1_initial", "v2_formal", "v3_credibility", "v4_moral"]


def extract_metrics_from_eval(eval_path: Path) -> dict | None:
    """Extract metrics from a .eval file (zip containing JSON samples)."""
    try:
        samples = []
        with zipfile.ZipFile(str(eval_path), "r") as z:
            sample_files = sorted(
                f for f in z.namelist() if f.startswith("samples/") and f.endswith(".json")
            )
            for sample_file in sample_files:
                with z.open(sample_file) as f:
                    sample = json.load(f)
                samples.append(sample)

        if not samples:
            return None

        nash_scores = []
        util_scores = []

        for sample in samples:
            scores = sample.get("scores", {})
            scorer = scores.get("all_strategies_scorer", {})
            value = scorer.get("value", {})

            if isinstance(value, dict):
                if "nash" in value:
                    nash_scores.append(value["nash"])
                if "utilitarian" in value:
                    util_scores.append(value["utilitarian"])

        return {
            "nash_accuracy": np.mean(nash_scores) if nash_scores else 0.0,
            "utilitarian_accuracy": np.mean(util_scores) if util_scores else 0.0,
        }

    except Exception as e:
        print(f"Error processing {eval_path.name}: {e}")
        return None


def load_data(logs_dir: Path = Path("logs/mech-design"), verbose: bool = True,
               aggregate_variants: bool = False) -> pd.DataFrame:
    """Load evaluation data from logs directory.

    Args:
        logs_dir: Path to logs directory containing model subdirectories
        verbose: Print progress information
        aggregate_variants: If True, average across variants. If False, keep each variant separate.

    Returns:
        DataFrame with columns: model, mechanism, variant, nash_accuracy, utilitarian_accuracy
    """
    all_results = []

    for model_dir, model_display in MODEL_DISPLAY.items():
        model_path = logs_dir / model_dir
        if not model_path.exists():
            if verbose:
                print(f"Skipping {model_dir} - directory not found")
            continue

        if verbose:
            print(f"Processing {model_display}...")

        # Load baseline
        baseline_path = model_path / "baseline.eval"
        if baseline_path.exists():
            metrics = extract_metrics_from_eval(baseline_path)
            if metrics:
                all_results.append({
                    "model": model_display,
                    "mechanism": "Baseline",
                    "variant": "N/A",
                    **metrics
                })

        # Load mechanism variants
        for mech_prefix, mech_display in MECHANISM_FILE_MAP.items():
            mech_metrics_by_variant = {}

            for var in VARIANTS:
                eval_path = model_path / f"{mech_prefix}-{var}.eval"
                if eval_path.exists():
                    metrics = extract_metrics_from_eval(eval_path)
                    if metrics:
                        mech_metrics_by_variant[var] = metrics

            if not mech_metrics_by_variant:
                continue

            # Check for duplicates (contract vs contracts)
            existing = [r for r in all_results
                       if r["model"] == model_display and r["mechanism"] == mech_display]
            if existing:
                continue

            if aggregate_variants:
                # Average across variants for this mechanism
                avg_nash = np.mean([m["nash_accuracy"] for m in mech_metrics_by_variant.values()])
                avg_util = np.mean([m["utilitarian_accuracy"] for m in mech_metrics_by_variant.values()])
                all_results.append({
                    "model": model_display,
                    "mechanism": mech_display,
                    "variant": "Average",
                    "nash_accuracy": avg_nash,
                    "utilitarian_accuracy": avg_util,
                })
            else:
                # Keep each variant separate
                for var, metrics in mech_metrics_by_variant.items():
                    all_results.append({
                        "model": model_display,
                        "mechanism": mech_display,
                        "variant": VARIANT_DISPLAY[var],
                        **metrics
                    })

    if verbose:
        print(f"\nLoaded {len(all_results)} results")

    return pd.DataFrame(all_results)


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def get_diff_color(diff):
    """Return color based on whether diff is positive, negative, or zero."""
    if diff > 0.005:
        return COLOR_IMPROVE
    elif diff < -0.005:
        return COLOR_DECREASE
    else:
        return COLOR_NEUTRAL


# =============================================================================
# MAIN FIGURE: Grouped Bar Chart by Mechanism and Variant
# =============================================================================

# Colors for each variant
VARIANT_COLORS = {
    "Initial": "#1f77b4",      # Blue
    "Formal": "#ff7f0e",       # Orange
    "Credibility": "#2ca02c",  # Green
    "Moral": "#d62728",        # Red
}


def create_mechanism_comparison_figure(df, output_path=None, metric="utilitarian_accuracy"):
    """
    Create grouped bar chart comparing mechanisms with 4 bars per mechanism (one per variant).
    X-axis: Mechanisms (5 mechanisms x 4 variants = 20 bars)

    Args:
        df: DataFrame with columns: model, mechanism, variant, nash_accuracy, utilitarian_accuracy
        output_path: Path to save figure
        metric: Which metric to plot ("nash_accuracy" or "utilitarian_accuracy")

    This answers: "Which mechanism-variant combinations improve outcomes the most?"
    """
    # Get baseline value (average across models)
    baseline_df = df[df["mechanism"] == "Baseline"]
    baseline_val = baseline_df[metric].mean()

    # Filter to mechanisms only and average across models
    mechs_df = df[df["mechanism"] != "Baseline"].copy()
    avg_by_mech_var = mechs_df.groupby(["mechanism", "variant"]).agg({
        "nash_accuracy": "mean",
        "utilitarian_accuracy": "mean"
    }).reset_index()

    # Setup
    mechanisms = list(MECHANISM_DISPLAY.values())
    variants = list(VARIANT_DISPLAY.values())  # Initial, Formal, Credibility, Moral
    n_variants = len(variants)
    n_mechanisms = len(mechanisms)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 7))

    # Bar positioning
    bar_width = 0.18
    group_gap = 0.3

    # Calculate x positions for each bar
    x_positions = []
    x_group_centers = []

    for i, mech in enumerate(mechanisms):
        group_start = i * (n_variants * bar_width + group_gap)
        group_center = group_start + (n_variants - 1) * bar_width / 2
        x_group_centers.append(group_center)

        for j in range(n_variants):
            x_positions.append(group_start + j * bar_width)

    # Get values in correct order
    values = []
    colors = []
    for mech in mechanisms:
        for var in variants:
            row = avg_by_mech_var[(avg_by_mech_var["mechanism"] == mech) &
                                  (avg_by_mech_var["variant"] == var)]
            if len(row) > 0:
                values.append(row[metric].values[0])
            else:
                values.append(0)
            colors.append(VARIANT_COLORS[var])

    # Plot bars
    bars = ax.bar(x_positions, values, bar_width, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        diff = val - baseline_val

        # Value label
        ax.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        # Difference label (colored)
        if abs(diff) > 0.005:
            diff_color = COLOR_IMPROVE if diff > 0 else COLOR_DECREASE
            ax.annotate(f'{diff:+.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height + 0.035),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=7, fontweight='bold', color=diff_color)

    # Add baseline reference line
    ax.axhline(y=baseline_val, color='black', linestyle='--',
               alpha=0.7, linewidth=1.5, label=f'Baseline ({baseline_val:.2f})')

    # Add vertical separators between mechanism groups
    for i in range(1, n_mechanisms):
        sep_x = i * (n_variants * bar_width + group_gap) - group_gap / 2
        ax.axvline(x=sep_x, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    # Styling
    metric_label = "Utilitarian Accuracy" if metric == "utilitarian_accuracy" else "Nash Accuracy"
    ax.set_xlabel("Mechanism", fontsize=14)
    ax.set_ylabel(metric_label, fontsize=14)
    ax.set_title(f"Mechanism Design: Impact on Game Outcomes ({metric_label})", fontsize=16, pad=15)
    ax.set_xticks(x_group_centers)
    ax.set_xticklabels(mechanisms, fontsize=12)

    # Y-axis limits
    max_val = max(values) if values else 1.0
    ax.set_ylim(0, max_val + 0.12)
    ax.grid(axis="y", alpha=0.3)

    # Legend for variants
    legend_handles = [
        mpatches.Patch(color=VARIANT_COLORS[var], label=var) for var in variants
    ]
    legend_handles.append(plt.Line2D([0], [0], color='black', linestyle='--',
                                      alpha=0.7, label=f'Baseline ({baseline_val:.2f})'))
    legend_handles.append(mpatches.Patch(color=COLOR_IMPROVE, alpha=0.7, label='Improvement'))
    legend_handles.append(mpatches.Patch(color=COLOR_DECREASE, alpha=0.7, label='Decrease'))

    ax.legend(handles=legend_handles, loc='upper right', fontsize=10,
              frameon=True, fancybox=True, shadow=True, ncol=2)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")

    return fig, ax


# =============================================================================
# ALTERNATIVE: Stacked Bar Chart by Model (shows model heterogeneity)
# =============================================================================

def create_model_comparison_figure(df, output_path=None):
    """
    Create stacked bar chart comparing models.
    X-axis: Models
    Bars: Nash (left) and Utilitarian (right) showing BEST mechanism improvement
    
    This answers: "How do different models respond to mechanism design?"
    """
    models = [m for m in MODEL_DISPLAY.values() if m in df["model"].values]
    
    # For each model, get baseline and best mechanism improvement
    results = []
    for model in models:
        model_data = df[df["model"] == model]
        baseline_nash = model_data[model_data["mechanism"] == "Baseline"]["nash_accuracy"].values[0]
        baseline_util = model_data[model_data["mechanism"] == "Baseline"]["utilitarian_accuracy"].values[0]
        
        # Get best improvement for each metric
        mechs = model_data[model_data["mechanism"] != "Baseline"]
        best_nash = mechs["nash_accuracy"].max()
        best_util = mechs["utilitarian_accuracy"].max()
        
        # Also get average across all mechanisms
        avg_nash = mechs["nash_accuracy"].mean()
        avg_util = mechs["utilitarian_accuracy"].mean()
        
        results.append({
            "model": model,
            "baseline_nash": baseline_nash,
            "baseline_util": baseline_util,
            "avg_nash": avg_nash,
            "avg_util": avg_util,
            "nash_diff": avg_nash - baseline_nash,
            "util_diff": avg_util - baseline_util,
        })
    
    results_df = pd.DataFrame(results)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 7))
    
    x = np.arange(len(models))
    width = 0.35
    
    # --- Plot Nash bars ---
    ax.bar(x - width/2, results_df["baseline_nash"], width,
           color=COLOR_NASH_BASE, alpha=0.9, label="Nash Equilibrium")
    
    nash_colors = [get_diff_color(d) for d in results_df["nash_diff"]]
    ax.bar(x - width/2, results_df["nash_diff"], width,
           bottom=results_df["baseline_nash"],
           color=nash_colors, alpha=0.7)
    
    # --- Plot Utilitarian bars ---
    ax.bar(x + width/2, results_df["baseline_util"], width,
           color=COLOR_UTIL_BASE, alpha=0.9, label="Utilitarian")
    
    util_colors = [get_diff_color(d) for d in results_df["util_diff"]]
    ax.bar(x + width/2, results_df["util_diff"], width,
           bottom=results_df["baseline_util"],
           color=util_colors, alpha=0.7)
    
    # --- Add value labels ---
    for i, row in results_df.iterrows():
        # Nash
        nash_final = row["avg_nash"]
        ax.text(x[i] - width/2, nash_final + 0.01, f"{nash_final:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
        if abs(row["nash_diff"]) > 0.005:
            ax.text(x[i] - width/2, nash_final + 0.04, f"{row['nash_diff']:+.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=nash_colors[i])
        
        # Utilitarian
        util_final = row["avg_util"]
        ax.text(x[i] + width/2, util_final + 0.01, f"{util_final:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
        if abs(row["util_diff"]) > 0.005:
            ax.text(x[i] + width/2, util_final + 0.04, f"{row['util_diff']:+.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=util_colors[i])
    
    # --- Styling ---
    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_title("Mechanism Design Impact by Model (Average Across All Mechanisms)", 
                 fontsize=16, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, rotation=30, ha="right")
    
    max_val = max(results_df["avg_nash"].max(), results_df["avg_util"].max())
    ax.set_ylim(0, max_val + 0.12)
    ax.grid(axis="y", alpha=0.3)
    
    # Legend
    legend_handles = [
        mpatches.Patch(color=COLOR_NASH_BASE, alpha=0.9, label="Nash Equilibrium"),
        mpatches.Patch(color=COLOR_UTIL_BASE, alpha=0.9, label="Utilitarian"),
        mpatches.Patch(color=COLOR_IMPROVE, alpha=0.7, label="Improvement"),
        mpatches.Patch(color=COLOR_DECREASE, alpha=0.7, label="Decrease"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10,
              frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")
    
    return fig, ax


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    output_dir = Path("assets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data from logs folder (keep variants separate)
    print("Loading data from logs/mech-design/...")
    df = load_data(logs_dir=Path("logs/mech-design"), aggregate_variants=False)
    # save
    df.to_csv(output_dir / "mech_design_data.csv", index=False) 
    print(f"Loaded {len(df)} rows\n")

    # Figure 1: Mechanism comparison - Utilitarian (20 bars: 5 mechanisms x 4 variants)
    print("=" * 60)
    print("Creating Figure: Mechanism Comparison (Utilitarian)")
    print("=" * 60)
    fig1, ax1 = create_mechanism_comparison_figure(
        df,
        output_path=output_dir / "mech_design_mechanism_comparison_util.png",
        metric="utilitarian_accuracy"
    )
    plt.show()

    # Figure 2: Mechanism comparison - Nash (20 bars: 5 mechanisms x 4 variants)
    print("\n" + "=" * 60)
    print("Creating Figure: Mechanism Comparison (Nash)")
    print("=" * 60)
    fig2, ax2 = create_mechanism_comparison_figure(
        df,
        output_path=output_dir / "mech_design_mechanism_comparison_nash.png",
        metric="nash_accuracy"
    )
    plt.show()

    # Figure 3: Model comparison (uses aggregated data)
    print("\n" + "=" * 60)
    print("Creating Figure: Model Comparison")
    print("=" * 60)
    df_agg = load_data(logs_dir=Path("logs"), aggregate_variants=True, verbose=False)
    fig3, ax3 = create_model_comparison_figure(
        df_agg,
        output_path=output_dir / "mech_design_model_comparison.png"
    )
    plt.show()

    print("\n" + "=" * 60)
    print("Done! Figures saved to assets/")
    print("=" * 60)