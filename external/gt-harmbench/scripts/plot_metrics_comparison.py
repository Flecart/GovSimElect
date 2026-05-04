"""Compare metrics from two CSV files (gamify vs standard) with overlapping bar plots."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import utils for model name shortening
from eval.analysis.utils import shorten_model_name

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (6, 8)


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV and prepare for plotting."""
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    required_cols = ["model_name", "nash_accuracy", "utilitarian_accuracy"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    
    # Clean and map model names using utils
    df["model_clean"] = df["model_name"].apply(shorten_model_name)
    
    return df


def plot_comparison(gamify_path: Path, standard_path: Path, output_path: Path) -> None:
    """Create overlapping bar plot comparing gamify vs standard metrics."""
    # Load data
    gamify_df = load_and_prepare_data(gamify_path)
    standard_df = load_and_prepare_data(standard_path)
    
    print(f"Columns in gamify data: {gamify_df.columns.tolist()}")
    
    # Get unique models (union of both datasets)
    all_models = sorted(set(gamify_df["model_clean"].unique()) | set(standard_df["model_clean"].unique()))
    
    # Prepare data for plotting
    nash_gamify = []
    nash_standard = []
    nash_social_gamify = []
    nash_social_standard = []
    
    for model in all_models:
        # Get gamify values
        gamify_row = gamify_df[gamify_df["model_clean"] == model]
        if not gamify_row.empty:
            nash_gamify.append(gamify_row.iloc[0]["nash_accuracy"])
            nash_social_gamify.append(gamify_row.iloc[0]["utilitarian_accuracy"])
        else:
            nash_gamify.append(0.0)
            nash_social_gamify.append(0.0)
        
        # Get standard values
        standard_row = standard_df[standard_df["model_clean"] == model]
        if not standard_row.empty:
            nash_standard.append(standard_row.iloc[0]["nash_accuracy"])
            nash_social_standard.append(standard_row.iloc[0]["utilitarian_accuracy"])
        else:
            nash_standard.append(0.0)
            nash_social_standard.append(0.0)
    
    # Calculate differences for color coding
    nash_diff = [g - s for g, s in zip(nash_gamify, nash_standard)]
    nash_social_diff = [g - s for g, s in zip(nash_social_gamify, nash_social_standard)]
    
    # Calculate mean variation (absolute differences) across all models
    # Exclude cases where either gamify or standard is 0 (missing data)
    nash_valid_diffs = [
        abs(d) for d, g, s in zip(nash_diff, nash_gamify, nash_standard)
        if not np.isnan(d) and g > 0 and s > 0
    ]
    nash_social_valid_diffs = [
        abs(d) for d, g, s in zip(nash_social_diff, nash_social_gamify, nash_social_standard)
        if not np.isnan(d) and g > 0 and s > 0
    ]
    
    nash_mean_variation = np.mean(nash_valid_diffs) if nash_valid_diffs else 0.0
    nash_social_mean_variation = np.mean(nash_social_valid_diffs) if nash_social_valid_diffs else 0.0
    
    # Also calculate mean signed difference (to see if gamify is generally better or worse)
    nash_valid_signed = [
        d for d, g, s in zip(nash_diff, nash_gamify, nash_standard)
        if not np.isnan(d) and g > 0 and s > 0
    ]
    nash_social_valid_signed = [
        d for d, g, s in zip(nash_social_diff, nash_social_gamify, nash_social_standard)
        if not np.isnan(d) and g > 0 and s > 0
    ]
    
    nash_mean_diff = np.mean(nash_valid_signed) if nash_valid_signed else 0.0
    nash_social_mean_diff = np.mean(nash_social_valid_signed) if nash_social_valid_signed else 0.0
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    
    x = np.arange(len(all_models))
    width = 0.25  # Width of each bar group
    
    # Plot Nash Equilibrium (blue)
    # Standard bars
    bars_ne_std = ax.bar(
        x - width/2,
        nash_standard,
        width,
        alpha=0.9,
        color="steelblue",
        label="Nash Equilibrium",
    )
    # Gamify bars starting from standard bar top, showing only the difference
    nash_colors = ["#2ecc71" if diff > 0 else "#e74c3c" if diff < 0 else "#95a5a6" for diff in nash_diff]
    bars_ne_gam = ax.bar(
        x - width/2,
        nash_diff,
        width,
        bottom=nash_standard,
        alpha=0.7,
        color=nash_colors,
    )
    
    # Plot Nash Social Welfare (deep orange)
    # Standard bars
    bars_nsw_std = ax.bar(
        x + width/2,
        nash_social_standard,
        width,
        alpha=0.9,
        color="#ff8c00",  # Deep orange
        label="Utilitarian Accuracy",
    )
    # Gamify bars starting from standard bar top, showing only the difference
    nash_social_colors = ["#2ecc71" if diff > 0 else "#e74c3c" if diff < 0 else "#95a5a6" for diff in nash_social_diff]
    bars_nsw_gam = ax.bar(
        x + width/2,
        nash_social_diff,
        width,
        bottom=nash_social_standard,
        alpha=0.7,
        color=nash_social_colors,
    )
    
    ax.set_xlabel("Model", fontsize=16, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=16, fontweight="bold")
    ax.set_title("Game Theoretic vs Prosaic: Nash Equilibrium and Utilitarian Accuracy", fontsize=18, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_models, rotation=0, ha="center", fontsize=12)
    
    # Add legend
    ax.legend(loc="upper left", fontsize=14, frameon=True, fancybox=True, shadow=True)
    
    # Set ylim to accommodate negative differences if any
    all_values = nash_gamify + nash_standard + nash_social_gamify + nash_social_standard
    max_val = max(all_values)
    min_val = min(all_values)
    ax.set_ylim(min(0, min_val - 0.1), max(1.1, max_val + 0.1))
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on standard bars and gamify bars with overlap detection
    # Process pairs together to detect overlaps
    bar_pairs = [
        (bars_ne_std, nash_standard, bars_ne_gam, nash_gamify),
        (bars_nsw_std, nash_social_standard, bars_nsw_gam, nash_social_gamify),
    ]
    
    for bars_std, std_values, bars_gam, gam_values in bar_pairs:
        for i, (bar_std, bar_gam) in enumerate(zip(bars_std, bars_gam)):
            std_height = std_values[i]
            gam_height = gam_values[i]
            
            # Check if values are too close (overlap threshold)
            overlap_threshold = 0.02  # If bars are within 3% of each other, adjust positioning
            
            if std_height > 0:
                # Standard bar label
                if abs(std_height - gam_height) < overlap_threshold and gam_height > 0:
                    # Values are very close, put standard label inside the bar
                    ax.text(
                        bar_std.get_x() + bar_std.get_width() / 2.0,
                        std_height,  # Middle of bar
                        f"{std_height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
                else:
                    # Normal positioning at top
                    ax.text(
                        bar_std.get_x() + bar_std.get_width() / 2.0,
                        std_height,
                        f"{std_height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
            
            if gam_height > 0:
                # Gamify bar label
                if abs(std_height - gam_height) < overlap_threshold and std_height > 0:
                    pass
                else:
                    # Normal positioning at top
                    ax.text(
                        bar_gam.get_x() + bar_gam.get_width() / 2.0,
                        gam_height,
                        f"{gam_height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
    
    # Add difference labels above/below gamify bars with overlap detection
    bar_diff_pairs = [
        (bars_ne_gam, nash_diff, nash_gamify, nash_standard, nash_colors),
        (bars_nsw_gam, nash_social_diff, nash_social_gamify, nash_social_standard, nash_social_colors),
    ]
    
    for bars_gam, diffs, final_vals, std_vals, colors in bar_diff_pairs:
        for i, (bar, diff, final_val, std_val) in enumerate(zip(bars_gam, diffs, final_vals, std_vals)):
            if abs(diff) > 0.001:  # Only show if meaningful difference
                diff_str = f"{diff:+.2f}"  # + or - sign
                
                # Check if difference label would overlap with value labels
                overlap_threshold = 0.03
                values_close = abs(final_val - std_val) < overlap_threshold
                
                y_pos = std_val + 0.03
                ax.text(
                    bar.get_x(), # + bar.get_width() / 2.0,
                    y_pos,
                    diff_str,
                    ha="center",
                    va="top",
                    fontsize=12,
                    fontweight="bold",
                    color=colors[i],
                )
                
    plt.ylim(0, 1)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved comparison plot to: {output_path}")
    print("\n" + "="*60)
    print("Mean Variation Across All Models:")
    print("="*60)
    print(f"Nash Equilibrium:")
    print(f"  Mean absolute variation: {nash_mean_variation:.4f}")
    print(f"  Mean difference (gamify - standard): {nash_mean_diff:+.4f}")
    print(f"  ({'Gamify better' if nash_mean_diff > 0 else 'Standard better' if nash_mean_diff < 0 else 'Equal'})")
    print(f"\nNash Social Welfare:")
    print(f"  Mean absolute variation: {nash_social_mean_variation:.4f}")
    print(f"  Mean difference (gamify - standard): {nash_social_mean_diff:+.4f}")
    print(f"  ({'Gamify better' if nash_social_mean_diff > 0 else 'Standard better' if nash_social_mean_diff < 0 else 'Equal'})")
    print("="*60)


def main():
    """Main entry point."""
    gamify_path = Path("results/gamify.csv")
    standard_path = Path("results/standard.csv")
    output_path = Path("assets/metrics_comparison.png")
    
    # Check if files exist
    if not gamify_path.exists():
        print(f"Error: {gamify_path} not found", file=sys.stderr)
        sys.exit(1)
    
    if not standard_path.exists():
        print(f"Error: {standard_path} not found", file=sys.stderr)
        sys.exit(1)
    
    plot_comparison(gamify_path, standard_path, output_path)


if __name__ == "__main__":
    main()
