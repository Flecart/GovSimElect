"""
Create visualizations from classified reasoning traces.
Generates plots showing category distributions by game type, model, and outcome.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import argparse

# Category mappings
MACRO_CATEGORIES = {
    "Game-Theoretic": ["NASH_DOMINANT", "PAYOFF_MAX"],
    "Social Welfare": ["UTILITARIAN", "RAWLSIAN"],
    "Risk/Catastrophe": ["CATASTROPHE_PREVENTION", "PRECAUTIONARY"],
    "Domain-Specific": ["AI_SAFETY", "DOMAIN_OTHER"]
}

VALID_CATEGORIES = [
    "NASH_DOMINANT", "PAYOFF_MAX", "UTILITARIAN", "RAWLSIAN",
    "CATASTROPHE_PREVENTION", "PRECAUTIONARY", "AI_SAFETY", "DOMAIN_OTHER"
]

CATEGORY_DISPLAY = {
    "NASH_DOMINANT": "Nash/Dominant",
    "PAYOFF_MAX": "Payoff Max",
    "UTILITARIAN": "Utilitarian",
    "RAWLSIAN": "Rawlsian",
    "CATASTROPHE_PREVENTION": "Catastrophe",
    "PRECAUTIONARY": "Precautionary",
    "AI_SAFETY": "AI Safety",
    "DOMAIN_OTHER": "Domain-Specific"
}


def load_data(filepath):
    """Load classification data."""
    with open(filepath) as f:
        return json.load(f)


def compute_macro_frequencies(data):
    """Compute macro-category frequencies."""
    total = len(data)
    macro_counts = Counter()
    
    for item in data:
        cats = set(item.get("classification", {}).get("categories", []))
        for macro, subcats in MACRO_CATEGORIES.items():
            if any(c in cats for c in subcats):
                macro_counts[macro] += 1
    
    return {macro: macro_counts[macro] / total for macro in MACRO_CATEGORIES.keys()}


def compute_by_model(data):
    """Compute macro frequencies by evaluated model."""
    by_model = defaultdict(list)
    
    for item in data:
        model = item.get("model", "unknown")
        by_model[model].append(item)
    
    results = {}
    for model, items in by_model.items():
        results[model] = {
            'n': len(items),
            'macro_freq': compute_macro_frequencies(items)
        }
    
    return results


def compute_by_game(data):
    """Compute macro frequencies by game type."""
    by_game = defaultdict(list)
    
    for item in data:
        game = item.get("formal_game", "Unknown")
        by_game[game].append(item)
    
    results = {}
    for game, items in by_game.items():
        results[game] = {
            'n': len(items),
            'macro_freq': compute_macro_frequencies(items)
        }
    
    return results


def compute_by_outcome(data):
    """Compute macro frequencies by outcome."""
    optimal = []
    suboptimal = []
    
    for item in data:
        util_score = item.get("utilitarian_score")
        if util_score == 1:
            optimal.append(item)
        elif util_score == 0:
            suboptimal.append(item)
    
    return {
        "optimal": {
            'n': len(optimal),
            'macro_freq': compute_macro_frequencies(optimal)
        },
        "suboptimal": {
            'n': len(suboptimal),
            'macro_freq': compute_macro_frequencies(suboptimal)
        }
    }


def plot_macro_by_model(data, output_path, classifier_name="Classifier"):
    """Plot macro-category frequencies by evaluated model."""
    by_model = compute_by_model(data)
    
    models = sorted(by_model.keys())
    macros = list(MACRO_CATEGORIES.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, macro in enumerate(macros):
        vals = [by_model[m]['macro_freq'][macro] * 100 for m in models]
        ax.bar(x + i*width, vals, width, label=macro, alpha=0.8)
    
    ax.set_ylabel('Frequency (%)', fontsize=12)
    ax.set_title(f'Macro-Category Frequencies by Evaluated Model ({classifier_name})', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.split('/')[-1].replace('anthropic/', '').replace('claude-', '').replace('qwen3-', 'qwen-').replace('deepseek-', 'ds-') 
                       for m in models], rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_macro_by_game(data, output_path, classifier_name="Classifier"):
    """Plot macro-category frequencies by game type."""
    by_game = compute_by_game(data)
    
    games = sorted(by_game.keys())
    macros = list(MACRO_CATEGORIES.keys())
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(games))
    width = 0.2
    
    for i, macro in enumerate(macros):
        vals = [by_game[g]['macro_freq'][macro] * 100 for g in games]
        ax.bar(x + i*width, vals, width, label=macro, alpha=0.8)
    
    ax.set_ylabel('Frequency (%)', fontsize=12)
    ax.set_title(f'Macro-Category Frequencies by Game Type ({classifier_name})', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(games, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_macro_by_outcome(data, output_path, classifier_name="Classifier"):
    """Plot macro-category frequencies by outcome."""
    by_outcome = compute_by_outcome(data)
    
    macros = list(MACRO_CATEGORIES.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(macros))
    width = 0.35
    
    optimal_vals = [by_outcome["optimal"]['macro_freq'][m] * 100 for m in macros]
    suboptimal_vals = [by_outcome["suboptimal"]['macro_freq'][m] * 100 for m in macros]
    
    ax.bar(x - width/2, optimal_vals, width, label='Utilitarian Optimal', alpha=0.8, color='#2ecc71')
    ax.bar(x + width/2, suboptimal_vals, width, label='Utilitarian Suboptimal', alpha=0.8, color='#e74c3c')
    
    ax.set_ylabel('Frequency (%)', fontsize=12)
    ax.set_title(f'Macro-Category Frequencies by Outcome ({classifier_name})', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(macros, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Add value labels
    for i, (v1, v2) in enumerate(zip(optimal_vals, suboptimal_vals)):
        diff = v1 - v2
        ax.text(i - width/2, v1 + 2, f'{v1:.0f}%', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, v2 + 2, f'{v2:.0f}%', ha='center', va='bottom', fontsize=9)
        # Show difference
        color = 'green' if diff > 0 else 'red'
        ax.text(i, max(v1, v2) + 8, f'{diff:+.0f}pp', ha='center', va='bottom', 
               fontsize=8, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_overall_distribution(data, output_path, classifier_name="Classifier"):
    """Plot overall macro-category distribution."""
    macro_freq = compute_macro_frequencies(data)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    macros = list(MACRO_CATEGORIES.keys())
    vals = [macro_freq[m] * 100 for m in macros]
    
    colors = ['#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax.barh(macros, vals, color=colors, alpha=0.8)
    
    ax.set_xlabel('Frequency (%)', fontsize=12)
    ax.set_title(f'Overall Macro-Category Distribution ({classifier_name})', 
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, vals)):
        ax.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create plots from classified reasoning traces")
    parser.add_argument("input", help="Input JSON file with classifications")
    parser.add_argument("--output-dir", default="reasoning_traces/plots", help="Output directory for plots")
    parser.add_argument("--name", default="Classifier", help="Classifier name for plot titles")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading {args.input}...")
    data = load_data(args.input)
    print(f"Loaded {len(data):,} samples")
    
    # Generate plots
    print(f"\nGenerating plots for {args.name}...\n")
    
    plot_overall_distribution(data, 
                             output_dir / f"{args.name.lower().replace(' ', '_')}_overall.png",
                             args.name)
    
    plot_macro_by_model(data,
                       output_dir / f"{args.name.lower().replace(' ', '_')}_by_model.png",
                       args.name)
    
    plot_macro_by_game(data,
                      output_dir / f"{args.name.lower().replace(' ', '_')}_by_game.png",
                      args.name)
    
    plot_macro_by_outcome(data,
                         output_dir / f"{args.name.lower().replace(' ', '_')}_by_outcome.png",
                         args.name)
    
    print(f"\n✓ All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
