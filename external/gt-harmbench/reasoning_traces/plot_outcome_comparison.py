"""
Visualize category frequencies by outcome (optimal vs suboptimal).
Shows which reasoning patterns are associated with better game-theoretic outcomes.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter

VALID_CATEGORIES = [
    "NASH_DOMINANT", "PAYOFF_MAX", "UTILITARIAN", "RAWLSIAN",
    "CATASTROPHE_PREVENTION", "PRECAUTIONARY", "AI_SAFETY", "DOMAIN_OTHER"
]

CATEGORY_DISPLAY = {
    "NASH_DOMINANT": "Nash/Dominant",
    "PAYOFF_MAX": "Payoff Max",
    "UTILITARIAN": "Utilitarian",
    "RAWLSIAN": "Rawlsian",
    "CATASTROPHE_PREVENTION": "Catastrophe Prevention",
    "PRECAUTIONARY": "Precautionary",
    "AI_SAFETY": "AI Safety",
    "DOMAIN_OTHER": "Domain-Specific"
}


def load_data(filepath):
    """Load classification data."""
    with open(filepath) as f:
        return json.load(f)


def compute_by_outcome(data):
    """Compute category frequencies by outcome."""
    optimal = []
    suboptimal = []
    
    for item in data:
        util_score = item.get("utilitarian_score")
        if util_score == 1:
            optimal.append(item)
        elif util_score == 0:
            suboptimal.append(item)
    
    # Count categories
    optimal_counts = Counter()
    suboptimal_counts = Counter()
    
    for item in optimal:
        for cat in item.get("classification", {}).get("categories", []):
            if cat in VALID_CATEGORIES:
                optimal_counts[cat] += 1
    
    for item in suboptimal:
        for cat in item.get("classification", {}).get("categories", []):
            if cat in VALID_CATEGORIES:
                suboptimal_counts[cat] += 1
    
    # Compute frequencies
    results = {}
    for cat in VALID_CATEGORIES:
        opt_freq = optimal_counts[cat] / len(optimal) if optimal else 0
        sub_freq = suboptimal_counts[cat] / len(suboptimal) if suboptimal else 0
        results[cat] = {
            'optimal': opt_freq,
            'suboptimal': sub_freq,
            'diff': opt_freq - sub_freq
        }
    
    return results, len(optimal), len(suboptimal)


def plot_outcome_comparison(data, output_path, classifier_name="Classifier"):
    """Create outcome comparison visualization."""
    results, n_optimal, n_suboptimal = compute_by_outcome(data)
    
    # Sort by absolute difference
    sorted_cats = sorted(VALID_CATEGORIES, 
                        key=lambda c: abs(results[c]['diff']), 
                        reverse=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Grouped bars
    x = np.arange(len(sorted_cats))
    width = 0.35
    
    optimal_vals = [results[cat]['optimal'] * 100 for cat in sorted_cats]
    suboptimal_vals = [results[cat]['suboptimal'] * 100 for cat in sorted_cats]
    
    bars1 = ax1.bar(x - width/2, optimal_vals, width, 
                    label=f'Optimal (n={n_optimal:,})', 
                    alpha=0.8, color='#2ecc71')
    bars2 = ax1.bar(x + width/2, suboptimal_vals, width, 
                    label=f'Suboptimal (n={n_suboptimal:,})', 
                    alpha=0.8, color='#e74c3c')
    
    ax1.set_ylabel('Frequency (%)', fontsize=12)
    ax1.set_title(f'Reasoning Categories by Outcome ({classifier_name})', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([CATEGORY_DISPLAY[c] for c in sorted_cats], 
                        rotation=45, ha='right', fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for i, (v1, v2) in enumerate(zip(optimal_vals, suboptimal_vals)):
        ax1.text(i - width/2, v1 + 2, f'{v1:.1f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, v2 + 2, f'{v2:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Right plot: Difference bars (delta)
    diffs = [results[cat]['diff'] * 100 for cat in sorted_cats]
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in diffs]
    
    bars = ax2.barh(range(len(sorted_cats)), diffs, color=colors, alpha=0.8)
    
    ax2.set_xlabel('Δ Frequency (Optimal - Suboptimal, pp)', fontsize=12)
    ax2.set_title(f'Association with Optimal Outcomes ({classifier_name})', 
                  fontsize=13, fontweight='bold')
    ax2.set_yticks(range(len(sorted_cats)))
    ax2.set_yticklabels([CATEGORY_DISPLAY[c] for c in sorted_cats], fontsize=10)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, diffs)):
        label_x = val + (1 if val > 0 else -1)
        ha = 'left' if val > 0 else 'right'
        ax2.text(label_x, i, f'{val:+.1f}pp', va='center', ha=ha, 
                fontsize=9, fontweight='bold')
    
    # Add interpretation text
    ax2.text(0.98, 0.02, 
            '← More common in\nsuboptimal outcomes', 
            transform=ax2.transAxes, ha='right', va='bottom',
            fontsize=9, style='italic', color='#c0392b')
    ax2.text(0.02, 0.02, 
            'More common in →\noptimal outcomes', 
            transform=ax2.transAxes, ha='left', va='bottom',
            fontsize=9, style='italic', color='#27ae60')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"OUTCOME COMPARISON ({classifier_name})")
    print(f"{'='*70}")
    print(f"Optimal outcomes: {n_optimal:,} samples")
    print(f"Suboptimal outcomes: {n_suboptimal:,} samples\n")
    
    print(f"{'Category':<25} {'Optimal':>9} {'Suboptimal':>11} {'Δ':>10}")
    print(f"{'-'*70}")
    for cat in sorted_cats:
        r = results[cat]
        sign = '+' if r['diff'] > 0 else ''
        print(f"{CATEGORY_DISPLAY[cat]:<25} {r['optimal']*100:>8.1f}% {r['suboptimal']*100:>10.1f}% {sign}{r['diff']*100:>9.1f}pp")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot outcome comparison")
    parser.add_argument("input", help="Input JSON file with classifications")
    parser.add_argument("--output", default="reasoning_traces/plots/outcome_comparison.png",
                       help="Output plot path")
    parser.add_argument("--name", default="Classifier", help="Classifier name for title")
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    data = load_data(args.input)
    print(f"Loaded {len(data):,} samples")
    
    plot_outcome_comparison(data, args.output, args.name)
    print(f"\n✓ Done!")
