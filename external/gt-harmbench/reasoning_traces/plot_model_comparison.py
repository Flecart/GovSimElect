"""
Plot model comparison using macro-categories.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

# Macro-categories
MACRO_CATEGORIES = {
    "Game-Theoretic": ["NASH_DOMINANT", "PAYOFF_MAX"],
    "Social Welfare": ["UTILITARIAN", "RAWLSIAN"],
    "Risk/Catastrophe": ["CATASTROPHE_PREVENTION", "PRECAUTIONARY"],
    "Domain-Specific": ["AI_SAFETY", "DOMAIN_OTHER"]
}

def compute_macro_frequencies(items):
    """Compute frequency of each macro-category."""
    n = len(items)
    macro_freq = {}
    
    for macro, categories in MACRO_CATEGORIES.items():
        # Count items that have ANY category from this macro-category
        count = 0
        for item in items:
            item_cats = set(item.get("classification", {}).get("categories", []))
            if any(cat in item_cats for cat in categories):
                count += 1
        macro_freq[macro] = 100 * count / n
    
    return macro_freq

# Load data
print("Loading data...")
with open("reasoning_traces/classified_all_gpt4o.json") as f:
    data = json.load(f)

# Group by model
by_model = defaultdict(list)
for item in data:
    model = item.get("model", "unknown")
    by_model[model].append(item)

# Simplify model names for display
MODEL_NAMES = {
    "anthropic/claude-opus-4.5": "Claude Opus 4.5",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "deepseek/deepseek-v3.2": "DeepSeek v3.2",
    "qwen/qwen3-30b-a3b": "Qwen3 30B"
}

# Compute macro-category frequencies
results = {}
for model, items in sorted(by_model.items()):
    display_name = MODEL_NAMES.get(model, model)
    results[display_name] = compute_macro_frequencies(items)

# Print table
print(f"\n{'='*70}")
print("MACRO-CATEGORY FREQUENCIES BY MODEL")
print(f"{'='*70}\n")
print(f"{'Model':<20} {'Game-Theoretic':>15} {'Social Welfare':>15} {'Risk/Catastrophe':>16} {'Domain-Specific':>17}")
print("-" * 85)
for model, freqs in results.items():
    print(f"{model:<20} {freqs['Game-Theoretic']:>14.1f}% {freqs['Social Welfare']:>14.1f}% {freqs['Risk/Catastrophe']:>15.1f}% {freqs['Domain-Specific']:>16.1f}%")

# Create visualization
fig, ax = plt.subplots(figsize=(12, 6))

models = list(results.keys())
macro_cats = list(MACRO_CATEGORIES.keys())
x = np.arange(len(models))
width = 0.2

# Color scheme
colors = ['#2563eb', '#16a34a', '#dc2626', '#ea580c']

# Plot bars
for i, macro in enumerate(macro_cats):
    values = [results[model][macro] for model in models]
    ax.bar(x + i * width, values, width, label=macro, color=colors[i], alpha=0.9)

# Customize plot
ax.set_ylabel('Frequency (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_title('Reasoning Category Usage by Model', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.legend(loc='upper left', frameon=True, shadow=True)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 105)

# Add value labels on bars
for i, macro in enumerate(macro_cats):
    values = [results[model][macro] for model in models]
    for j, v in enumerate(values):
        ax.text(j + i * width, v + 2, f'{v:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()

# Save
output_path = "reasoning_traces/analysis_output/model_comparison_macro.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot to {output_path}")

# Also create a stacked version
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Prepare data for stacking
data_matrix = np.array([[results[model][macro] for macro in macro_cats] for model in models])

# Create stacked bar chart
bottom = np.zeros(len(models))
for i, macro in enumerate(macro_cats):
    values = data_matrix[:, i]
    ax2.barh(models, values, left=bottom, label=macro, color=colors[i], alpha=0.9)
    
    # Add labels
    for j, (model, value) in enumerate(zip(models, values)):
        if value > 5:  # Only show label if bar is wide enough
            ax2.text(bottom[j] + value/2, j, f'{value:.0f}%', 
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    bottom += values

ax2.set_xlabel('Cumulative Frequency (%)', fontsize=12, fontweight='bold')
ax2.set_title('Reasoning Category Distribution by Model (Stacked)', fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='lower right', frameon=True, shadow=True)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()

# Save stacked version
output_path2 = "reasoning_traces/analysis_output/model_comparison_stacked.png"
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✓ Saved stacked plot to {output_path2}")

plt.show()
