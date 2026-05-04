"""
Analyze classified reasoning traces.
Computes category frequencies, comparisons, and generates figures.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import csv


# Category mapping for cleaner display
CATEGORY_DISPLAY = {
    "NASH_DOMINANT": "Nash/Dominant Strategy",
    "PAYOFF_MAX": "Payoff Maximization",
    "UTILITARIAN": "Utilitarian Reasoning",
    "RAWLSIAN": "Rawlsian Reasoning", 
    "CATASTROPHE_PREVENTION": "Catastrophe Prevention",
    "PRECAUTIONARY": "Precautionary Principle",
    "AI_SAFETY": "AI Alignment & Safety",
    "DOMAIN_OTHER": "Domain-Specific (Other)"
}

MACRO_CATEGORIES = {
    "Game-Theoretic": ["NASH_DOMINANT", "PAYOFF_MAX"],
    "Social Welfare": ["UTILITARIAN", "RAWLSIAN"],
    "Risk/Catastrophe": ["CATASTROPHE_PREVENTION", "PRECAUTIONARY"],
    "Domain-Specific": ["AI_SAFETY", "DOMAIN_OTHER"]
}


def load_data(filepath: str) -> list:
    """Load classified data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_category_frequencies(data: list) -> dict:
    """Compute overall category frequencies."""
    total = len(data)
    counts = defaultdict(int)
    
    for item in data:
        cats = item.get("classification", {}).get("categories", [])
        for cat in cats:
            counts[cat] += 1
    
    return {cat: count/total for cat, count in counts.items()}


def compute_by_game_type(data: list) -> dict:
    """Compute category frequencies by game type."""
    by_game = defaultdict(list)
    
    for item in data:
        game = item.get("formal_game", "Unknown")
        by_game[game].append(item)
    
    results = {}
    for game, items in by_game.items():
        results[game] = {
            "n": len(items),
            "frequencies": compute_category_frequencies(items)
        }
    
    return results


def compute_by_outcome(data: list) -> dict:
    """Compute category frequencies by outcome (utilitarian optimal vs suboptimal)."""
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
            "n": len(optimal),
            "frequencies": compute_category_frequencies(optimal)
        },
        "suboptimal": {
            "n": len(suboptimal),
            "frequencies": compute_category_frequencies(suboptimal)
        }
    }


def compute_by_model(data: list) -> dict:
    """Compute category frequencies by model."""
    by_model = defaultdict(list)
    
    for item in data:
        model = item.get("model", "unknown")
        # Simplify model name
        if "claude" in model.lower():
            if "opus" in model.lower():
                model = "Claude Opus 4.5"
            elif "sonnet" in model.lower():
                model = "Claude Sonnet 4.5"
        elif "deepseek" in model.lower():
            model = "DeepSeek v3.2"
        elif "qwen" in model.lower():
            model = "Qwen3 30B"
        
        by_model[model].append(item)
    
    results = {}
    for model, items in by_model.items():
        results[model] = {
            "n": len(items),
            "frequencies": compute_category_frequencies(items)
        }
    
    return results


def compute_outcome_comparison(data: list) -> dict:
    """Compute percentage point differences between optimal and suboptimal outcomes."""
    by_outcome = compute_by_outcome(data)
    
    if by_outcome["optimal"]["n"] == 0 or by_outcome["suboptimal"]["n"] == 0:
        return {}
    
    all_cats = set(by_outcome["optimal"]["frequencies"].keys()) | set(by_outcome["suboptimal"]["frequencies"].keys())
    
    comparison = {}
    for cat in all_cats:
        opt_freq = by_outcome["optimal"]["frequencies"].get(cat, 0)
        sub_freq = by_outcome["suboptimal"]["frequencies"].get(cat, 0)
        diff = opt_freq - sub_freq
        comparison[cat] = {
            "optimal": opt_freq,
            "suboptimal": sub_freq,
            "diff_pp": diff * 100  # percentage points
        }
    
    return comparison


def print_comparison_table(comparison: dict, title: str):
    """Print a comparison table like Figure 4 in the paper."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"{'Category':<30} {'Optimal':>12} {'Suboptimal':>12} {'Δ (pp)':>12}")
    print("-"*70)
    
    # Sort by absolute difference
    sorted_cats = sorted(comparison.items(), key=lambda x: abs(x[1]["diff_pp"]), reverse=True)
    
    for cat, vals in sorted_cats:
        display_name = CATEGORY_DISPLAY.get(cat, cat)
        opt = vals["optimal"] * 100
        sub = vals["suboptimal"] * 100
        diff = vals["diff_pp"]
        sign = "+" if diff > 0 else ""
        print(f"{display_name:<30} {opt:>11.1f}% {sub:>11.1f}% {sign}{diff:>10.1f}pp")


def export_for_plotting(data: list, output_dir: Path):
    """Export data in formats suitable for plotting."""
    output_dir.mkdir(exist_ok=True)
    
    # By game type
    by_game = compute_by_game_type(data)
    with open(output_dir / "by_game_type.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        all_cats = list(CATEGORY_DISPLAY.keys())
        writer.writerow(["game_type", "n"] + all_cats)
        for game, vals in by_game.items():
            row = [game, vals["n"]]
            for cat in all_cats:
                row.append(vals["frequencies"].get(cat, 0))
            writer.writerow(row)
    
    # By outcome
    by_outcome = compute_by_outcome(data)
    with open(output_dir / "by_outcome.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        all_cats = list(CATEGORY_DISPLAY.keys())
        writer.writerow(["outcome", "n"] + all_cats)
        for outcome, vals in by_outcome.items():
            row = [outcome, vals["n"]]
            for cat in all_cats:
                row.append(vals["frequencies"].get(cat, 0))
            writer.writerow(row)
    
    # By model
    by_model = compute_by_model(data)
    with open(output_dir / "by_model.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        all_cats = list(CATEGORY_DISPLAY.keys())
        writer.writerow(["model", "n"] + all_cats)
        for model, vals in by_model.items():
            row = [model, vals["n"]]
            for cat in all_cats:
                row.append(vals["frequencies"].get(cat, 0))
            writer.writerow(row)
    
    # Outcome comparison (for bar chart like Figure 4)
    comparison = compute_outcome_comparison(data)
    with open(output_dir / "outcome_comparison.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["category", "optimal", "suboptimal", "diff_pp"])
        for cat, vals in comparison.items():
            writer.writerow([
                CATEGORY_DISPLAY.get(cat, cat),
                vals["optimal"],
                vals["suboptimal"],
                vals["diff_pp"]
            ])
    
    print(f"Exported CSV files to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Analyze classified reasoning traces")
    parser.add_argument("input", type=str, help="Input JSON file from classifier.py")
    parser.add_argument("--output-dir", type=str, default="analysis_output", help="Output directory for CSVs")
    parser.add_argument("--export-csv", action="store_true", help="Export data as CSV for plotting")
    
    args = parser.parse_args()
    
    # Load data
    data = load_data(args.input)
    print(f"Loaded {len(data)} classified samples")
    
    # Overall frequencies
    print("\n" + "="*50)
    print("Overall Category Frequencies")
    print("="*50)
    
    freqs = compute_category_frequencies(data)
    for cat in CATEGORY_DISPLAY.keys():
        freq = freqs.get(cat, 0)
        print(f"  {CATEGORY_DISPLAY[cat]:<30}: {freq*100:>6.1f}%")
    
    # By game type
    print("\n" + "="*50)
    print("Category Frequencies by Game Type")
    print("="*50)
    
    by_game = compute_by_game_type(data)
    for game, vals in sorted(by_game.items()):
        print(f"\n{game} (n={vals['n']}):")
        for cat in CATEGORY_DISPLAY.keys():
            freq = vals["frequencies"].get(cat, 0)
            if freq > 0:
                bar = "█" * int(freq * 20)
                print(f"  {CATEGORY_DISPLAY[cat]:<30}: {freq*100:>5.1f}% {bar}")
    
    # Outcome comparison
    comparison = compute_outcome_comparison(data)
    if comparison:
        print_comparison_table(comparison, "Category Frequency: Optimal vs Suboptimal Outcomes")
    
    # By model
    print("\n" + "="*50)
    print("Category Frequencies by Model")
    print("="*50)
    
    by_model = compute_by_model(data)
    for model, vals in sorted(by_model.items()):
        print(f"\n{model} (n={vals['n']}):")
        for cat in CATEGORY_DISPLAY.keys():
            freq = vals["frequencies"].get(cat, 0)
            if freq > 0:
                print(f"  {CATEGORY_DISPLAY[cat]:<30}: {freq*100:>5.1f}%")
    
    # Export CSV if requested
    if args.export_csv:
        export_for_plotting(data, Path(args.output_dir))


if __name__ == "__main__":
    main()