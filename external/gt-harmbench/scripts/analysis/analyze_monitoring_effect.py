#!/usr/bin/env python3
"""Analyze monitoring effect experiment results.

Compares violation rates between enforced and unenforced conditions.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


def load_compliance_metrics(log_dir: str) -> list[dict[str, Any]]:
    """Load all compliance metrics CSVs from log directory."""
    log_path = Path(log_dir)
    metrics = []

    # Find all compliance-metrics-*.csv files
    for csv_file in sorted(log_path.glob("**/compliance-metrics-*.csv")):
        print(f"Loading: {csv_file}")
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract condition from file path or metadata
                row['_source_file'] = str(csv_file)
                metrics.append(row)

    return metrics


def analyze_monitoring_effect(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze the monitoring effect by comparing enforced vs unenforced."""

    # Group by condition
    enforced = [m for m in metrics if m.get('contract_mode') == 'welfare_optimal_enforced']
    unenforced = [m for m in metrics if m.get('contract_mode') == 'welfare_optimal_unenforced']

    print(f"\n=== Sample Counts ===")
    print(f"Enforced: {len(enforced)} samples")
    print(f"Unenforced: {len(unenforced)} samples")

    # Calculate violation rates
    enforced_violations = sum(1 for m in enforced if not m.get('both_complied', 'False').lower() == 'true')
    unenforced_violations = sum(1 for m in unenforced if not m.get('both_complied', 'False').lower() == 'true')

    enforced_rate = enforced_violations / len(enforced) if enforced else 0
    unenforced_rate = unenforced_violations / len(unenforced) if unenforced else 0

    print(f"\n=== Violation Rates ===")
    print(f"Enforced:   {enforced_rate:.1%} ({enforced_violations}/{len(enforced)})")
    print(f"Unenforced: {unenforced_rate:.1%} ({unenforced_violations}/{len(unenforced)})")
    print(f"Gap:        {abs(enforced_rate - unenforced_rate):.1%} points")

    # Calculate Nash accuracy
    enforced_nash = sum(1 for m in enforced if m.get('is_nash', 'False').lower() == 'true')
    unenforced_nash = sum(1 for m in unenforced if m.get('is_nash', 'False').lower() == 'true')

    enforced_nash_rate = enforced_nash / len(enforced) if enforced else 0
    unenforced_nash_rate = unenforced_nash / len(unenforced) if unenforced else 0

    print(f"\n=== Nash Accuracy ===")
    print(f"Enforced:   {enforced_nash_rate:.1%} ({enforced_nash}/{len(enforced)})")
    print(f"Unenforced: {unenforced_nash_rate:.1%} ({unenforced_nash}/{len(unenforced)})")

    # Calculate average payoffs
    enforced_util = sum(float(m.get('utilitarian_payoff', 0) or 0) for m in enforced) / len(enforced) if enforced else 0
    unenforced_util = sum(float(m.get('utilitarian_payoff', 0) or 0) for m in unenforced) / len(unenforced) if unenforced else 0

    print(f"\n=== Average Utilitarian Payoff ===")
    print(f"Enforced:   {enforced_util:.2f}")
    print(f"Unenforced: {unenforced_util:.2f}")

    # Break down by game type
    print(f"\n=== By Game Type ===")
    for game_type in ['Prisoner\'s Dilemma', 'Stag hunt']:
        game_enforced = [m for m in enforced if m.get('formal_game') == game_type]
        game_unenforced = [m for m in unenforced if m.get('formal_game') == game_type]

        if game_enforced and game_unenforced:
            game_enf_viol = sum(1 for m in game_enforced if not m.get('both_complied', 'False').lower() == 'true')
            game_unenf_viol = sum(1 for m in game_unenforced if not m.get('both_complied', 'False').lower() == 'true')

            game_enf_rate = game_enf_viol / len(game_enforced)
            game_unenf_rate = game_unenf_viol / len(game_unenforced)

            print(f"\n{game_type}:")
            print(f"  Enforced:   {game_enf_rate:.1%} ({game_enf_viol}/{len(game_enforced)})")
            print(f"  Unenforced: {game_unenf_rate:.1%} ({game_unenf_viol}/{len(game_unenforced)})")

    return {
        'enforced_violation_rate': enforced_rate,
        'unenforced_violation_rate': unenforced_rate,
        'enforced_nash_accuracy': enforced_nash_rate,
        'unenforced_nash_accuracy': unenforced_nash_rate,
        'enforced_avg_utilitarian': enforced_util,
        'unenforced_avg_utilitarian': unenforced_util,
        'gap': abs(enforced_rate - unenforced_rate),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze monitoring effect experiment results")
    parser.add_argument("log_dir", help="Log directory containing compliance-metrics CSV files")
    args = parser.parse_args()

    metrics = load_compliance_metrics(args.log_dir)

    if not metrics:
        print("No compliance metrics found!", file=sys.stderr)
        return 1

    results = analyze_monitoring_effect(metrics)

    print(f"\n=== Summary ===")
    print(f"Monitoring effect gap: {results['gap']:.1%} violation rate difference")

    return 0


if __name__ == "__main__":
    sys.exit(main())
