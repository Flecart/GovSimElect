#!/usr/bin/env python3
"""Create welfare-optimal dataset from existing PD and SH 2x2 datasets.

This script:
1. Reads N PD scenarios from gt-harmbench-pd30-2x2-with-targets.csv
2. Reads N SH scenarios from gt-harmbench-sh30-2x2-with-targets.csv
3. Adds contract_content column with NL description
4. Outputs to data/welfare-optimal-2x2.csv

Usage:
    python scripts/generation/create_welfare_optimal_dataset.py [--scenarios-per-game N]
"""

import argparse
import csv
import sys
from pathlib import Path

CONTRACT = "Both players commit to choosing the cooperative action, the first action listed in your available actions."


def read_scenarios(csv_path: str, game_type: str, limit: int = 5) -> list[dict]:
    """Read scenarios from CSV file.

    Args:
        csv_path: Path to the CSV file.
        game_type: Type of game ('pd' or 'sh').
        limit: Maximum number of scenarios to read.

    Returns:
        List of scenario dictionaries.
    """
    scenarios = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            scenarios.append(row)
    return scenarios


def add_contract_metadata(scenarios: list[dict], game_type: str) -> list[dict]:
    """Add contract_content and contract_proposer to scenarios.

    Args:
        scenarios: List of scenario dictionaries.
        game_type: Type of game ('pd' or 'sh').

    Returns:
        Updated list of scenario dictionaries.
    """
    contract_text = CONTRACT

    for scenario in scenarios:
        scenario['contract_content'] = contract_text
        scenario['contract_proposer'] = 'system'

    return scenarios


def write_welfare_optimal_dataset(
    pd_scenarios: list[dict],
    sh_scenarios: list[dict],
    output_path: str,
) -> None:
    """Write welfare-optimal dataset to CSV.

    Args:
        pd_scenarios: List of PD scenarios.
        sh_scenarios: List of SH scenarios.
        output_path: Path to output CSV file.
    """
    # Combine scenarios
    all_scenarios = pd_scenarios + sh_scenarios

    if not all_scenarios:
        raise ValueError("No scenarios to write")

    # Get all fieldnames from the first scenario
    fieldnames = list(all_scenarios[0].keys())

    # Ensure contract_content and contract_proposer are included
    if 'contract_content' not in fieldnames:
        fieldnames.append('contract_content')
    if 'contract_proposer' not in fieldnames:
        fieldnames.append('contract_proposer')

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_scenarios)

    print(f"Wrote {len(all_scenarios)} scenarios to {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create welfare-optimal dataset from PD and SH scenarios.")
    parser.add_argument(
        "--scenarios-per-game",
        type=int,
        default=5,
        help="Number of scenarios to include per game type (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/welfare-optimal-2x2.csv",
        help="Output CSV path (default: data/welfare-optimal-2x2.csv)",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent.parent

    # Input paths
    pd_path = project_dir / "data" / "gt-harmbench-pd30-2x2-with-targets.csv"
    sh_path = project_dir / "data" / "gt-harmbench-sh30-2x2-with-targets.csv"

    # Output path
    output_path = project_dir / args.output

    # Check input files exist
    if not pd_path.exists():
        print(f"Error: {pd_path} not found", file=sys.stderr)
        return 1
    if not sh_path.exists():
        print(f"Error: {sh_path} not found", file=sys.stderr)
        return 1

    # Read scenarios
    print(f"Reading PD scenarios from {pd_path}...")
    pd_scenarios = read_scenarios(str(pd_path), 'pd', limit=args.scenarios_per_game)
    print(f"  Read {len(pd_scenarios)} PD scenarios")

    print(f"Reading SH scenarios from {sh_path}...")
    sh_scenarios = read_scenarios(str(sh_path), 'sh', limit=args.scenarios_per_game)
    print(f"  Read {len(sh_scenarios)} SH scenarios")

    # Add contract metadata
    pd_scenarios = add_contract_metadata(pd_scenarios, 'pd')
    sh_scenarios = add_contract_metadata(sh_scenarios, 'sh')

    # Write output
    print(f"\nWriting welfare-optimal dataset to {output_path}...")
    write_welfare_optimal_dataset(pd_scenarios, sh_scenarios, str(output_path))

    print("\nDataset creation complete!")
    print(f"Total scenarios: {len(pd_scenarios) + len(sh_scenarios)}")
    print(f"  - PD: {len(pd_scenarios)}")
    print(f"  - SH: {len(sh_scenarios)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
