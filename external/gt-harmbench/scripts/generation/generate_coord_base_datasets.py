#!/usr/bin/env python3
"""Generate Coordination 2x2 datasets for contracting experiments.

Samples N coordination games from gt-harmbench.csv, gamifies stories,
and adds targets.

Usage:
    python generate_coord_base_datasets.py --sample 30
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import click
import pandas as pd

# Import from add_targets module
from scripts.generation.add_targets import get_targets
from scripts.generation.variation_game_theoretic import gamify
from src.metrics import utilitarian, fairness, nash_social_welfare


@click.command()
@click.option(
    "--sample", type=int, default=30,
    help="Number of games to sample from base dataset.",
)
@click.option(
    "--base-dataset", type=click.Path(exists=True), default="data/gt-harmbench.csv",
    help="Base dataset to sample from. Default: data/gt-harmbench.csv",
)
def main(sample, base_dataset):
    """Generate Coordination 2x2 datasets with targets.

    Samples N coordination games from the base dataset and adds targets.
    """
    print(f"Reading {base_dataset}...")
    df = pd.read_csv(base_dataset)
    original_count = len(df)

    # Filter to coordination games only
    coord_games = df[df["formal_game"] == "Coordination"].copy()
    print(f"Filtered to {len(coord_games)} Coordination games (from {original_count} total)")

    # Sample
    if sample > len(coord_games):
        raise click.ClickException(f"Cannot sample {sample} games from {len(coord_games)} available")
    coord_sampled = coord_games.sample(sample, random_state=42)
    print(f"Sampled {len(coord_sampled)} games")

    # Gamify stories (add explicit payoff matrices)
    print("\nGamifying 2x2 stories...")
    gamified_rows = []
    for _, row in coord_sampled.iterrows():
        gamified_row = gamify(row)
        gamified_rows.append(gamified_row)
    coord_sampled = pd.DataFrame(gamified_rows)
    print(f"Gamified {len(gamified_rows)} 2x2 stories")

    # Save 2x2 dataset
    output_path = Path(f"data/gt-harmbench-coordination{sample}-2x2.csv")
    coord_sampled.to_csv(output_path, index=False)
    print(f"Saved 2x2 subset to {output_path}")

    # Add targets using the same logic as add_targets.py
    print("\nAdding targets to 2x2 subset...")

    nash_equilibria_list = []
    utility_maximizing_list = []
    rawlsian_list = []
    nash_social_welfare_list = []
    max_utilitarian_list = []
    max_rawlsian_list = []
    max_social_welfare_list = []

    for _, row in coord_sampled.iterrows():
        payoff_matrix = [
            [eval(row['1_1_payoff']), eval(row['1_2_payoff'])],
            [eval(row['2_1_payoff']), eval(row['2_2_payoff'])]
        ]

        targets = get_targets(row)
        nash_equilibria_list.append("|".join(targets.nash_equilibria) if targets.nash_equilibria else "None")
        utility_maximizing_list.append("|".join(targets.utility_maximizing) if targets.utility_maximizing else "None")
        rawlsian_list.append("|".join(targets.rawlsian) if targets.rawlsian else "None")
        nash_social_welfare_list.append("|".join(targets.nash_social_welfare) if targets.nash_social_welfare else "None")
        max_utilitarian_list.append(utilitarian(payoff_matrix))
        max_rawlsian_list.append(fairness(payoff_matrix))
        max_social_welfare_list.append(nash_social_welfare(payoff_matrix))

    coord_sampled['target_nash_equilibria'] = nash_equilibria_list
    coord_sampled['target_utility_maximizing'] = utility_maximizing_list
    coord_sampled['target_rawlsian'] = rawlsian_list
    coord_sampled['target_nash_social_welfare'] = nash_social_welfare_list
    coord_sampled['nash_social_welfare'] = max_social_welfare_list
    coord_sampled['max_utilitarian'] = max_utilitarian_list
    coord_sampled['max_rawlsian'] = max_rawlsian_list

    output_with_targets = Path(f"data/gt-harmbench-coordination{sample}-2x2-with-targets.csv")
    coord_sampled.to_csv(output_with_targets, index=False)
    print(f"Saved 2x2 with targets to {output_with_targets}")

    print("\nDone!")


if __name__ == "__main__":
    main()
