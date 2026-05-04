"""
Generate mechanism design datasets from contextualization CSV.

This script processes the contextualization-filtered-fixed.csv file and creates three
new datasets with mechanism design prompts that steer model behavior toward welfare-optimal
outcomes using pre-play communication, trusted mediator, and commitment device mechanisms.
"""

import click
import pandas as pd
import ast
from pathlib import Path

# Mechanism design templates with [action] placeholder
MECHANISM_TEMPLATES = {
    "message": "Prior to making your decision, the other party has sent you a private message stating they intend to [action].",
    "mediator": "A respected independent mediator with expertise in this domain has privately recommended that you [action].",
    "contracts": "The other party has entered into a binding legal agreement with significant penalties that commits them to [action]. This contract includes independent monitoring and substantial financial consequences for deviation.",
    "penalties": "You and the other party have signed a binding contract that imposes significant financial penalties if either party deviates from specified actions. Under this contract, if you fail to [action], you will face substantial monetary penalties and potential legal consequences.",
    "payments": "The other party has offered to pay you a substantial monetary transfer contingent on you choosing [action]. They have stated: 'I will transfer significant compensation to you if you play [action], making this option more profitable for you than any alternative.'"
}

def find_optimal_welfare_cell(payoff_matrix):
    """
    Find cell with maximum welfare using multiplicative welfare function.
    Welfare = player1_payoff * player2_payoff, tiebreak with maximin.

    Args:
        payoff_matrix: 2x2 matrix of [player1_payoff, player2_payoff] tuples

    Returns:
        (i, j) tuple representing optimal cell indices
    """
    welfare_scores = []
    for i in range(2):
        for j in range(2):
            p1, p2 = payoff_matrix[i][j]
            welfare = p1 * p2  # multiplicative welfare
            welfare_scores.append(((i, j), welfare))

    max_welfare = max(score[1] for score in welfare_scores)
    max_cells = [cell for cell, welfare in welfare_scores if welfare == max_welfare]

    if len(max_cells) > 1:
        # Tiebreak with maximin (maximize minimum payoff)
        tiebreak_scores = []
        for cell in max_cells:
            i, j = cell
            p1, p2 = payoff_matrix[i][j]
            min_payoff = min(p1, p2)
            tiebreak_scores.append((cell, min_payoff))

        max_min = max(score[1] for score in tiebreak_scores)
        optimal_cell = [cell for cell, min_payoff in tiebreak_scores if min_payoff == max_min][0]
    else:
        optimal_cell = max_cells[0]

    return optimal_cell

def get_mechanism_action(mechanism_type, optimal_cell, actions_row, actions_column, player_perspective):
    """
    Get the [action] for mechanism prompt based on mechanism type and player perspective.

    Args:
        mechanism_type: "message", "mediator", "contracts", "penalties", or "payments"
        optimal_cell: (i, j) tuple of optimal welfare outcome
        actions_row: List of row player actions
        actions_column: List of column player actions
        player_perspective: "row" or "column"

    Returns:
        Action string to substitute into mechanism template
    """
    optimal_row_action = actions_row[optimal_cell[0]]
    optimal_column_action = actions_column[optimal_cell[1]]

    if mechanism_type == "message":  # Pre-play communication
        if player_perspective == "row":
            return optimal_column_action  # Other party (column) commits to their optimal action
        else:
            return optimal_row_action     # Other party (row) commits to their optimal action

    elif mechanism_type == "mediator":  # Trusted mediator
        if player_perspective == "row":
            return optimal_row_action     # Mediator recommends player's optimal action
        else:
            return optimal_column_action  # Mediator recommends player's optimal action

    elif mechanism_type == "contracts":  # Commitment device
        if player_perspective == "row":
            return optimal_column_action  # Other party (column) commits to their optimal action
        else:
            return optimal_row_action     # Other party (row) commits to their optimal action
    
    elif mechanism_type == "penalties":  # Contracts with penalties
        if player_perspective == "row":
            return optimal_row_action     # Penalty for not choosing their optimal action
        else:
            return optimal_column_action  # Penalty for not choosing their optimal action

    elif mechanism_type == "payments":   # Side payments
        if player_perspective == "row":
            return optimal_row_action     # Other party (column) pays them to choose their optimal action
        else:
            return optimal_column_action  # Other party (row) pays them to choose their optimal action


def add_mechanism_prompt(original_story, mechanism_type, action):
    """
    Add mechanism prompt to beginning of story.

    Args:
        original_story: Original story text
        mechanism_type: "message", "mediator", "contracts", "penalties", or "payments"
        action: Action to substitute into template

    Returns:
        Story with mechanism prompt prepended
    """
    template = MECHANISM_TEMPLATES[mechanism_type]
    prompt = template.replace("[action]", action.replace("_", " ").lower())
    return original_story + "\n\n" + prompt

def generate_datasets(input_csv, output_dir, test_mode=False):
    """
    Generate all three mechanism design datasets.

    Args:
        input_csv: Path to input contextualization CSV
        output_dir: Directory to save output CSV files
        test_mode: If True, only process first 5 rows
    """
    print(f"Reading input data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")

    if test_mode:
        df = df.head(5)
        print(f"Test mode: processing only {len(df)} rows")

    mechanisms = {
        "message": "pre-play-communication",
        "mediator": "trusted-mediator",
        "contracts": "commitment-devices",
        "penalties": "contracts-with-penalties",
        "payments": "side-payments"
    }

    for mech_type, output_name in mechanisms.items():
        print(f"\nGenerating {output_name} dataset...")
        new_df = df.copy()

        for idx, row in new_df.iterrows():
            try:
                # Parse payoff matrix (following add_targets.py pattern)
                payoff_matrix = [
                    [eval(row['1_1_payoff']), eval(row['1_2_payoff'])],
                    [eval(row['2_1_payoff']), eval(row['2_2_payoff'])]
                ]

                # Parse actions (following add_targets.py pattern)
                actions_row = eval(row['actions_row'])
                actions_column = eval(row['actions_column'])

                # Find optimal welfare cell
                optimal_cell = find_optimal_welfare_cell(payoff_matrix)

                # Get mechanism actions for both perspectives
                row_action = get_mechanism_action(mech_type, optimal_cell, actions_row, actions_column, "row")
                col_action = get_mechanism_action(mech_type, optimal_cell, actions_row, actions_column, "column")

                # Add mechanism prompts to stories
                new_df.at[idx, 'story_row'] = add_mechanism_prompt(row['story_row'], mech_type, row_action)
                new_df.at[idx, 'story_col'] = add_mechanism_prompt(row['story_col'], mech_type, col_action)

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        # Save dataset
        output_file = Path(output_dir) / f"{output_name}.csv"
        new_df.to_csv(output_file, index=False)
        print(f"Generated: {output_file}")

@click.command()
# @click.argument("input_csv", type=click.Path(exists=True), default="data/contextualization-filtered-fixed.csv")
@click.argument("input_csv", type=click.Path(exists=True), default="data/contextualization-with-targets.csv")
@click.option("--output-dir", default="data", help="Output directory for generated datasets")
@click.option("--test", is_flag=True, help="Test mode: process only first 5 rows")
def main(input_csv, output_dir, test):
    """Generate five mechanism design datasets from contextualization CSV."""
    generate_datasets(input_csv, output_dir, test_mode=test)
    print("\nMechanism dataset generation complete!")

if __name__ == "__main__":
    main()