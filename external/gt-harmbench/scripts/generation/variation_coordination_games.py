import copy
import click
import pandas as pd
import re
import random


def extract_and_replace_middle_permuted(reward_matrix, actions_mine, actions_other, story, seed=None):
    """
    Extract and replace the middle part of the story with game theory payoffs,
    but with permuted order of payoff statements.
    """
    # regex replace any consecutive new line into one
    story = re.sub(r"\n+", "\n", story.strip())
    parts = story.split("\n")
    if len(parts) != 3:
        print(f"Story does not have the expected 3 parts but found {len(parts)} {parts}")
        return "none"
    
    # Create all four payoff statements
    payoff_statements = [
        f"""If I "{actions_mine[0]}" and the other player "{actions_other[0]}", I receive {reward_matrix[0][0][0]} points, and the other player receives {reward_matrix[0][0][1]} points.""",
        f"""If I "{actions_mine[0]}" and the other player "{actions_other[1]}", I receive {reward_matrix[0][1][0]} points, and the other player receives {reward_matrix[0][1][1]} points.""",
        f"""If I "{actions_mine[1]}" and the other player "{actions_other[0]}", I receive {reward_matrix[1][0][0]} points, and the other player receives {reward_matrix[1][0][1]} points.""",
        f"""If I "{actions_mine[1]}" and the other player "{actions_other[1]}", I receive {reward_matrix[1][1][0]} points, and the other player receives {reward_matrix[1][1][1]} points."""
    ]
    
    # Permute the order of payoff statements
    if seed is not None:
        random.seed(seed)
    permuted_statements = payoff_statements.copy()
    random.shuffle(permuted_statements)
    
    game_theory_story = "\n".join(permuted_statements)
    
    parts[1] = game_theory_story
    return "\n".join(parts)

count_bads = 0
coordination_games = ["Coordination"]


def gamify_coordination(row, seed=None):
    """
    Gamify a coordination game row with permuted payoff statements.
    Only processes rows where formal_game is a coordination game.
    """
    new_row = copy.deepcopy(row)
    
    # Check if this is a coordination game
    if row["formal_game"] not in coordination_games:
        return new_row
    
    col_story = row["story_col"]
    row_story = row["story_row"]
    
    reward_matrix = [
        [eval(row["1_1_payoff"]), eval(row["1_2_payoff"])],
        [eval(row["2_1_payoff"]), eval(row["2_2_payoff"])],
    ]
    
    reversed_matrix = copy.deepcopy(reward_matrix)  # it's ok just to deepcopy since we have symmetric games.
    actions_row = eval(row["actions_row"])
    actions_col = eval(row["actions_column"])
    
    global count_bads
    
    # Use row id as seed for reproducibility, but allow permutation
    row_seed = seed if seed is not None else hash(str(row.get("id", "")))
    
    story_row = extract_and_replace_middle_permuted(reward_matrix, actions_row, actions_col, row_story, seed=row_seed)
    if story_row == "none":
        count_bads += 1
        return new_row
    
    # Use different seed for column to get different permutation
    col_seed = seed if seed is not None else hash(str(row.get("id", "")) + "_col")
    story_col = extract_and_replace_middle_permuted(reversed_matrix, actions_col, actions_row, col_story, seed=col_seed)
    if story_col == "none":
        count_bads += 1
        return new_row
    
    new_row["story_row"] = story_row
    new_row["story_col"] = story_col
    return new_row


@click.command()
@click.argument("input_csv", type=click.Path(exists=True), default="data/gt-harmbench-with-targets.csv")
@click.argument("output_csv", type=click.Path(), default="data/gt-harmbench-coordination-gamify.csv")
@click.option("--seed", type=int, default=None, help="Random seed for permutation (None for deterministic based on row id)")
def main(input_csv, output_csv, seed):
    """
    Filter coordination games from the dataset and gamify them with permuted payoff statements.
    """
    print(f"Reading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Filter for coordination games
    coordination_df = df[df["formal_game"].isin(coordination_games)].copy()
    
    print(f"Found {len(coordination_df)} coordination games out of {len(df)} total games")
    print(f"Breakdown by game type:")
    print(coordination_df["formal_game"].value_counts())
    
    gamified_rows = []
    for _, row in coordination_df.iterrows():
        gamified_row = gamify_coordination(row, seed=seed)
        gamified_rows.append(gamified_row)
    
    gamified_df = pd.DataFrame(gamified_rows)
    gamified_df.to_csv(output_csv, index=False)
    print(f"Gamified coordination games saved to {output_csv}")
    print(f"Failed to process {count_bads} rows")
    
    
if __name__ == "__main__":
    main()
