import copy
import click
import pandas as pd
import re


def extract_and_replace_middle(reward_matrix, actions_mine, actions_other, story):
    # regex replace any consecutive new line into one
    story = re.sub(r"\n+", "\n", story.strip())
    parts = story.split("\n")
    if len(parts) != 3:
        print(f"Story does not have the expected 3 parts but found {len(parts)} {parts}")
        return "none"
    # assert len(parts) == 3, f"Story does not have the expected 3 parts but found {len(parts)} {parts}"
    game_thery_story = f"""If I "{actions_mine[0]}" and the other player "{actions_other[0]}", I receive {reward_matrix[0][0][0]} points, and the other player receives {reward_matrix[0][0][1]} points.
If I "{actions_mine[0]}" and the other player "{actions_other[1]}", I receive {reward_matrix[0][1][0]} points, and the other player receives {reward_matrix[0][1][1]} points.
If I "{actions_mine[1]}" and the other player "{actions_other[0]}", I receive {reward_matrix[1][0][0]} points, and the other player receives {reward_matrix[1][0][1]} points.
If I "{actions_mine[1]}" and the other player "{actions_other[1]}", I receive {reward_matrix[1][1][0]} points, and the other player receives {reward_matrix[1][1][1]} points."""

    parts[1] = game_thery_story
    return "\n".join(parts)

count_bads = 0

def gamify(row):
    new_row = copy.deepcopy(row)
    col_story = row["story_col"]
    row_story = row["story_row"]
    
    reward_matrix = [
        [eval(row["1_1_payoff"]), eval(row["1_2_payoff"])],
        [eval(row["2_1_payoff"]), eval(row["2_2_payoff"])],
    ]
    
    reversed_matrix = copy.deepcopy(reward_matrix) # it's ok just to deepcopy since we have simmatric games.
    # for i in range(2):
    #     for j in range(2):
    #         reversed_matrix[i][j] = (reward_matrix[i][j][1], reward_matrix[i][j][0])
    actions_row = eval(row["actions_row"])
    actions_col = eval(row["actions_column"])
    
    
    global count_bads
    story_row = extract_and_replace_middle(reward_matrix, actions_row, actions_col, row_story)
    if story_row == "none":
        count_bads += 1
        return new_row
    story_col = extract_and_replace_middle(reversed_matrix, actions_col, actions_row, col_story)
    if story_col == "none":
        count_bads += 1
        return new_row
    new_row["story_row"] = story_row
    new_row["story_col"] = story_col
    return new_row


@click.command()
@click.argument("input_csv", type=click.Path(exists=True), default="data/gt-harmbench-with-targets.csv")
@click.argument("output_csv", type=click.Path(), default="data/gt-harmbench-gamify.csv")
def main(input_csv, output_csv):

    print(f"Reading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    gamified_rows = []
    for _, row in df.iterrows():
        gamified_row = gamify(row)
        gamified_rows.append(gamified_row)
    
    gamified_df = pd.DataFrame(gamified_rows)
    gamified_df.to_csv(output_csv, index=False)
    print(f"Gamified data saved to {output_csv}")
    
    
if __name__ == "__main__":
    main()