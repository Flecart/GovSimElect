"""
This script is used to add the targets to a given csv file in the format produced by the contextualizations.
This is done such that at the end we can have better downstream evaluation.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import click
import pandas as pd
from src.utils import find_nash_equilibria, find_utility_maximizing, find_Rawlsian_actions, find_nash_social_welfare
from src.metrics import utilitarian, fairness, nash_social_welfare
from dataclasses import dataclass

@dataclass
class TargetResults:
    nash_equilibria: list
    utility_maximizing: list
    rawlsian: list
    nash_social_welfare: list

def get_targets(row) -> TargetResults:
    
    indices = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1)
    ]

    payoff_matrix = [
        [eval(row['1_1_payoff']), eval(row['1_2_payoff'])],
        [eval(row['2_1_payoff']), eval(row['2_2_payoff'])]
    ]

    nash_equilibria = find_nash_equilibria(payoff_matrix)
    utility_maximizing = find_utility_maximizing(payoff_matrix)
    rawlsian = find_Rawlsian_actions(payoff_matrix)
    nash_social_welfare = find_nash_social_welfare(payoff_matrix)

    nash_list  = []
    utility_list = []
    rawlsian_list = []
    nash_social_welfare_list = []
    for (i, j) in indices:
        action_row = eval(row["actions_row"])   
        action_column = eval(row["actions_column"])
        if nash_equilibria is not None and (i, j) in nash_equilibria:
            nash_list.append(str((action_row[i], action_column[j])).lower())
        
        if (i, j) in utility_maximizing:
            utility_list.append(str((action_row[i], action_column[j])).lower())

        if (i,j) in rawlsian:
            rawlsian_list.append(str((action_row[i], action_column[j])).lower())
            
        if (i,j) in nash_social_welfare:
            nash_social_welfare_list.append(str((action_row[i], action_column[j])).lower())
            
            
    return TargetResults(
        nash_equilibria=nash_list,  
        utility_maximizing=utility_list,
        rawlsian=rawlsian_list,
        nash_social_welfare=nash_social_welfare_list
    )

@click.command()
@click.argument("input_csv", type=click.Path(exists=True), default="data/contextualization-sampled.csv")
@click.argument("output_csv", type=click.Path(), default="data/contextualization-with-targets.csv")
def main(input_csv, output_csv):

    df = pd.read_csv(input_csv)
    
    nash_equilibria_list = []
    utility_maximizing_list = []
    rawlsian_list_of_lists = []
    nash_social_welfare_list_of_lists = []
    max_utilitarian_list = []
    max_rawlsian_list = []
    max_social_welfare_list = []

    for index, row in df.iterrows():
        payoff_matrix = [
            [eval(row['1_1_payoff']), eval(row['1_2_payoff'])],
            [eval(row['2_1_payoff']), eval(row['2_2_payoff'])]
        ]
        targets = get_targets(row)
        nash_list = targets.nash_equilibria
        utility_list = targets.utility_maximizing
        rawlsian_list = targets.rawlsian
        nash_social_welfare_list = targets.nash_social_welfare
        nash_equilibria_list.append("|".join(nash_list) if nash_list else "None")
        utility_maximizing_list.append("|".join(utility_list) if utility_list else "None")
        rawlsian_list_of_lists.append("|".join(rawlsian_list) if rawlsian_list else "None")
        nash_social_welfare_list_of_lists.append("|".join(nash_social_welfare_list) if nash_social_welfare_list else "None")        
        max_utilitarian_list.append(utilitarian(payoff_matrix))
        max_rawlsian_list.append(fairness(payoff_matrix))     
        max_social_welfare_list.append(nash_social_welfare(payoff_matrix))                          

    df['target_nash_equilibria'] = nash_equilibria_list
    df['target_utility_maximizing'] = utility_maximizing_list
    df['target_rawlsian'] = rawlsian_list_of_lists
    df['target_nash_social_welfare'] = nash_social_welfare_list_of_lists
    df['nash_social_welfare'] = max_social_welfare_list
    df['max_utilitarian'] = max_utilitarian_list
    df['max_rawlsian'] = max_rawlsian_list

    df.to_csv(output_csv, index=False)
    print(f"Output saved to {output_csv}")
    
if __name__ == "__main__":
    main()
    