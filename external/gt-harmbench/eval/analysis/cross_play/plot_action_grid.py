import glob
import os
import sys
import json
from inspect_ai.log import read_eval_log
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from eval.analysis.plots.shared import render_action_probability_grid

def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def parse_action_logs(log_dir):
    files = glob.glob(os.path.join(log_dir, "*.eval"))
    if not files:
        print(f"No eval logs found in {log_dir}")
        return {}
    
    # Nested dict: model_game_actions[model_combo][game] = list of (row_idx, col_idx)
    model_game_actions = {}
    
    for f in files:
        try:
            logs = read_eval_log(f)
            task_name = logs.eval.task
            
            if task_name.startswith("cross_play_"):
                model_combo = task_name.replace("cross_play_", "").replace("_vs_", " vs\n")
            else:
                model_combo = task_name
                
            if model_combo not in model_game_actions:
                model_game_actions[model_combo] = {}
                
            for sample in logs.samples:
                formal_game = sample.metadata.get("formal_game", "Unknown")
                if formal_game not in model_game_actions[model_combo]:
                    model_game_actions[model_combo][formal_game] = []
                
                if getattr(sample, 'scores', None) and 'all_strategies_scorer' in sample.scores:
                    answer_str = sample.scores['all_strategies_scorer'].answer
                    if not answer_str: continue
                    
                    row_actions = sample.metadata.get("actions_row", [])
                    col_actions = sample.metadata.get("actions_column", [])
                    
                    if len(row_actions) < 2 or len(col_actions) < 2:
                        continue
                        
                    # Parse answer lines
                    for line in answer_str.strip().split("\n"):
                        if "Row choice:" in line and "Column choice:" in line:
                            parts = line.split(", Column choice:")
                            row_part = parts[0].replace("Row choice:", "").strip().lower()
                            col_part = parts[1].strip().lower()
                            
                            # Find index using edit distance just like custom_all_scorer
                            try:
                                r_idx = min(range(2), key=lambda idx: edit_distance(row_actions[idx].lower(), row_part))
                                c_idx = min(range(2), key=lambda idx: edit_distance(col_actions[idx].lower(), col_part))
                                
                                model_game_actions[model_combo][formal_game].append((r_idx, c_idx))
                            except Exception as e:
                                pass
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    return model_game_actions


def generate_action_grid(log_dir, output_path):
    model_game_actions = parse_action_logs(log_dir)
    if not model_game_actions:
        return
        
    models = sorted(list(model_game_actions.keys()))
    
    # Collect all unique games across models
    all_games = set()
    for m in models:
        for g in model_game_actions[m].keys():
            all_games.add(g)
            
    games = sorted(list(all_games))
    
    if not models or not games:
        print("No valid data to plot grid.")
        return
        
    model_game_matrices = {}
    model_game_counts = {}
    
    for i, model in enumerate(models):
        for j, game in enumerate(games):
            if game in model_game_actions[model]:
                action_pairs = model_game_actions[model][game]
                matrix = np.zeros((2, 2))
                total = len(action_pairs)
                
                for r_idx, c_idx in action_pairs:
                    matrix[r_idx, c_idx] += 1
                    
                if total > 0:
                    matrix = matrix / total
                    
                key = (model, game)
                model_game_matrices[key] = matrix
                model_game_counts[key] = total
                
    suptitle = "Action Probability Distribution: Cross-Play Models × Formal Games\n(Row 0=UP, Row 1=DOWN, Col 0=LEFT, Col 1=RIGHT)"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    render_action_probability_grid(model_game_matrices, model_game_counts, models, games, suptitle, str(output_path))
    print(f"Action probability grid saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs/cross_play")
    parser.add_argument("--output", default="results/cross_play_action_grid.png")
    args = parser.parse_args()
    
    generate_action_grid(args.log_dir, args.output)
