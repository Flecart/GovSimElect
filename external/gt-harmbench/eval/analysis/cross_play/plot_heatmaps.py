import glob
import os
import sys
from inspect_ai.log import read_eval_log
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def generate_heatmap(log_dir, output_path, metric="nash"):
    files = glob.glob(os.path.join(log_dir, "*.eval"))
    if not files:
        print(f"No eval logs found in {log_dir}")
        return
    
    data = []
    
    for f in files:
        try:
            logs = read_eval_log(f)
            task_name = logs.eval.task
            
            # The task name format is 'cross_play_Model1_vs_Model2'
            if task_name.startswith("cross_play_"):
                model_combo = task_name.replace("cross_play_", "")
            else:
                model_combo = task_name
                
            for sample in logs.samples:
                formal_game = sample.metadata.get("formal_game", "Unknown")
                
                # Check if there is a 'all_strategies_scorer' score
                if getattr(sample, 'scores', None) and 'all_strategies_scorer' in sample.scores:
                    score_dict = sample.scores['all_strategies_scorer'].value
                    if isinstance(score_dict, dict) and metric in score_dict:
                        val = score_dict[metric]
                        data.append({
                            "Model Combination (Row vs Col)": model_combo.replace("_vs_", " vs "),
                            "Game": formal_game,
                            "Value": val
                        })
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not data:
        print("No valid data found to plot.")
        return
        
    df = pd.DataFrame(data)
    
    # Calculate means
    pivot_df = df.pivot_table(index="Model Combination (Row vs Col)", columns="Game", values="Value", aggfunc='mean')
    
    # Fill NaN with 0 or drop them based on preference, here we'll keep them as NaN but mask in seaborn
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f", vmin=0, vmax=1)
    plt.title(f"Cross-Play Results: Mean {metric.capitalize()} Score")
    plt.ylabel("Model Combination (Row vs Col)")
    plt.xlabel("Game")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Heatmap saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs/cross_play")
    parser.add_argument("--output", default="results/cross_play_heatmap_nash.png")
    parser.add_argument("--metric", default="nash", help="Metric to plot (e.g., nash, avg_utilitarian_score)")
    args = parser.parse_args()
    
    generate_heatmap(args.log_dir, args.output, args.metric)
    
    # Also generate utilitarian and social welfare
    base, ext = os.path.splitext(args.output)
    generate_heatmap(args.log_dir, f"{base}_utilitarian{ext}", "avg_utilitarian_score")
    generate_heatmap(args.log_dir, f"{base}_rawlsian{ext}", "avg_rawlsian_score")
