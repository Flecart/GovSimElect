import json
import os
import argparse
from pathlib import Path
from eval.analysis.core import list_logs, load_log_samples_from_path, is_due_diligence_log, build_dataframe, compute_metrics
from eval.analysis.plots.single_log import plot_accuracy_by_game

def sanitize_name(name):
    if not name:
        return "Unknown"
    # Replace problematic characters
    return "".join([c if c.isalnum() else "_" for c in str(name)]).strip("_")

def main():
    parser = argparse.ArgumentParser(description="Plot accuracy by game for each security group.")
    parser.add_argument('--output_dir', type=str, default="data/plots/security_groups", help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get all logs
    logs = list_logs(max_logs=100)
    
    # Filter for standard big logs
    std_logs = []
    for l in logs:
        if l["type"] == "eval" and l["num_samples"] > 100:
            if not is_due_diligence_log(l["path"], "eval"):
                std_logs.append(l)

    print(f"Found {len(std_logs)} standard model logs.")

    processed_models = set()

    for log_entry in std_logs:
        path = log_entry["path"]
        model_name = log_entry["model_name"]
        
        # In case we have multiple runs for the same model, we can either aggregate or just use the first one
        if model_name in processed_models:
            print(f"Skipping {model_name} as it is already processed.")
            continue
            
        print(f"\nProcessing log for model: {model_name} from {path}")
        samples, _ = load_log_samples_from_path(path, "eval")
        
        # Group samples by Risk category
        group_samples = {}
        for sample in samples:
            target_str = sample.get("target")
            risk_category = "Unknown"
            if target_str:
                try:
                    target = json.loads(target_str) if isinstance(target_str, str) else target_str
                    risk_category = target.get("Risk category", "Unknown")
                    if not risk_category or str(risk_category).strip() == "" or str(risk_category) == "nan":
                        risk_category = "Unknown"
                except Exception as e:
                    pass
                    
            if risk_category not in group_samples:
                group_samples[risk_category] = []
            group_samples[risk_category].append(sample)
            
        print(f"Found {len(group_samples)} security groups (Risk categories) for {model_name}.")

        for group, g_samples in group_samples.items():
            if len(g_samples) < 5:
                # Skip groups with too few samples to make a meaningful plot
                continue
                
            print(f"  - Group '{group}': {len(g_samples)} samples")
            df = build_dataframe(g_samples)
            if df.empty:
                continue
                
            metrics_df, accuracy_by_game, welfare_by_game, _, overall_accuracy, refusal_ratio = compute_metrics(df)
            
            if accuracy_by_game.empty:
                print(f"    No valid accuracy data for group {group}")
                continue
                
            sanitized_group = sanitize_name(group)
            
            try:
                # Need to run plot_accuracy_by_game
                plot_accuracy_by_game(
                    accuracy_by_game,
                    model_name=model_name,
                    output_dir=args.output_dir,
                    prefix=f"group_{sanitized_group}"
                )
            except Exception as e:
                print(f"    Failed to plot for group {group}: {e}")
                
        processed_models.add(model_name)

    print(f"\nAll plots saved to {args.output_dir}/")

if __name__ == '__main__':
    main()
