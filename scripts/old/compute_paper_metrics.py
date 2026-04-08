import os
import json
import argparse
import numpy as np
from glob import glob

def compute_gini(x):
    """Calculate the Gini coefficient of a numpy array."""
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0.0
    if np.amin(x) < 0:
        x -= np.amin(x)
    x = np.sort(x)
    n = len(x)
    if n == 0 or np.sum(x) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))

def process_run(results_file):
    with open(results_file, 'r') as f:
        lines = f.readlines()
        
    harvest_data = None
    env_config = None
    
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
            
        if record.get("type") == "initialization":
            # Extract config
            # config is a string representation of a dict, need to use eval or string matching if it's strictly formatted.
            # Using a simplified approach: we know default capacity=100, regen=2.0
            pass
        elif record.get("type") == "harvest":
            harvest_data = record.get("data")
            
    if harvest_data is None:
        return None

    # Simulation Configuration
    CARRYING_CAPACITY = 100.0
    REGEN_FACTOR = 2.0
    MAX_ROUNDS = 12
    
    current_pool = CARRYING_CAPACITY
    survival_time = MAX_ROUNDS
    
    total_harvest_by_agent = {}
    total_catch_all_months = 0.0
    sustainable_catch_all_months = 0.0
    
    # Process each month's harvest
    # harvest_data keys are string representing months "0", "1", ...
    sorted_months = sorted([int(m) for m in harvest_data.keys()])
    
    for month in sorted_months:
        month_str = str(month)
        agent_catches = harvest_data[month_str]
        
        month_total_catch = sum(agent_catches.values())
        
        for agent, catch in agent_catches.items():
            total_harvest_by_agent[agent] = total_harvest_by_agent.get(agent, 0.0) + catch
            
        # The sustainable catch is the amount that can be regenerated to full capacity.
        # If pool is C, sustainable = C - C/regen = 100 - 50 = 50.
        sustainable_catch_month = max(0.0, current_pool - (CARRYING_CAPACITY / REGEN_FACTOR))
        sustainable_catch_all_months += sustainable_catch_month
        
        total_catch_all_months += month_total_catch
        
        current_pool -= month_total_catch
        
        if current_pool <= 0:
            current_pool = 0
            survival_time = min(survival_time, month + 1)
            break
            
        current_pool = min(CARRYING_CAPACITY, current_pool * REGEN_FACTOR)
        
    survival_rate = 1.0 if survival_time == MAX_ROUNDS else 0.0
    gain = sum(total_harvest_by_agent.values())
    
    # Efficiency: max theoretical gain is sustainable_catch * MAX_ROUNDS = 50 * 12 = 600
    MAX_THEORETICAL_GAIN = 600.0
    efficiency = gain / MAX_THEORETICAL_GAIN
    
    # Equality
    agent_totals = list(total_harvest_by_agent.values())
    equality = 1.0 - compute_gini(agent_totals)
    
    # Over-usage
    over_usage = total_catch_all_months / sustainable_catch_all_months if sustainable_catch_all_months > 0 else 0.0

    return {
        "survival_rate": survival_rate,
        "survival_time": survival_time,
        "gain": gain,
        "efficiency": efficiency,
        "equality": equality,
        "over_usage": over_usage
    }

def main():
    parser = argparse.ArgumentParser(description="Compute GovSim metrics from simulation outputs.")
    parser.add_argument("results_dir", type=str, help="Directory containing run folders (e.g. simulation/results/fishing_v7.0/code_law)")
    args = parser.parse_args()
    
    results_files = glob(os.path.join(args.results_dir, "**", "consolidated_results.json"), recursive=True)
    
    if not results_files:
        print(f"No consolidated_results.json found in {args.results_dir}")
        return
        
    print(f"Found {len(results_files)} run(s) to process.")
    
    metrics_list = {
        "survival_rate": [],
        "survival_time": [],
        "gain": [],
        "efficiency": [],
        "equality": [],
        "over_usage": []
    }
    
    for rf in results_files:
        res = process_run(rf)
        if res:
            for k in metrics_list:
                metrics_list[k].append(res[k])
                
    if not metrics_list["survival_time"]:
        print("No valid harvest data could be parsed.")
        return
        
    print("\n--- Aggregated Metrics ---")
    for k, v in metrics_list.items():
        mean_val = np.mean(v)
        std_val = np.std(v)
        print(f"{k.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")

if __name__ == "__main__":
    main()
