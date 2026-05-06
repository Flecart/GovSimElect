import pandas as pd
import json
import glob
import os
from inspect_ai.log import read_eval_log
import argparse


def load_csv_data(csv_path: str):
    return pd.read_csv(csv_path)

def parse_logs(log_dir: str):
    records = []
    log_files = glob.glob(os.path.join(log_dir, "*.eval"))
    
    for log_file in log_files:
        try:
            log = read_eval_log(log_file)
            if not log.samples:
                continue
                
            model = log.eval.model
            
            for sample in log.samples:
                if not sample.target:
                    continue
                try:
                    target_data = json.loads(sample.target)
                except json.JSONDecodeError:
                    continue
                
                # Extract identifiers and categories from target
                sample_id = target_data.get('id', sample.id)
                domain = target_data.get('Risk category', 'Unknown')
                game_type = target_data.get('formal_game', 'Unknown')
                
                # Extract performance scores
                scores = {}
                if sample.scores and 'all_strategies_scorer' in sample.scores:
                    scorer_val = sample.scores['all_strategies_scorer'].value
                    if isinstance(scorer_val, dict):
                        scores = scorer_val
                
                record = {
                    'id': sample_id,
                    'model': model,
                    'domain': domain,
                    'game_type': game_type,
                    **scores
                }
                records.append(record)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="Analyze inspect_ai eval logs Performance.")
    parser.add_argument('--output', type=str, default=None, help="Path to save output CSV instead of printing.")
    args = parser.parse_args()

    # Setup paths
    base_dir = "/home/anonymous/Desktop/work/gt-harmbench"
    csv_path = os.path.join(base_dir, "data", "gt-harmbench.csv")
    log_dir = os.path.join(base_dir, "logs")
    
    print("Loading gt-harmbench.csv...")
    df_csv = load_csv_data(csv_path)
    
    # We rename columns to be clear
    df_csv_sub = df_csv[['id', 'Risk category', 'formal_game']].copy()
    df_csv_sub.rename(columns={'Risk category': 'domain_csv', 'formal_game': 'game_type_csv'}, inplace=True)
    
    print("Loading data/generation/mit.csv...")
    mit_path = os.path.join(base_dir, "data", "generation", "mit.csv")
    df_mit = load_csv_data(mit_path)
    df_mit_sub = df_mit[['Ev_ID', 'Domain']].copy()
    df_mit_sub = df_mit_sub.dropna(subset=['Ev_ID']).rename(columns={'Ev_ID': 'id', 'Domain': 'mit_domain'})
    df_mit_sub = df_mit_sub.drop_duplicates(subset=['id'])
    
    print("Parsing inspect_ai logs...")
    df_logs = parse_logs(log_dir)
    
    if df_logs.empty:
        print("No log data found or parsed.")
        return
        
    print(f"Loaded {len(df_logs)} records from logs.")
    
    # Merge on id
    df_logs['id'] = df_logs['id'].astype(str)
    df_csv_sub['id'] = df_csv_sub['id'].astype(str)
    
    df_merged = pd.merge(df_logs, df_csv_sub, on='id', how='left')
    df_merged = pd.merge(df_merged, df_mit_sub, on='id', how='left')
    
    # Some targets might have missed domain or formal_game, so we fill from CSV if needed
    df_merged['domain'] = df_merged['mit_domain'].fillna(df_merged['domain_csv'])
    df_merged['game_type'] = df_merged['game_type'].fillna(df_merged['game_type_csv'])
    
    # Select numerical score columns only
    ignore_cols = ['id', 'model', 'domain', 'mit_domain', 'game_type', 'domain_csv', 'game_type_csv']
    score_cols = [c for c in df_merged.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df_merged[c])]
    
    if not score_cols:
        print("No numerical score columns found.")
        return
        
    # Generate reports
    metrics = {}
    
    print("\n--- PERFORMANCE ACROSS MODELS ---")
    model_df = df_merged.groupby('model')[score_cols].mean()
    print(model_df.to_string())
    
    print("\n--- PERFORMANCE ACROSS GAME TYPES ---")
    game_df = df_merged.groupby('game_type')[score_cols].mean()
    print(game_df.to_string())
    
    print("\n--- PERFORMANCE ACROSS DOMAINS ---")
    domain_df = df_merged.groupby('domain')[score_cols].mean()
    print(domain_df.to_string())
    
    print("\n--- PERFORMANCE ACROSS GAME TYPES AND DOMAINS ---")
    g_d_df = df_merged.groupby(['game_type', 'domain'])[score_cols].mean()
    print(g_d_df.to_string())
    
    print("\n--- COMBINED PERFORMANCE: MODEL x GAME TYPE ---")
    m_g_df = df_merged.groupby(['model', 'game_type'])[score_cols].mean()
    print(m_g_df.to_string())

    print("\n--- COMBINED PERFORMANCE: MODEL x DOMAIN ---")
    m_d_df = df_merged.groupby(['model', 'domain'])[score_cols].mean()
    print(m_d_df.to_string())

    if args.output:
        df_merged.to_csv(args.output, index=False)
        print(f"\nSaved raw merged dataset to {args.output}")

if __name__ == '__main__':
    main()
