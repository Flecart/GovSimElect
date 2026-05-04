import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_single_evals(df: pd.DataFrame, metric: str, group_col: str, title: str, output_path: str):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df.sort_values(by=metric, ascending=False), x=metric, y=group_col, palette="viridis")
    plt.title(title)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_multi_evals(df: pd.DataFrame, metric: str, index_col: str, columns_col: str, title: str, output_path: str):
    pivot_df = df.pivot(index=index_col, columns=columns_col, values=metric)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot performance metrics.")
    parser.add_argument('--input', type=str, default="data/results_performance.csv", help="Input CSV path")
    parser.add_argument('--output_dir', type=str, default="data/plots", help="Directory to save plots")
    parser.add_argument('--metric', type=str, default="nash_social_welfare", help="Metric to plot")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    df = pd.read_csv(args.input)
    
    # Preprocess
    # Identify score columns or use passed metric
    metric = args.metric
    if metric not in df.columns:
        print(f"Metric {metric} not found in columns. Using first numerical column.")
        ignore_cols = ['id', 'model', 'domain', 'game_type', 'domain_csv', 'game_type_csv']
        num_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]
        metric = num_cols[0]

    print(f"Generating plots for metric: {metric}")

    # Plot Across Models
    print("Plotting Performance across Models...")
    model_df = df.groupby('model', as_index=False)[metric].mean()
    plot_single_evals(
        df=model_df, metric=metric, group_col='model', 
        title=f"Mean {metric} across Models", 
        output_path=os.path.join(args.output_dir, "model_performance.png")
    )

    # Plot Across Game Types
    print("Plotting Performance across Game Types...")
    game_df = df.groupby('game_type', as_index=False)[metric].mean()
    plot_single_evals(
        df=game_df, metric=metric, group_col='game_type',
        title=f"Mean {metric} across Game Types",
        output_path=os.path.join(args.output_dir, "game_type_performance.png")
    )

    # Plot Across Domains
    print("Plotting Performance across Domains...")
    domain_df = df.groupby('domain', as_index=False)[metric].mean()
    plot_single_evals(
        df=domain_df, metric=metric, group_col='domain',
        title=f"Mean {metric} across Domains",
        output_path=os.path.join(args.output_dir, "domain_performance.png")
    )

    # Combined: Model x Game Type
    print("Plotting Combined: Model x Game Type...")
    m_g_df = df.groupby(['model', 'game_type'], as_index=False)[metric].mean()
    plot_multi_evals(
        df=m_g_df, metric=metric, index_col='model', columns_col='game_type',
        title=f"{metric} by Model and Game Type",
        output_path=os.path.join(args.output_dir, "model_x_game_type.png")
    )

    # Combined: Model x Domain
    print("Plotting Combined: Model x Domain...")
    m_d_df = df.groupby(['model', 'domain'], as_index=False)[metric].mean()
    plot_multi_evals(
        df=m_d_df, metric=metric, index_col='model', columns_col='domain',
        title=f"{metric} by Model and Domain",
        output_path=os.path.join(args.output_dir, "model_x_domain.png")
    )

    # Combined: Game Type x Domain
    print("Plotting Combined: Game Type x Domain...")
    g_d_df = df.groupby(['game_type', 'domain'], as_index=False)[metric].mean()
    plot_multi_evals(
        df=g_d_df, metric=metric, index_col='game_type', columns_col='domain',
        title=f"{metric} by Game Type and Domain",
        output_path=os.path.join(args.output_dir, "game_type_x_domain.png")
    )
    
    print(f"All plots saved to {args.output_dir}/")

if __name__ == '__main__':
    main()
