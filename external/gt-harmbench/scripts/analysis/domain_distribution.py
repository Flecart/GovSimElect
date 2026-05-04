"""Compute domain distribution in GT-HarmBench by joining with MIT AI Risk Database."""

import click
import pandas as pd


@click.command()
@click.option(
    "--gt-csv",
    default="data/gt-harmbench-with-targets.csv",
    help="Path to GT-HarmBench CSV file.",
)
@click.option(
    "--mit-csv",
    default="data/generation/mit.csv",
    help="Path to MIT AI Risk Database CSV file.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output CSV file path. If not specified, only prints to stdout.",
)
def main(gt_csv: str, mit_csv: str, output: str | None):
    """Compute domain distribution in GT-HarmBench by joining with MIT AI Risk Database.

    Joins GT-HarmBench dataset with MIT CSV on Ev_ID to get domain information,
    then computes the distribution of domains.
    """
    # Load both CSVs
    mit = pd.read_csv(mit_csv)
    gt = pd.read_csv(gt_csv)
    
    # drop matching pennies game
    gt = gt[gt["formal_game"] != "Matching pennies"]

    # Select only the relevant columns from MIT
    mit_domains = mit[["Ev_ID", "Domain"]].drop_duplicates()

    # Merge on Ev_ID
    merged = gt.merge(mit_domains, on="Ev_ID", how="left")

    # Compute distribution
    domain_counts = merged["Domain"].value_counts(dropna=True)
    domain_pcts = merged["Domain"].value_counts(dropna=True, normalize=True) * 100
    


    # Build results dataframe
    results = pd.DataFrame({
        "domain": [str(d) if pd.notna(d) else "NaN (missing)" for d in domain_counts.index],
        "count": domain_counts.values,
        "percentage": [round(p, 1) for p in domain_pcts.values],
    })
    # drop NaN
    results = results[results["domain"] != "NaN (missing)"]
    
    # Print results
    print("=== Domain Distribution in GT-HarmBench ===")
    print(f"Total samples: {len(merged)}")
    print()
    print("Count and Percentage:")
    print("-" * 70)
    for _, row in results.iterrows():
        print(f"{row['domain']:50} {row['count']:5} ({row['percentage']:5.1f}%)")

    print()
    print("=== Summary ===")
    matched = merged["Domain"].notna().sum()
    missing = merged["Domain"].isna().sum()
    print(f"Matched with domain: {matched}")
    print(f"Missing domain: {missing}")

    # Save to CSV if output specified
    if output:
        results.to_csv(output, index=False)
        print(f"\nResults saved to: {output}")
        
def save_piechart():
    import matplotlib.pyplot as plt

# Data
    categories = [
        "1. Discrimination & Toxicity",
        "2. Privacy & Security",
        "3. Misinformation"
        "4. Malicious Actors & Misuse",
        "5. Human-Computer Interaction",
        "6. Socioeconomic and Environmental",
        "7. AI System Safety, Failures, & Limitations",
    ]
    counts = [384, 283, 272, 257, 148, 145, 87]

    # RGB Color (normalized to 0-1)
    bg_color = (255/255, 249/255, 196/255)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(bg_color) # Set figure background
    ax.set_facecolor(bg_color) # Set axes background (though pie chart doesn't really use axes bg)

    colors = plt.get_cmap('Pastel1').colors

    # Create pie chart
    patches, texts, autotexts = ax.pie(
        counts, 
        labels=None, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        textprops={'fontsize': 12}
    )

    # Style the percentages
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')

    # Create a custom legend
    legend = ax.legend(
        patches, 
        categories, 
        title="Categories", 
        loc="center left", 
        bbox_to_anchor=(1, 0, 0.5, 1),
        frameon=False 
    )

    # Equal aspect ratio
    ax.axis('equal')

    plt.tight_layout()

# Save with the specific background color
# Note: transparent=False is default, but we ensure facecolor is used.
# plt.savefig('ai_issues_pie_chart_colored_bg.png', facecolor=bg_color, bbox_inches='tight')
    plt.savefig('assets/ai_issues_pie_chart.png')


def game_distribution():
    import matplotlib.pyplot as plt

    # Data extracted from the image
    categories = [
        "Prisoner's Dilemma",
        "Chicken",
        "Stag hunt",
        "Matching pennies",
        "Coordination",
        "Bach or Stravinski",
        "(test) No conflict"
    ]
    counts = [654, 491, 403, 256, 252, 170, 39]

    # Background color: rgb(255, 249, 196)
    bg_color = (255/255, 249/255, 196/255)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Use the same color map as before
    colors = plt.get_cmap('Pastel1').colors

    # Create pie chart
    patches, texts, autotexts = ax.pie(
        counts, 
        labels=None, # Labels will be in the legend
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        textprops={'fontsize': 12}
    )

    # Style the percentages
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')

    # Create the custom legend
    legend = ax.legend(
        patches, 
        categories, 
        title="Formal Games", 
        loc="center left", 
        bbox_to_anchor=(1, 0, 0.5, 1),
        frameon=False 
    )

    # Equal aspect ratio
    ax.axis('equal')

    plt.tight_layout()

    # Save with the specific background color
    plt.savefig('assets/formal_games_pie_chart.png', transparent=True) #, facecolor=bg_color, bbox_inches='tight')

if __name__ == "__main__":
    main()
    # save_piechart()
    # game_distribution()
