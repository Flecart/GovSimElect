"""Paper figure: average total gain R and average survival months m,
grouped bars per model, violet gradient bars for prosocial levels p0..p5,
one panel per contract condition, two rows (R and m)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

agg = pd.read_csv('aggregated.csv')

# Collapse duplicate (model, prosocial, condition) rows (some "nl" cells appear
# twice from concatenated parser runs) using the same unweighted-mean convention
# as scripts/logs/parser.py.
numeric_cols = agg.select_dtypes(include='number').columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'prosocial']
agg = (
    agg.groupby(['model', 'prosocial', 'condition'], as_index=False)[numeric_cols]
       .mean()
)

# -- Styling --
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.color': '#dddddd',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.6,
    'axes.axisbelow': True,
})

# Violet gradient for prosocial levels p0..p5 (0 = lightest, 5 = darkest)
violet_colors = [
    "#c9b8de",  # p0 - light violet
    "#b19bd1",  # p1
    "#9a7fc4",  # p2
    "#8364b7",  # p3
    "#6c4aaa",  # p4
    "#553093",  # p5 - deep purple
]

p_levels = [0, 1, 2, 3, 4, 5]

model_display = {
    'gemma-4-31b':    'Gemma-4-31B',
    'gemma-4-31b-it': 'Gemma-4-31B-IT',
    'gpt-4o':         'GPT-4o',
    'gpt-5.4':        'GPT-5.4',
    'gpt-5.4-mini':   'GPT-5.4-Mini',
    'grok-4.1-fast':  'Grok-4.1-Fast',
}
model_order = ['gemma-4-31b', 'gemma-4-31b-it', 'gpt-4o', 'gpt-5.4', 'gpt-5.4-mini', 'grok-4.1-fast']

cond_order = ['no-contract', 'nl', 'code-law']
cond_title = {
    'no-contract': 'Baseline',
    'nl':          'Constitution',
    'code-law':    'Code Law',
}

# --- Figure layout: 2 rows (gain, survival) x 3 cols (conditions) ---
fig, axes = plt.subplots(2, 3, figsize=(14, 7.0), sharex='col', sharey='row')

n_models = len(model_order)
n_p = len(p_levels)
bar_width = 0.13
group_centers = np.arange(n_models)

metrics = [
    ('R',  'Total gain $R$',         [0, 625]),
    ('m',  'Survival months $m$',    [0, 12.9]),
]

for row_idx, (metric, ylabel, ylim) in enumerate(metrics):
    hw_col = f'{metric}_hi'
    lw_col = f'{metric}_lo'
    for col_idx, cond in enumerate(cond_order):
        ax = axes[row_idx, col_idx]
        sub = agg[agg['condition'] == cond]

        for pi, p in enumerate(p_levels):
            vals, high_endpoints, low_endpoints = [], [], []
            for m in model_order:
                row = sub[(sub['model'] == m) & (sub['prosocial'] == p)]
                if row.empty:
                    vals.append(np.nan)
                    high_endpoints.append(np.nan)
                    low_endpoints.append(np.nan)
                else:
                    vals.append(float(row[metric].iloc[0]))
                    high_endpoints.append(float(row[hw_col].iloc[0]))
                    low_endpoints.append(float(row[lw_col].iloc[0]))

            vals = np.array(vals)
            high_endpoints = np.array(high_endpoints)
            low_endpoints = np.array(low_endpoints)
            lower = np.clip(vals - low_endpoints, 0, None)
            upper = np.clip(high_endpoints - vals, 0, None)
            yerr = np.vstack([lower, upper])
            offset = (pi - (n_p - 1) / 2) * bar_width
            # Replace NaN with 0 for plotting (NaN bars render as missing).
            plot_vals = np.where(np.isnan(vals), 0.0, vals)
            plot_yerr = np.where(np.isnan(yerr), 0.0, yerr)
            ax.bar(
                group_centers + offset, plot_vals,
                width=bar_width,
                color=violet_colors[pi],
                edgecolor='#2a1a4a',
                linewidth=0.5,
                yerr=plot_yerr,
                error_kw=dict(ecolor='#444444', elinewidth=0.6, capsize=1.2, alpha=0.7),
                label=f'$p_{p}$' if (row_idx == 0 and col_idx == 0) else None,
            )

        if row_idx == 0:
            ax.set_title(cond_title[cond], pad=8, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(ylabel)
        ax.set_xticks(group_centers)
        ax.set_xticklabels([model_display[m] for m in model_order], rotation=20, ha='right')
        ax.set_ylim(ylim)
        ax.tick_params(axis='x', length=0)

        if metric == 'm':
            ax.axhline(12, color='#888', linestyle=':', linewidth=0.7, zorder=0)
        if metric == 'R':
            ax.axhline(600, color='#888', linestyle=':', linewidth=0.7, zorder=0)

# --- Shared legend (prosocial levels) ---
legend_handles = [
    Patch(facecolor=violet_colors[i], edgecolor='#2a1a4a', linewidth=0.5,
          label=f'$p_{p_levels[i]}$')
    for i in range(n_p)
]
fig.legend(
    handles=legend_handles,
    title='Number of prosocial agents',
    loc='lower center',
    ncol=n_p,
    bbox_to_anchor=(0.5, -0.015),
    frameon=False,
    columnspacing=1.8,
    handlelength=1.2,
)

fig.tight_layout(rect=[0, 0.05, 1, 1])

import os
os.makedirs('outputs', exist_ok=True)
out_pdf = 'outputs/govsim_gain_survival.pdf'
out_png = 'outputs/govsim_gain_survival.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
