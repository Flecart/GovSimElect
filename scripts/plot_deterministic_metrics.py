#!/usr/bin/env python3
"""Paper figure: commons metrics under deterministic regeneration (factor = 2).

Extracted from the notebook cell that plotted deterministic vs stochastic
regeneration side by side. The stochastic regime is dropped here; instead the
equality metric e is added, so the figure shows all five metrics on a 2x3 grid
with the legend occupying the sixth cell:

  +--------+--------+--------+
  |   m    |   R    |   u    |   row 0
  +--------+--------+--------+
  |   o    |   e    | LEGEND |   row 1
  +--------+--------+--------+

x-axis: number of "we"-frame agents, 0..5 (the `prosocial` column).
One line per contract condition, band = mean of per-model CI half-widths.
u, o and e all share the [0, 1] axis; pass --equality-scale auto to zoom e to
its observed band instead.

Usage:
    python scripts/plot_deterministic_metrics.py
    python scripts/plot_deterministic_metrics.py -i aggregated.csv \
        -o paper/images/det_regen_metrics --show
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Styling ---
STYLE = {
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#e2e2e2',
    'grid.linewidth': 0.7,
    'grid.alpha': 0.9,
    'axes.axisbelow': True,
}

TINT_DET = '#fbf6ee'   # warm beige, marks the deterministic regime
HEADER_DET = '#a86326'

COND_STYLE = {
    'no-contract': {'color': '#d48c42', 'label': 'No contract',   'marker': 'o'},
    'nl':          {'color': '#4a7bc1', 'label': 'NL contract',   'marker': 's'},
    'code-law':    {'color': '#4f9d69', 'label': 'Code contract', 'marker': '^'},
}
COND_ORDER = ['no-contract', 'nl', 'code-law']

P_LEVELS = [0, 1, 2, 3, 4, 5]
XLABEL = 'Number of “we”-frame agents'
METRIC_TYPES = ['m', 'q', 'R', 'u', 'e', 'o']

HORIZON = 12       # max survival months
MAX_GAIN = 600     # max total gain over the horizon

# gemma-4-31b-it duplicates gemma-4-31b, so it is dropped everywhere rather
# than double-counted in the cross-model average.
EXCLUDE_MODELS = {'gemma-4-31b-it'}


def value_columns():
    return list(METRIC_TYPES) + [f'{mt}_hw' for mt in METRIC_TYPES]


def load_per_model(path):
    """Read the aggregate CSV as one row per (model, condition, p).

    Some `nl` cells appear twice from concatenated parser runs; those are
    collapsed here (same unweighted-mean convention as scripts/logs/parser.py)
    so duplicated models do not get double weight downstream.
    """
    df = pd.read_csv(path)
    df = df[~df['model'].isin(EXCLUDE_MODELS)]

    for mt in METRIC_TYPES:
        if f'{mt}_hw' not in df.columns:
            df[f'{mt}_hw'] = (df[f'{mt}_hi'] - df[f'{mt}_lo']) / 2.0

    return (
        df.groupby(['model', 'condition', 'prosocial'], as_index=False)[value_columns()]
          .mean()
    )


def load_aggregate(path):
    """Read the aggregate CSV and average it over models, down to (condition, p)."""
    df = load_per_model(path)
    return (
        df.groupby(['condition', 'prosocial'], as_index=False)[value_columns()]
          .mean()
    )


def clip_band(lo, hi, metric):
    """Keep CI bands inside the range each metric can actually take."""
    if metric == 'm':
        return np.clip(lo, 0, HORIZON), np.clip(hi, 0, HORIZON)
    if metric == 'R':
        return np.clip(lo, 0, MAX_GAIN), np.clip(hi, 0, MAX_GAIN)
    if metric in ('u', 'e', 'o'):
        return np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    return lo, hi


def plot_panel(ax, agg, metric, ylabel, ylim):
    hw_col = f'{metric}_hw'
    for cond in COND_ORDER:
        s = agg[agg['condition'] == cond].sort_values('prosocial')
        x = s['prosocial'].values
        y = s[metric].values
        lo, hi = clip_band(y - s[hw_col].values, y + s[hw_col].values, metric)
        st = COND_STYLE[cond]
        ax.fill_between(x, lo, hi, color=st['color'], alpha=0.18,
                        linewidth=0, zorder=2)
        ax.plot(x, lo, color=st['color'], alpha=0.35, linewidth=0.6, zorder=3)
        ax.plot(x, hi, color=st['color'], alpha=0.35, linewidth=0.6, zorder=3)
        ax.plot(x, y,
                color=st['color'], marker=st['marker'], markersize=6.0,
                markerfacecolor='white', markeredgewidth=1.5,
                markeredgecolor=st['color'], linewidth=2.0,
                zorder=5, label=st['label'])
    ax.set_xticks(P_LEVELS)
    ax.set_xticklabels([str(p) for p in P_LEVELS], fontsize=11)
    if ylim is not None:
        ax.set_ylim(ylim)
    if metric == 'm':
        ax.axhline(HORIZON, color='#888', linestyle=':', linewidth=0.8, zorder=0)
    if metric == 'R':
        ax.axhline(MAX_GAIN, color='#888', linestyle=':', linewidth=0.8, zorder=0)
    ax.set_ylabel(ylabel, fontsize=15)


def gain_ylim(agg):
    """Round the total-gain axis up to the next 50, capped at the horizon max."""
    top = (agg['R'] + agg['R_hw']).max()
    return [0, min(np.ceil(top / 50) * 50, MAX_GAIN * 1.05)]


def equality_ylim(agg, mode):
    """Equality lives in a narrow high band, so zoom it unless asked not to."""
    if mode == 'full':
        return [0, 1.05]
    lo = (agg['e'] - agg['e_hw']).min()
    hi = (agg['e'] + agg['e_hw']).max()
    pad = max(0.02, 0.12 * (hi - lo))
    return [max(0.0, lo - pad), min(1.02, hi + pad)]


def build_figure(agg, tint=True, equality_scale='full', header=False):
    fig, axes = plt.subplots(
        2, 3,
        figsize=(13.0, 6.4),
        gridspec_kw={'wspace': 0.28, 'hspace': 0.26},
    )

    if tint:
        for ax in axes.flat:
            ax.set_facecolor(TINT_DET)
        axes[1, 2].set_facecolor('white')

    panels = [
        (axes[0, 0], 'm', r'Survival months  $m$', [0, HORIZON + 0.5]),
        (axes[0, 1], 'R', r'Total gain  $R$',      gain_ylim(agg)),
        (axes[0, 2], 'u', r'Efficiency  $u$',      [0, 1.05]),
        (axes[1, 0], 'o', r'Over-usage  $o$',      [0, 1.05]),
        (axes[1, 1], 'e', r'Equality  $e$',        equality_ylim(agg, equality_scale)),
    ]
    for ax, metric, ylabel, ylim in panels:
        plot_panel(ax, agg, metric, ylabel, ylim)

    for ax in (axes[1, 0], axes[1, 1], axes[0, 2]):
        ax.set_xlabel(XLABEL, fontsize=15)

    # --- Legend in the free bottom-right cell ---
    axes[1, 2].axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = axes[1, 2].legend(
        handles, labels,
        title='Contract condition',
        loc='center', bbox_to_anchor=(0.5, 0.5),
        frameon=False,
        handlelength=2.6,
        labelspacing=1.4,
        title_fontsize=18,
        fontsize=15,
        borderaxespad=0.5,
    )
    leg.get_title().set_fontweight('bold')

    if header:
        fig.text(
            0.5, 0.965,
            r'Deterministic regeneration  (factor $= 2$)',
            ha='center', va='center', fontsize=15, fontweight='bold',
            color=HEADER_DET,
        )
        underline = Line2D([0.30, 0.70], [0.941, 0.941],
                           transform=fig.transFigure,
                           color=HEADER_DET, linewidth=2.0, alpha=0.6,
                           figure=fig)
        fig.lines.append(underline)

    fig.subplots_adjust(top=0.915 if header else 0.97,
                        bottom=0.10, left=0.06, right=0.99)
    return fig


def parse_args():
    p = argparse.ArgumentParser(
        description='Plot commons metrics under deterministic regeneration '
                    '(factor = 2), including the equality metric.',
    )
    p.add_argument('-i', '--input', default='aggregated.csv',
                   help='per-model aggregate CSV (default: aggregated.csv)')
    p.add_argument('-o', '--out', default='paper/images/det_regen_metrics',
                   help='output path without extension; one file per --format '
                        '(default: paper/images/det_regen_metrics)')
    p.add_argument('-f', '--format', nargs='+', default=['pdf', 'png'],
                   help='output formats (default: pdf png)')
    p.add_argument('--dpi', type=int, default=200,
                   help='raster DPI (default: 200)')
    p.add_argument('--equality-scale', choices=['full', 'auto'], default='full',
                   help="y-scale for the equality panel: 'full' uses [0, 1] like "
                        "u and o, 'auto' zooms to the observed band "
                        "(default: full)")
    p.add_argument('--no-tint', action='store_true',
                   help='drop the warm background tint on the panels')
    p.add_argument('--header', action='store_true',
                   help='add the "Deterministic regeneration (factor = 2)" '
                        'header above the grid (off by default)')
    p.add_argument('--show', action='store_true',
                   help='open an interactive window instead of only saving')
    return p.parse_args()


def main():
    args = parse_args()
    if not args.show:
        plt.switch_backend('Agg')
    plt.rcParams.update(STYLE)

    agg = load_aggregate(args.input)
    fig = build_figure(agg,
                       tint=not args.no_tint,
                       equality_scale=args.equality_scale,
                       header=args.header)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for ext in args.format:
        path = f'{args.out}.{ext}'
        fig.savefig(path, bbox_inches='tight', dpi=args.dpi)
        print(f'wrote {path}')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
