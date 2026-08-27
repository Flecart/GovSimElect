#!/usr/bin/env python3
"""Paper model-breakdown figure using grouped coloured bars.

The figure shows average total gain R and survival months m, with one column
per contract condition and one colour per model. Within each model, bar opacity
encodes the number of "we"-frame agents.

Usage:
    uv run python scripts/plot_model_bars.py
    uv run python scripts/plot_model_bars.py -o /tmp/bars -f png --show
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_deterministic_metrics import (  # noqa: E402
    HORIZON,
    MAX_GAIN,
    P_LEVELS,
    load_per_model,
)

# Fixed model order and print-safe colour slots. Keep this roster explicit so
# an unexpected model cannot silently change the visual encoding.
MODEL_STYLE = {
    'gemma-4-31b': {'color': '#3f6fa8', 'label': 'Gemma-4-31B'},
    'gpt-4o': {'color': '#c87d33', 'label': 'GPT-4o'},
    'gpt-5.4': {'color': '#2f95a4', 'label': 'GPT-5.4'},
    'gpt-5.4-mini': {'color': '#8a5296', 'label': 'GPT-5.4-Mini'},
    'deepseek-v4-flash': {
        'color': '#5b9a48',
        'label': 'DeepSeek-V4-Flash',
    },
}
MODEL_ORDER = list(MODEL_STYLE)

ALPHA_MIN, ALPHA_MAX = 0.42, 1.0


def alpha_for(prosocial_count: int) -> float:
    """Map the number of "we"-frame agents to a bounded bar opacity."""
    span = max(P_LEVELS) - min(P_LEVELS)
    return ALPHA_MIN + (
        (ALPHA_MAX - ALPHA_MIN)
        * (prosocial_count - min(P_LEVELS))
        / span
    )

# --- Styling (as in the original figure: y-grid only, compact fonts) ---
STYLE = {
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
}

COND_ORDER = ['no-contract', 'nl', 'code-law']
COND_TITLE = {
    'no-contract': 'Baseline',
    'nl':          'NL Contract',
    'code-law':    'Code Contract',
}

METRICS = [
    ('R', r'Total gain $R$',      [0, MAX_GAIN * 1.04]),
    ('m', r'Survival months $m$', [0, HORIZON * 1.075]),
]
# Ceiling each metric cannot exceed; CI whiskers are cropped there.
METRIC_CEILING = {'R': MAX_GAIN, 'm': HORIZON}

BAR_WIDTH = 0.13
LEGEND_CAPTION = 'Bar opacity (alpha) = number of “we”-frame agents:'
RAMP_GREY = '#4a4a4a'


def plot_bars(ax, sub, metric, models):
    """Grouped bars: one x slot per model, one bar per "we"-frame level."""
    centers = np.arange(len(models))
    ceiling = METRIC_CEILING[metric]
    for pi, p in enumerate(P_LEVELS):
        vals, err_lo, err_hi = [], [], []
        for model in models:
            row = sub[(sub['model'] == model) & (sub['prosocial'] == p)]
            if row.empty:
                vals.append(np.nan)
                err_lo.append(np.nan)
                err_hi.append(np.nan)
                continue
            v = float(row[metric].iloc[0])
            hw = float(row[f'{metric}_hw'].iloc[0])
            vals.append(v)
            # Crop the whiskers to the range the metric can take: no negative
            # gain / survival, and nothing past the horizon or the max gain.
            err_lo.append(min(hw, v))
            err_hi.append(min(hw, max(0.0, ceiling - v)))

        vals = np.asarray(vals)
        yerr = np.vstack([np.asarray(err_lo), np.asarray(err_hi)])
        offset = (pi - (len(P_LEVELS) - 1) / 2) * BAR_WIDTH
        # NaN bars render as missing; their error bars must be zeroed.
        plot_vals = np.where(np.isnan(vals), 0.0, vals)
        plot_yerr = np.where(np.isnan(yerr), 0.0, yerr)

        for ci, model in enumerate(models):
            st = MODEL_STYLE[model]
            ax.bar(
                centers[ci] + offset, plot_vals[ci],
                width=BAR_WIDTH,
                color=st['color'], alpha=alpha_for(p),
                edgecolor=st['color'], linewidth=0.5,
                yerr=plot_yerr[:, ci:ci + 1],
                error_kw=dict(ecolor=st['color'], elinewidth=0.8, capsize=1.4,
                              alpha=0.9),
            )

    ax.set_xticks(centers)
    ax.set_xticklabels([MODEL_STYLE[m]['label'] for m in models],
                       rotation=20, ha='right')
    ax.tick_params(axis='x', length=0)
    if metric == 'm':
        ax.axhline(HORIZON, color='#888', linestyle=':', linewidth=0.7, zorder=0)
    if metric == 'R':
        ax.axhline(MAX_GAIN, color='#888', linestyle=':', linewidth=0.7, zorder=0)


def add_single_line_legend(fig):
    """Caption + 0..5 opacity ramp on one row.

    The caption is placed by measuring the drawn legend rather than by using
    the legend title, which would occupy a second line.
    """
    handles = [Patch(facecolor=RAMP_GREY, alpha=alpha_for(p),
                     edgecolor='none', label=str(p))
               for p in P_LEVELS]
    leg = fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.56, -0.012),
        ncol=len(P_LEVELS),
        frameon=False,
        columnspacing=1.8,
        handlelength=1.2,
        handletextpad=0.5,
    )
    fig.canvas.draw()
    box = leg.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(box.x0 - 0.012, (box.y0 + box.y1) / 2.0, LEGEND_CAPTION,
             ha='right', va='center', fontsize=10)
    return leg


def build_figure(df, models=None):
    models = models or [m for m in MODEL_ORDER if m in set(df['model'])]
    fig, axes = plt.subplots(len(METRICS), len(COND_ORDER), figsize=(13, 7.0),
                             sharex='col', sharey='row', squeeze=False)

    for r, (metric, ylabel, ylim) in enumerate(METRICS):
        for c, cond in enumerate(COND_ORDER):
            ax = axes[r][c]
            sub = df[df['condition'] == cond]
            plot_bars(ax, sub, metric, models)
            if r == 0:
                ax.set_title(COND_TITLE[cond], pad=8, fontweight='bold')
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.set_ylim(ylim)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    add_single_line_legend(fig)
    return fig


def parse_args():
    p = argparse.ArgumentParser(
        description='Grouped-bar figure of total gain R and survival months m, '
                    'coloured by model, shaded by "we"-frame count.',
    )
    p.add_argument('-i', '--input', default='aggregated.csv',
                   help='per-model aggregate CSV (default: aggregated.csv)')
    p.add_argument('-o', '--out', default='paper/images/model-results-2',
                   help='output path without extension '
                        '(default: paper/images/model-results-2)')
    p.add_argument('-f', '--format', nargs='+', default=['pdf', 'png'],
                   help='output formats (default: pdf png)')
    p.add_argument('--dpi', type=int, default=200,
                   help='raster DPI (default: 200)')
    p.add_argument('--show', action='store_true',
                   help='open an interactive window instead of only saving')
    return p.parse_args()


def main():
    args = parse_args()
    if not args.show:
        plt.switch_backend('Agg')
    plt.rcParams.update(STYLE)

    fig = build_figure(load_per_model(args.input))

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
