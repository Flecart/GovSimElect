"""Plotting utilities for GT-HarmBench contracting experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .contracting import normalize_welfare, nash_deviation


def _setup_tueplot_style() -> None:
    try:
        from tueplots import bundles, markers, axes
        plt.rcParams.update(bundles.icml2024(
            family="serif",
            usetex=True,
            column="half",
            nrows=1, ncols=1,
        ))
        plt.rcParams.update(markers.with_edge())
        plt.rcParams.update(axes.lines())

        font_scale = 1.5
        font_keys = [
            'font.size',
            'axes.labelsize',
            'axes.titlesize',
            'xtick.labelsize',
            'ytick.labelsize',
            'legend.fontsize',
            'legend.title_fontsize',
            'figure.titlesize',
        ]
        for key in font_keys:
            if key in plt.rcParams and isinstance(plt.rcParams[key], (int, float)):
                plt.rcParams[key] = int(plt.rcParams[key] * font_scale)
    except ImportError:
        pass


def plot_variance_comparison(
    summary_df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    _setup_tueplot_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    datasets = ['4x4', '2x2']
    modes = ['no_comm', 'code_nl', 'code_law']
    mode_labels = ['No Contract', 'NL Contract', 'Code Contract']

    x = range(len(modes))
    width = 0.35

    if 'utilitarian_payoff_variance' not in summary_df.columns:
        ax.text(0.5, 0.5, 'No variance data', ha='center', va='center')
        return ax

    for i, dataset in enumerate(datasets):
        subset = summary_df[summary_df['dataset'] == dataset]

        util_var_values = []
        rawls_var_values = []

        for mode in modes:
            mode_data = subset[subset['mode'] == mode]
            if not mode_data.empty:
                util_var_values.append(mode_data['utilitarian_payoff_variance'].mean())
                rawls_var_values.append(mode_data['rawlsian_payoff_variance'].mean())
            else:
                util_var_values.append(0.0)
                rawls_var_values.append(0.0)

        offset = -width/2 if i == 0 else width/2
        ax.bar([p + offset for p in x], util_var_values, width/2, label=f'{dataset} Util Var', alpha=0.8)
        ax.bar([p + offset + width/4 for p in x], rawls_var_values, width/2, label=f'{dataset} Rawls Var', alpha=0.8)

    ax.set_ylabel('Variance', fontsize=11)
    ax.set_title('Welfare Variance Across Experiments', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, rotation=15, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    return ax


def plot_interaction_effects(
    metrics_df: pd.DataFrame,
    outcome_col: str = 'avg_utilitarian_payoff',
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot 2x2 factorial interaction (prompt mode x contract mode) for complementarity."""
    _setup_tueplot_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    modes = ['no_comm', 'code_nl', 'code_law']
    mode_labels = ['No Contract', 'NL Contract', 'Code Contract']
    prompt_modes = sorted(metrics_df['prompt_mode'].unique())
    colors = {'4x4': '#1f77b4', '2x2': '#ff7f0e'}
    x = range(len(modes))
    width = 0.35

    for i, (dataset, color) in enumerate(colors.items()):
        subset = metrics_df[metrics_df['dataset'] == dataset]

        values = []
        for mode in modes:
            mode_data = subset[subset['mode'] == mode]
            if not mode_data.empty:
                val = mode_data[outcome_col].mean()
                values.append(val)
            else:
                values.append(0.0)

        offset = -width/2 if i == 0 else width/2
        ax.bar([p + offset for p in x], values, width, label=dataset, color=color, alpha=0.8)

    ax.set_ylabel(outcome_col.replace('_', ' ').title(), fontsize=11)
    ax.set_title('Interaction Effects: Prompt Mode x Contract Mode', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)

    return ax


def plot_greenwashing_detection(
    df_4x4: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart of greenwashing rate (cooperate action + Low Effort) by prompt mode."""
    _setup_tueplot_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if 'prompt_mode' not in df_4x4.columns:
        ax.text(0.5, 0.5, 'No prompt_mode data', ha='center', va='center')
        return ax

    rates = {}
    for mode in df_4x4['prompt_mode'].unique():
        mode_col = f'{mode}_rate'
        if mode_col in df_4x4.columns:
            rates[mode] = df_4x4[mode_col].iloc[0] if not df_4x4.empty else 0.0

    if not rates:
        ax.text(0.5, 0.5, 'No greenwashing data', ha='center', va='center')
        return ax

    modes = list(rates.keys())
    values = list(rates.values())
    colors = ['#d62728' if 'selfish' in m else '#2ca02c' for m in modes]

    ax.bar(modes, values, color=colors, alpha=0.8)
    ax.set_ylabel('Greenwashing Rate', fontsize=11)
    ax.set_title('Greenwashing Detection: Cooperate Output + Low Effort', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (mode, val) in enumerate(zip(modes, values)):
        ax.text(i, val + 0.02, f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

    return ax


def plot_formation_rates(
    summary_df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart of contract formation rate by contract mode and dataset."""
    _setup_tueplot_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    datasets = ['4x4', '2x2']
    modes = ['no_comm', 'code_nl', 'code_law']
    mode_labels = ['No Contract', 'NL Contract', 'Code Contract']

    x = range(len(modes))
    width = 0.35

    for i, dataset in enumerate(datasets):
        subset = summary_df[summary_df['dataset'] == dataset]

        values = []
        for mode in modes:
            mode_data = subset[subset['mode'] == mode]
            if not mode_data.empty:
                val = mode_data['contract_formation_rate'].mean()
                values.append(val)
            else:
                values.append(0.0)

        offset = -width/2 if i == 0 else width/2
        ax.bar([p + offset for p in x], values, width, label=dataset, alpha=0.8)

    ax.set_ylabel('Formation Rate', fontsize=11)
    ax.set_title('Contract Formation Rate', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    return ax


def plot_activation_rates(
    summary_df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart of contract activation rate (enforceable contracts only)."""
    _setup_tueplot_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    subset = summary_df[summary_df['mode'] != 'no_comm']

    if subset.empty:
        ax.text(0.5, 0.5, 'No contract data', ha='center', va='center')
        return ax

    datasets = ['4x4', '2x2']
    modes = ['code_nl', 'code_law']
    mode_labels = ['NL Contract', 'Code Contract']

    x = range(len(modes))
    width = 0.35

    for i, dataset in enumerate(datasets):
        data = subset[subset['dataset'] == dataset]

        values = []
        for mode in modes:
            mode_data = data[data['mode'] == mode]
            if not mode_data.empty:
                val = mode_data['contract_activation_rate'].mean()
                values.append(val)
            else:
                values.append(0.0)

        offset = -width/2 if i == 0 else width/2
        ax.bar([p + offset for p in x], values, width, label=dataset, alpha=0.8)

    ax.set_ylabel('Activation Rate', fontsize=11)
    ax.set_title('Contract Activation Rate', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    return ax


def plot_effort_distribution(
    metrics_df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Stacked bar chart of High/Low effort choices across contract modes."""
    _setup_tueplot_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if 'mode' not in metrics_df.columns or 'prompt_mode' not in metrics_df.columns:
        ax.text(0.5, 0.5, 'No effort distribution data', ha='center', va='center')
        return ax

    modes = ['no_comm', 'code_nl', 'code_law']
    mode_labels = ['No Contract', 'NL Contract', 'Code Contract']
    prompt_modes = sorted(metrics_df['prompt_mode'].unique())

    x = range(len(modes))
    width = 0.8

    bottom_high = [0] * len(modes)
    bottom_low = [0] * len(modes)

    for i, prompt_mode in enumerate(prompt_modes):
        high_effort = []
        low_effort = []

        for mode in modes:
            subset = metrics_df[
                (metrics_df['mode'] == mode) &
                (metrics_df['prompt_mode'] == prompt_mode) &
                (metrics_df['row_effort_level'] == 'High Effort')
            ]
            high_count = subset['count'].sum() if not subset.empty else 0

            subset = metrics_df[
                (metrics_df['mode'] == mode) &
                (metrics_df['prompt_mode'] == prompt_mode) &
                (metrics_df['row_effort_level'] == 'Low Effort')
            ]
            low_count = subset['count'].sum() if not subset.empty else 0

            high_effort.append(high_count)
            low_effort.append(low_count)

        alpha = 0.6 if i > 0 else 0.8
        ax.bar(x, high_effort, width, label=f'{prompt_mode} (High)', bottom=bottom_high, alpha=alpha)

        new_bottom_low = [h + l for h, l in zip(bottom_high, low_effort)]
        ax.bar(x, low_effort, width, label=f'{prompt_mode} (Low)', bottom=bottom_high, alpha=alpha)

        bottom_high = new_bottom_low

    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Effort Level Distribution (4x4 Scenarios)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    return ax


def plot_welfare_comparison(
    summary_df: pd.DataFrame,
    normalized: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Grouped bar chart of utilitarian and Rawlsian welfare across contract modes."""
    _setup_tueplot_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    df = summary_df.copy()
    datasets = ['4x4', '2x2']
    modes = ['no_comm', 'code_nl', 'code_law']
    mode_labels = ['No Contract', 'NL Contract', 'Code Contract']

    x = range(len(modes))
    width = 0.35

    if normalized:
        util_col = 'avg_norm_utilitarian'
        rawls_col = 'avg_norm_rawlsian'
    else:
        util_col = 'avg_utilitarian_payoff'
        rawls_col = 'avg_rawlsian_payoff'

    for i, dataset in enumerate(datasets):
        subset = df[df['dataset'] == dataset]

        util_values = []
        rawls_values = []

        for mode in modes:
            mode_data = subset[subset['mode'] == mode]
            if not mode_data.empty:
                util_val = mode_data[util_col].mean()
                rawls_val = mode_data[rawls_col].mean()
                util_values.append(util_val if util_val is not None and not pd.isna(util_val) else 0.0)
                rawls_values.append(rawls_val if rawls_val is not None and not pd.isna(rawls_val) else 0.0)
            else:
                util_values.append(0.0)
                rawls_values.append(0.0)

        offset = -width/2 if i == 0 else width/2
        ax.bar([p + offset for p in x], util_values, width/2, label=f'{dataset} Util', alpha=0.8)
        ax.bar([p + offset + width/4 for p in x], rawls_values, width/2, label=f'{dataset} Rawls', alpha=0.8)

    ylabel = 'Welfare Score (Normalized 0-1)' if normalized else 'Welfare Score'
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title('Welfare Comparison Across Experiments', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, rotation=15, ha='right')

    if normalized:
        ax.set_ylim(0, 1.1)
    else:
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    return ax


def create_summary_table(
    summary_df: pd.DataFrame,
    normalized: bool = True,
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Render a formatted summary table of contracting metrics."""
    _setup_tueplot_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    for _, row in summary_df.iterrows():
        row_data = [
            row['dataset'],
            row['mode'],
            row.get('prompt_mode', 'base'),
            f"{row['contract_formation_rate']:.1%}",
            f"{row['avg_utilitarian_payoff']:+.2f}",
            f"{row['avg_rawlsian_payoff']:+.2f}",
        ]

        if normalized:
            norm_util = row.get('avg_norm_utilitarian')
            norm_rawls = row.get('avg_norm_rawlsian')
            nash_dev = row.get('avg_nash_deviation_util')

            row_data.extend([
                f"{norm_util:.2f}" if norm_util is not None else "N/A",
                f"{norm_rawls:.2f}" if norm_rawls is not None else "N/A",
                f"{nash_dev:.2f}" if nash_dev is not None else "N/A",
            ])

        table_data.append(row_data)

    columns = [
        'Dataset', 'Mode', 'Prompt',
        'Formation', 'Util (Raw)', 'Rawls (Raw)',
    ]

    if normalized:
        columns.extend(['Util (Norm)', 'Rawls (Norm)', 'NE Dev'])

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')

    plt.title(
        'GT-HarmBench Contracting Evaluation Summary',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    return fig
