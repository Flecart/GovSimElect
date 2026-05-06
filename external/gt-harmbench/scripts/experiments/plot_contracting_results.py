"""Plot contracting evaluation results from summary CSV."""

import sys
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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
        plt.rcParams['figure.autolayout'] = True

        # Scale up fonts for readability
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


def extract_joint_action_frequencies(summary_csv: Path, log_base: Path) -> tuple:
    """Read pre-cached joint action matrices from summary CSV.

    Aggregates by action category (cooperate/defect) and effort level across
    scenarios, returning (action_data, prompt_mode).
    """
    df = pd.read_csv(summary_csv)

    # 4x4 label ordering: Coop+High, Coop+Low, Defect+High, Defect+Low
    ACTION_4X4_LABELS = [
        'Cooperate (High Effort)',
        'Cooperate (Low Effort)',
        'Defect (High Effort)',
        'Defect (Low Effort)'
    ]

    result = {
        '4x4': {'by_mode': {}, 'action_labels': {'row': ACTION_4X4_LABELS, 'col': ACTION_4X4_LABELS}},
        '2x2': {'by_mode': {}, 'action_labels': {'row': ['Cooperate', 'Defect'], 'col': ['Cooperate', 'Defect']}},
    }

    # Infer prompt_mode from analysis/{prompt_mode}/ path
    prompt_mode = "base"
    path_parts = summary_csv.parts
    if 'analysis' in path_parts:
        analysis_idx = path_parts.index('analysis')
        if analysis_idx + 1 < len(path_parts):
            prompt_mode = path_parts[analysis_idx + 1]

    for _, row in df.iterrows():
        exp = row['experiment']
        dataset = row.get('joint_action_dataset')
        if not isinstance(dataset, str) or pd.isna(dataset):
            dataset = '4x4' if '4x4' in exp else '2x2'
        mode = 'No Contract' if 'no_comm' in exp or 'no-comm' in exp else (
            'NL Contract' if 'code_n' in exp or 'code-nl' in exp else 'Code Contract'
        )

        try:
            matrix_raw = row.get('joint_action_matrix')
            if not isinstance(matrix_raw, str) or not matrix_raw or pd.isna(matrix_raw):
                continue
            matrix = json.loads(matrix_raw)
            existing = result[dataset]['by_mode'].get(mode)
            if existing is None:
                result[dataset]['by_mode'][mode] = matrix
            else:
                result[dataset]['by_mode'][mode] = [
                    [old + new for old, new in zip(old_row, new_row)]
                    for old_row, new_row in zip(existing, matrix)
                ]
        except Exception as e:
            print(f"Warning: Could not load cached action matrix from {exp}: {e}")

    return result, prompt_mode


def extract_model_name_from_log_base(log_base: Path) -> str:
    """Parse model identifier from a log directory like eval-20260419-143000-openai-gpt-4o."""
    dir_name = log_base.name
    parts = dir_name.split('-')
    if len(parts) >= 3 and parts[0] == 'eval':
        model_parts = [p for p in parts[2:] if p and not p.isdigit()]
        if model_parts:
            return '-'.join(model_parts)
    return "unknown"


def shorten_model_name(model_name: str) -> str:
    """Map internal model identifiers to display names."""
    if not model_name or model_name == "unknown":
        return "Unknown Model"

    name_map = {
        "gpt-4o": "GPT-4o",
        "gpt-4o-mini": "GPT-4o Mini",
        "gpt-5.1": "GPT-5.1",
        "gpt-5.2": "GPT-5.2",
        "claude-sonnet-4.5": "Sonnet 4.5",
        "claude-3.5-sonnet": "Claude 3.5 Sonnet",
        "claude-3-opus": "Claude 3 Opus",
        "grok": "Grok 4.1 fast",
        "gemini-flash": "Gemini Flash",
        "gemini-pro": "Gemini Pro",
        "llama-3": "Llama 3",
        "qwen": "Qwen",
    }

    normalized = model_name.lower().replace("anthropic-", "").replace("openai-", "").replace("meta-", "")

    for key, value in name_map.items():
        if key.lower() in normalized:
            return value

    # Capitalize as fallback
    parts = normalized.replace("_", "-").split("-")
    return " ".join(p.capitalize() for p in parts if p)


def plot_contracting_results(summary_csv: Path, output_dir: Path, model_name: str = "unknown") -> None:
    df = pd.read_csv(summary_csv)
    df['dataset'] = df['experiment'].apply(lambda x: '4x4' if '4x4' in x else '2x2')

    def extract_mode(exp_name):
        if 'no_comm' in exp_name or 'no-comm' in exp_name:
            return 'No Contract'
        elif 'code_n' in exp_name or 'code-nl' in exp_name:
            return 'NL Contract'
        elif 'code_l' in exp_name or 'code-law' in exp_name:
            return 'Code Contract'
        return 'Unknown'

    df['mode_label'] = df['experiment'].apply(extract_mode)

    # summary.csv lives at logs/eval-xxx/analysis/{prompt_mode}/summary.csv
    log_base = summary_csv.parent.parent.parent
    action_data, prompt_mode = extract_joint_action_frequencies(summary_csv, log_base)

    if not model_name or model_name == "unknown":
        model_name = extract_model_name_from_log_base(log_base)
    short_model = shorten_model_name(model_name)

    n_per_exp = 0
    for dataset in ['4x4', '2x2']:
        for mode, matrix in action_data[dataset]['by_mode'].items():
            matrix_sum = sum(sum(row) for row in matrix)
            if matrix_sum > 0:
                n_per_exp = matrix_sum
                break
        if n_per_exp > 0:
            break

    title_parts = [short_model]
    if prompt_mode != "base":
        title_parts.append(prompt_mode)
    title_suffix = f" ({', '.join(title_parts)}, n={n_per_exp})"

    output_dir.mkdir(parents=True, exist_ok=True)

    _setup_tueplot_style()

    mode_order = ['No Contract', 'NL Contract', 'Code Contract']
    dataset_colors = {'4x4': 'steelblue', '2x2': 'coral'}
    df_ordered = df.set_index('mode_label').loc[mode_order].reset_index()
    has_variance = {
        'utilitarian_payoff_variance',
        'rawlsian_payoff_variance',
    }.issubset(df.columns)

    def values_by_mode(dataset: str, column: str) -> list[float]:
        """Return values for a dataset in canonical contract-mode order."""
        data = df_ordered[df_ordered['dataset'] == dataset]
        return [
            data[data['mode_label'] == mode][column].values[0]
            for mode in mode_order
            if mode in data['mode_label'].values
        ]

    def positions_by_mode(dataset: str) -> list[int]:
        """Return x positions for modes available in a dataset."""
        data = df_ordered[df_ordered['dataset'] == dataset]
        return [
            idx
            for idx, mode in enumerate(mode_order)
            if mode in data['mode_label'].values
        ]

    # ============================================================================
    # PLOT 1: Contract Formation Rate
    # ============================================================================
    fig, ax = plt.subplots(figsize=(8, 6))

    x = range(len(mode_order))
    width = 0.35

    for i, dataset in enumerate(['4x4', '2x2']):
        data = df_ordered[df_ordered['dataset'] == dataset]
        positions = [x[j] for j, mode in enumerate(mode_order) if mode in data['mode_label'].values]
        values = [data[data['mode_label'] == mode]['contract_formation_rate'].values[0]
                  for mode in mode_order if mode in data['mode_label'].values]
        offset = -width/2 if i == 0 else width/2
        ax.bar([p + offset for p in positions], values, width, label=dataset, alpha=0.8)

    ax.set_ylabel('Formation Rate', fontsize=11)
    ax.set_title(f'Contract Formation Rate{title_suffix}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_order, rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    formation_path = output_dir / "z_formation_rate.png"
    plt.savefig(formation_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {formation_path}")
    plt.close()

    # ============================================================================
    # PLOT 2: Joint Action Heatmaps
    # ============================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for dataset_idx, dataset in enumerate(['2x2', '4x4']):
        for mode_idx, mode in enumerate(mode_order):
            ax = axes[dataset_idx, mode_idx]

            row_labels = action_data[dataset]['action_labels']['row']
            col_labels = action_data[dataset]['action_labels']['col']
            matrix = action_data[dataset]['by_mode'].get(mode, [[0]*len(col_labels) for _ in row_labels])

            if dataset == '4x4':
                def format_action(action):
                    return action.replace('Cooperate', 'Coop').replace('Defect', 'Defect')
                row_display = [format_action(label) for label in row_labels]
                col_display = [format_action(label) for label in col_labels]
            else:
                row_display = row_labels
                col_display = col_labels

            total_samples = sum(sum(row) for row in matrix)
            im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, interpolation='nearest')
            ax.grid(False)

            ax.set_xticks(range(len(col_labels)))
            ax.set_yticks(range(len(row_labels)))
            ax.set_xticklabels(col_display, fontsize=7, rotation=45, ha='right')
            ax.set_yticklabels(row_display, fontsize=7, rotation=0)

            for i in range(len(row_labels)):
                for j in range(len(col_labels)):
                    value = matrix[i][j]
                    if value > 0:
                        rel_freq = value / total_samples if total_samples > 0 else 0
                        max_val = max(map(max, matrix)) if any(any(row) for row in matrix) else 1
                        text_color = 'white' if value > max_val * 0.5 else 'black'
                        ax.text(j, i, f'{rel_freq:.2f}\n({value})', ha='center', va='center',
                               color=text_color, fontsize=8, fontweight='bold')

            if mode_idx == 0:
                ax.set_ylabel(dataset, fontsize=11, fontweight='bold')
            if dataset_idx == 0:
                ax.set_title(mode, fontsize=10, fontweight='bold')

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Frequency', fontsize=7)

    plt.suptitle(f'Joint Action Selection Frequencies{title_suffix}', fontsize=14, fontweight='bold', y=0.995)
    action_heatmaps_path = output_dir / "action_heatmaps.png"
    plt.savefig(action_heatmaps_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {action_heatmaps_path}")
    plt.close()

    # ============================================================================
    # PLOT 3: Raw Payoffs (Utilitarian only - Rawlsian is identical for PD)
    # ============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, dataset in enumerate(['4x4', '2x2']):
        ax = axes[ax_idx]
        data = df_ordered[df_ordered['dataset'] == dataset]
        x_pos = range(len(data))

        ax.bar(x_pos, data['avg_utilitarian_payoff'],
               width=0.6, label='Utilitarian', alpha=0.8, color='steelblue')

        ax.set_ylabel('Payoff', fontsize=11)
        ax.set_title(f'{dataset} Raw Payoffs', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(data['mode_label'], rotation=15, ha='right')
        ax.legend(loc='upper right')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    raw_path = output_dir / "z_raw_payoffs.png"
    plt.savefig(raw_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {raw_path}")
    plt.close()

    # ============================================================================
    # PLOT 4: Variance Comparison (Utilitarian only - Rawlsian is identical for PD)
    # ============================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    if has_variance:
        x = range(len(mode_order))
        width = 0.35

        for i, dataset in enumerate(['4x4', '2x2']):
            positions = positions_by_mode(dataset)
            values = values_by_mode(dataset, 'utilitarian_payoff_variance')
            offset = -width / 2 if i == 0 else width / 2
            ax.bar(
                [p + offset for p in positions],
                values,
                width,
                label=dataset,
                alpha=0.8,
                color=dataset_colors[dataset],
            )

        ax.set_ylabel('Payoff Variance', fontsize=11)
        ax.set_title('Utilitarian Payoff Variance', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(mode_order, rotation=15, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No variance data', ha='center', va='center')
        ax.axis('off')

    variance_path = output_dir / "z_variance_payoffs.png"
    plt.savefig(variance_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {variance_path}")
    plt.close()

    # ============================================================================
    # PLOT 5: Summary Table
    # ============================================================================
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    for _, row in df.iterrows():
        activation = 'N/A' if pd.isna(row.get('contract_activation_rate')) or row.get('contract_activation_rate') is None else f"{row['contract_activation_rate']:.1%}"
        effort = 'N/A' if pd.isna(row.get('high_effort_rate')) or row.get('high_effort_rate') is None else f"{row['high_effort_rate']:.1%}"

        nash_acc = row.get('nash_accuracy', None)
        util_acc = row.get('utilitarian_accuracy', None)
        nash_acc_str = f"{nash_acc:.1%}" if nash_acc is not None else 'N/A'
        util_acc_str = f"{util_acc:.1%}" if util_acc is not None else 'N/A'

        util_var = row.get('utilitarian_payoff_variance', 0)
        util_var_str = f"{util_var:.3f}" if has_variance else 'N/A'

        table_data.append([
            row['dataset'],
            row['mode_label'],
            f"{row['contract_formation_rate']:.1%}",
            activation,
            nash_acc_str,
            util_acc_str,
            f"{row['avg_utilitarian_payoff']:+.2f}",
            f"{row.get('cooperation_rate', 0):.1%}",
            effort,
            util_var_str,
        ])

    col_labels = [
        'Dataset', 'Mode', 'Formation', 'Activation',
        'Nash Acc', 'Util Acc',
        'Util Payoff',
        'Coop Rate', 'High Effort',
        'Util Var',
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')

    plt.title('Contracting Evaluation Summary (Including Nash/Util Accuracy)', fontsize=14, fontweight='bold', pad=20)

    table_path = output_dir / "z_summary_table.png"
    plt.savefig(table_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {table_path}")
    plt.close()

    # ============================================================================
    # PLOT 6: Combined Grid (all plots in one figure)
    # ============================================================================
    # 3x2 grid (Rawlsian omitted since identical to Utilitarian for PD)
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.30, left=0.06, right=0.96, top=0.95, bottom=0.05)

    x = range(len(mode_order))
    width = 0.35

    # Subplot 1: Formation Rate (top-left)
    ax1 = fig.add_subplot(gs[0, 0])

    for i, dataset in enumerate(['4x4', '2x2']):
        data = df_ordered[df_ordered['dataset'] == dataset]
        positions = [x[j] for j, mode in enumerate(mode_order) if mode in data['mode_label'].values]
        values = [data[data['mode_label'] == mode]['contract_formation_rate'].values[0]
                  for mode in mode_order if mode in data['mode_label'].values]
        offset = -width/2 if i == 0 else width/2
        ax1.bar([p + offset for p in positions], values, width, label=dataset, alpha=0.8)

    ax1.set_ylabel('Formation Rate', fontsize=10, fontweight='bold')
    ax1.set_title('Contract Formation Rate', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_order, rotation=15, ha='right', fontsize=8)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Raw Payoffs 4x4 (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    data_4x4 = df_ordered[df_ordered['dataset'] == '4x4']
    x_pos = range(len(data_4x4))

    ax2.bar(x_pos, data_4x4['avg_utilitarian_payoff'],
            width=0.6, label='Utilitarian', alpha=0.8, color='steelblue')

    ax2.set_ylabel('Payoff', fontsize=10, fontweight='bold')
    ax2.set_title('4x4 Raw Payoffs', fontsize=11, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(data_4x4['mode_label'], rotation=15, ha='right', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Utilitarian variance (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])

    if has_variance:
        for i, dataset in enumerate(['4x4', '2x2']):
            positions = positions_by_mode(dataset)
            values = values_by_mode(dataset, 'utilitarian_payoff_variance')
            offset = -width/2 if i == 0 else width/2
            ax3.bar([p + offset for p in positions], values, width, label=dataset, alpha=0.8, color=dataset_colors[dataset])

        ax3.set_ylabel('Payoff Variance', fontsize=10, fontweight='bold')
        ax3.set_title('Utilitarian Variance', fontsize=11, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(mode_order, rotation=15, ha='right', fontsize=8)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No variance data', ha='center', va='center', fontsize=9)

    # Subplot 4: Raw Payoffs 2x2 (middle-right)
    ax4 = fig.add_subplot(gs[1, 1])
    data_2x2 = df_ordered[df_ordered['dataset'] == '2x2']
    x_pos = range(len(data_2x2))

    ax4.bar(x_pos, data_2x2['avg_utilitarian_payoff'],
            width=0.6, label='Utilitarian', alpha=0.8, color='steelblue')

    ax4.set_ylabel('Payoff', fontsize=10, fontweight='bold')
    ax4.set_title('2x2 Raw Payoffs', fontsize=11, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(data_2x2['mode_label'], rotation=15, ha='right', fontsize=8)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)

    # Subplot 5: Nash Accuracy (bottom-left)
    ax5 = fig.add_subplot(gs[2, 0])

    has_nash_acc = 'nash_accuracy' in df.columns and df['nash_accuracy'].notna().any()

    if has_nash_acc:
        for i, dataset in enumerate(['4x4', '2x2']):
            data = df_ordered[df_ordered['dataset'] == dataset]
            positions = [x[j] for j, mode in enumerate(mode_order) if mode in data['mode_label'].values]
            values = [data[data['mode_label'] == mode]['nash_accuracy'].values[0]
                      for mode in mode_order if mode in data['mode_label'].values]
            offset = -width/2 if i == 0 else width/2
            ax5.bar([p + offset for p in positions], values, width, label=dataset, alpha=0.8)

        ax5.set_ylabel('Nash Accuracy', fontsize=10, fontweight='bold')
        ax5.set_title('Nash Equilibrium Selection Rate', fontsize=11, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(mode_order, rotation=15, ha='right', fontsize=8)
        ax5.set_ylim(0, 1.1)
        ax5.legend(loc='upper right', fontsize=8)
        ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No Nash accuracy data', ha='center', va='center', fontsize=9)

    # Subplot 6: Utilitarian Accuracy (bottom-right)
    ax6 = fig.add_subplot(gs[2, 1])

    has_util_acc = 'utilitarian_accuracy' in df.columns and df['utilitarian_accuracy'].notna().any()

    if has_util_acc:
        for i, dataset in enumerate(['4x4', '2x2']):
            data = df_ordered[df_ordered['dataset'] == dataset]
            positions = [x[j] for j, mode in enumerate(mode_order) if mode in data['mode_label'].values]
            values = [data[data['mode_label'] == mode]['utilitarian_accuracy'].values[0]
                      for mode in mode_order if mode in data['mode_label'].values]
            offset = -width/2 if i == 0 else width/2
            ax6.bar([p + offset for p in positions], values, width, label=dataset, alpha=0.8)

        ax6.set_ylabel('Utilitarian Accuracy', fontsize=10, fontweight='bold')
        ax6.set_title('Utilitarian Optimum Selection Rate', fontsize=11, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(mode_order, rotation=15, ha='right', fontsize=8)
        ax6.set_ylim(0, 1.1)
        ax6.legend(loc='upper right', fontsize=8)
        ax6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No Util accuracy data', ha='center', va='center', fontsize=9)

    fig.suptitle(f'GT-HarmBench Contracting Evaluation Results{title_suffix}', fontsize=16, fontweight='bold')

    combined_path = output_dir / "combined_plots.png"
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {combined_path}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_contracting_results.py <summary.csv> <output_dir> [model_name]")
        sys.exit(1)

    summary_csv = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    model_name = sys.argv[3] if len(sys.argv) > 3 else "unknown"

    if not summary_csv.exists():
        print(f"Error: {summary_csv} not found")
        sys.exit(1)

    plot_contracting_results(summary_csv, output_dir, model_name)
