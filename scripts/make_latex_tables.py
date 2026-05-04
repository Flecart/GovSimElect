"""Emit LaTeX tables for GovSim commons results.

Reads aggregated.csv (deterministic) and aggregated-sto.csv (stochastic),
collapses any duplicate (model, prosocial, condition) rows via unweighted
mean (matching scripts/logs/parser.py), then prints two tables to stdout:

  1. Deterministic-only table that matches the format in the paper draft.
  2. Combined table with an extra "Regen" column (Det / Sto) interleaving rows.

Usage:
    python3 scripts/make_latex_tables.py            # both tables to stdout
    python3 scripts/make_latex_tables.py --det      # deterministic only
    python3 scripts/make_latex_tables.py --combined # combined only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DET_CSV = REPO_ROOT / 'aggregated.csv'
STO_CSV = REPO_ROOT / 'aggregated-sto.csv'

MODEL_ORDER = [
    'gemma-4-31b',
    'gemma-4-31b-it',
    'gpt-4o',
    'gpt-5.4',
    'gpt-5.4-mini',
    'grok-4.1-fast',
]
MODEL_DISPLAY = {
    'gemma-4-31b':    'Gemma-4-31B',
    'gemma-4-31b-it': 'Gemma-4-31B-IT',
    'gpt-4o':         'GPT-4o',
    'gpt-5.4':        'GPT-5.4',
    'gpt-5.4-mini':   'GPT-5.4-Mini',
    'grok-4.1-fast':  'Grok-4.1-Fast',
}

PROSOCIAL_LEVELS = [0, 1, 2, 3, 4, 5]
COND_ORDER = ['no-contract', 'nl', 'code-law']
COND_DISPLAY = {
    'no-contract': 'No contract',
    'nl':          'NL contract',
    'code-law':    'Code contract',
}


def load_and_dedup(path: Path) -> pd.DataFrame:
    """Read a results CSV and collapse duplicate cells via unweighted mean."""
    df = pd.read_csv(path)
    numeric = [c for c in df.select_dtypes(include='number').columns if c != 'prosocial']
    return (
        df.groupby(['model', 'prosocial', 'condition'], as_index=False)[numeric].mean()
    )


def fmt_val(val: float, hw: float | None, decimals: int, show_hw: bool = True) -> str:
    """Format a metric value. Optionally append a scriptscriptstyle CI half-width."""
    if pd.isna(val):
        return '--'
    body = f'{val:.{decimals}f}'
    if show_hw and hw is not None and not pd.isna(hw):
        body += rf'{{\scriptscriptstyle\,\pm{hw:.{decimals}f}}}'
    return f'${body}$'


def fmt_full_surv(full: float, total: float) -> str:
    if pd.isna(full) or pd.isna(total):
        return '--'
    return f'${int(round(full))}/{int(round(total))}$'


def get_row(df: pd.DataFrame, model: str, p: int, cond: str) -> pd.Series | None:
    sel = df[(df['model'] == model) & (df['prosocial'] == p) & (df['condition'] == cond)]
    if sel.empty:
        return None
    return sel.iloc[0]


def metric_cells(row: pd.Series | None) -> list[str]:
    """Return the seven metric cells: q, m, full_surv, R, u, e, o."""
    if row is None:
        return ['--'] * 7
    return [
        fmt_val(row['q'], row.get('q_hw'), 3, show_hw=False),
        fmt_val(row['m'], row.get('m_hw'), 1),
        fmt_full_surv(row['full_horizon'], row['full_horizon'] + row['below_horizon']),
        fmt_val(row['R'], row.get('R_hw'), 1),
        fmt_val(row['u'], row.get('u_hw'), 3),
        fmt_val(row['e'], row.get('e_hw'), 3),
        fmt_val(row['o'], row.get('o_hw'), 3),
    ]


HEADER_DET = r"""\begin{table}[ht]
\centering
\caption{GovSim commons results by model, prosocial composition $p_k$ (number of prosocial agents out of 5), and contract condition. Point estimates are means across runs; values in small font report the bootstrap 95\% CI half-width (symmetrised). \textbf{Full surv.} is the number of runs that reached the 12-month horizon over the total number of runs.}
\label{tab:commons_results}
\small
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llccccccc}
\toprule
\textbf{Model} & \textbf{Condition} & Surv.\ Rate $q$ & Surv.\ Time $m$ & Full surv. & Gain $R$ & Eff.\ $u$ & Eq.\ $e$ & Over-usage $o$ \\
\midrule"""

FOOTER = r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}"""


def render_det_table(det: pd.DataFrame) -> str:
    out = [HEADER_DET]
    n_rows_per_model = len(PROSOCIAL_LEVELS) * len(COND_ORDER)  # 18
    for mi, model in enumerate(MODEL_ORDER):
        for pi, p in enumerate(PROSOCIAL_LEVELS):
            for ci, cond in enumerate(COND_ORDER):
                row = get_row(det, model, p, cond)
                cells = metric_cells(row)
                model_cell = (
                    rf'\multirow{{{n_rows_per_model}}}{{*}}{{{MODEL_DISPLAY[model]}}}'
                    if (pi == 0 and ci == 0) else ''
                )
                cond_cell = f'$p_{p}$, {COND_DISPLAY[cond]}'
                out.append(' & '.join([model_cell, cond_cell, *cells]) + r' \\')
            if pi < len(PROSOCIAL_LEVELS) - 1:
                out.append(r'\cmidrule(l){2-9}')
        if mi < len(MODEL_ORDER) - 1:
            out.append(r'\midrule')
    out.append(FOOTER)
    return '\n'.join(out)


HEADER_COMBINED = r"""\begin{table}[ht]
\centering
\caption{GovSim commons results by model, prosocial composition $p_k$, contract condition, and regeneration regime (Det = deterministic $r=2$; Sto = i.i.d.\ stochastic $r\sim U\{1.5,2.5\}$). Point estimates are means; values in small font report the bootstrap 95\% CI half-width (symmetrised). Cells marked -- have no runs in that regime.}
\label{tab:commons_results_combined}
\scriptsize
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lllccccccc}
\toprule
\textbf{Model} & \textbf{Condition} & \textbf{Regen} & Surv.\ Rate $q$ & Surv.\ Time $m$ & Full surv. & Gain $R$ & Eff.\ $u$ & Eq.\ $e$ & Over-usage $o$ \\
\midrule"""

FOOTER_COMBINED = r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}"""


def render_combined_table(det: pd.DataFrame, sto: pd.DataFrame) -> str:
    out = [HEADER_COMBINED]
    n_rows_per_model = len(PROSOCIAL_LEVELS) * len(COND_ORDER) * 2  # 36
    for mi, model in enumerate(MODEL_ORDER):
        for pi, p in enumerate(PROSOCIAL_LEVELS):
            for ci, cond in enumerate(COND_ORDER):
                for ri, (regen_label, regen_df) in enumerate([('Det', det), ('Sto', sto)]):
                    row = get_row(regen_df, model, p, cond)
                    cells = metric_cells(row)
                    if pi == 0 and ci == 0 and ri == 0:
                        model_cell = (
                            rf'\multirow{{{n_rows_per_model}}}{{*}}{{{MODEL_DISPLAY[model]}}}'
                        )
                    else:
                        model_cell = ''
                    cond_cell = f'$p_{p}$, {COND_DISPLAY[cond]}' if ri == 0 else ''
                    out.append(
                        ' & '.join([model_cell, cond_cell, regen_label, *cells]) + r' \\'
                    )
            if pi < len(PROSOCIAL_LEVELS) - 1:
                out.append(r'\cmidrule(l){2-10}')
        if mi < len(MODEL_ORDER) - 1:
            out.append(r'\midrule')
    out.append(FOOTER_COMBINED)
    return '\n'.join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--det', action='store_true', help='Print only the deterministic table.')
    parser.add_argument('--combined', action='store_true', help='Print only the combined det+sto table.')
    args = parser.parse_args(argv)

    det = load_and_dedup(DET_CSV)
    sto = load_and_dedup(STO_CSV)

    print_det = args.det or not (args.det or args.combined)
    print_combined = args.combined or not (args.det or args.combined)

    parts = []
    if print_det:
        parts.append('% === Deterministic-only table ===')
        parts.append(render_det_table(det))
    if print_combined:
        if parts:
            parts.append('')
        parts.append('% === Combined deterministic + stochastic table ===')
        parts.append(render_combined_table(det, sto))
    print('\n'.join(parts))
    return 0


if __name__ == '__main__':
    sys.exit(main())
