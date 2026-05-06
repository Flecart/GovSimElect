"""Generate the LaTeX table in the requested format."""
import pandas as pd
import numpy as np

agg = pd.read_csv('/home/claude/govsim/aggregated.csv')

# Model display names (preserve order matching the data)
model_display = {
    'gemma-4-31b': 'Gemma-4-31B',
    'gpt-4o':       'GPT-4o',
    'gpt-5.4':      'GPT-5.4',
    'gpt-5.4-mini': 'GPT-5.4-Mini',
    'grok-4.1-fast': 'Grok-4.1-Fast',
}
# Table row order: for each model, we iterate p0..p5 and within each p, conditions in order
# "Selfish/Prosocial" maps to prosocial count (p0 = fully selfish, p5 = fully prosocial).
# Instead we label by prosocial count.
cond_order = ['no-contract', 'nl', 'code-law']
cond_label = {
    'no-contract': 'No contract',
    'nl':          'NL contract',
    'code-law':    'Code contract',
}

model_order = ['gemma-4-31b', 'gpt-4o', 'gpt-5.4', 'gpt-5.4-mini', 'grok-4.1-fast']

def fmt_val(v, hw, decimals=3, width=None):
    """Format value with CI half-width in smaller font."""
    if decimals == 1:
        val_s = f"{v:.1f}"
        hw_s = f"{hw:.1f}"
    elif decimals == 2:
        val_s = f"{v:.2f}"
        hw_s = f"{hw:.2f}"
    else:
        val_s = f"{v:.3f}"
        hw_s = f"{hw:.3f}"
    return f"${val_s}{{\\scriptscriptstyle\\,\\pm{hw_s}}}$"

def fmt_gain(v, hw):
    return f"${v:.1f}{{\\scriptscriptstyle\\,\\pm{hw:.1f}}}$"

def fmt_surv_time(v, hw):
    return f"${v:.1f}{{\\scriptscriptstyle\\,\\pm{hw:.1f}}}$"

def fmt_rate(v):
    return f"${v:.3f}$"

def fmt_full(full, total):
    return f"${int(full)}/{int(total)}$"

# Build rows
lines = []
lines.append(r"\begin{table}[ht]")
lines.append(r"\centering")
lines.append(r"\caption{GovSim commons results by model, prosocial composition $p_k$ (number of prosocial agents out of 5), and contract condition. Point estimates are means across runs; values in small font report the bootstrap 95\% CI half-width (symmetrised).  \textbf{Full surv.} is the number of runs that reached the 12-month horizon over the total number of runs.}")
lines.append(r"\label{tab:commons_results}")
lines.append(r"\small")
lines.append(r"\begin{adjustbox}{max width=\textwidth}")
lines.append(r"\begin{tabular}{llccccccc}")
lines.append(r"\toprule")
lines.append(r"\textbf{Model} & \textbf{Condition} & Surv.\ Rate $q$ & Surv.\ Time $m$ & Full surv. & Gain $R$ & Eff.\ $u$ & Eq.\ $e$ & Over-usage $o$ \\")
lines.append(r"\midrule")

for mi, model in enumerate(model_order):
    sub = agg[agg['model'] == model].copy()
    # Count unique prosocial levels
    p_levels = sorted(sub['prosocial'].unique())
    n_rows = len(p_levels) * len(cond_order)
    mname = model_display[model]
    first_in_block = True
    for p in p_levels:
        for ci, cond in enumerate(cond_order):
            row = sub[(sub['prosocial'] == p) & (sub['condition'] == cond)]
            if row.empty:
                continue
            r = row.iloc[0]
            if first_in_block:
                model_cell = f"\\multirow{{{n_rows}}}{{*}}{{{mname}}}"
                first_in_block = False
            else:
                model_cell = ""
            # condition label with prosocial level
            cond_cell = f"$p_{int(p)}$, {cond_label[cond]}"
            line = " & ".join([
                model_cell,
                cond_cell,
                fmt_rate(r['q']),
                fmt_surv_time(r['m'], r['m_hw']),
                fmt_full(r['full_horizon'], r['total_runs']),
                fmt_gain(r['R'], r['R_hw']),
                fmt_val(r['u'], r['u_hw']),
                fmt_val(r['e'], r['e_hw']),
                fmt_val(r['o'], r['o_hw']),
            ]) + r" \\"
            lines.append(line)
        # subtle separator between prosocial blocks for readability
        if p != p_levels[-1]:
            lines.append(r"\cmidrule(l){2-9}")
    if mi < len(model_order) - 1:
        lines.append(r"\midrule")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{adjustbox}")
lines.append(r"\end{table}")

table_tex = "\n".join(lines)
with open('/home/claude/govsim/commons_results.tex', 'w') as f:
    f.write(table_tex)
print(table_tex)