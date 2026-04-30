"""Stratified paired permutation tests for GovSim cell-mean data.

Inputs: two CSVs produced by earlier aggregation scripts:
    /home/claude/govsim/aggregated.csv       (deterministic regen=2)
    /home/claude/govsim/aggregated_sto.csv   (stochastic regen ~ U{1.5,2.5})

Each CSV has one row per (model, prosocial, condition) with columns
    m, q, R, u, e, o (means) and *_hw (CI half-widths from the original run-level bootstrap).

Hypotheses tested:
  H1: Prosociality increases total gain R (p5 vs p0).
  H2: Prosociality increases survival months m (p5 vs p0).
  H3: Prosociality decreases over-usage o (p5 vs p0).
  H4: Prosociality decreases inequality, i.e. increases equality e (p5 vs p0).
  H5a (sto): Contracts increase R vs no-contract.
  H5b (det): Contracts' effect on R in det is different from zero (two-sided).
  H5c (DiD): Contract-effect-on-R is larger in sto than in det.

Test design for H1-H4:
  Statistic: T = mean_{(model, condition)} [ y_{p=5} - y_{p=0} ],
  where "better" is sign-corrected (higher y for R, m, e; lower y for o).
  Null: within each (model, condition) stratum, the p=5 and p=0 labels are
  exchangeable. This is a paired-stratified permutation test on cell means.
  We run the test separately for each regime (det, sto).
  One-sided p-value (upper tail) since the hypotheses are directional.

Test design for H5:
  H5a / H5b: define contract_effect_{model,p} = mean(R_NL, R_code) - R_nocontract.
  Statistic: T = mean_{(model,p)} contract_effect_{model,p}.
  Null: within each (model,p), the three condition labels are exchangeable
  over R. We enumerate/sample permutations of (R_nc, R_nl, R_code) triples
  independently per stratum, recompute T*, and compare.
  H5c: define diff_{model,p} = contract_effect_sto - contract_effect_det (only
  for models present in both regimes). Statistic: mean diff. Null: within each
  (model,p), the regime label is exchangeable on contract_effect. Paired
  stratified permutation.

All tests operate on cell means; there are no raw per-run values here.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(12345)
N_PERM = 10000
N_BOOT = 10000
# Tune this list manually:
# - ["code-law", "nl"] -> contract arm is mean(code-law, nl)
# - ["code-law"]       -> contract arm is code-law only
H5_CONTRACT_CONDITIONS = ["code-law"] #, "nl"]
# H5_CONTRACT_CONDITIONS = ["nl"]

# ---------- Data loading ----------

def format_p_value(p):
    """Format p-values with scientific notation for very small values."""
    return f"{p:.2e}" if p < 1e-4 else f"{p:.7f}"


def perm_pvalue_upper_tail(t_perms, t_obs):
    """Monte Carlo permutation p-value with +1 correction (never exactly 0)."""
    b = int(np.sum(t_perms >= t_obs))
    m = int(len(t_perms))
    return (b + 1) / (m + 1)


def perm_pvalue_two_sided(t_perms, t_obs):
    """Two-sided Monte Carlo p-value with +1 correction (never exactly 0)."""
    b = int(np.sum(np.abs(t_perms) >= abs(t_obs)))
    m = int(len(t_perms))
    return (b + 1) / (m + 1)


def bootstrap_mean_ci(values, *, n_boot=N_BOOT, alpha=0.05, rng=RNG):
    """Percentile bootstrap CI for mean(values)."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return np.nan, np.nan
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def _fmt_or_blank(value, formatter):
    if pd.isna(value):
        return "-"
    return formatter(value)


def print_summary_table(df):
    """Pretty-print a compact summary table with consistent formatting."""
    columns = [
        ("test", "test"),
        ("regime", "regime"),
        ("metric", "metric"),
        ("direction", "dir"),
        ("T_obs", "T_obs"),
        ("T_ci", "T_ci_95"),
        ("T_ci_hw", "T_ci_hw"),
        ("p_value", "p_value"),
        ("n_strata", "n"),
        ("description", "description"),
    ]

    rows = []
    for _, row in df.iterrows():
        formatted = {
            "test": str(row["test"]),
            "regime": str(row["regime"]),
            "metric": _fmt_or_blank(row["metric"], str),
            "direction": _fmt_or_blank(row["direction"], lambda x: f"{int(x):+d}"),
            "T_obs": f"{row['T_obs']:+.4f}",
            "T_ci": (
                "-"
                if pd.isna(row.get("T_ci_low", np.nan)) or pd.isna(row.get("T_ci_high", np.nan))
                else f"[{row['T_ci_low']:+.4f}, {row['T_ci_high']:+.4f}]"
            ),
            "T_ci_hw": _fmt_or_blank(
                row.get("T_ci_hw", np.nan),
                lambda x: f"{float(x):.4f}",
            ),
            "p_value": format_p_value(float(row["p_value"])),
            "n_strata": f"{int(row['n_strata'])}",
            "description": str(row["description"]),
        }
        rows.append(formatted)

    widths = {}
    for key, header in columns:
        max_cell = max((len(r[key]) for r in rows), default=0)
        widths[key] = max(len(header), max_cell)

    header_line = "  ".join(header.ljust(widths[key]) for key, header in columns)
    sep_line = "  ".join("-" * widths[key] for key, _ in columns)
    print(header_line)
    print(sep_line)
    for r in rows:
        line = "  ".join(r[key].ljust(widths[key]) for key, _ in columns)
        print(line)

def load_agg(path):
    df = pd.read_csv(path)
    return df[['model','prosocial','condition','m','q','R','u','e','o',
               'm_hw','q_hw','R_hw','u_hw','e_hw','o_hw',
               'full_horizon','total_runs']].copy()

det = load_agg('aggregated.csv')
sto = load_agg('aggregated-sto.csv')

# ---------- H1-H4: p5 vs p0 paired stratified permutation ----------

def paired_p5_vs_p0(df, metric, direction, n_perm=N_PERM, rng=RNG):
    """
    df: aggregated DataFrame for one regime.
    metric: column name (e.g. 'R', 'm', 'o', 'e').
    direction: +1 if "prosociality helps" means higher metric
               -1 if "prosociality helps" means lower metric.
    Returns dict with observed T, permutation p-value (one-sided, upper tail),
    and the per-stratum deltas used.
    """
    # Build paired table: one row per (model, condition) with y0 (p=0) and y5 (p=5).
    sub = df[df['prosocial'].isin([0, 5])][['model','prosocial','condition', metric]]
    piv = sub.pivot_table(index=['model','condition'], columns='prosocial',
                          values=metric).reset_index()
    piv.columns.name = None
    piv = piv.rename(columns={0: 'y0', 5: 'y5'}).dropna(subset=['y0','y5'])
    deltas = direction * (piv['y5'].values - piv['y0'].values)  # sign-corrected
    T_obs = float(deltas.mean())
    T_ci_low, T_ci_high = bootstrap_mean_ci(deltas, rng=rng)

    # Permute: within each stratum (row), flip the p-label (swap y5/y0) with prob 0.5
    n_strata = len(deltas)
    T_perms = np.empty(n_perm)
    for k in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n_strata)
        T_perms[k] = float(np.mean(signs * deltas))
    p_val = perm_pvalue_upper_tail(T_perms, T_obs)
    return {
        'metric': metric,
        'direction': direction,
        'T_obs': T_obs,
        'T_ci_low': T_ci_low,
        'T_ci_high': T_ci_high,
        'p_value': p_val,
        'n_strata': n_strata,
        'deltas': deltas.tolist(),
        'pivot': piv,
    }


def run_H1_to_H4(regime_name, df):
    print(f"\n{'='*68}")
    print(f"  H1-H4: p5 vs p0 effect   [{regime_name}]")
    print(f"{'='*68}")
    tests = [
        ('H1', 'R', +1, 'Prosociality increases total gain R'),
        ('H2', 'm', +1, 'Prosociality increases survival months m'),
        ('H3', 'o', -1, 'Prosociality decreases over-usage o'),
        ('H4', 'e', +1, 'Prosociality increases equality e (reduces inequality)'),
    ]
    rows = []
    for tag, metric, direction, desc in tests:
        res = paired_p5_vs_p0(df, metric, direction)
        sign_str = "+" if direction > 0 else "-"
        print(f"\n  [{tag}] {desc}")
        print(f"    metric = {metric}  (sign {sign_str}), {res['n_strata']} strata (model x condition)")
        print(f"    observed T  = {res['T_obs']:+.7f}  "
              f"(mean sign-corrected delta over strata)")
        print(f"    95% CI(T_obs) = [{res['T_ci_low']:+.7f}, {res['T_ci_high']:+.7f}]")
        print(f"    perm p-value (one-sided) = {format_p_value(res['p_value'])}")
        rows.append({
            'test': tag, 'regime': regime_name, 'metric': metric,
            'direction': direction, 'T_obs': res['T_obs'],
            'T_ci_low': res['T_ci_low'], 'T_ci_high': res['T_ci_high'],
            'p_value': res['p_value'], 'n_strata': res['n_strata'],
            'description': desc,
        })
    return pd.DataFrame(rows)


# ---------- H5: contracts and regime interaction on gain R ----------

def contract_effect_table(df):
    """Per (model, prosocial) compute contract effect vs no-contract."""
    piv = df.pivot_table(
        index=['model','prosocial'], columns='condition', values='R'
    ).reset_index()
    piv.columns.name = None

    required_cols = ['no-contract'] + list(H5_CONTRACT_CONDITIONS)
    piv = piv.dropna(subset=required_cols)

    piv['R_nc'] = piv['no-contract']
    piv['R_contract_mean'] = np.mean(piv[H5_CONTRACT_CONDITIONS].to_numpy(), axis=1)
    piv['contract_effect'] = piv['R_contract_mean'] - piv['R_nc']
    return piv

def H5a_contracts_help_sto(sto_df, n_perm=N_PERM, rng=RNG):
    """Permutation test for contracts helping in stochastic regime."""
    piv = contract_effect_table(sto_df)
    T_obs = float(piv['contract_effect'].mean())
    T_ci_low, T_ci_high = bootstrap_mean_ci(piv['contract_effect'].values, rng=rng)

    n_strata = len(piv)
    value_cols = ['no-contract'] + list(H5_CONTRACT_CONDITIONS)
    stacked = piv[value_cols].to_numpy()
    T_perms = np.empty(n_perm)
    for k in range(n_perm):
        perms = rng.permuted(stacked, axis=1)
        ce = np.mean(perms[:, 1:], axis=1) - perms[:, 0]
        T_perms[k] = float(np.mean(ce))

    p_val = perm_pvalue_upper_tail(T_perms, T_obs)
    return {'T_obs': T_obs, 'T_ci_low': T_ci_low, 'T_ci_high': T_ci_high,
            'p_value': p_val, 'n_strata': n_strata,
            'per_stratum_effect': piv[['model','prosocial','contract_effect']]}


def H5b_contracts_effect_det(det_df, n_perm=N_PERM, rng=RNG):
    """Two-sided test for deterministic contract effect vs zero."""
    piv = contract_effect_table(det_df)
    T_obs = float(piv['contract_effect'].mean())
    T_ci_low, T_ci_high = bootstrap_mean_ci(piv['contract_effect'].values, rng=rng)
    n_strata = len(piv)
    value_cols = ['no-contract'] + list(H5_CONTRACT_CONDITIONS)
    stacked = piv[value_cols].to_numpy()
    T_perms = np.empty(n_perm)
    for k in range(n_perm):
        perms = rng.permuted(stacked, axis=1)
        ce = np.mean(perms[:, 1:], axis=1) - perms[:, 0]
        T_perms[k] = float(np.mean(ce))

    p_val = perm_pvalue_two_sided(T_perms, T_obs)
    return {'T_obs': T_obs, 'T_ci_low': T_ci_low, 'T_ci_high': T_ci_high,
            'p_value': p_val, 'n_strata': n_strata,
            'per_stratum_effect': piv[['model','prosocial','contract_effect']]}


def H5c_DiD_sto_vs_det(det_df, sto_df, n_perm=N_PERM, rng=RNG):
    """Difference-of-differences:
       contract_effect_sto > contract_effect_det across shared (model, p) strata.
       Null: regime label exchangeable on contract_effect within (model, p).
    """
    det_ce = contract_effect_table(det_df)[['model','prosocial','contract_effect']]
    sto_ce = contract_effect_table(sto_df)[['model','prosocial','contract_effect']]
    det_ce = det_ce.rename(columns={'contract_effect': 'ce_det'})
    sto_ce = sto_ce.rename(columns={'contract_effect': 'ce_sto'})
    merged = det_ce.merge(sto_ce, on=['model','prosocial'], how='inner')
    diffs = merged['ce_sto'].values - merged['ce_det'].values
    T_obs = float(diffs.mean())
    T_ci_low, T_ci_high = bootstrap_mean_ci(diffs, rng=rng)
    n_strata = len(merged)

    # Null: regime label exchangeable -> flip sign of diff with prob 0.5
    T_perms = np.empty(n_perm)
    for k in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n_strata)
        T_perms[k] = float(np.mean(signs * diffs))
    p_val = perm_pvalue_upper_tail(T_perms, T_obs)
    return {'T_obs': T_obs, 'T_ci_low': T_ci_low, 'T_ci_high': T_ci_high,
            'p_value': p_val, 'n_strata': n_strata,
            'merged': merged}


def run_H5():
    print(f"\n{'='*68}")
    print("  H5: Contracts and regime interaction on gain R")
    print(f"{'='*68}")
    mode_label = f"mean({', '.join(H5_CONTRACT_CONDITIONS)}) vs no-contract"
    print(f"  Contract effect mode: {mode_label}")

    print("\n  [H5a] Contracts help vs no-contract in STOCHASTIC")
    a = H5a_contracts_help_sto(sto)
    print(f"    T_obs = {a['T_obs']:+.2f}  over {a['n_strata']} (model, p) strata")
    print(f"    95% CI(T_obs) = [{a['T_ci_low']:+.2f}, {a['T_ci_high']:+.2f}]")
    print(f"    perm p-value (one-sided) = {format_p_value(a['p_value'])}")
    print("    Per-stratum contract effects:")
    print(a['per_stratum_effect'].to_string(index=False))

    print("\n  [H5b] Contracts' effect in DETERMINISTIC (two-sided)")
    b = H5b_contracts_effect_det(det)
    print(f"    T_obs = {b['T_obs']:+.2f}  over {b['n_strata']} (model, p) strata")
    print(f"    95% CI(T_obs) = [{b['T_ci_low']:+.2f}, {b['T_ci_high']:+.2f}]")
    print(f"    perm p-value (two-sided) = {format_p_value(b['p_value'])}")
    print("    Per-stratum contract effects:")
    print(b['per_stratum_effect'].to_string(index=False))

    print("\n  [H5c] Difference-of-differences: sto contract effect > det contract effect")
    c = H5c_DiD_sto_vs_det(det, sto)
    print(f"    T_obs = {c['T_obs']:+.2f}  over {c['n_strata']} matched (model, p) strata")
    print(f"    95% CI(T_obs) = [{c['T_ci_low']:+.2f}, {c['T_ci_high']:+.2f}]")
    print(f"    perm p-value (one-sided) = {format_p_value(c['p_value'])}")
    print("    Per-stratum (ce_sto - ce_det):")
    show = c['merged'].copy()
    show['delta'] = show['ce_sto'] - show['ce_det']
    print(show.to_string(index=False))

    return pd.DataFrame([
        {'test':'H5a', 'regime':'sto',     'T_obs': a['T_obs'], 'T_ci_low': a['T_ci_low'], 'T_ci_high': a['T_ci_high'], 'p_value': a['p_value'], 'n_strata': a['n_strata'],
         'description': 'Contracts help vs no-contract on R, stochastic (one-sided)'},
        {'test':'H5b', 'regime':'det',     'T_obs': b['T_obs'], 'T_ci_low': b['T_ci_low'], 'T_ci_high': b['T_ci_high'], 'p_value': b['p_value'], 'n_strata': b['n_strata'],
         'description': "Contracts' R effect in det differs from zero (two-sided)"},
        {'test':'H5c', 'regime':'sto-det', 'T_obs': c['T_obs'], 'T_ci_low': c['T_ci_low'], 'T_ci_high': c['T_ci_high'], 'p_value': c['p_value'], 'n_strata': c['n_strata'],
         'description': 'sto contract effect > det contract effect (DiD, one-sided)'},
    ])

# ---------- D5: code-law reduces per-month gain (R/m) vs NL ----------

def D5_code_vs_nl_per_month(df, n_perm=N_PERM, rng=RNG):
    """
    Within each (model, prosocial) stratum, test whether code-law reduces
    per-month productivity (R/m) relative to NL.

    Statistic: T = mean over strata of [ (R/m)_NL - (R/m)_code ].
    Null: within each stratum, the {NL, code-law} labels are exchangeable
    on R/m (paired sign-flip permutation).
    One-sided upper-tail p-value (we predict NL > code-law on R/m).
    """
    sub = df[df['condition'].isin(['nl', 'code-law'])].copy()
    sub['R_per_m'] = sub['R'] / sub['m']

    piv = sub.pivot_table(index=['model', 'prosocial'],
                          columns='condition', values='R_per_m').reset_index()
    piv.columns.name = None
    piv = piv.dropna(subset=['nl', 'code-law'])

    diffs = piv['nl'].values - piv['code-law'].values  # NL - code, per stratum
    T_obs = float(diffs.mean())
    T_ci_low, T_ci_high = bootstrap_mean_ci(diffs, rng=rng)
    n_strata = len(diffs)

    T_perms = np.empty(n_perm)
    for k in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n_strata)
        T_perms[k] = float(np.mean(signs * diffs))
    p_val = perm_pvalue_upper_tail(T_perms, T_obs)

    return {
        'T_obs': T_obs,
        'T_ci_low': T_ci_low,
        'T_ci_high': T_ci_high,
        'p_value': p_val,
        'n_strata': n_strata,
        'per_stratum': piv.assign(diff_nl_minus_code=diffs),
    }


def run_D5():
    print(f"\n{'='*68}")
    print("  D5: NL > code-law on per-month gain (R/m)")
    print(f"{'='*68}")

    rows = []
    for regime_name, df in [('deterministic', det), ('stochastic', sto)]:
        print(f"\n  [D5 / {regime_name}]  NL vs code-law on R/m")
        r = D5_code_vs_nl_per_month(df)
        print(f"    T_obs = {r['T_obs']:+.3f} units of R/m  "
              f"(over {r['n_strata']} model x p strata)")
        print(f"    95% CI(T_obs) = [{r['T_ci_low']:+.3f}, {r['T_ci_high']:+.3f}]")
        print(f"    perm p-value (one-sided) = {format_p_value(r['p_value'])}")
        print("    Per-stratum (NL - code-law):")
        print(r['per_stratum'].to_string(index=False))
        rows.append({
            'test': 'D5', 'regime': regime_name,
            'metric': 'R_per_m', 'direction': +1,
            'T_obs': r['T_obs'], 'p_value': r['p_value'],
            'T_ci_low': r['T_ci_low'], 'T_ci_high': r['T_ci_high'],
            'n_strata': r['n_strata'],
            'description': 'NL > code-law on per-month gain R/m (one-sided)',
        })
    return pd.DataFrame(rows)

if __name__ == '__main__':
    rows_det = run_H1_to_H4('deterministic', det)
    rows_sto = run_H1_to_H4('stochastic',    sto)
    rows_h5 = run_H5()
    rows_d5 = run_D5()
    out = pd.concat([rows_det, rows_sto, rows_h5, rows_d5], ignore_index=True)
    out['T_ci_hw'] = (out['T_ci_high'] - out['T_ci_low']) / 2.0
    out_path = 'permutation_test_results.csv'
    out.to_csv(out_path, index=False)
    print(f"\n\nSaved: {out_path}")
    avg_ci_hw = float(out['T_ci_hw'].mean()) if len(out) > 0 else np.nan
    print(f"Average 95% CI half-width (mean T_ci_hw): {avg_ci_hw:.6f}")
    print("\nSummary:")
    print_summary_table(out)