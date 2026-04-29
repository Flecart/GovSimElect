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

import itertools
import numpy as np
import pandas as pd

RNG = np.random.default_rng(12345)
N_PERM = 10000

# ---------- Data loading ----------

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

    # Permute: within each stratum (row), flip the p-label (swap y5/y0) with prob 0.5
    n_strata = len(deltas)
    T_perms = np.empty(n_perm)
    for k in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n_strata)
        T_perms[k] = float(np.mean(signs * deltas))
    p_val = float(np.mean(T_perms >= T_obs))  # one-sided upper tail
    return {
        'metric': metric,
        'direction': direction,
        'T_obs': T_obs,
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
        print(f"    observed T  = {res['T_obs']:+.4f}  "
              f"(mean sign-corrected delta over strata)")
        print(f"    perm p-value (one-sided) = {res['p_value']:.4f}")
        rows.append({
            'test': tag, 'regime': regime_name, 'metric': metric,
            'direction': direction, 'T_obs': res['T_obs'],
            'p_value': res['p_value'], 'n_strata': res['n_strata'],
            'description': desc,
        })
    return pd.DataFrame(rows)


# ---------- H5: contracts and regime interaction on gain R ----------

def contract_effect_table(df):
    """Per (model, prosocial) compute R_nc, mean(R_NL, R_code), and their diff."""
    piv = df.pivot_table(
        index=['model','prosocial'], columns='condition', values='R'
    ).reset_index()
    piv.columns.name = None
    # Drop any (model, p) missing a condition
    piv = piv.dropna(subset=['no-contract','nl','code-law'])
    piv['R_nc'] = piv['no-contract']
    piv['R_contract_mean'] = (piv['nl'] + piv['code-law']) / 2.0
    piv['contract_effect'] = piv['R_contract_mean'] - piv['R_nc']
    return piv

def H5a_contracts_help_sto(sto_df, n_perm=N_PERM, rng=RNG):
    """Within each (model,p), permute the three condition labels on R.
    T_obs = mean(contract_effect)."""
    piv = contract_effect_table(sto_df)
    T_obs = float(piv['contract_effect'].mean())

    # For each stratum, enumerate the 6 label permutations of (R_nc, R_nl, R_code);
    # per permutation, recompute contract_effect = mean(perm[1], perm[2]) - perm[0].
    # With n strata and 6 perms each, we sample n_perm combinations.
    R_nc_col  = piv['no-contract'].values
    R_nl_col  = piv['nl'].values
    R_cd_col  = piv['code-law'].values
    stacked = np.stack([R_nc_col, R_nl_col, R_cd_col], axis=1)  # (n_strata, 3)
    n_strata = len(piv)

    T_perms = np.empty(n_perm)
    for k in range(n_perm):
        # Independently permute each row's 3 values
        # Sample a random permutation index 0..5 per row
        perms = rng.permuted(stacked, axis=1)
        # "No-contract" placeholder is column 0, contracts are columns 1 and 2
        ce = perms[:,1:].mean(axis=1) - perms[:,0]
        T_perms[k] = ce.mean()
    p_val = float(np.mean(T_perms >= T_obs))  # one-sided upper
    return {'T_obs': T_obs, 'p_value': p_val, 'n_strata': n_strata,
            'per_stratum_effect': piv[['model','prosocial','contract_effect']]}


def H5b_contracts_effect_det(det_df, n_perm=N_PERM, rng=RNG):
    """Two-sided test: is det contract_effect different from zero?"""
    piv = contract_effect_table(det_df)
    T_obs = float(piv['contract_effect'].mean())
    R_nc_col  = piv['no-contract'].values
    R_nl_col  = piv['nl'].values
    R_cd_col  = piv['code-law'].values
    stacked = np.stack([R_nc_col, R_nl_col, R_cd_col], axis=1) # (n_strata, 3)
    n_strata = len(piv)

    T_perms = np.empty(n_perm)
    for k in range(n_perm):
        perms = rng.permuted(stacked, axis=1)
        ce = perms[:,1:].mean(axis=1) - perms[:,0]
        T_perms[k] = ce.mean()
    p_val = float(np.mean(np.abs(T_perms) >= abs(T_obs)))  # two-sided
    return {'T_obs': T_obs, 'p_value': p_val, 'n_strata': n_strata,
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
    n_strata = len(merged)

    # Null: regime label exchangeable -> flip sign of diff with prob 0.5
    T_perms = np.empty(n_perm)
    for k in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n_strata)
        T_perms[k] = float(np.mean(signs * diffs))
    p_val = float(np.mean(T_perms >= T_obs))  # one-sided upper
    return {'T_obs': T_obs, 'p_value': p_val, 'n_strata': n_strata,
            'merged': merged}


def run_H5():
    print(f"\n{'='*68}")
    print(f"  H5: Contracts and regime interaction on gain R")
    print(f"{'='*68}")

    print("\n  [H5a] Contracts help vs no-contract in STOCHASTIC")
    a = H5a_contracts_help_sto(sto)
    print(f"    T_obs = {a['T_obs']:+.2f}  over {a['n_strata']} (model, p) strata")
    print(f"    perm p-value (one-sided) = {a['p_value']:.4f}")
    print("    Per-stratum contract effects:")
    print(a['per_stratum_effect'].to_string(index=False))

    print("\n  [H5b] Contracts' effect in DETERMINISTIC (two-sided)")
    b = H5b_contracts_effect_det(det)
    print(f"    T_obs = {b['T_obs']:+.2f}  over {b['n_strata']} (model, p) strata")
    print(f"    perm p-value (two-sided) = {b['p_value']:.4f}")
    print("    Per-stratum contract effects:")
    print(b['per_stratum_effect'].to_string(index=False))

    print("\n  [H5c] Difference-of-differences: sto contract effect > det contract effect")
    c = H5c_DiD_sto_vs_det(det, sto)
    print(f"    T_obs = {c['T_obs']:+.2f}  over {c['n_strata']} matched (model, p) strata")
    print(f"    perm p-value (one-sided) = {c['p_value']:.4f}")
    print("    Per-stratum (ce_sto - ce_det):")
    show = c['merged'].copy()
    show['delta'] = show['ce_sto'] - show['ce_det']
    print(show.to_string(index=False))

    return pd.DataFrame([
        {'test':'H5a', 'regime':'sto',     'T_obs': a['T_obs'], 'p_value': a['p_value'], 'n_strata': a['n_strata'],
         'description': 'Contracts help vs no-contract on R, stochastic (one-sided)'},
        {'test':'H5b', 'regime':'det',     'T_obs': b['T_obs'], 'p_value': b['p_value'], 'n_strata': b['n_strata'],
         'description': "Contracts' R effect in det differs from zero (two-sided)"},
        {'test':'H5c', 'regime':'sto-det', 'T_obs': c['T_obs'], 'p_value': c['p_value'], 'n_strata': c['n_strata'],
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
    n_strata = len(diffs)

    T_perms = np.empty(n_perm)
    for k in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n_strata)
        T_perms[k] = float(np.mean(signs * diffs))
    p_val = float(np.mean(T_perms >= T_obs))  # one-sided upper

    return {
        'T_obs': T_obs,
        'p_value': p_val,
        'n_strata': n_strata,
        'per_stratum': piv.assign(diff_nl_minus_code=diffs),
    }


def run_D5():
    print(f"\n{'='*68}")
    print(f"  D5: NL > code-law on per-month gain (R/m)")
    print(f"{'='*68}")

    rows = []
    for regime_name, df in [('deterministic', det), ('stochastic', sto)]:
        print(f"\n  [D5 / {regime_name}]  NL vs code-law on R/m")
        r = D5_code_vs_nl_per_month(df)
        print(f"    T_obs = {r['T_obs']:+.3f} units of R/m  "
              f"(over {r['n_strata']} model x p strata)")
        print(f"    perm p-value (one-sided) = {r['p_value']:.4f}")
        print("    Per-stratum (NL - code-law):")
        print(r['per_stratum'].to_string(index=False))
        rows.append({
            'test': 'D5', 'regime': regime_name,
            'metric': 'R_per_m', 'direction': +1,
            'T_obs': r['T_obs'], 'p_value': r['p_value'],
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
    out_path = 'permutation_test_results.csv'
    out.to_csv(out_path, index=False)
    print(f"\n\nSaved: {out_path}")
    print("\nSummary:")
    print(out.to_string(index=False))