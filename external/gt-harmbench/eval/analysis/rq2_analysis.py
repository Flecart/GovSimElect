"""RQ2 analysis: Complementarity vs Substitutability of Contracting and Prosociality.

This module implements statistical tests for RQ2: "Are Contracting and Prosociality
Complementary or Substitutive?"

The paper's theoretical claim is that contracting and prosociality are:
- **Substitutes when contractibility is full** (2×2): Contracts alone can close the gap
- **Complements when incontractibility is introduced** (4×4): Contracts hit a structural
  floor that prosocial agents close

The key empirical prediction is a three-way interaction: variant × regime × prosociality.
The regime-dependence of the prosociality effect should differ between 2×2 and 4×4.

Three Statistical Tests:
1. Substitutability in 2×2: Does prosociality effect shrink as contracts strengthen?
2. Persistence in 4×4: Does prosociality effect persist across regimes?
3. Three-way interaction: Does the regime-dependence differ between variants?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from scipy import stats


# =============================================================================
# Data Preparation (reuse from RQ1)
# =============================================================================

def expand_to_sample_level(
    scenario_stats_df: pd.DataFrame,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Expand scenario-level summary stats to synthetic sample-level records.

    This is a convenience import from rq1_analysis. Import directly from there
    in the notebook to avoid duplication.
    """
    from rq1_analysis import expand_to_sample_level as _expand
    return _expand(scenario_stats_df, rng)


# =============================================================================
# Test 1: Substitutability in 2×2
# =============================================================================

def test_substitutability_2x2(
    sample_df: pd.DataFrame,
    game: str = "pd",
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """Test if prosociality effect shrinks as contracts get stronger in 2×2.

    Tests the substitutability hypothesis: in contractible (2×2) environments,
    contracts can substitute for prosociality, so the prosociality effect should
    be smaller under strong contracts.

    Quantity: Δ(2×2, no-comm) - Δ(2×2, code-law) — should be positive

    Args:
        sample_df: Sample-level DataFrame from expand_to_sample_level()
        game: "pd" or "sh"
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed

    Returns:
        Dict with keys:
            - interaction_p: p-value from likelihood-ratio test on prosociality × regime
            - contrast_no_comm: Δ at no-comm regime
            - contrast_code_law: Δ at code-law regime
            - contrast_diff: Δ(no-comm) - Δ(code-law)
            - ci_lower, ci_upper: Bootstrap CI for contrast_diff
            - selfish_code_law_acc: P(util-correct | selfish, 2×2, code-law)
            - has_ceiling_effect: True if selfish_code_law_acc ≈ 1
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError("statsmodels is required. Install with: pip install statsmodels")

    # Filter to 2×2, {selfish, cooperative}
    df = sample_df[
        (sample_df['dataset_size'] == '2x2') &
        (sample_df['game'] == game) &
        (sample_df['prompt_mode'].isin(['selfish', 'cooperative']))
    ].copy()

    if df.empty:
        raise ValueError(f"No data for substitutability test in {game} 2×2")

    # Encode variables
    df['prosociality'] = df['prompt_mode'].map({'selfish': 0, 'cooperative': 1})
    df['regime'] = pd.Categorical(
        df['contract_mode'],
        categories=['no_comm', 'code_nl', 'code_law']
    )

    # Fit full model with interaction
    formula_full = "util_correct ~ prosociality * C(regime)"
    formula_reduced = "util_correct ~ prosociality + C(regime)"

    # Try mixed effects, fall back to fixed effects
    try:
        full_model = smf.mixedlm(
            formula_full,
            df,
            groups=df['model'],
            re_formula="1",
        )
        full_result = full_model.fit(reml=False, disp=0)

        reduced_model = smf.mixedlm(
            formula_reduced,
            df,
            groups=df['model'],
            re_formula="1",
        )
        reduced_result = reduced_model.fit(reml=False, disp=0)
    except Exception:
        # Fallback to fixed effects
        full_result = smf.logit(formula_full, data=df).fit(disp=0)
        reduced_result = smf.logit(formula_reduced, data=df).fit(disp=0)

    # Likelihood-ratio test for interaction
    lr_stat = 2 * (full_result.llf - reduced_result.llf)
    lr_df = 2  # 2 interaction terms (prosociality × code_nl, prosociality × code_law)
    from scipy.stats import chi2
    interaction_p = 1 - chi2.cdf(lr_stat, lr_df)

    # Compute marginal prosociality effects at each regime
    marginal_effects = {}
    for regime_val in ['no_comm', 'code_nl', 'code_law']:
        regime_df = df[df['contract_mode'] == regime_val]

        coop_acc = regime_df[regime_df['prompt_mode'] == 'cooperative']['util_correct'].mean()
        selfish_acc = regime_df[regime_df['prompt_mode'] == 'selfish']['util_correct'].mean()

        marginal_effects[regime_val] = {
            'cooperative': coop_acc,
            'selfish': selfish_acc,
            'delta': coop_acc - selfish_acc,
        }

    # Check for ceiling effect in selfish+code-law
    selfish_code_law_acc = marginal_effects['code_law']['selfish']
    has_ceiling_effect = selfish_code_law_acc >= 0.95

    # Bootstrap CI for contrast difference
    rng = np.random.default_rng(seed)
    scenarios = df['scenario_id'].unique()
    n_scenarios = len(scenarios)

    boot_contrasts = []
    for _ in range(n_bootstrap):
        resampled_scenarios = rng.choice(scenarios, size=n_scenarios, replace=True)
        boot_df = df[df['scenario_id'].isin(resampled_scenarios)]

        # Compute Δ at no-comm and code-law
        deltas = []
        for regime_val in ['no_comm', 'code_law']:
            regime_df = boot_df[boot_df['contract_mode'] == regime_val]
            coop_acc = regime_df[regime_df['prompt_mode'] == 'cooperative']['util_correct'].mean()
            selfish_acc = regime_df[regime_df['prompt_mode'] == 'selfish']['util_correct'].mean()

            if pd.notna(coop_acc) and pd.notna(selfish_acc):
                deltas.append(coop_acc - selfish_acc)
            else:
                deltas.append(0.0)

        if len(deltas) == 2:
            boot_contrasts.append(deltas[0] - deltas[1])

    boot_contrasts = np.array(boot_contrasts)

    # CI for the difference
    ci_lower = float(np.percentile(boot_contrasts, 2.5))
    ci_upper = float(np.percentile(boot_contrasts, 97.5))

    return {
        'interaction_p': interaction_p,
        'contrast_no_comm': marginal_effects['no_comm']['delta'],
        'contrast_code_law': marginal_effects['code_law']['delta'],
        'contrast_diff': marginal_effects['no_comm']['delta'] - marginal_effects['code_law']['delta'],
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'selfish_code_law_acc': selfish_code_law_acc,
        'has_ceiling_effect': has_ceiling_effect,
        'n_scenarios': n_scenarios,
    }


# =============================================================================
# Test 2: Persistence in 4×4
# =============================================================================

def test_persistence_4x4(
    sample_df: pd.DataFrame,
    game: str = "pd",
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """Test if prosociality effect persists across regimes in 4×4.

    Tests the complementarity hypothesis: in incontractible (4×4) environments,
    contracts cannot fully substitute for prosociality, so the prosociality effect
    should persist even under the strongest contracts.

    Key comparison: Δ(4×4, code-law) > 0

    Args:
        sample_df: Sample-level DataFrame from expand_to_sample_level()
        game: "pd" or "sh"
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed

    Returns:
        Dict with keys:
            - delta_by_regime: Dict of Δ values at each regime
            - delta_code_law: Δ at code-law regime (headline comparison)
            - ci_lower_code_law, ci_upper_code_law: Bootstrap CI for Δ(code-law)
            - p_code_law: One-sided p-value for Δ(code-law) > 0
            - invariance_test: Comparison of Δ across regimes
    """
    # Filter to 4×4, {selfish, cooperative}
    df = sample_df[
        (sample_df['dataset_size'] == '4x4') &
        (sample_df['game'] == game) &
        (sample_df['prompt_mode'].isin(['selfish', 'cooperative']))
    ].copy()

    if df.empty:
        raise ValueError(f"No data for persistence test in {game} 4×4")

    # Compute prosociality effects at each regime
    delta_by_regime = {}
    for regime_val in ['no_comm', 'code_nl', 'code_law']:
        regime_df = df[df['contract_mode'] == regime_val]

        coop_acc = regime_df[regime_df['prompt_mode'] == 'cooperative']['util_correct'].mean()
        selfish_acc = regime_df[regime_df['prompt_mode'] == 'selfish']['util_correct'].mean()

        delta_by_regime[regime_val] = {
            'cooperative': coop_acc,
            'selfish': selfish_acc,
            'delta': coop_acc - selfish_acc,
        }

    # Bootstrap CIs for each regime
    rng = np.random.default_rng(seed)
    scenarios = df['scenario_id'].unique()
    n_scenarios = len(scenarios)

    boot_deltas = {regime: [] for regime in ['no_comm', 'code_nl', 'code_law']}

    for _ in range(n_bootstrap):
        resampled_scenarios = rng.choice(scenarios, size=n_scenarios, replace=True)
        boot_df = df[df['scenario_id'].isin(resampled_scenarios)]

        for regime_val in ['no_comm', 'code_nl', 'code_law']:
            regime_df = boot_df[boot_df['contract_mode'] == regime_val]

            coop_acc = regime_df[regime_df['prompt_mode'] == 'cooperative']['util_correct'].mean()
            selfish_acc = regime_df[regime_df['prompt_mode'] == 'selfish']['util_correct'].mean()

            if pd.notna(coop_acc) and pd.notna(selfish_acc):
                boot_deltas[regime_val].append(coop_acc - selfish_acc)

    # Compute CIs
    results = {
        'delta_by_regime': delta_by_regime,
        'delta_code_law': delta_by_regime['code_law']['delta'],
        'ci_lower_code_law': float(np.percentile(boot_deltas['code_law'], 2.5)),
        'ci_upper_code_law': float(np.percentile(boot_deltas['code_law'], 97.5)),
        'p_code_law': float(np.mean(np.array(boot_deltas['code_law']) <= 0)),
    }

    # Add CIs for other regimes
    for regime_val in ['no_comm', 'code_nl']:
        boot_arr = np.array(boot_deltas[regime_val])
        results[f'ci_lower_{regime_val}'] = float(np.percentile(boot_arr, 2.5))
        results[f'ci_upper_{regime_val}'] = float(np.percentile(boot_arr, 97.5))
        results[f'p_{regime_val}'] = float(np.mean(boot_arr <= 0))

    # Invariance test: compare Δ(no-comm) vs Δ(code-law)
    # If prosociality persists, these should be similar
    invariance_contrasts = [
        delta_by_regime['no_comm']['delta'] - delta_by_regime['code_law']['delta']
        for _ in range(len(boot_deltas['no_comm']))
    ]

    results['invariance_contrast'] = delta_by_regime['no_comm']['delta'] - delta_by_regime['code_law']['delta']
    results['invariance_ci_lower'] = float(np.percentile(invariance_contrasts, 2.5))
    results['invariance_ci_upper'] = float(np.percentile(invariance_contrasts, 97.5))

    results['n_scenarios'] = n_scenarios

    return results


# =============================================================================
# Test 3: Three-way Interaction (Headline)
# =============================================================================

def test_three_way_interaction(
    sample_df: pd.DataFrame,
    game: str = "pd",
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """Test if regime-dependence of prosociality effect differs between 2×2 and 4×4.

    This is the headline test for RQ2. Tests the three-way interaction:
    variant × regime × prosociality.

    Quantity: [Δ(2×2, no-comm) - Δ(2×2, code-law)] - [Δ(4×4, no-comm) - Δ(4×4, code-law)]

    Theory predicts this is positive: substitution happens in 2×2 but not in 4×4.

    Args:
        sample_df: Sample-level DataFrame from expand_to_sample_level()
        game: "pd" or "sh"
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed

    Returns:
        Dict with keys:
            - three_way_p: p-value from likelihood-ratio test on three-way interaction
            - contrast_2x2: Δ(2×2, no-comm) - Δ(2×2, code-law)
            - contrast_4x4: Δ(4×4, no-comm) - Δ(4×4, code-law)
            - three_way_contrast: contrast_2x2 - contrast_4x4
            - ci_lower, ci_upper: Bootstrap CI for three_way_contrast
            - marginal_effects: Full breakdown of effects by variant and regime
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError("statsmodels is required. Install with: pip install statsmodels")

    # Filter to {selfish, cooperative}, both variants
    df = sample_df[
        (sample_df['game'] == game) &
        (sample_df['prompt_mode'].isin(['selfish', 'cooperative']))
    ].copy()

    if df.empty:
        raise ValueError(f"No data for three-way interaction test in {game}")

    # Encode variables
    df['prosociality'] = df['prompt_mode'].map({'selfish': 0, 'cooperative': 1})
    df['variant'] = pd.Categorical(df['dataset_size'], categories=['2x2', '4x4'])
    df['regime'] = pd.Categorical(
        df['contract_mode'],
        categories=['no_comm', 'code_nl', 'code_law']
    )

    # Fit full model with three-way interaction
    formula_full = "util_correct ~ variant * prosociality * C(regime)"
    formula_reduced = "util_correct ~ variant * prosociality + variant * C(regime) + prosociality * C(regime)"

    # Try mixed effects, fall back to fixed effects
    try:
        full_model = smf.mixedlm(
            formula_full,
            df,
            groups=df['model'],
            re_formula="1",
        )
        full_result = full_model.fit(reml=False, disp=0)

        reduced_model = smf.mixedlm(
            formula_reduced,
            df,
            groups=df['model'],
            re_formula="1",
        )
        reduced_result = reduced_model.fit(reml=False, disp=0)
    except Exception:
        # Fallback to fixed effects
        full_result = smf.logit(formula_full, data=df).fit(disp=0)
        reduced_result = smf.logit(formula_reduced, data=df).fit(disp=0)

    # Likelihood-ratio test for three-way interaction
    lr_stat = 2 * (full_result.llf - reduced_result.llf)
    lr_df = 2  # 2 three-way interaction terms
    from scipy.stats import chi2
    three_way_p = 1 - chi2.cdf(lr_stat, lr_df)

    # Compute marginal prosociality effects by variant and regime
    marginal_effects = {}
    for variant_val in ['2x2', '4x4']:
        marginal_effects[variant_val] = {}
        for regime_val in ['no_comm', 'code_nl', 'code_law']:
            cell_df = df[
                (df['dataset_size'] == variant_val) &
                (df['contract_mode'] == regime_val)
            ]

            coop_acc = cell_df[cell_df['prompt_mode'] == 'cooperative']['util_correct'].mean()
            selfish_acc = cell_df[cell_df['prompt_mode'] == 'selfish']['util_correct'].mean()

            marginal_effects[variant_val][regime_val] = {
                'cooperative': coop_acc,
                'selfish': selfish_acc,
                'delta': coop_acc - selfish_acc,
            }

    # Compute the key contrasts
    contrast_2x2 = (
        marginal_effects['2x2']['no_comm']['delta'] -
        marginal_effects['2x2']['code_law']['delta']
    )
    contrast_4x4 = (
        marginal_effects['4x4']['no_comm']['delta'] -
        marginal_effects['4x4']['code_law']['delta']
    )
    three_way_contrast = contrast_2x2 - contrast_4x4

    # Bootstrap CI for three-way contrast
    rng = np.random.default_rng(seed)
    scenarios = df['scenario_id'].unique()
    n_scenarios = len(scenarios)

    boot_three_way = []
    for _ in range(n_bootstrap):
        resampled_scenarios = rng.choice(scenarios, size=n_scenarios, replace=True)
        boot_df = df[df['scenario_id'].isin(resampled_scenarios)]

        boot_contrasts_2x2 = []
        boot_contrasts_4x4 = []

        for variant_val in ['2x2', '4x4']:
            for regime_val in ['no_comm', 'code_law']:
                cell_df = boot_df[
                    (boot_df['dataset_size'] == variant_val) &
                    (boot_df['contract_mode'] == regime_val)
                ]

                coop_acc = cell_df[cell_df['prompt_mode'] == 'cooperative']['util_correct'].mean()
                selfish_acc = cell_df[cell_df['prompt_mode'] == 'selfish']['util_correct'].mean()

                delta = coop_acc - selfish_acc if (pd.notna(coop_acc) and pd.notna(selfish_acc)) else 0

                if variant_val == '2x2':
                    boot_contrasts_2x2.append(delta)
                else:
                    boot_contrasts_4x4.append(delta)

        if boot_contrasts_2x2 and boot_contrasts_4x4:
            boot_contrast_2x2 = boot_contrasts_2x2[0] - boot_contrasts_2x2[1]
            boot_contrast_4x4 = boot_contrasts_4x4[0] - boot_contrasts_4x4[1]
            boot_three_way.append(boot_contrast_2x2 - boot_contrast_4x4)

    boot_three_way = np.array(boot_three_way)

    ci_lower = float(np.percentile(boot_three_way, 2.5))
    ci_upper = float(np.percentile(boot_three_way, 97.5))
    p_value = float(np.mean(boot_three_way <= 0))

    return {
        'three_way_p': three_way_p,
        'contrast_2x2': contrast_2x2,
        'contrast_4x4': contrast_4x4,
        'three_way_contrast': three_way_contrast,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'marginal_effects': marginal_effects,
        'n_scenarios': n_scenarios,
    }


# =============================================================================
# Summary Table Generation
# =============================================================================

def generate_rq2_summary_table(
    sample_df: pd.DataFrame,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate summary table for all RQ2 tests.

    Args:
        sample_df: Sample-level DataFrame
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed

    Returns:
        DataFrame with one row per test. Columns include:
            - test, game, variant, regime, estimate, ci_lower, ci_upper, p_value
            - headline_test (bool), corrected_significant (bool)
    """
    results = []

    # Test 1: Substitutability in 2×2
    for game in ['pd', 'sh']:
        try:
            result = test_substitutability_2x2(
                sample_df, game=game, n_bootstrap=n_bootstrap, seed=seed
            )

            # Add main result
            results.append({
                'test': 'Substitutability',
                'game': game,
                'variant': '2x2',
                'regime': 'interaction',
                'estimate': result['contrast_diff'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'p_value': result['interaction_p'],
                'headline_test': False,
                'note': 'ceiling effect' if result['has_ceiling_effect'] else '',
            })

            # Add regime-specific effects
            for regime_val in ['no_comm', 'code_law']:
                regime_results = result['contrast_no_comm'] if regime_val == 'no_comm' else result['contrast_code_law']
                results.append({
                    'test': 'Substitutability (marginal)',
                    'game': game,
                    'variant': '2x2',
                    'regime': regime_val,
                    'estimate': regime_results,
                    'ci_lower': None,
                    'ci_upper': None,
                    'p_value': None,
                    'headline_test': False,
                    'note': '',
                })
        except Exception as e:
            print(f"Warning: Failed substitutability test for {game}: {e}")
            continue

    # Test 2: Persistence in 4×4
    for game in ['pd', 'sh']:
        try:
            result = test_persistence_4x4(
                sample_df, game=game, n_bootstrap=n_bootstrap, seed=seed
            )

            # Add headline result: Δ(code-law)
            results.append({
                'test': 'Persistence',
                'game': game,
                'variant': '4x4',
                'regime': 'code-law',
                'estimate': result['delta_code_law'],
                'ci_lower': result['ci_lower_code_law'],
                'ci_upper': result['ci_upper_code_law'],
                'p_value': result['p_code_law'],
                'headline_test': True,
                'note': '',
            })

            # Add other regimes
            for regime_val in ['no_comm', 'code_nl']:
                results.append({
                    'test': 'Persistence (marginal)',
                    'game': game,
                    'variant': '4x4',
                    'regime': regime_val,
                    'estimate': result['delta_by_regime'][regime_val]['delta'],
                    'ci_lower': result[f'ci_lower_{regime_val}'],
                    'ci_upper': result[f'ci_upper_{regime_val}'],
                    'p_value': result[f'p_{regime_val}'],
                    'headline_test': False,
                    'note': '',
                })
        except Exception as e:
            print(f"Warning: Failed persistence test for {game}: {e}")
            continue

    # Test 3: Three-way interaction
    for game in ['pd', 'sh']:
        try:
            result = test_three_way_interaction(
                sample_df, game=game, n_bootstrap=n_bootstrap, seed=seed
            )

            results.append({
                'test': 'Three-way interaction',
                'game': game,
                'variant': 'both',
                'regime': 'no-comm vs code-law',
                'estimate': result['three_way_contrast'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'p_value': result['p_value'],
                'headline_test': True,
                'note': f"LRT p={result['three_way_p']:.4e}",
            })
        except Exception as e:
            print(f"Warning: Failed three-way test for {game}: {e}")
            continue

    df = pd.DataFrame(results)

    # Apply multiple testing correction to headline tests
    headline_mask = df['headline_test'] == True
    headline_p_values = df[headline_mask]['p_value'].dropna().tolist()

    if headline_p_values:
        from rq1_analysis import holm_bonferroni_correction
        headline_significant = holm_bonferroni_correction(headline_p_values, alpha=0.005)

        df['corrected_significant'] = None
        significant_idx = df[headline_mask].index
        for idx, sig in zip(significant_idx, headline_significant):
            df.loc[idx, 'corrected_significant'] = sig

    return df
