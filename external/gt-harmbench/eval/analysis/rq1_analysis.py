"""RQ1 analysis: Testing the Cooperation Gap.

This module implements statistical tests for RQ1: "Does the Cooperation Gap Manifest?"

Based on *The Case for Moral Agents* paper, which proves that in multi-agent
interactions where contracts cannot fully describe the state space (incontractibility)
and the game has a social-dilemma structure, there is an irreducible welfare loss
called the *cooperation gap* — no contracting mechanism can close it, but prosocial
agents can.

Three Statistical Tests:
1. Irreducibility: Is P(utilitarian-correct) < 1 in (selfish, 4x4, code-law)?
2. Realizability: Does cooperative prompting improve utilitarian accuracy?
3. Structural Specificity: Is 4x4 harder than 2x2 under (selfish, code-law)?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from scipy import stats


# =============================================================================
# Data Preparation
# =============================================================================

def expand_to_sample_level(
    scenario_stats_df: pd.DataFrame,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Expand scenario-level summary stats to synthetic sample-level records.

    Reconstructs per-sample data from aggregated scenario statistics.
    For each scenario with util_accuracy_mean = 0.6 and n_repetitions = 5:
    - Creates 5 rows with util_correct = [1, 1, 1, 0, 0] (approximately)
    - Preserves scenario clustering for bootstrap

    Args:
        scenario_stats_df: DataFrame with columns:
            - model, model_display, game, prompt_mode, dataset_size, contract_mode
            - utilitarian_accuracy_mean, n_repetitions
        rng: Random number generator for reproducibility

    Returns:
        DataFrame with per-sample rows. Columns:
            - model, model_display, game, prompt_mode, dataset_size, contract_mode
            - scenario_id, rep_id, util_correct (0/1)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sample_rows = []

    for _, row in scenario_stats_df.iterrows():
        n_reps = int(row['n_repetitions'])
        util_acc_mean = row['utilitarian_accuracy_mean']

        # Handle NaN values
        if pd.isna(util_acc_mean) or pd.isna(n_reps) or n_reps == 0:
            continue

        n_correct = int(round(util_acc_mean * n_reps))
        n_correct = max(0, min(n_correct, n_reps))  # Clamp to valid range

        # Create scenario_id from available fields
        # Use index or combination of fields to create unique ID
        scenario_base = f"{row['game']}_{row.get('scenario_idx', _)}"

        # Create n_reps rows with n_correct=1 and (n_reps-n_correct)=0
        correct_flags = [1] * n_correct + [0] * (n_reps - n_correct)
        rng.shuffle(correct_flags)  # Random order

        for rep_id, util_correct in enumerate(correct_flags):
            sample_rows.append({
                'model': row['model'],
                'model_display': row['model_display'],
                'game': row['game'],
                'prompt_mode': row['prompt_mode'],
                'dataset_size': row['dataset_size'],
                'contract_mode': row['contract_mode'],
                'scenario_id': scenario_base,
                'rep_id': rep_id,
                'util_correct': util_correct,
            })

    return pd.DataFrame(sample_rows)


# =============================================================================
# Test 1: Irreducibility (Cluster Bootstrap)
# =============================================================================

def test_irreducibility(
    sample_df: pd.DataFrame,
    game: str = "pd",
    dataset_size: str = "4x4",
    prompt_mode: str = "selfish",
    contract_mode: str = "code_law",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    """Test if P(util-correct) < 1 in a single cell (cooperation gap exists).

    Quantity: Γ̂ = 1 - P(util-correct)
    Test: One-sided 95% lower bound on Γ̂ via scenario-clustered bootstrap

    This tests the Irreducibility claim: when the contracting language cannot
    distinguish all welfare-relevant states (incontractibility), no mechanism
    in the language can close the gap.

    Args:
        sample_df: Sample-level DataFrame from expand_to_sample_level()
        game: "pd" or "sh"
        dataset_size: "4x4" or "2x2"
        prompt_mode: "selfish"
        contract_mode: "code_law" or "code_nl"
        n_bootstrap: Number of bootstrap resamples
        ci: Confidence level (default 0.95 for 95% CI)
        seed: Random seed

    Returns:
        Dict with keys:
            - point_estimate: Γ̂
            - ci_lower: One-sided lower bound
            - ci_upper: Upper bound
            - p_value: Proportion of bootstrap estimates <= 0
            - n_scenarios: Number of scenarios
            - n_samples: Total number of samples
    """
    # Filter to target cell
    cell_df = sample_df[
        (sample_df['game'] == game) &
        (sample_df['dataset_size'] == dataset_size) &
        (sample_df['prompt_mode'] == prompt_mode) &
        (sample_df['contract_mode'] == contract_mode)
    ].copy()

    if cell_df.empty:
        raise ValueError(f"No data for cell ({game}, {dataset_size}, {prompt_mode}, {contract_mode})")

    # Point estimate
    point_estimate = 1 - cell_df['util_correct'].mean()

    # Scenario-clustered bootstrap
    rng = np.random.default_rng(seed)
    scenarios = cell_df['scenario_id'].unique()
    n_scenarios = len(scenarios)

    if n_scenarios == 0:
        raise ValueError(f"No scenarios found in cell")

    boot_estimates = []
    for _ in range(n_bootstrap):
        # Resample scenarios with replacement
        resampled_scenarios = rng.choice(scenarios, size=n_scenarios, replace=True)
        resampled_df = cell_df[cell_df['scenario_id'].isin(resampled_scenarios)]

        # Compute Γ̂ on resampled data
        boot_gamma = 1 - resampled_df['util_correct'].mean()
        boot_estimates.append(boot_gamma)

    boot_estimates = np.array(boot_estimates)

    # Percentile CI (one-sided: lower bound only)
    alpha = 1 - ci
    ci_lower = float(np.percentile(boot_estimates, alpha * 100))
    ci_upper = float(np.percentile(boot_estimates, (1 - alpha) * 100))

    # P-value: proportion of bootstrap estimates <= 0
    p_value = float(np.mean(boot_estimates <= 0))

    return {
        'point_estimate': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'n_scenarios': n_scenarios,
        'n_samples': len(cell_df),
    }


# =============================================================================
# Test 2: Realizability (GLMM + Bootstrap CI)
# =============================================================================

def fit_realizability_glmm(
    sample_df: pd.DataFrame,
    dataset_size: str = "4x4",
) -> Dict:
    """Fit logistic mixed-effects model for realizability test.

    Model: util_correct ~ prosociality * regime + (1 | model) + (1 | scenario)

    This tests the Realizability claim: sufficiently prosocial agents play the
    social optimum, closing the gap that contracts cannot.

    Args:
        sample_df: Sample-level DataFrame
        dataset_size: "4x4" or "2x2"

    Returns:
        Dict with keys:
            - model_result: Fitted model object
            - marginal_probs: Dict of marginal probabilities for each (regime, prosociality)
            - differences: Dict of differences (cooperative - selfish) for each regime
            - n_samples: Number of samples used
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError(
            "statsmodels is required for GLMM. "
            "Install with: pip install statsmodels"
        )

    # Filter to dataset_size and relevant regimes/prosociality levels
    df = sample_df[
        (sample_df['dataset_size'] == dataset_size) &
        (sample_df['prompt_mode'].isin(['cooperative', 'selfish'])) &
        (sample_df['contract_mode'].isin(['no_comm', 'code_nl', 'code_law']))
    ].copy()

    if df.empty:
        raise ValueError(f"No data for realizability test with dataset_size={dataset_size}")

    # Encode variables
    df['prosociality'] = df['prompt_mode'].map({'selfish': 0, 'cooperative': 1})
    df['regime'] = pd.Categorical(
        df['contract_mode'],
        categories=['no_comm', 'code_nl', 'code_law']
    )

    # Fit model (try mixed effects, fall back to fixed effects)
    formula = "util_correct ~ prosociality * C(regime)"

    try:
        # Try with model random effect
        model = smf.mixedlm(
            formula,
            df,
            groups=df['model'],
            re_formula="1",
        )
        result = model.fit(reml=False, disp=0)
    except Exception:
        # Fallback: fixed-effects logistic regression
        result = smf.logit(formula, data=df).fit(disp=0)

    # Compute marginal probabilities for each regime
    marginal_probs = {}
    for regime_val in ['no_comm', 'code_nl', 'code_law']:
        for prosociality_val in [0, 1]:
            # Create prediction data
            pred_df = pd.DataFrame({
                'prosociality': [prosociality_val],
                'regime': [regime_val],
            })

            pred = result.predict(pred_df)
            key = f"{regime_val}_{'cooperative' if prosociality_val else 'selfish'}"
            marginal_probs[key] = float(pred.iloc[0])

    # Compute differences (cooperative - selfish) within each regime
    differences = {}
    for regime_val in ['no_comm', 'code_nl', 'code_law']:
        coop_key = f"{regime_val}_cooperative"
        selfish_key = f"{regime_val}_selfish"
        differences[regime_val] = marginal_probs[coop_key] - marginal_probs[selfish_key]

    return {
        'model_result': result,
        'marginal_probs': marginal_probs,
        'differences': differences,
        'n_samples': len(df),
    }


def bootstrap_realizability_differences(
    sample_df: pd.DataFrame,
    dataset_size: str = "4x4",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict:
    """Bootstrap CIs for realizability differences.

    Args:
        sample_df: Sample-level DataFrame
        dataset_size: "4x4" or "2x2"
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed

    Returns:
        Dict with keys for each regime:
            - mean: Mean difference
            - ci_lower: 95% CI lower bound
            - ci_upper: 95% CI upper bound
            - p_value: One-sided p-value (proportion <= 0)
    """
    rng = np.random.default_rng(seed)

    # Get unique scenarios for clustering
    df = sample_df[
        (sample_df['dataset_size'] == dataset_size) &
        (sample_df['prompt_mode'].isin(['cooperative', 'selfish']))
    ].copy()

    if df.empty:
        raise ValueError(f"No data for realizability bootstrap with dataset_size={dataset_size}")

    scenarios = df['scenario_id'].unique()
    n_scenarios = len(scenarios)

    # Storage for bootstrap differences
    boot_diffs = {regime: [] for regime in ['no_comm', 'code_nl', 'code_law']}

    for _ in range(n_bootstrap):
        # Resample scenarios
        resampled_scenarios = rng.choice(scenarios, size=n_scenarios, replace=True)
        boot_df = df[df['scenario_id'].isin(resampled_scenarios)]

        # Compute differences in this bootstrap sample
        for regime_val in ['no_comm', 'code_nl', 'code_law']:
            regime_df = boot_df[boot_df['contract_mode'] == regime_val]

            if regime_df.empty:
                continue

            coop_acc = regime_df[regime_df['prompt_mode'] == 'cooperative']['util_correct'].mean()
            selfish_acc = regime_df[regime_df['prompt_mode'] == 'selfish']['util_correct'].mean()

            # Handle NaN
            if pd.isna(coop_acc) or pd.isna(selfish_acc):
                continue

            boot_diffs[regime_val].append(coop_acc - selfish_acc)

    # Compute CIs
    results = {}
    for regime_val in ['no_comm', 'code_nl', 'code_law']:
        diffs = np.array(boot_diffs[regime_val])
        if len(diffs) == 0:
            results[regime_val] = {
                'mean': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'p_value': np.nan,
            }
            continue

        results[regime_val] = {
            'mean': float(np.mean(diffs)),
            'ci_lower': float(np.percentile(diffs, 2.5)),
            'ci_upper': float(np.percentile(diffs, 97.5)),
            'p_value': float(np.mean(diffs <= 0)),  # One-sided test
        }

    return results


# =============================================================================
# Test 3: Structural Specificity (Paired Bootstrap)
# =============================================================================

def test_structural_specificity(
    sample_df: pd.DataFrame,
    prompt_mode: str = "selfish",
    contract_mode: str = "code_law",
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """Test if 4x4 is harder than 2x2 under selfish/code-law.

    Quantity: Δ̂ = P(util-correct|2x2) - P(util-correct|4x4)
    Test: One-sided test that Δ̂ > 0 via paired bootstrap (resample scenarios)

    This tests the Structural Specificity claim: the gap is a feature of
    incontractibility specifically, not a generic capability failure. In a
    fully contractible variant (2x2), sufficiently strong contracts should
    close the gap.

    Args:
        sample_df: Sample-level DataFrame
        prompt_mode: "selfish"
        contract_mode: "code_law"
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed

    Returns:
        Dict with keys:
            - point_estimate: Δ̂
            - ci_lower: One-sided 90% lower bound
            - p_value: One-sided p-value
            - n_scenarios: Number of scenarios
    """
    # Filter to target cells
    df_2x2 = sample_df[
        (sample_df['dataset_size'] == '2x2') &
        (sample_df['prompt_mode'] == prompt_mode) &
        (sample_df['contract_mode'] == contract_mode)
    ].copy()

    df_4x4 = sample_df[
        (sample_df['dataset_size'] == '4x4') &
        (sample_df['prompt_mode'] == prompt_mode) &
        (sample_df['contract_mode'] == contract_mode)
    ].copy()

    if df_2x2.empty or df_4x4.empty:
        raise ValueError(f"No data for structural specificity test")

    # Point estimate
    point_estimate = df_2x2['util_correct'].mean() - df_4x4['util_correct'].mean()

    # Get common scenarios (matched by scenario_id prefix)
    scenarios_2x2 = set(df_2x2['scenario_id'].apply(lambda x: x.split('_')[0] if '_' in str(x) else str(x)))
    scenarios_4x4 = set(df_4x4['scenario_id'].apply(lambda x: x.split('_')[0] if '_' in str(x) else str(x)))
    common_scenarios = list(scenarios_2x2 & scenarios_4x4)

    if not common_scenarios:
        # Fallback: use all scenarios (unpaired)
        common_scenarios = list(set(df_2x2['scenario_id']) | set(df_4x4['scenario_id']))

    # Bootstrap by resampling scenarios
    rng = np.random.default_rng(seed)
    n_scenarios = len(common_scenarios)

    boot_diffs = []
    for _ in range(n_bootstrap):
        # Resample scenarios
        resampled_scenarios = rng.choice(common_scenarios, size=n_scenarios, replace=True)

        # Match scenarios in both datasets
        mask_2x2 = df_2x2['scenario_id'].apply(
            lambda x: any(str(s) in str(x) for s in resampled_scenarios)
        )
        mask_4x4 = df_4x4['scenario_id'].apply(
            lambda x: any(str(s) in str(x) for s in resampled_scenarios)
        )

        acc_2x2 = df_2x2[mask_2x2]['util_correct'].mean()
        acc_4x4 = df_4x4[mask_4x4]['util_correct'].mean()

        if pd.notna(acc_2x2) and pd.notna(acc_4x4):
            boot_diffs.append(acc_2x2 - acc_4x4)

    boot_diffs = np.array(boot_diffs)

    # One-sided CI and p-value
    ci_lower = float(np.percentile(boot_diffs, 5.0))  # 90% one-sided CI
    p_value = float(np.mean(boot_diffs <= 0))

    return {
        'point_estimate': point_estimate,
        'ci_lower': ci_lower,
        'p_value': p_value,
        'n_scenarios': n_scenarios,
    }


# =============================================================================
# Multiple Testing Correction
# =============================================================================

def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> List[bool]:
    """Apply Holm-Bonferroni correction for multiple testing.

    Args:
        p_values: List of p-values
        alpha: Family-wise error rate

    Returns:
        List of booleans indicating which tests are significant after correction
    """
    if not p_values:
        return []

    # Sort p-values with indices
    sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])

    # Apply Holm procedure
    rejected = [False] * len(p_values)
    for rank, (idx, p) in enumerate(sorted_p):
        threshold = alpha / (len(p_values) - rank)
        if p < threshold:
            rejected[idx] = True
        else:
            break  # Stop at first non-rejection

    return rejected


# =============================================================================
# Summary Table Generation
# =============================================================================

def generate_rq1_summary_table(
    sample_df: pd.DataFrame,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate summary table for all RQ1 tests.

    Args:
        sample_df: Sample-level DataFrame
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed

    Returns:
        DataFrame with one row per test. Columns:
            - test, game, cell, estimate, ci_lower, ci_upper, p_value, n_scenarios
            - headline_test (bool), corrected_significant (bool)
    """
    results = []

    # Test 1: Irreducibility (primary and supporting)
    for game in ['pd', 'sh']:
        # Primary: (selfish, 4x4, code_law)
        result = test_irreducibility(
            sample_df, game=game, dataset_size='4x4',
            prompt_mode='selfish', contract_mode='code_law',
            n_bootstrap=n_bootstrap, seed=seed
        )
        results.append({
            'test': 'Irreducibility',
            'game': game,
            'cell': '(selfish, 4×4, code-law)',
            'estimate': result['point_estimate'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'p_value': result['p_value'],
            'n_scenarios': result['n_scenarios'],
            'headline_test': True,
        })

        # Supporting: (selfish, 4x4, code_nl)
        result_nl = test_irreducibility(
            sample_df, game=game, dataset_size='4x4',
            prompt_mode='selfish', contract_mode='code_nl',
            n_bootstrap=n_bootstrap, seed=seed
        )
        results.append({
            'test': 'Irreducibility (supporting)',
            'game': game,
            'cell': '(selfish, 4×4, NL)',
            'estimate': result_nl['point_estimate'],
            'ci_lower': result_nl['ci_lower'],
            'ci_upper': result_nl['ci_upper'],
            'p_value': result_nl['p_value'],
            'n_scenarios': result_nl['n_scenarios'],
            'headline_test': False,
        })

    # Test 2: Realizability (GLMM + bootstrap)
    for game in ['pd', 'sh']:
        game_df = sample_df[sample_df['game'] == game]

        for dataset_size in ['4x4', '2x2']:
            try:
                # Fit GLMM
                glmm_result = fit_realizability_glmm(game_df, dataset_size)

                # Bootstrap CIs
                boot_result = bootstrap_realizability_differences(
                    game_df, dataset_size, n_bootstrap=n_bootstrap, seed=seed
                )

                for regime_val in ['no_comm', 'code_nl', 'code_law']:
                    diff = glmm_result['differences'][regime_val]
                    boot = boot_result[regime_val]

                    # Headline tests: code-law in 4x4
                    is_headline = (dataset_size == '4x4' and regime_val == 'code_law')

                    results.append({
                        'test': 'Realizability',
                        'game': game,
                        'cell': f'{dataset_size}, {regime_val}',
                        'estimate': diff,
                        'ci_lower': boot['ci_lower'],
                        'ci_upper': boot['ci_upper'],
                        'p_value': boot['p_value'],
                        'n_scenarios': glmm_result['n_samples'],
                        'headline_test': is_headline,
                    })
            except Exception as e:
                print(f"Warning: Failed to compute realizability for {game} {dataset_size}: {e}")
                continue

    # Test 3: Structural specificity
    for game in ['pd', 'sh']:
        game_df = sample_df[sample_df['game'] == game]
        result = test_structural_specificity(
            game_df, prompt_mode='selfish', contract_mode='code_law',
            n_bootstrap=n_bootstrap, seed=seed
        )
        results.append({
            'test': 'Structural specificity',
            'game': game,
            'cell': '(selfish, code-law)',
            'estimate': result['point_estimate'],
            'ci_lower': result['ci_lower'],
            'ci_upper': None,  # One-sided CI
            'p_value': result['p_value'],
            'n_scenarios': result['n_scenarios'],
            'headline_test': True,
        })

    df = pd.DataFrame(results)

    # Apply multiple testing correction to headline tests
    headline_mask = df['headline_test'] == True
    headline_p_values = df[headline_mask]['p_value'].tolist()

    if headline_p_values:
        headline_significant = holm_bonferroni_correction(headline_p_values, alpha=0.005)

        df['corrected_significant'] = None
        df.loc[headline_mask, 'corrected_significant'] = headline_significant

    return df
