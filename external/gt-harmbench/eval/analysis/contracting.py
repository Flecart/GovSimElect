"""Shared analysis utilities for GT-HarmBench contracting experiments.

This module provides functions for:
- Loading contracting logs from multiple experiments
- Computing metrics for formation, activation, and welfare
- Testing complementarity between prosocial preferences and contracting
- Detecting Greenwashing behavior in 4x4 moral hazard scenarios
- Extracting negotiation transcripts and reasoning

Functions are shared between:
- Bash scripts (analyze_contracting_results.sh)
- Jupyter notebooks (notebooks/contracting_analysis.ipynb)
"""

from __future__ import annotations

import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.metrics import WelfareBounds, compute_welfare_bounds

CONTRACTING_SCORER_KEYS = (
    "contracting_scorer",
    "contracting_scorer_pd",
    "contracting_scorer_sh",
)


def get_contracting_score(scores: Dict[str, Any]) -> Dict[str, Any]:
    """Return contracting score data regardless of game-specific scorer name."""
    if not isinstance(scores, dict):
        return {}

    for key in CONTRACTING_SCORER_KEYS:
        score = scores.get(key)
        if isinstance(score, dict):
            return score

    for key in sorted(scores):
        if key.startswith("contracting_scorer") and isinstance(scores[key], dict):
            return scores[key]

    return {}


def get_contracting_score_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """Return scorer metadata from an inspect summary/sample item."""
    score = get_contracting_score(item.get("scores", {}))
    metadata = score.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def load_contracting_logs(
    log_base_dir: Path,
    experiments: tuple[str, ...] = (
        "4x4-no-comm", "4x4-code-nl", "4x4-code-law",
        "2x2-no-comm", "2x2-code-nl", "2x2-code-law",
    ),
    prompt_modes: tuple[str, ...] = ("base",),
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load contracting logs from multiple experiments.

    Args:
        log_base_dir: Base directory containing experiment subdirectories
        experiments: Tuple of experiment names (e.g., "4x4-code-nl")
        prompt_modes: Tuple of prompt modes to load (e.g., "base", "selfish", "cooperative")
        verbose: Whether to print loading progress

    Returns:
        Dictionary mapping experiment names to DataFrames with sample-level data
    """
    log_base_dir = Path(log_base_dir)
    model_data: Dict[str, pd.DataFrame] = {}

    for prompt_mode in prompt_modes:
        for exp in experiments:
            # Construct path for this experiment
            # New folder structure: log_base_dir/{prompt_mode}/{exp}
            exp_path = log_base_dir / prompt_mode / exp

            if not exp_path.exists():
                if verbose:
                    print(f"Skipping {exp_path} (not found)")
                continue

            # Find .eval files
            eval_files = sorted(exp_path.glob("*.eval"))

            if not eval_files:
                if verbose:
                    print(f"No .eval files in {exp_path}")
                continue

            # Extract data from each .eval file
            samples = []
            for eval_file in eval_files:
                try:
                    extracted = _extract_eval_samples(eval_file)
                    if extracted:
                        samples.extend(extracted)
                except Exception as e:
                    if verbose:
                        print(f"Error loading {eval_file.name}: {e}", file=sys.stderr)
                    continue

            if not samples:
                if verbose:
                    print(f"No samples found in {exp_path}")
                continue

            df = pd.DataFrame(samples)

            # Extract task-level metadata from first .eval file
            task_meta = {}
            if eval_files:
                task_meta = _extract_task_metadata(eval_files[0])

            # Add experiment metadata
            df['experiment'] = exp
            df['prompt_mode'] = prompt_mode
            df['dataset'] = '4x4' if '4x4' in exp else '2x2'

            # Extract mode label
            df['mode'] = (
                'no_comm' if 'no-comm' in exp or 'no_comm' in exp else
                'code_nl' if 'code-n' in exp or 'code_nl' in exp else
                'code_law'
            )

            # Add task-level metadata as columns (if available)
            # Core experiment variables
            if 'model_name' in task_meta:
                df['model_name'] = task_meta['model_name']
            if 'negotiation_mode' in task_meta:
                df['negotiation_mode'] = task_meta['negotiation_mode']
            if 'num_stories' in task_meta:
                df['num_stories'] = task_meta['num_stories']
            if 'num_times' in task_meta:
                df['num_times'] = task_meta['num_times']

            # Additional recommended fields
            if 'experiment_id' in task_meta:
                df['experiment_id'] = task_meta['experiment_id']
            if 'dataset_name' in task_meta:
                df['dataset_name'] = task_meta['dataset_name']
            if 'sample_limit' in task_meta:
                df['sample_limit'] = task_meta['sample_limit']
            if 'execution_timestamp' in task_meta:
                df['execution_timestamp'] = task_meta['execution_timestamp']
            if 'git_commit_hash' in task_meta:
                df['git_commit_hash'] = task_meta['git_commit_hash']

            # Use experiment name as key
            key = f"{exp}-{prompt_mode}" if prompt_mode != "base" else exp

            if verbose:
                print(f"Loaded {len(df)} samples from {key}")

            model_data[key] = df

    return model_data


def _extract_task_metadata(eval_file: Path) -> Dict[str, Any]:
    """Extract task-level metadata from header.json in .eval file.

    Args:
        eval_file: Path to .eval file

    Returns:
        Dictionary with task metadata (model_name, experiment_id, etc.)
    """
    try:
        with zipfile.ZipFile(eval_file, 'r') as zf:
            # Read header.json which contains task metadata
            if 'header.json' in zf.namelist():
                with zf.open('header.json') as f:
                    header_data = json.load(f)
                    # Metadata is nested under 'eval' key in inspect_ai v2 format
                    return header_data.get('eval', {}).get('metadata', {})
    except Exception:
        pass
    return {}


def _parse_payoff_matrix_from_metadata(input_meta: Dict[str, Any]) -> list | None:
    """Parse payoff matrix from sample metadata.

    Handles both 4x4 (payoff_matrix_4x4 JSON string) and 2x2
    (1_1_payoff etc.) formats. Returns a list-of-lists matrix
    where matrix[i][j] = [row_payoff, col_payoff], or None on failure.
    """
    # 4x4 format
    pm4_raw = input_meta.get("payoff_matrix_4x4")
    if pm4_raw:
        try:
            matrix = json.loads(pm4_raw) if isinstance(pm4_raw, str) else pm4_raw
            # Validate shape: should be NxN with each cell being a 2-element list
            if (
                isinstance(matrix, list)
                and len(matrix) > 0
                and isinstance(matrix[0], list)
                and isinstance(matrix[0][0], (list, tuple))
                and len(matrix[0][0]) >= 2
            ):
                return matrix
        except (json.JSONDecodeError, TypeError, IndexError):
            pass

    # 2x2 format: reconstruct from 1_1_payoff, etc.
    keys = ["1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff"]
    if all(k in input_meta for k in keys):
        try:
            def _parse_pair(val: Any) -> list[float]:
                if isinstance(val, str):
                    # "(3, -3)" or "[3, -3]"
                    cleaned = val.strip("()[] ")
                    parts = cleaned.split(",")
                    return [float(p.strip()) for p in parts]
                elif isinstance(val, (list, tuple)):
                    return [float(v) for v in val]
                return [0.0, 0.0]

            return [
                [_parse_pair(input_meta["1_1_payoff"]), _parse_pair(input_meta["1_2_payoff"])],
                [_parse_pair(input_meta["2_1_payoff"]), _parse_pair(input_meta["2_2_payoff"])],
            ]
        except (ValueError, TypeError, IndexError):
            pass

    return None


def _extract_eval_samples(eval_file: Path) -> List[Dict[str, Any]]:
    """Extract sample-level data from a .eval file.

    Each sample dict includes welfare_bounds computed from the payoff
    matrix, enabling per-scenario normalization.

    Args:
        eval_file: Path to .eval file

    Returns:
        List of sample dictionaries with metadata, scores, and welfare bounds
    """
    samples = []

    try:
        with zipfile.ZipFile(eval_file, 'r') as zf:
            summary_files = sorted([name for name in zf.namelist() if name.startswith('_journal/summaries/')])

            for summary_file in summary_files:
                with zf.open(summary_file) as f:
                    data = json.load(f)
                    items = data if isinstance(data, list) else [data]

                    for item in items:
                        # Extract input metadata
                        input_meta = item.get('metadata', {})
                        score_meta = get_contracting_score_metadata(item)

                        # Combine into sample dict
                        sample = {
                            'sample_id': input_meta.get('id', ''),
                            'base_scenario_id': input_meta.get('base_scenario_id', input_meta.get('id', '')),
                            'input_metadata': input_meta,
                            'score_metadata': score_meta,
                            'trace_usage': input_meta.get('trace_usage', {}),
                        }
                        sample.update(_flatten_trace_usage(input_meta.get('trace_usage', {})))

                        # Add all score metadata as top-level fields
                        sample.update(score_meta)

                        # Compute per-scenario welfare bounds from payoff matrix
                        matrix = _parse_payoff_matrix_from_metadata(input_meta)
                        if matrix is not None:
                            bounds = compute_welfare_bounds(matrix)
                            sample['welfare_bounds'] = bounds
                        else:
                            sample['welfare_bounds'] = None

                        samples.append(sample)

    except Exception as e:
        print(f"Error extracting from {eval_file.name}: {e}", file=sys.stderr)

    return samples


def _flatten_trace_usage(trace_usage: Any) -> Dict[str, Any]:
    """Flatten persisted trace usage into DataFrame-friendly columns."""
    if not isinstance(trace_usage, dict):
        return {}

    result: Dict[str, Any] = {}
    sections = {"trace": trace_usage.get("total", {})}
    by_phase = trace_usage.get("by_phase", {})
    if isinstance(by_phase, dict):
        for phase in ("negotiation", "decision", "coding"):
            sections[phase] = by_phase.get(phase, {})

    for prefix, values in sections.items():
        if not isinstance(values, dict):
            continue
        for field in (
            "call_count",
            "elapsed_seconds",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
            "calls_with_usage",
            "errors",
        ):
            result[f"{prefix}_{field}"] = values.get(field, 0)

    return result


def compute_contracting_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute contracting metrics from a DataFrame.

    Includes per-scenario normalized welfare (normalize each sample against
    its own payoff matrix bounds, then average). This produces robust
    cross-scenario comparisons even when payoff scales diverge.

    Args:
        df: DataFrame from load_contracting_logs()

    Returns:
        Dictionary with all contracting metrics including normalized welfare
    """
    if df.empty:
        return {
            'total_samples': 0,
            'contract_formation_rate': 0.0,
            'contract_activation_rate': None,
            'avg_utilitarian_payoff': 0.0,
            'avg_rawlsian_payoff': 0.0,
            'avg_norm_utilitarian': None,
            'avg_norm_rawlsian': None,
            'avg_nash_deviation_util': None,
            'avg_nash_deviation_rawls': None,
            'nash_accuracy': 0.0,
            'utilitarian_accuracy': 0.0,
            'rawlsian_accuracy': 0.0,
            'formation_failure_reasons': {},
            'compliance_failure_reasons': {},
        }

    total = len(df)

    # Basic metrics
    contract_formed = df.get('contract_formed', pd.Series([False] * total)).sum()
    activation_series = df.get('contract_complied', pd.Series([None] * total))
    activation_defined = activation_series.dropna()
    contract_activated = (activation_defined == False).sum()

    # Formation failure reasons
    formation_reasons = df.get('formation_failure_reason', pd.Series([None] * total)).dropna()
    formation_reason_counts = formation_reasons.value_counts().to_dict() if not formation_reasons.empty else {}

    # Compliance failure reasons
    compliance_reasons = df.get('compliance_failure_reason', pd.Series([None] * total)).dropna()
    compliance_reason_counts = compliance_reasons.value_counts().to_dict() if not compliance_reasons.empty else {}

    # Raw payoffs
    avg_util_payoff = df.get('utilitarian_payoff', pd.Series([0] * total)).mean()
    avg_rawls_payoff = df.get('rawlsian_payoff', pd.Series([0] * total)).mean()

    # Variance of payoffs
    var_util_payoff = df.get('utilitarian_payoff', pd.Series([0] * total)).var()
    var_rawls_payoff = df.get('rawlsian_payoff', pd.Series([0] * total)).var()

    # Per-scenario normalized welfare: normalize each sample against its own
    # payoff matrix range, then average. Falls back to None for samples
    # without welfare_bounds (e.g. if payoff matrix was missing).
    norm_utils: list[float] = []
    norm_rawls: list[float] = []
    nash_dev_utils: list[float] = []
    nash_dev_rawls: list[float] = []

    util_col = df.get('utilitarian_payoff', pd.Series([None] * total))
    rawls_col = df.get('rawlsian_payoff', pd.Series([None] * total))
    bounds_col = df.get('welfare_bounds', pd.Series([None] * total))

    for idx in range(total):
        bounds = bounds_col.iloc[idx] if idx < len(bounds_col) else None
        util_val = util_col.iloc[idx] if idx < len(util_col) else None
        rawls_val = rawls_col.iloc[idx] if idx < len(rawls_col) else None

        if bounds is None or util_val is None or rawls_val is None:
            continue

        norm_util = normalize_welfare(util_val, bounds, 'utilitarian')
        norm_rawls_val = normalize_welfare(rawls_val, bounds, 'rawlsian')
        norm_utils.append(norm_util)
        norm_rawls.append(norm_rawls_val)

        nd_util = nash_deviation(util_val, bounds, 'utilitarian')
        nd_rawls = nash_deviation(rawls_val, bounds, 'rawlsian')
        if nd_util is not None:
            nash_dev_utils.append(nd_util)
        if nd_rawls is not None:
            nash_dev_rawls.append(nd_rawls)

    avg_norm_util = sum(norm_utils) / len(norm_utils) if norm_utils else None
    avg_norm_rawls = sum(norm_rawls) / len(norm_rawls) if norm_rawls else None
    avg_nd_util = sum(nash_dev_utils) / len(nash_dev_utils) if nash_dev_utils else None
    avg_nd_rawls = sum(nash_dev_rawls) / len(nash_dev_rawls) if nash_dev_rawls else None

    # Effort level distribution (4x4 only)
    effort_distribution = {}
    if 'row_effort_level' in df.columns:
        effort_dist = df.get('row_effort_level', pd.Series([None] * total)).value_counts()
        effort_distribution = effort_dist.to_dict()

    usage_metrics = _aggregate_usage_columns(df)

    # Nash/utilitarian/rawlsian accuracy (percentage of samples with targets where equilibrium was achieved)
    # Filter to samples with targets (answer_type != 'neutral' means targets exist)
    samples_with_targets = df[df.get('answer_type', 'neutral') != 'neutral']
    samples_with_targets_count = len(samples_with_targets)

    if samples_with_targets_count > 0:
        nash_count = samples_with_targets.get('is_nash', pd.Series([False] * samples_with_targets_count)).sum()
        utilitarian_count = samples_with_targets.get('is_utilitarian', pd.Series([False] * samples_with_targets_count)).sum()
        rawlsian_count = samples_with_targets.get('is_rawlsian', pd.Series([False] * samples_with_targets_count)).sum()

        nash_accuracy = float(nash_count / samples_with_targets_count) if samples_with_targets_count > 0 else 0.0
        utilitarian_accuracy = float(utilitarian_count / samples_with_targets_count) if samples_with_targets_count > 0 else 0.0
        rawlsian_accuracy = float(rawlsian_count / samples_with_targets_count) if samples_with_targets_count > 0 else 0.0
    else:
        nash_accuracy = 0.0
        utilitarian_accuracy = 0.0
        rawlsian_accuracy = 0.0

    return {
        'total_samples': int(total),
        'contracts_formed': int(contract_formed),
        'contract_formation_rate': float(contract_formed / total) if total > 0 else 0.0,
        'contracts_activated': int(contract_activated),
        'contract_activation_rate': (
            float(contract_activated / len(activation_defined))
            if len(activation_defined) > 0
            else None
        ),
        'avg_utilitarian_payoff': float(avg_util_payoff),
        'avg_rawlsian_payoff': float(avg_rawls_payoff),
        'avg_norm_utilitarian': avg_norm_util,
        'avg_norm_rawlsian': avg_norm_rawls,
        'avg_nash_deviation_util': avg_nd_util,
        'avg_nash_deviation_rawls': avg_nd_rawls,
        'utilitarian_payoff_variance': float(var_util_payoff) if not pd.isna(var_util_payoff) else 0.0,
        'rawlsian_payoff_variance': float(var_rawls_payoff) if not pd.isna(var_rawls_payoff) else 0.0,
        'nash_accuracy': nash_accuracy,
        'utilitarian_accuracy': utilitarian_accuracy,
        'rawlsian_accuracy': rawlsian_accuracy,
        'formation_failure_reasons': formation_reason_counts,
        'compliance_failure_reasons': compliance_reason_counts,
        'effort_distribution': effort_distribution,
        **usage_metrics,
    }


def _aggregate_usage_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """Sum persisted usage columns if they are present in a sample DataFrame."""
    fields = (
        "trace_call_count",
        "trace_elapsed_seconds",
        "trace_input_tokens",
        "trace_output_tokens",
        "trace_total_tokens",
        "trace_reasoning_tokens",
        "trace_calls_with_usage",
        "trace_errors",
        "negotiation_call_count",
        "negotiation_total_tokens",
        "negotiation_elapsed_seconds",
        "decision_call_count",
        "decision_total_tokens",
        "decision_elapsed_seconds",
        "coding_call_count",
        "coding_total_tokens",
        "coding_elapsed_seconds",
    )
    result: Dict[str, Any] = {}
    for field in fields:
        if field in df.columns:
            value = df[field].fillna(0).sum()
            result[field] = float(value) if "seconds" in field else int(value)
    return result


def aggregate_experiment_metrics(
    log_base_dir: Path,
    prompt_modes: tuple[str, ...] = ("base",),
) -> pd.DataFrame:
    """Aggregate metrics across all experiments.

    Args:
        log_base_dir: Base directory containing experiment subdirectories
        prompt_modes: Prompt modes to include

    Returns:
        DataFrame with one row per experiment and all metrics
    """
    # Load all experiments
    data = load_contracting_logs(log_base_dir, prompt_modes=prompt_modes, verbose=False)

    if not data:
        return pd.DataFrame()

    # Compute metrics for each experiment
    results = []
    for exp_name, df in data.items():
        metrics = compute_contracting_metrics(df)
        metrics['experiment'] = exp_name
        metrics['dataset'] = df['dataset'].iloc[0] if not df.empty else 'unknown'
        metrics['mode'] = df['mode'].iloc[0] if not df.empty else 'unknown'
        metrics['prompt_mode'] = df['prompt_mode'].iloc[0] if not df.empty else 'base'

        # Add new metadata fields if available
        if 'model_name' in df.columns:
            metrics['model_name'] = df['model_name'].iloc[0]
        if 'experiment_id' in df.columns:
            metrics['experiment_id'] = df['experiment_id'].iloc[0]
        if 'num_stories' in df.columns:
            metrics['num_stories'] = df['num_stories'].iloc[0]
        if 'num_times' in df.columns:
            metrics['num_times'] = df['num_times'].iloc[0]
        if 'execution_timestamp' in df.columns:
            metrics['execution_timestamp'] = df['execution_timestamp'].iloc[0]
        if 'git_commit_hash' in df.columns:
            metrics['git_commit_hash'] = df['git_commit_hash'].iloc[0]

        results.append(metrics)

    summary_df = pd.DataFrame(results)

    # Reorder columns for readability
    column_order = [
        # Core experiment metadata
        'experiment', 'dataset', 'mode', 'prompt_mode',
        'model_name', 'experiment_id', 'num_stories', 'num_times',
        # Metrics
        'total_samples', 'contracts_formed', 'contract_formation_rate',
        'contracts_activated', 'contract_activation_rate',
        'nash_accuracy', 'utilitarian_accuracy', 'rawlsian_accuracy',
        # Raw welfare
        'avg_utilitarian_payoff', 'avg_rawlsian_payoff',
        # Per-scenario normalized welfare
        'avg_norm_utilitarian', 'avg_norm_rawlsian',
        'avg_nash_deviation_util', 'avg_nash_deviation_rawls',
        'utilitarian_payoff_variance', 'rawlsian_payoff_variance',
        # Additional metadata (will be added below if present)
        'execution_timestamp', 'git_commit_hash',
    ]

    # Add any remaining columns
    for col in summary_df.columns:
        if col not in column_order:
            column_order.append(col)

    # Ensure all columns exist
    for col in column_order:
        if col not in summary_df.columns:
            summary_df[col] = None

    return summary_df[column_order]


def compute_interaction_effects(
    metrics_df: pd.DataFrame,
    outcome_col: str = 'avg_utilitarian_payoff',
) -> Dict[str, float]:
    """Compute superadditive interaction effects for complementarity test.

    Interaction effect formula:
    (cooperative+contract) - (cooperative alone) - (contract alone) + (baseline)

    A positive interaction indicates complementarity (the whole > sum of parts).

    Args:
        metrics_df: DataFrame from aggregate_experiment_metrics()
        outcome_col: Column to use as outcome measure

    Returns:
        Dict with interaction effects for 4x4 and 2x2 datasets
    """
    results = {}

    for dataset in ['4x4', '2x2']:
        subset = metrics_df[metrics_df['dataset'] == dataset]

        if subset.empty:
            results[dataset] = 0.0
            continue

        # Get values for each condition
        # We need: baseline (no_comm, base), contract (no_comm + contract modes, base),
        # cooperative (no_comm, cooperative), cooperative+contract (contract modes, cooperative)

        # For this implementation, we'll compute a simplified interaction
        # comparing contract modes with/without cooperative prompting

        # Filter by mode and prompt_mode
        no_comm_base = subset[(subset['mode'] == 'no_comm') & (subset['prompt_mode'] == 'base')]
        contract_base = subset[(subset['mode'] != 'no_comm') & (subset['prompt_mode'] == 'base')]
        no_comm_coop = subset[(subset['mode'] == 'no_comm') & (subset['prompt_mode'] == 'cooperative')]
        contract_coop = subset[(subset['mode'] != 'no_comm') & (subset['prompt_mode'] == 'cooperative')]

        # Get average values
        baseline_val = no_comm_base[outcome_col].mean() if not no_comm_base.empty else 0.0
        contract_val = contract_base[outcome_col].mean() if not contract_base.empty else 0.0
        coop_val = no_comm_coop[outcome_col].mean() if not no_comm_coop.empty else 0.0
        coop_contract_val = contract_coop[outcome_col].mean() if not contract_coop.empty else 0.0

        # Compute interaction effect
        interaction = (coop_contract_val - baseline_val) - (coop_val - baseline_val) - (contract_val - baseline_val)

        results[dataset] = float(interaction)

    return results


def detect_greenwashing(
    df: pd.DataFrame,
    prompt_mode: str | None = None,
) -> pd.DataFrame:
    """Detect Greenwashing behavior in 4x4 moral hazard scenarios.

    Greenwashing = Cooperate on output + Low Effort (exploiting moral hazard)

    Works identically for PD and Stag Hunt:
    - PD: greenwashing = defect while declaring cooperation (classic moral hazard)
    - Stag Hunt: greenwashing = slack on effort while cooperating (undermines coordination)

    Args:
        df: DataFrame from load_contracting_logs() (must be 4x4 data)
        prompt_mode: Filter by prompt mode (None = all)

    Returns:
        DataFrame with Greenwashing instances and metadata
    """
    if df.empty:
        return pd.DataFrame()

    # Filter to 4x4 only
    if 'dataset' in df.columns:
        df = df[df['dataset'] == '4x4'].copy()

    # Filter by prompt mode if specified
    if prompt_mode and 'prompt_mode' in df.columns:
        df = df[df['prompt_mode'] == prompt_mode].copy()

    # Need action category and effort level
    if 'row_action_category' not in df.columns or 'row_effort_level' not in df.columns:
        return pd.DataFrame()

    # Detect Greenwashing: cooperate on output but low effort
    greenwashing_mask = (
        (df['row_action_category'] == 'cooperate') &
        (df['row_effort_level'] == 'Low Effort')
    )

    greenwashing_df = df[greenwashing_mask].copy()

    # Add flag
    greenwashing_df['is_greenwashing'] = True

    # Compute rates by prompt mode
    if 'prompt_mode' in df.columns:
        total_by_mode = df.groupby('prompt_mode').size()
        greenwash_by_mode = greenwashing_df.groupby('prompt_mode').size()

        greenwashing_rates = {}
        for mode in df['prompt_mode'].unique():
            total = total_by_mode.get(mode, 0)
            gw_count = greenwash_by_mode.get(mode, 0)
            greenwashing_rates[f'{mode}_rate'] = float(gw_count / total) if total > 0 else 0.0

        # Add to results
        for key, value in greenwashing_rates.items():
            greenwashing_df[key] = value

    return greenwashing_df


def compute_coordination_rate(
    df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute coordination rate for Stag Hunt and other coordination games.

    Coordination = both players choose the same base action category
    (both cooperate or both defect). Miscoordination = different categories.

    For Stag Hunt, this measures whether players coordinate on the same
    action (both cooperate or both defect) vs. miscoordinate. High
    miscoordination rate indicates Stag Hunt risk is being realized.

    Args:
        df: DataFrame from load_contracting_logs()

    Returns:
        Dict with coordination metrics:
        - coordination_rate: % of samples where both players coordinate
        - both_cooperate_rate: % where both cooperate
        - both_defect_rate: % where both defect
        - miscoordination_rate: % where actions differ
    """
    if df.empty:
        return {
            'coordination_rate': None,
            'both_cooperate_rate': None,
            'both_defect_rate': None,
            'miscoordination_rate': None,
        }

    if 'row_action_category' not in df.columns or 'col_action_category' not in df.columns:
        return {
            'coordination_rate': None,
            'both_cooperate_rate': None,
            'both_defect_rate': None,
            'miscoordination_rate': None,
        }

    total = len(df)
    if total == 0:
        return {
            'coordination_rate': None,
            'both_cooperate_rate': None,
            'both_defect_rate': None,
            'miscoordination_rate': None,
        }

    # Coordination: both players choose same action category
    coordinated = (df['row_action_category'] == df['col_action_category']).sum()

    # Both cooperate
    both_coop = ((df['row_action_category'] == 'cooperate') &
                 (df['col_action_category'] == 'cooperate')).sum()

    # Both defect
    both_defect = ((df['row_action_category'] == 'defect') &
                   (df['col_action_category'] == 'defect')).sum()

    return {
        'coordination_rate': float(coordinated / total),
        'both_cooperate_rate': float(both_coop / total),
        'both_defect_rate': float(both_defect / total),
        'miscoordination_rate': float(1 - (coordinated / total)),
    }


def get_effort_distribution(
    df: pd.DataFrame,
    by_mode: bool = True,
    by_prompt_mode: bool = True,
) -> pd.DataFrame:
    """Extract effort level distribution for 4x4 scenarios.

    Args:
        df: DataFrame from load_contracting_logs()
        by_mode: Break down by contract mode
        by_prompt_mode: Break down by prompt mode

    Returns:
        DataFrame with effort distribution counts and percentages
    """
    if df.empty:
        return pd.DataFrame()

    # Filter to 4x4
    if 'dataset' in df.columns:
        df = df[df['dataset'] == '4x4'].copy()

    if 'row_effort_level' not in df.columns:
        return pd.DataFrame()

    # Build grouping
    group_cols = []
    if by_mode and 'mode' in df.columns:
        group_cols.append('mode')
    if by_prompt_mode and 'prompt_mode' in df.columns:
        group_cols.append('prompt_mode')

    if not group_cols:
        # Just count total effort distribution
        effort_dist = df['row_effort_level'].value_counts().reset_index()
        effort_dist.columns = ['effort_level', 'count']
        effort_dist['percentage'] = (effort_dist['count'] / effort_dist['count'].sum() * 100).round(2)
        return effort_dist

    # Group by specified columns
    effort_dist = df.groupby(group_cols + ['row_effort_level']).size().reset_index(name='count')

    # Compute percentages within each group
    effort_dist['percentage'] = (
        effort_dist.groupby(group_cols)['count'].transform(lambda x: x / x.sum() * 100).round(2)
    )

    return effort_dist


def get_negotiation_transcripts(
    df: pd.DataFrame,
    sample_ids: List[str] | None = None,
) -> pd.DataFrame:
    """Extract negotiation transcripts from metadata.

    Args:
        df: DataFrame from load_contracting_logs()
        sample_ids: Optional list of sample IDs to filter

    Returns:
        DataFrame with transcript entries (turn, player, message, etc.)
    """
    if df.empty:
        return pd.DataFrame()

    transcripts = []

    for _, row in df.iterrows():
        sample_id = row.get('sample_id', '')
        if sample_ids and sample_id not in sample_ids:
            continue

        # Parse negotiation history
        negotiation_json = row.get('negotiation_history', '')
        if not negotiation_json:
            continue

        try:
            conversations = json.loads(negotiation_json)

            for conv in conversations:
                transcripts.append({
                    'sample_id': sample_id,
                    'turn': conv.get('turn'),
                    'player': conv.get('player'),
                    'action': conv.get('action'),
                    'contract_text': conv.get('contract_text'),
                    'reasoning': conv.get('reasoning'),
                    'experiment': row.get('experiment', ''),
                    'prompt_mode': row.get('prompt_mode', ''),
                    'mode': row.get('mode', ''),
                })
        except json.JSONDecodeError:
            continue

    if not transcripts:
        return pd.DataFrame()

    return pd.DataFrame(transcripts)


def normalize_welfare(score: float, bounds: WelfareBounds, metric: str = 'utilitarian') -> float:
    """Normalize a welfare score to [0, 1] using per-scenario bounds.

    Maps the score from [min, max] to [0, 1] where min and max are the
    worst and best possible welfare values for this specific scenario's
    payoff matrix.

    Args:
        score: The achieved welfare value (utilitarian or rawlsian)
        bounds: Per-scenario WelfareBounds from compute_welfare_bounds()
        metric: 'utilitarian' or 'rawlsian'

    Returns:
        Normalized score in [0, 1], or 0.5 if all outcomes are identical.
    """
    if metric == 'utilitarian':
        span = bounds['util_max'] - bounds['util_min']
        min_val = bounds['util_min']
    else:
        span = bounds['rawls_max'] - bounds['rawls_min']
        min_val = bounds['rawls_min']

    if span == 0:
        return 0.5  # All outcomes produce identical welfare

    return (score - min_val) / span


def nash_deviation(score: float, bounds: WelfareBounds, metric: str = 'utilitarian') -> float | None:
    """Compute normalized deviation from Nash equilibrium welfare.

    Returns a value in [0, 1] where:
      - 0.0 = welfare at the Nash equilibrium
      - 1.0 = best possible welfare (utilitarian/rawlsian optimum)

    This captures how much better than Nash the achieved outcome is,
    as a fraction of the gap between Nash and optimum.

    Args:
        score: The achieved welfare value
        bounds: Per-scenario WelfareBounds
        metric: 'utilitarian' or 'rawlsian'

    Returns:
        Normalized deviation in [0, 1], or None if no Nash equilibrium
        exists or Nash equals optimum (no room for improvement).
    """
    if metric == 'utilitarian':
        nash_val = bounds['nash_util']
        opt_val = bounds['util_max']
    else:
        nash_val = bounds['nash_rawls']
        opt_val = bounds['rawls_max']

    if nash_val is None:
        return None

    gap = opt_val - nash_val
    if gap == 0:
        return 0.0  # Nash is already optimal

    return (score - nash_val) / gap


# ---------------------------------------------------------------------------
# Backward-compatible aliases (used by plot scripts)
# ---------------------------------------------------------------------------

def normalize_score(score: float, bounds: WelfareBounds, metric: str = 'utilitarian') -> float:
    """Normalize a welfare score using per-scenario bounds.

    Backward-compatible wrapper around normalize_welfare().
    """
    return normalize_welfare(score, bounds, metric)


def compute_normalized_nash_deviation(score: float, bounds: WelfareBounds, welfare: str = 'utilitarian') -> float | None:
    """Compute normalized Nash deviation using per-scenario bounds.

    Backward-compatible wrapper around nash_deviation().
    """
    return nash_deviation(score, bounds, welfare)
