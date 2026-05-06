"""Analysis utilities for GT-HarmBench contracting experiments."""

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
    """Find the contracting score dict regardless of game-specific scorer name."""
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
    """Extract scorer metadata from a summary or sample item."""
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
    """Load contracting logs from disk, returning {experiment_name: DataFrame} per condition."""
    model_data: Dict[str, pd.DataFrame] = {}

    for prompt_mode in prompt_modes:
        for exp in experiments:
            exp_path = log_base_dir / prompt_mode / exp

            if not exp_path.exists():
                if verbose:
                    print(f"Skipping {exp_path} (not found)")
                continue

            eval_files = sorted(exp_path.glob("*.eval"))

            if not eval_files:
                if verbose:
                    print(f"No .eval files in {exp_path}")
                continue

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

            task_meta = {}
            if eval_files:
                task_meta = _extract_task_metadata(eval_files[0])

            df['experiment'] = exp
            df['prompt_mode'] = prompt_mode
            df['dataset'] = '4x4' if '4x4' in exp else '2x2'

            df['mode'] = (
                'no_comm' if 'no-comm' in exp or 'no_comm' in exp else
                'code_nl' if 'code-n' in exp or 'code_nl' in exp else
                'code_law'
            )

            if 'model_name' in task_meta:
                df['model_name'] = task_meta['model_name']
            if 'negotiation_mode' in task_meta:
                df['negotiation_mode'] = task_meta['negotiation_mode']
            if 'num_stories' in task_meta:
                df['num_stories'] = task_meta['num_stories']
            if 'num_times' in task_meta:
                df['num_times'] = task_meta['num_times']
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

            key = f"{exp}-{prompt_mode}" if prompt_mode != "base" else exp

            if verbose:
                print(f"Loaded {len(df)} samples from {key}")

            model_data[key] = df

    return model_data


def _extract_task_metadata(eval_file: Path) -> Dict[str, Any]:
    """Read task metadata from header.json inside an .eval archive."""
    try:
        with zipfile.ZipFile(eval_file, 'r') as zf:
            if 'header.json' in zf.namelist():
                with zf.open('header.json') as f:
                    header_data = json.load(f)
                    return header_data.get('eval', {}).get('metadata', {})
    except Exception:
        pass
    return {}


def _parse_payoff_matrix_from_metadata(input_meta: Dict[str, Any]) -> list | None:
    """Parse payoff matrix from sample metadata into a list-of-lists, or None on failure.

    Handles both 4x4 (payoff_matrix_4x4 JSON string) and 2x2 (1_1_payoff etc.) formats.
    """
    pm4_raw = input_meta.get("payoff_matrix_4x4")
    if pm4_raw:
        try:
            matrix = json.loads(pm4_raw) if isinstance(pm4_raw, str) else pm4_raw
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

    # 2x2 fallback
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
    """Extract per-sample data from an .eval archive, including welfare bounds."""
    samples = []

    try:
        with zipfile.ZipFile(eval_file, 'r') as zf:
            summary_files = sorted([name for name in zf.namelist() if name.startswith('_journal/summaries/')])

            for summary_file in summary_files:
                with zf.open(summary_file) as f:
                    data = json.load(f)
                    items = data if isinstance(data, list) else [data]

                    for item in items:
                        input_meta = item.get('metadata', {})
                        score_meta = get_contracting_score_metadata(item)

                        sample = {
                            'sample_id': input_meta.get('id', ''),
                            'base_scenario_id': input_meta.get('base_scenario_id', input_meta.get('id', '')),
                            'input_metadata': input_meta,
                            'score_metadata': score_meta,
                            'trace_usage': input_meta.get('trace_usage', {}),
                        }
                        sample.update(_flatten_trace_usage(input_meta.get('trace_usage', {})))
                        sample.update(score_meta)

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
    """Flatten trace usage dict into {prefix_field: value} columns for DataFrames."""
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
    """Compute formation/activation rates, welfare metrics, and per-scenario normalized scores."""
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

    contract_formed = df.get('contract_formed', pd.Series([False] * total)).sum()
    activation_series = df.get('contract_complied', pd.Series([None] * total))
    activation_defined = activation_series.dropna()
    contract_activated = (activation_defined == False).sum()

    formation_reasons = df.get('formation_failure_reason', pd.Series([None] * total)).dropna()
    formation_reason_counts = formation_reasons.value_counts().to_dict() if not formation_reasons.empty else {}

    compliance_reasons = df.get('compliance_failure_reason', pd.Series([None] * total)).dropna()
    compliance_reason_counts = compliance_reasons.value_counts().to_dict() if not compliance_reasons.empty else {}

    avg_util_payoff = df.get('utilitarian_payoff', pd.Series([0] * total)).mean()
    avg_rawls_payoff = df.get('rawlsian_payoff', pd.Series([0] * total)).mean()

    var_util_payoff = df.get('utilitarian_payoff', pd.Series([0] * total)).var()
    var_rawls_payoff = df.get('rawlsian_payoff', pd.Series([0] * total)).var()

    # Per-scenario normalized welfare
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

    effort_distribution = {}
    if 'row_effort_level' in df.columns:
        effort_dist = df.get('row_effort_level', pd.Series([None] * total)).value_counts()
        effort_distribution = effort_dist.to_dict()

    usage_metrics = _aggregate_usage_columns(df)

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
    """Sum usage columns (tokens, call counts, elapsed time) if present."""
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
    """Load all experiments and return a summary DataFrame with one row per experiment."""
    data = load_contracting_logs(log_base_dir, prompt_modes=prompt_modes, verbose=False)

    if not data:
        return pd.DataFrame()

    results = []
    for exp_name, df in data.items():
        metrics = compute_contracting_metrics(df)
        metrics['experiment'] = exp_name
        metrics['dataset'] = df['dataset'].iloc[0] if not df.empty else 'unknown'
        metrics['mode'] = df['mode'].iloc[0] if not df.empty else 'unknown'
        metrics['prompt_mode'] = df['prompt_mode'].iloc[0] if not df.empty else 'base'

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

    column_order = [
        'experiment', 'dataset', 'mode', 'prompt_mode',
        'model_name', 'experiment_id', 'num_stories', 'num_times',
        'total_samples', 'contracts_formed', 'contract_formation_rate',
        'contracts_activated', 'contract_activation_rate',
        'nash_accuracy', 'utilitarian_accuracy', 'rawlsian_accuracy',
        'avg_utilitarian_payoff', 'avg_rawlsian_payoff',
        'avg_norm_utilitarian', 'avg_norm_rawlsian',
        'avg_nash_deviation_util', 'avg_nash_deviation_rawls',
        'utilitarian_payoff_variance', 'rawlsian_payoff_variance',
        'execution_timestamp', 'git_commit_hash',
    ]

    for col in summary_df.columns:
        if col not in column_order:
            column_order.append(col)

    for col in column_order:
        if col not in summary_df.columns:
            summary_df[col] = None

    return summary_df[column_order]


def compute_interaction_effects(
    metrics_df: pd.DataFrame,
    outcome_col: str = 'avg_utilitarian_payoff',
) -> Dict[str, float]:
    """Compute superadditive interaction effects for complementarity.

    Interaction = (cooperative+contract) - (cooperative alone) - (contract alone) + (baseline).
    A positive value indicates complementarity: the joint effect exceeds the sum of parts.
    """
    results = {}

    for dataset in ['4x4', '2x2']:
        subset = metrics_df[metrics_df['dataset'] == dataset]

        if subset.empty:
            results[dataset] = 0.0
            continue

        no_comm_base = subset[(subset['mode'] == 'no_comm') & (subset['prompt_mode'] == 'base')]
        contract_base = subset[(subset['mode'] != 'no_comm') & (subset['prompt_mode'] == 'base')]
        no_comm_coop = subset[(subset['mode'] == 'no_comm') & (subset['prompt_mode'] == 'cooperative')]
        contract_coop = subset[(subset['mode'] != 'no_comm') & (subset['prompt_mode'] == 'cooperative')]

        baseline_val = no_comm_base[outcome_col].mean() if not no_comm_base.empty else 0.0
        contract_val = contract_base[outcome_col].mean() if not contract_base.empty else 0.0
        coop_val = no_comm_coop[outcome_col].mean() if not no_comm_coop.empty else 0.0
        coop_contract_val = contract_coop[outcome_col].mean() if not contract_coop.empty else 0.0

        interaction = (coop_contract_val - baseline_val) - (coop_val - baseline_val) - (contract_val - baseline_val)

        results[dataset] = float(interaction)

    return results


def detect_greenwashing(
    df: pd.DataFrame,
    prompt_mode: str | None = None,
) -> pd.DataFrame:
    """Detect greenwashing: cooperate action paired with Low Effort, exploiting moral hazard.

    Works for both PD (defecting while declaring cooperation) and Stag Hunt
    (slacking on effort while cooperating), returning greenwashing instances
    with per-prompt-mode rates.
    """
    if df.empty:
        return pd.DataFrame()

    if 'dataset' in df.columns:
        df = df[df['dataset'] == '4x4'].copy()

    if prompt_mode and 'prompt_mode' in df.columns:
        df = df[df['prompt_mode'] == prompt_mode].copy()

    if 'row_action_category' not in df.columns or 'row_effort_level' not in df.columns:
        return pd.DataFrame()

    greenwashing_mask = (
        (df['row_action_category'] == 'cooperate') &
        (df['row_effort_level'] == 'Low Effort')
    )

    greenwashing_df = df[greenwashing_mask].copy()
    greenwashing_df['is_greenwashing'] = True

    if 'prompt_mode' in df.columns:
        total_by_mode = df.groupby('prompt_mode').size()
        greenwash_by_mode = greenwashing_df.groupby('prompt_mode').size()

        greenwashing_rates = {}
        for mode in df['prompt_mode'].unique():
            total = total_by_mode.get(mode, 0)
            gw_count = greenwash_by_mode.get(mode, 0)
            greenwashing_rates[f'{mode}_rate'] = float(gw_count / total) if total > 0 else 0.0

        for key, value in greenwashing_rates.items():
            greenwashing_df[key] = value

    return greenwashing_df


def compute_coordination_rate(
    df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute coordination metrics: rate of same-category choices, both-cooperate, both-defect."""
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

    coordinated = (df['row_action_category'] == df['col_action_category']).sum()

    both_coop = ((df['row_action_category'] == 'cooperate') &
                 (df['col_action_category'] == 'cooperate')).sum()

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
    """Count High/Low effort choices, optionally broken down by contract and prompt mode."""
    if df.empty:
        return pd.DataFrame()

    if 'dataset' in df.columns:
        df = df[df['dataset'] == '4x4'].copy()

    if 'row_effort_level' not in df.columns:
        return pd.DataFrame()

    group_cols = []
    if by_mode and 'mode' in df.columns:
        group_cols.append('mode')
    if by_prompt_mode and 'prompt_mode' in df.columns:
        group_cols.append('prompt_mode')

    if not group_cols:
        effort_dist = df['row_effort_level'].value_counts().reset_index()
        effort_dist.columns = ['effort_level', 'count']
        effort_dist['percentage'] = (effort_dist['count'] / effort_dist['count'].sum() * 100).round(2)
        return effort_dist

    effort_dist = df.groupby(group_cols + ['row_effort_level']).size().reset_index(name='count')

    effort_dist['percentage'] = (
        effort_dist.groupby(group_cols)['count'].transform(lambda x: x / x.sum() * 100).round(2)
    )

    return effort_dist


def get_negotiation_transcripts(
    df: pd.DataFrame,
    sample_ids: List[str] | None = None,
) -> pd.DataFrame:
    """Parse negotiation_history JSON into a turn-by-turn DataFrame."""
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
    """Normalize a welfare score to [0, 1] using per-scenario payoff bounds.

    Returns 0.5 when all outcomes produce identical welfare (degenerate case).
    """
    if metric == 'utilitarian':
        span = bounds['util_max'] - bounds['util_min']
        min_val = bounds['util_min']
    else:
        span = bounds['rawls_max'] - bounds['rawls_min']
        min_val = bounds['rawls_min']

    if span == 0:
        return 0.5

    return (score - min_val) / span


def nash_deviation(score: float, bounds: WelfareBounds, metric: str = 'utilitarian') -> float | None:
    """Fractional improvement over Nash: 0 at Nash, 1 at optimum, None if degenerate."""
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
        return 0.0

    return (score - nash_val) / gap