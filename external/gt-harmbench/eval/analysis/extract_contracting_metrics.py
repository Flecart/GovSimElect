"""Extract contracting-specific metrics from eval logs."""

import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import click
import pandas as pd

from eval.analysis.contracting import get_contracting_score

USAGE_PHASES = ("negotiation", "decision", "coding")
USAGE_FIELDS = (
    "call_count",
    "elapsed_seconds",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "calls_with_usage",
    "errors",
)


def _action_category_index(category: Any) -> int | None:
    if not isinstance(category, str):
        return None

    category_norm = category.strip().lower()
    if category_norm == "cooperate":
        return 0
    if category_norm == "defect":
        return 1
    return None


def _action_category_effort_index(category: Any, effort: Any) -> int | None:
    base_idx = _action_category_index(category)
    if base_idx is None or not isinstance(effort, str):
        return None

    effort_norm = effort.strip().lower()
    if "high" in effort_norm:
        return base_idx * 2
    if "low" in effort_norm:
        return base_idx * 2 + 1
    return None


def _empty_usage_totals() -> Dict[str, Dict[str, float]]:
    """Initialize zeroed usage totals for trace + all phases."""
    totals = {"trace": {field: 0.0 for field in USAGE_FIELDS}}
    for phase in USAGE_PHASES:
        totals[phase] = {field: 0.0 for field in USAGE_FIELDS}
    return totals


def _add_usage_summary(totals: Dict[str, Dict[str, float]], trace_usage: Any) -> None:
    if not isinstance(trace_usage, dict):
        return

    _add_usage_fields(totals["trace"], trace_usage.get("total", {}))
    by_phase = trace_usage.get("by_phase", {})
    if not isinstance(by_phase, dict):
        return

    for phase in USAGE_PHASES:
        _add_usage_fields(totals[phase], by_phase.get(phase, {}))


def _add_usage_fields(target: Dict[str, float], source: Any) -> None:
    if not isinstance(source, dict):
        return
    for field in USAGE_FIELDS:
        target[field] += float(source.get(field) or 0)


def _write_usage_results(result: Dict[str, Any], totals: Dict[str, Dict[str, float]]) -> None:
    for prefix, values in totals.items():
        for field, value in values.items():
            column = f"{prefix}_{field}"
            if field in {"call_count", "input_tokens", "output_tokens", "total_tokens", "reasoning_tokens", "calls_with_usage", "errors"}:
                result[column] = int(value)
            else:
                result[column] = value


def extract_contracting_metrics(log_path: Path) -> Dict[str, Any]:
    """Extract formation, activation, welfare, effort, and usage metrics from one .eval file."""
    log_id = log_path.stem

    try:
        with zipfile.ZipFile(log_path, 'r') as zf:
            summary_files = [name for name in zf.namelist() if name.startswith('_journal/summaries/')]

            if not summary_files:
                print(f"No summary files found in {log_path.name}", file=sys.stderr)
                return None

            summaries = []
            for summary_file in sorted(summary_files):
                with zf.open(summary_file) as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        summaries.extend(data)
                    else:
                        summaries.append(data)

            sample_usage_by_id: Dict[str, List[Dict[str, Any]]] = {}
            sample_files = sorted(
                name
                for name in zf.namelist()
                if name.startswith("samples/") and name.endswith(".json")
            )
            for sample_file in sample_files:
                with zf.open(sample_file) as f:
                    sample = json.load(f)
                sample_meta = sample.get("metadata", {})
                sample_id = str(sample_meta.get("id", sample.get("id", "")))
                sample_usage = sample_meta.get("trace_usage")
                if sample_usage is None:
                    sample_usage = _get_sample_score_metadata(sample).get("trace_usage")
                if isinstance(sample_usage, dict):
                    sample_usage_by_id.setdefault(sample_id, []).append(sample_usage)

            # Read header for model name
            model_name = 'unknown'
            if '_journal/start.json' in zf.namelist():
                with zf.open('_journal/start.json') as f:
                    header = json.load(f)
                    if 'eval' in header and 'model' in header['eval']:
                        model_name = header['eval']['model']

    except Exception as e:
        print(f"Error loading {log_path.name}: {e}", file=sys.stderr)
        return None

    if not summaries:
        print(f"No summaries found in {log_path.name}", file=sys.stderr)
        return None

    result = {
        "log_id": log_id,
        "model_name": model_name,
    }

    total_samples = 0
    contracts_formed = 0
    contracts_activated = 0
    contracts_with_activation = 0
    enforcement_success = 0
    enforcement_attempted = 0
    nash_count = 0
    utilitarian_count = 0
    rawlsian_count = 0
    samples_with_targets = 0
    samples_without_targets = 0
    total_utilitarian_payoff = 0.0
    total_rawlsian_payoff = 0.0
    cooperation_count = 0
    high_effort_count = 0
    high_effort_eligible = 0
    total_row_payoff = 0.0
    total_col_payoff = 0.0
    total_turns_to_agreement = 0.0
    negotiation_count = 0
    formation_failure_reasons: Dict[str, int] = {}
    activation_failure_reasons: Dict[str, int] = {}
    utilitarian_payoffs: List[float] = []
    rawlsian_payoffs: List[float] = []
    dataset = "4x4" if "4x4" in log_path.parent.name else "2x2"
    matrix_size = 4 if dataset == "4x4" else 2
    joint_action_matrix = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]
    joint_action_count = 0
    usage_totals = _empty_usage_totals()

    for summary in summaries:
        input_metadata = summary.get("metadata", {})

        scorer_data = get_contracting_score(summary.get('scores', {}))
        if not scorer_data:
            continue

        if 'metadata' not in scorer_data:
            continue

        metadata = scorer_data['metadata']
        _add_usage_summary(
            usage_totals,
            _first_trace_usage(
                input_metadata.get("trace_usage"),
                metadata.get("trace_usage"),
                _pop_sample_usage(sample_usage_by_id, input_metadata.get("id")),
            ),
        )
        total_samples += 1

        formed = metadata.get('contract_formed', False)
        if formed:
            contracts_formed += 1
            if metadata.get('contract_complied') is not None:
                contracts_with_activation += 1
            if metadata.get('contract_complied') is False:
                contracts_activated += 1

        if metadata.get('enforcement_attempted', False):
            enforcement_attempted += 1
            if metadata.get('enforcement_success', False):
                enforcement_success += 1

        if not formed:
            reason = metadata.get('formation_failure_reason')
            if reason:
                formation_failure_reasons[reason] = formation_failure_reasons.get(reason, 0) + 1
        elif metadata.get('contract_complied') is False:
            reason = metadata.get('compliance_failure_reason')
            if reason:
                activation_failure_reasons[reason] = activation_failure_reasons.get(reason, 0) + 1

        answer_type = metadata.get('answer_type', 'neutral')
        if answer_type != 'neutral':
            samples_with_targets += 1
            if metadata.get('is_nash', False):
                nash_count += 1
            if metadata.get('is_utilitarian', False):
                utilitarian_count += 1
            if metadata.get('is_rawlsian', False):
                rawlsian_count += 1
        else:
            samples_without_targets += 1

        util_payoff = metadata.get('utilitarian_payoff', 0)
        rawls_payoff = metadata.get('rawlsian_payoff', 0)
        total_utilitarian_payoff += util_payoff
        total_rawlsian_payoff += rawls_payoff
        utilitarian_payoffs.append(util_payoff)
        rawlsian_payoffs.append(rawls_payoff)

        total_row_payoff += metadata.get('row_payoff', 0) or 0
        total_col_payoff += metadata.get('col_payoff', 0) or 0

        row_cat = metadata.get('row_action_category', '')
        col_cat = metadata.get('col_action_category', '')
        if row_cat == 'cooperate' or col_cat == 'cooperate':
            cooperation_count += 1

        r_effort = metadata.get('row_effort_level', '')
        c_effort = metadata.get('col_effort_level', '')
        is_4x4 = bool(r_effort or c_effort)
        if is_4x4:
            high_effort_eligible += 1
            if 'High' in r_effort or 'High' in c_effort:
                high_effort_count += 1

        if dataset == "4x4":
            r_idx = _action_category_effort_index(row_cat, r_effort)
            c_idx = _action_category_effort_index(col_cat, c_effort)
        else:
            r_idx = _action_category_index(row_cat)
            c_idx = _action_category_index(col_cat)
        if r_idx is not None and c_idx is not None:
            joint_action_matrix[r_idx][c_idx] += 1
            joint_action_count += 1

        turns = metadata.get('turns_to_agreement')
        if turns is not None and turns > 0:
            total_turns_to_agreement += turns
            negotiation_count += 1

    result['total_evaluated'] = total_samples
    result['contracts_formed'] = contracts_formed
    result['contract_formation_rate'] = contracts_formed / total_samples if total_samples > 0 else 0.0
    result['contracts_activated'] = contracts_activated
    result['contracts_with_enforcement'] = contracts_with_activation
    if contracts_with_activation > 0:
        result['contract_activation_rate'] = contracts_activated / contracts_with_activation
    else:
        result['contract_activation_rate'] = None
    result['enforcement_attempted'] = enforcement_attempted
    result['enforcement_success'] = enforcement_success
    result['enforcement_success_rate'] = enforcement_success / enforcement_attempted if enforcement_attempted > 0 else None

    result['samples_with_targets'] = samples_with_targets
    result['samples_without_targets'] = samples_without_targets
    result['nash_count'] = nash_count
    result['utilitarian_count'] = utilitarian_count
    result['rawlsian_count'] = rawlsian_count
    result['nash_accuracy_with_contract'] = nash_count / samples_with_targets if samples_with_targets > 0 else 0.0
    result['utilitarian_accuracy_with_contract'] = utilitarian_count / samples_with_targets if samples_with_targets > 0 else 0.0
    result['rawlsian_accuracy_with_contract'] = rawlsian_count / samples_with_targets if samples_with_targets > 0 else 0.0

    result['avg_utilitarian_payoff'] = total_utilitarian_payoff / total_samples if total_samples > 0 else 0.0
    result['avg_rawlsian_payoff'] = total_rawlsian_payoff / total_samples if total_samples > 0 else 0.0

    if len(utilitarian_payoffs) > 1:
        result['utilitarian_payoff_variance'] = sum((x - result['avg_utilitarian_payoff']) ** 2 for x in utilitarian_payoffs) / len(utilitarian_payoffs)
    else:
        result['utilitarian_payoff_variance'] = 0.0

    if len(rawlsian_payoffs) > 1:
        result['rawlsian_payoff_variance'] = sum((x - result['avg_rawlsian_payoff']) ** 2 for x in rawlsian_payoffs) / len(rawlsian_payoffs)
    else:
        result['rawlsian_payoff_variance'] = 0.0

    result['avg_row_payoff'] = total_row_payoff / total_samples if total_samples > 0 else 0.0
    result['avg_col_payoff'] = total_col_payoff / total_samples if total_samples > 0 else 0.0
    result['cooperation_rate'] = cooperation_count / total_samples if total_samples > 0 else 0.0
    result['high_effort_rate'] = high_effort_count / high_effort_eligible if high_effort_eligible > 0 else None
    result['avg_turns_to_agreement'] = total_turns_to_agreement / negotiation_count if negotiation_count > 0 else None
    result['formation_failure_reasons'] = json.dumps(formation_failure_reasons) if formation_failure_reasons else None
    result['activation_failure_reasons'] = json.dumps(activation_failure_reasons) if activation_failure_reasons else None
    result['joint_action_dataset'] = dataset
    result['joint_action_count'] = joint_action_count
    result['joint_action_matrix'] = json.dumps(joint_action_matrix)
    _write_usage_results(result, usage_totals)

    return result


def _get_sample_score_metadata(sample: Dict[str, Any]) -> Dict[str, Any]:
    scorer_data = get_contracting_score(sample.get("scores", {}))
    metadata = scorer_data.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _first_trace_usage(*values: Any) -> Dict[str, Any] | None:
    for value in values:
        if isinstance(value, dict):
            return value
    return None


def _pop_sample_usage(
    sample_usage_by_id: Dict[str, List[Dict[str, Any]]],
    sample_id: Any,
) -> Dict[str, Any] | None:
    usage_entries = sample_usage_by_id.get(str(sample_id))
    if not usage_entries:
        return None
    return usage_entries.pop(0)


@click.command()
@click.option(
    "--directory",
    "-d",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default="logs",
    help="Directory containing .eval files",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="contracting_metrics.csv",
    help="Output CSV file path",
)
def main(directory: Path, output: Path) -> None:
    """Extract contracting metrics from all .eval files in a directory to CSV."""
    eval_files = sorted(directory.glob("*.eval"))

    if not eval_files:
        print(f"No .eval files found in {directory}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(eval_files)} .eval files in {directory}")
    print("Processing files...")

    results: List[Dict[str, Any]] = []

    for eval_file in eval_files:
        print(f"Processing {eval_file.name}...")
        metrics = extract_contracting_metrics(eval_file)
        if metrics:
            results.append(metrics)
        else:
            print(f"  Skipped {eval_file.name} due to errors")

    if not results:
        print("No valid results extracted.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(results)

    column_order = [
        "log_id",
        "model_name",
        "contract_formation_rate",
        "contract_activation_rate",
        "enforcement_success_rate",
        "nash_accuracy_with_contract",
        "utilitarian_accuracy_with_contract",
        "rawlsian_accuracy_with_contract",
        "avg_utilitarian_payoff",
        "avg_rawlsian_payoff",
        "avg_row_payoff",
        "avg_col_payoff",
        "cooperation_rate",
        "high_effort_rate",
        "avg_turns_to_agreement",
        "contracts_formed",
        "contracts_activated",
        "enforcement_success",
        "enforcement_attempted",
        "formation_failure_reasons",
        "activation_failure_reasons",
        "joint_action_dataset",
        "joint_action_count",
        "joint_action_matrix",
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
    ]

    for col in df.columns:
        if col not in column_order:
            column_order.append(col)

    for col in column_order:
        if col not in df.columns:
            df[col] = None

    df = df[column_order]

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\nExtracted metrics from {len(results)} logs")
    print(f"Saved to: {output}")
    print(f"\nSummary:")
    print(df.describe())


if __name__ == "__main__":
    main()
