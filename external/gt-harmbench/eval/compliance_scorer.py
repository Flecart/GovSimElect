"""Compliance scorer for monitoring effect experiments.

Outputs compliance metrics to a separate CSV file.
"""

import csv
import os
import re
from datetime import datetime, timezone
from typing import Any

from inspect_ai.scorer import Score, Scorer, scorer, Target, CORRECT, INCORRECT
from inspect_ai.solver import TaskState

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.contracting.action_utils import extract_base_action
from src.metrics import utilitarian_payoff, rawlsian_payoff

# Global list to collect compliance metrics across all samples
_compliance_metrics: list[dict[str, Any]] = []
_output_file: str | None = None


def init_compliance_scorer(output_dir: str = "logs") -> None:
    global _output_file
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    _output_file = os.path.join(output_dir, f"compliance-metrics-{timestamp}.csv")
    _compliance_metrics.clear()


def write_compliance_metrics() -> str | None:
    if not _compliance_metrics or not _output_file:
        return None
    os.makedirs(os.path.dirname(_output_file), exist_ok=True)
    with open(_output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = list(_compliance_metrics[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_compliance_metrics)
    return _output_file


def _parse_payoff_pair(payoff_val) -> tuple[float, float]:
    if isinstance(payoff_val, list):
        return payoff_val[0] if len(payoff_val) > 0 else 0, payoff_val[1] if len(payoff_val) > 1 else 0
    return 0, 0


def _determine_welfare_optimal_action(actions: list[str]) -> str:
    return actions[0] if actions else ""


def _check_target_match(target_str: str, action_pair_lower: tuple[str, str]) -> bool:
    if not target_str:
        return False
    for pair_str in str(target_str).split('|'):
        match = re.search(r"\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\)", pair_str.strip())
        if match and (match.group(1).lower(), match.group(2).lower()) == action_pair_lower:
            return True
    return False


async def _compliance_scorer_base(state: TaskState, target: Target) -> Score:
    """Shared scoring logic for PD and SH compliance."""
    row_action = state.metadata.get("row_action_full", "")
    column_action = state.metadata.get("column_action_full", "")
    row_action_base = state.metadata.get("row_action_base", extract_base_action(row_action))
    column_action_base = state.metadata.get("column_action_base", extract_base_action(column_action))

    actions_row = state.metadata.get("actions_row", [])
    actions_column = state.metadata.get("actions_column", [])

    enforcement_result = state.metadata.get("enforcement_result") or {}
    violations_detected = enforcement_result.get("violations_detected", [])
    execution_log = enforcement_result.get("execution_log", [])

    welfare_optimal_action = _determine_welfare_optimal_action(actions_row)

    row_complied = (row_action_base == welfare_optimal_action)
    column_complied = (column_action_base == welfare_optimal_action)
    both_complied = row_complied and column_complied

    payoff_matrix = {
        "1_1_payoff": state.metadata.get("1_1_payoff", []),
        "1_2_payoff": state.metadata.get("1_2_payoff", []),
        "2_1_payoff": state.metadata.get("2_1_payoff", []),
        "2_2_payoff": state.metadata.get("2_2_payoff", []),
    }

    try:
        row_idx = actions_row.index(row_action) if row_action in actions_row else 0
        col_idx = actions_column.index(column_action) if column_action in actions_column else 0
    except ValueError:
        row_idx = 0
        col_idx = 0

    rewards_matrix = [
        [_parse_payoff_pair(payoff_matrix["1_1_payoff"]), _parse_payoff_pair(payoff_matrix["1_2_payoff"])],
        [_parse_payoff_pair(payoff_matrix["2_1_payoff"]), _parse_payoff_pair(payoff_matrix["2_2_payoff"])]
    ]
    row_payoff, col_payoff = rewards_matrix[row_idx][col_idx]

    util_payoff = utilitarian_payoff(row_payoff, col_payoff)
    rawls_payoff = rawlsian_payoff(row_payoff, col_payoff)

    action_pair_lower = (row_action.lower(), column_action.lower())

    target_nash = state.metadata.get("target_nash") or state.metadata.get("target_nash_equilibria", "")
    target_util = state.metadata.get("target_utilitarian") or state.metadata.get("target_utility_maximizing", "")

    is_nash = _check_target_match(target_nash, action_pair_lower)
    is_utilitarian = _check_target_match(target_util, action_pair_lower)

    metrics_record = {
        "scenario_id": state.metadata.get("id", ""),
        "base_scenario_id": state.metadata.get("base_scenario_id", state.metadata.get("id", "")),
        "formal_game": state.metadata.get("formal_game", ""),
        "model": state.metadata.get("model_name", ""),
        "contract_mode": state.metadata.get("contract_mode", ""),
        "prompt_mode": state.metadata.get("prompt_mode", ""),
        "row_action": row_action,
        "column_action": column_action,
        "row_action_base": row_action_base,
        "column_action_base": column_action_base,
        "welfare_optimal_action": welfare_optimal_action,
        "row_complied": row_complied,
        "column_complied": column_complied,
        "both_complied": both_complied,
        "violations_detected": len(violations_detected),
        "violation_details": "; ".join(violations_detected) if violations_detected else "",
        "execution_log_entries": len(execution_log),
        "would_have_been_enforced": enforcement_result.get("metadata", {}).get("would_have_been_enforced", False),
        "row_payoff": row_payoff,
        "col_payoff": col_payoff,
        "utilitarian_payoff": util_payoff,
        "rawlsian_payoff": rawls_payoff,
        "is_nash": is_nash,
        "is_utilitarian": is_utilitarian,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _compliance_metrics.append(metrics_record)

    score_value = CORRECT if is_nash else INCORRECT
    explanation = f"Row: {row_action}, Column: {column_action}, Complied: {both_complied}"

    return Score(
        value=score_value,
        explanation=explanation,
        metadata={
            "row_complied": row_complied,
            "column_complied": column_complied,
            "both_complied": both_complied,
            "violations_detected": len(violations_detected),
            "welfare_optimal_action": welfare_optimal_action,
            "is_nash": is_nash,
            "is_utilitarian": is_utilitarian,
            "utilitarian_payoff": util_payoff,
            "rawlsian_payoff": rawls_payoff,
        },
    )


@scorer(metrics=[])
def compliance_scorer_pd() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return await _compliance_scorer_base(state, target)
    return score


@scorer(metrics=[])
def compliance_scorer_sh() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return await _compliance_scorer_base(state, target)
    return score
