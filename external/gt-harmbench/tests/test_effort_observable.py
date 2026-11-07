"""Tests for the observable-effort (contractible) 4x4 variant vs. the hidden-effort default."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contracting.contract import Contract, ContractType, GameContractState
from src.contracting.enforcers import LawEnforcer, LawExecutionRuntime
from src.contracting.prompts import coding_agent_prompt, negotiation_system_prompt, negotiation_user_prompt

ACTIONS = ["Engage (High Effort)", "Engage (Low Effort)", "Impose (High Effort)", "Impose (Low Effort)"]
PAYOFF = [
    [[50, 50], [35, 60], [-70, 98], [-70, 88]],
    [[60, 35], [30, 30], [-90, 98], [-90, 88]],
    [[98, -70], [98, -90], [-10, -10], [0, -20]],
    [[88, -70], [88, -90], [-20, 0], [-30, -30]],
]


def _state(observable: bool) -> GameContractState:
    return GameContractState(
        scenario_id="t", formal_game="Prisoner's Dilemma",
        actions_row=ACTIONS, actions_column=ACTIONS, payoff_matrix=PAYOFF,
        is_4x4=True, effort_observable=observable,
    )


def _law(code: str) -> Contract:
    return Contract(contract_type=ContractType.PYTHON_LAW, content=code, proposer="test")


def test_hidden_effort_unchanged():
    rt = LawExecutionRuntime({"row": "Engage (Low Effort)", "column": "Impose (High Effort)"}, _state(False))
    assert rt.get_action("row") == "Engage"
    assert rt.available_actions("row") == ["Engage", "Impose"]
    # averaged over hidden effort: (50+35+60+30)/4
    assert rt.get_payoff("Engage", "Engage") == (43.75, 43.75)


def test_observable_effort_runtime():
    rt = LawExecutionRuntime({"row": "Engage (Low Effort)", "column": "Impose (High Effort)"}, _state(True))
    assert rt.get_action("row") == "Engage (Low Effort)"
    assert rt.available_actions("column") == ACTIONS
    assert rt.get_payoff("Engage (Low Effort)", "Engage (High Effort)") == (60, 35)
    rt.set_action("row", "Engage (High Effort)")
    assert rt.get_action("row") == "Engage (High Effort)"
    # bare base action keeps current effort
    rt.set_action("column", "Engage")
    assert rt.get_action("column") == "Engage (High Effort)"
    obs = rt.observable_state()
    assert obs.payoff_matrix["2_1_payoff"] == [60, 35]
    assert len(obs.payoff_matrix) == 16


def test_observable_effort_law_can_fine_low_effort():
    code = (
        "def enforce(actions, state, context):\n"
        "    if '(Low Effort)' in get_action('row'):\n"
        "        apply_fine('row', 40, 'shirked')\n"
        "        set_action('row', 'Engage (High Effort)', 'require high effort')\n"
    )
    res = LawEnforcer().enforce(
        _law(code), {"row": "Engage (Low Effort)", "column": "Engage (High Effort)"}, _state(True)
    )
    assert res.success, res.reasoning
    assert res.modified_actions["row"] == "Engage (High Effort)"
    assert res.payoff_adjustments["row"]["fines"] == [-40]


def test_hidden_effort_law_cannot_see_effort():
    code = (
        "def enforce(actions, state, context):\n"
        "    if '(Low Effort)' in get_action('row'):\n"
        "        apply_fine('row', 40, 'shirked')\n"
    )
    res = LawEnforcer().enforce(
        _law(code), {"row": "Engage (Low Effort)", "column": "Engage (High Effort)"}, _state(False)
    )
    assert res.success
    assert res.payoff_adjustments["row"]["fines"] == []


def test_prompts_switch():
    hidden = negotiation_system_prompt("row", "PD", "code_law", is_4x4=True)
    obs = negotiation_system_prompt("row", "PD", "code_law", is_4x4=True, effort_observable=True)
    assert "UNOBSERVABLE" in hidden and "UNOBSERVABLE" not in obs
    assert "OBSERVABLE and VERIFIABLE" in obs
    up = negotiation_user_prompt("row", "story", ACTIONS, True, 0, [], [], effort_observable=True)
    assert "NOT contractable" not in up and "Engage (Low Effort)" in up
    _, cp = coding_agent_prompt("x", _state(True))
    assert "Observable-Effort" in cp and "averaged across hidden effort" not in cp
    _, cp_hidden = coding_agent_prompt("x", _state(False))
    assert "averaged across hidden effort" in cp_hidden


def test_sample_metadata_flag():
    from eval.contracting_solver import record_to_sample_with_contracting
    rec = {"id": "1", "actions_row_4x4": json.dumps(ACTIONS).replace('"', "'"),
           "actions_col_4x4": json.dumps(ACTIONS).replace('"', "'"),
           "payoff_matrix_4x4": json.dumps(PAYOFF)}
    s = record_to_sample_with_contracting(rec, effort_observable=True)
    assert s.metadata["effort_observable"] is True
    assert record_to_sample_with_contracting(rec).metadata["effort_observable"] is False
