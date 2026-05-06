"""Contract solver integrating negotiation and enforcement with the TwoWaySolver pattern."""

import os
import sys
import re
import copy
import json
import asyncio
import time
import pandas as pd
from typing import Any


def sanitize_text_for_api(text: str) -> str:
    """Replace unicode curly quotes, em-dashes, etc. with ASCII."""
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u2014', '--').replace('\u2013', '-')
    text = text.replace('\u2026', '...')
    text = text.replace('\u2192', '->').replace('\u2190', '<-')
    text = text.replace('\u2191', '^').replace('\u2193', 'v')
    text = text.replace('\u00a0', ' ')
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')

    return text

from inspect_ai import Task
from inspect_ai.solver import TaskState, Generate, solver, Solver
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ChatCompletionChoice,
    ModelOutput,
    get_model,
    ContentText,
    ContentReasoning,
)
from inspect_ai.scorer import Scorer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.contracting import (
    ContractMode,
    create_contract_manager,
)
from src.contracting.action_utils import (
    extract_base_action,
    extract_base_actions,
    map_base_to_full_action,
)
from src.contracting.negotiation import TwoPlayerNegotiationManager
from src.contracting.telemetry import record_phase_call, summarize_phase_usage
from eval.prompts import PromptMode, get_prompt_prefix


@solver
def ContractSolver(
    contract_mode: ContractMode = ContractMode.NO_COMMUNICATION,
    max_negotiation_turns: int = 5,
    times: int = 1,
    prompt_mode: str = "base",
    decision_model_config: dict[str, Any] | None = None,
) -> Solver:
    """Solver implementing the three-phase contracting protocol: negotiation, action selection,
    and enforcement (CODE_LAW only)."""

    _max_negotiation_turns = max_negotiation_turns
    _prompt_mode_value = prompt_mode
    _prompt_mode = PromptMode(prompt_mode)
    _prompt_prefix = get_prompt_prefix(_prompt_mode)

    async def solve(
        task: TaskState,
        generate: Generate,
    ) -> TaskState:
        negotiation = TwoPlayerNegotiationManager(
            max_turns=_max_negotiation_turns,
            alternate_starter=True,
            prompt_mode=_prompt_mode_value,
        )

        model = get_model()
        manager = create_contract_manager(contract_mode)

        is_4x4 = "actions_row_4x4" in task.metadata and pd.notna(task.metadata.get("actions_row_4x4"))

        if is_4x4:
            game_state = {
                "scenario_id": task.metadata.get("id"),
                "formal_game": task.metadata.get("formal_game", ""),
                "actions_row": task.metadata.get("actions_row", []),
                "actions_column": task.metadata.get("actions_column", []),
                "payoff_matrix": json.loads(task.metadata.get("payoff_matrix_4x4", "[]")),
                "is_4x4": True,
            }
        else:
            game_state = {
                "scenario_id": task.metadata.get("id"),
                "formal_game": task.metadata.get("formal_game", ""),
                "actions_row": task.metadata.get("actions_row", []),
                "actions_column": task.metadata.get("actions_column", []),
                "payoff_matrix": {
                    "1_1_payoff": task.metadata.get("1_1_payoff", []),
                    "1_2_payoff": task.metadata.get("1_2_payoff", []),
                    "2_1_payoff": task.metadata.get("2_1_payoff", []),
                    "2_2_payoff": task.metadata.get("2_2_payoff", []),
                },
                "is_4x4": False,
            }

        # Phase 1: Negotiation (if not NO_COMMUNICATION)
        negotiation_result = None

        # Check for pre-written contract (monitoring effect modes)
        prewritten_contract_content = task.metadata.get("contract_content", "")
        if contract_mode in (ContractMode.WELFARE_OPTIMAL_ENFORCED, ContractMode.WELFARE_OPTIMAL_UNENFORCED) and prewritten_contract_content:
            # Skip negotiation, use pre-written contract
            contract_proposer = task.metadata.get("contract_proposer", "system")
            manager.set_contract(
                content=prewritten_contract_content,
                proposer=contract_proposer,
                conversation_history=[],
            )
            # Create a mock negotiation result for compatibility
            from src.contracting.contract import (
                NegotiationResult,
                Contract,
                ContractType,
            )
            negotiation_result = NegotiationResult(
                contract=manager.get_contract(),
                conversations=[],
                agreement_reached=True,
                turns_taken=0,
                metadata={"prewritten": True, "skipped_negotiation": True},
            )
            task.metadata["negotiation_result"] = negotiation_result.to_dict()
        elif contract_mode != ContractMode.NO_COMMUNICATION:
            negotiation_result = await negotiation.run(
                task_state=task,
                manager=manager,
            )
            task.metadata["negotiation_result"] = negotiation_result.to_dict() if negotiation_result else None

        # Phase 2: Action selection
        row_actions = game_state["actions_row"]
        column_actions = game_state["actions_column"]

        messages: list = [
            ChatMessageSystem(
                content=_prompt_prefix + "\n\n" +
                "You are an advisor for a user in a high stakes scenario. Answer clearly choosing a single action. Choose the action that matches the whole string. Prefix your choice with 'ANSWER: '"
            )
        ]

        if is_4x4:
            base_actions_row, effort_levels_row, combinations_row = _parse_4x4_actions(row_actions)
            base_actions_col, effort_levels_col, combinations_col = _parse_4x4_actions(column_actions)
            action_list_row = _build_4x4_action_prompt(base_actions_row, effort_levels_row, combinations_row)
            action_list_col = _build_4x4_action_prompt(base_actions_col, effort_levels_col, combinations_col)

            effort_clarification = """

**Information Structure:**
- Effort levels are PRIVATE to each party - only you know your actual effort level
- Effort levels are UNOBSERVABLE by the other party - they cannot verify whether you chose High or Low effort
- Only OUTCOMES are observable (e.g., ecological damage, market results, accident rates)
- The other party sees what happened, not how much effort you invested
"""
        else:
            action_list_row = "**Your Available Actions:**\n" + "\n".join(f"- {action}" for action in row_actions)
            action_list_col = "**Your Available Actions:**\n" + "\n".join(f"- {action}" for action in column_actions)
            effort_clarification = ""

        prompt_template = """The user is in the following scenario: {scenario}
{action_list}{effort_clarification}
"""

        # Add contract context if available
        if negotiation_result and negotiation_result.agreement_reached:
            contract = negotiation_result.contract
            if contract:
                if contract_mode == ContractMode.CODE_LAW:
                    nl_contract = contract.metadata.get("nl_contract", "")
                    safe_nl = nl_contract.replace("{", "{{").replace("}", "}}")
                    contract_context = f"""

**Constitution (enforceable):**
{safe_nl}

This constitution will be enforced after your decision. Actions that violate the constitution will be modified and penalties applied. Compliance is in your best interest.
"""
                elif contract_mode == ContractMode.WELFARE_OPTIMAL_ENFORCED:
                    safe_content = contract.content.replace("{", "{{").replace("}", "}}")
                    contract_context = f"""

**Welfare-Optimal Contract:**
{safe_content}

The constitution describes specific rules for the community. It will be enforced at its best effort.
"""
                    prompt_template = contract_context + prompt_template
                elif contract_mode == ContractMode.WELFARE_OPTIMAL_UNENFORCED:
                    safe_content = contract.content.replace("{", "{{").replace("}", "}}")
                    contract_context = f"""

**Welfare-Optimal Contract:**
{safe_content}

The constitution describes general rules of the community. It won't be enforced.
"""
                    prompt_template = contract_context + prompt_template
                else:
                    safe_content = contract.content.replace("{", "{{").replace("}", "}}")
                    contract_context = f"""

**Community Guidelines (mutual agreement):**
{safe_content}

Both parties have agreed to these guidelines. Honor your commitment and follow the agreed terms.
"""
                prompt_template = contract_context + prompt_template

        row_message = copy.deepcopy(messages) + [
            ChatMessageUser(
                role="user",
                content=prompt_template.format(
                    scenario=sanitize_text_for_api(task.metadata["story_row"]),
                    action_list=action_list_row,
                    effort_clarification=effort_clarification,
                )
            )
        ]

        column_message = copy.deepcopy(messages) + [
            ChatMessageUser(
                role="user",
                content=prompt_template.format(
                    scenario=sanitize_text_for_api(task.metadata["story_col"]),
                    action_list=action_list_col,
                    effort_clarification=effort_clarification,
                )
            )
        ]

        async def generate_and_catch(message, role: str):
            started_at = time.perf_counter()
            try:
                output = await model.generate(message)
                ended_at = time.perf_counter()
                telemetry = record_phase_call(
                    phase="decision",
                    role=role,
                    output=output,
                    started_at=started_at,
                    ended_at=ended_at,
                    model=model,
                    config=decision_model_config or {},
                )
                return output, telemetry
            except Exception as e:
                ended_at = time.perf_counter()
                print(f"Error during generation: {e}", file=sys.stderr)
                output = ModelOutput(choices=[ChatCompletionChoice(message=ChatMessageAssistant(role="assistant", content=f"ANSWER: none, error during generation, {e}"))])
                telemetry = record_phase_call(
                    phase="decision",
                    role=role,
                    started_at=started_at,
                    ended_at=ended_at,
                    model=model,
                    config=decision_model_config or {},
                    error=f"{type(e).__name__}: {e}",
                )
                return output, telemetry

        tasks_list = []
        for _ in range(times):
            tasks_list.append(generate_and_catch(row_message, "row"))
            tasks_list.append(generate_and_catch(column_message, "column"))

        result_records = await asyncio.gather(*tasks_list)
        results: list[ModelOutput] = [record[0] for record in result_records]
        decision_telemetry = [record[1] for record in result_records]

        row_combinations = combinations_row if is_4x4 else None
        col_combinations = combinations_col if is_4x4 else None

        original_row_action = _parse_action(results[0].completion, row_actions, is_4x4, row_combinations)
        original_column_action = _parse_action(results[1].completion, column_actions, is_4x4, col_combinations)

        task.metadata["original_row_action"] = original_row_action
        task.metadata["original_column_action"] = original_column_action
        task.metadata["decision_outputs"] = {
            "row": results[0].completion,
            "column": results[1].completion,
        }

        row_action_base = extract_base_action(original_row_action)
        column_action_base = extract_base_action(original_column_action)

        # Phase 3: Enforcement (CODE_LAW and WELFARE_OPTIMAL)
        enforcement_result = None
        row_action = original_row_action
        column_action = original_column_action

        if (
            negotiation_result
            and negotiation_result.agreement_reached
            and manager.has_contract()
        ):
            from src.contracting.contract import GameContractState
            contract_state = GameContractState(
                scenario_id=game_state["scenario_id"],
                formal_game=game_state["formal_game"],
                actions_row=game_state["actions_row"],
                actions_column=game_state["actions_column"],
                payoff_matrix=game_state["payoff_matrix"],
                is_4x4=game_state.get("is_4x4", False),
            )

            # Pass base actions to enforcement (effort levels hidden in 4x4)
            enforcement_result = manager.enforce(
                actions={"row": row_action_base, "column": column_action_base},
                state=contract_state,
            )

            enforced_row_base = enforcement_result.modified_actions.get("row", row_action_base)
            enforced_col_base = enforcement_result.modified_actions.get("column", column_action_base)

            # Map enforced base actions back to full actions, preserving original effort
            row_action = map_base_to_full_action(enforced_row_base, original_row_action)
            column_action = map_base_to_full_action(enforced_col_base, original_column_action)

            task.metadata["enforcement_result"] = enforcement_result.to_dict()
        else:
            task.metadata["enforcement_result"] = None

        row_action_base_final = extract_base_action(row_action)
        column_action_base_final = extract_base_action(column_action)
        task.metadata["row_action_full"] = row_action
        task.metadata["column_action_full"] = column_action
        task.metadata["row_action_base"] = row_action_base_final
        task.metadata["column_action_base"] = column_action_base_final
        task.metadata["row_action"] = row_action
        task.metadata["column_action"] = column_action
        task.metadata["contract_mode"] = contract_mode.value

        trace_calls = _collect_trace_calls(task.metadata.get("negotiation_result"))
        trace_calls.extend(decision_telemetry)
        task.metadata["trace_usage"] = summarize_phase_usage(trace_calls)

        choices = [r.choices[0] for r in results]

        # Embed enforced actions in the output for the scorer
        enforced_line = f"\n\nENFORCED: row={row_action}, col={column_action}"

        original_content = results[0].choices[0].message.content
        completion_text = results[0].completion

        if isinstance(original_content, list):
            final_content = original_content + [ContentText(text=enforced_line)]
            final_completion = f"{completion_text}{enforced_line}"
        else:
            final_content = f"{original_content}{enforced_line}"
            final_completion = f"{completion_text}{enforced_line}"

        task.output = ModelOutput(
            choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(
                    role="assistant",
                    content=final_content
                )
            )],
            completion=final_completion,
        )

        return task

    return solve


def _parse_4x4_actions(actions: list[str]) -> tuple[list[str], list[str], dict[str, str]]:
    """Split 4x4 action labels like 'Constrain AI (High Effort)' into base actions,
    effort levels, and a (base, effort) -> full_label mapping."""
    base_actions_set = set()
    combinations = {}

    for action in actions:
        if "(" in action and ")" in action:
            base = action[:action.index("(")].strip()
            effort_label = action[action.index("(")+1:action.index(")")].strip()
            effort = effort_label
            base_actions_set.add(base)
            combinations[(base, effort)] = action

    base_actions = sorted(list(base_actions_set))
    effort_levels = ["High Effort", "Low Effort"]

    return base_actions, effort_levels, combinations


def _build_4x4_action_prompt(
    base_actions: list[str],
    effort_levels: list[str],
    combinations: dict[tuple[str, str], str],
) -> str:
    base_actions_sorted = sorted(base_actions)

    prompt = "**Your Available Choices:**\n\n"
    prompt += "Base Actions:\n"
    for action in base_actions_sorted:
        prompt += f"- {action}\n"

    prompt += "\nEffort Levels:\n"
    for effort in effort_levels:
        prompt += f"- {effort}\n"

    prompt += "\n**Action Combinations:**\n"
    prompt += "When you choose (Base Action, Effort Level), it maps to:\n"

    # Sort combinations for consistent output
    sorted_combinations = sorted(combinations.items(), key=lambda x: (x[0][0], x[0][1]))
    for (base, effort), full_label in sorted_combinations:
        prompt += f"- ({base}, {effort}) → {full_label}\n"

    prompt += "\n**Your Task:**\n"
    prompt += "Choose (Base Action, Effort Level) for this scenario.\n\n"
    prompt += "Answer format: ANSWER: (Base Action, Effort Level)\n"

    # Add balanced examples (show both effort levels are valid)
    if base_actions_sorted and effort_levels:
        prompt += "\nExamples:\n"
        prompt += f"- ANSWER: ({base_actions_sorted[0]}, {effort_levels[0]})\n"
        prompt += f"- ANSWER: ({base_actions_sorted[0]}, {effort_levels[1]})\n"

    return prompt


def _parse_action(
    completion: str,
    available_actions: list[str],
    is_4x4: bool = False,
    combinations: dict[tuple[str, str], str] | None = None,
) -> str:
    completion_lower = completion.lower()

    if "answer:" in completion_lower:
        after_answer = completion_lower.split("answer:", 1)[1].strip()

        if is_4x4 and combinations:
            tuple_match = re.search(r'\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)', after_answer)
            if tuple_match:
                base = tuple_match.group(1).strip()
                effort = tuple_match.group(2).strip()

                for (combo_base, combo_effort), full_action in combinations.items():
                    if (base.lower() in combo_base.lower() and
                        effort.lower() in combo_effort.lower()):
                        return full_action

        for action in available_actions:
            if action.lower() in after_answer[:200]:
                return action

    for action in available_actions:
        if action.lower() in completion_lower:
            return action

    return available_actions[0]


def parse_actions(actions_str) -> list:
    """Parse actions from CSV storage format (string repr of list or actual list)."""
    import ast
    if isinstance(actions_str, list):
        return actions_str
    if isinstance(actions_str, str):
        try:
            return ast.literal_eval(actions_str)
        except (ValueError, SyntaxError):
            return [a.strip().strip("'\"") for a in actions_str.strip("[]").split(",")]
    return []


def _collect_trace_calls(negotiation_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Collect telemetry records from negotiation/coding conversation events."""
    if not negotiation_result:
        return []

    calls: list[dict[str, Any]] = []
    for event in negotiation_result.get("conversations", []):
        if not isinstance(event, dict):
            continue
        telemetry = event.get("telemetry")
        if isinstance(telemetry, dict) and telemetry:
            calls.append(telemetry)
    return calls


def record_to_sample_with_contracting(
    record: dict[str, Any],
    contract_mode: ContractMode = ContractMode.NO_COMMUNICATION,
    prompt_mode: str = "base",
) -> Sample:
    is_4x4 = "actions_row_4x4" in record and record.get("actions_row_4x4")

    if is_4x4:
        actions_row = parse_actions(record.get("actions_row_4x4", []))
        actions_column = parse_actions(record.get("actions_col_4x4", []))
        target = record.get("target_nash_4x4", "")

        metadata = {
            "id": record.get("id", ""),
            "base_scenario_id": record.get("base_scenario_id") or record.get("source_id") or record.get("id", ""),
            "story_row": record.get("story_row", ""),
            "story_col": record.get("story_col", ""),
            "actions_row": actions_row,
            "actions_column": actions_column,
            "actions_row_4x4": record.get("actions_row_4x4", []),
            "actions_col_4x4": record.get("actions_col_4x4", []),
            "payoff_matrix_4x4": record.get("payoff_matrix_4x4", "[]"),
            "formal_game": record.get("formal_game", ""),
            "target_nash": record.get("target_nash_4x4", ""),
            "target_utilitarian": record.get("target_utilitarian_4x4", ""),
            "target_rawlsian": record.get("target_rawlsian_4x4", ""),
            "contract_mode": contract_mode.value,
            "is_4x4": True,
            "prompt_mode": prompt_mode,
            "contract_content": record.get("contract_content", ""),
            "contract_proposer": record.get("contract_proposer", ""),
        }
    else:
        actions_row = parse_actions(record.get("actions_row", []))
        actions_column = parse_actions(record.get("actions_column", []))

        target_nash = record.get("target_nash_equilibria")
        target_util = record.get("target_utility_maximizing")
        target_rawlsian = record.get("target_rawlsian")

        target = target_nash

        metadata = {
            "id": record.get("id", ""),
            "base_scenario_id": record.get("base_scenario_id") or record.get("source_id") or record.get("id", ""),
            "story_row": record.get("story_row", ""),
            "story_col": record.get("story_col", ""),
            "actions_row": actions_row,
            "actions_column": actions_column,
            "formal_game": record.get("formal_game", ""),
            "1_1_payoff": record.get("1_1_payoff", []),
            "1_2_payoff": record.get("1_2_payoff", []),
            "2_1_payoff": record.get("2_1_payoff", []),
            "2_2_payoff": record.get("2_2_payoff", []),
            "target_nash": target_nash,
            "target_utilitarian": target_util,
            "target_rawlsian": target_rawlsian,
            "contract_mode": contract_mode.value,
            "is_4x4": False,
            "prompt_mode": prompt_mode,
            "contract_content": record.get("contract_content", ""),
            "contract_proposer": record.get("contract_proposer", ""),
        }

    return Sample(
        id=str(record.get("id", "")),
        input="",
        target=target,
        metadata=metadata,
    )
