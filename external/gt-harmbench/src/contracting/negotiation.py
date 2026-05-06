"""Negotiation managers for GT-HarmBench contracting evaluation.

Implements structured 2-player turn-based negotiation with <PROPOSE>, <REASONING>, and <ACCEPT> tags.
"""

import re
import time
from typing import Any

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    get_model,
)

from .contract import Contract, ContractMode, ContractType, NegotiationResult
from .managers import ContractManager
from .prompts import negotiation_system_prompt, negotiation_user_prompt
from .telemetry import record_phase_call


class TwoPlayerNegotiationManager:
    """Manager for 2-player turn-based negotiation with structured tags.

    Protocol:
    - Agent 1: <PROPOSE> contract text <REASONING> reasoning
    - Agent 2: <PROPOSE> counter-contract text <REASONING> reasoning
    - Agent 1: <ACCEPT> (accepts most recent proposal)

    If no agreement within max_turns, no contract is formed.
    """

    def __init__(
        self,
        max_turns: int = 5,
        alternate_starter: bool = True,
        prompt_mode: str = "base",
    ) -> None:
        """Initialize the negotiation manager.

        Args:
            max_turns: Maximum number of negotiation turns.
            alternate_starter: Whether to alternate which player starts first.
            prompt_mode: Prompt mode for preference induction (base/selfish/cooperative).
        """
        self.max_turns = max_turns
        self.alternate_starter = alternate_starter
        self.prompt_mode = prompt_mode

    async def run(
        self,
        task_state: Any,
        manager: ContractManager,
        starter: str | None = None,
    ) -> NegotiationResult:
        """Run the negotiation phase.

        Args:
            task_state: The inspect_ai TaskState with game metadata.
            manager: The contract manager for this negotiation.
            starter: Which player starts ("row" or "column"). If None and
                     alternate_starter is True, alternates based on scenario ID.

        Returns:
            NegotiationResult with the contract (if agreed) and conversation history.
        """
        # Determine starting player
        if starter is None and self.alternate_starter:
            scenario_id = task_state.metadata.get("id", 0)
            try:
                starter = "row" if int(scenario_id) % 2 == 0 else "column"
            except (ValueError, TypeError):
                starter = "row" if hash(str(scenario_id)) % 2 == 0 else "column"
        elif starter is None:
            starter = "row"

        conversations: list[dict[str, Any]] = []
        proposals: list[dict[str, str]] = []  # Track all proposals with reasoning

        current_player = starter

        for turn in range(self.max_turns):
            # Current player makes a proposal or responds
            result = await self._get_player_message(
                task_state=task_state,
                player=current_player,
                conversations=conversations,
                proposals=proposals,
                turn=turn,
            )

            # Parse the response
            parsed = self._parse_message(result["message"], current_player)

            conversations.append({
                "turn": turn,
                "player": current_player,
                "raw_message": result["message"],
                "action": parsed["action"],  # "PROPOSE" or "ACCEPT"
                "contract_text": parsed.get("contract_text"),
                "reasoning": parsed.get("reasoning"),
                "html": result.get("html", ""),
                "telemetry": result.get("telemetry", {}),
            })

            manager.add_conversation_turn(current_player, result["message"], turn)

            # Handle ACCEPT
            if parsed["action"] == "ACCEPT":
                # Accept the most recent proposal
                if proposals:
                    last_proposal = proposals[-1]
                    nl_contract = last_proposal["contract_text"]

                    contract_mode = task_state.metadata.get("contract_mode", "no_communication")

                    # Explicit handling for no_communication mode - should never reach here
                    # since negotiation is skipped for this mode, but fail explicitly if it does
                    if contract_mode == "no_communication":
                        return NegotiationResult(
                            contract=None,
                            conversations=conversations,
                            agreement_reached=False,
                            turns_taken=turn + 1,
                            metadata={"error": "negotiation_should_not_occur_in_no_communication_mode"},
                        )

                    if contract_mode == "code_nl":
                        contract = Contract(
                            contract_type=ContractType.NATURAL_LANGUAGE,
                            content=nl_contract,
                            proposer=last_proposal["proposer"],
                            enforcement_status="active",
                            conversation_history=list(conversations),
                            agreement_round=turn,
                            metadata={
                                "reasoning": last_proposal["reasoning"],
                                "proposer_reasoning": last_proposal["reasoning"],
                            },
                        )
                        manager.set_contract(nl_contract, last_proposal["proposer"], conversations)

                        return NegotiationResult(
                            contract=contract,
                            conversations=conversations,
                            agreement_reached=True,
                            turns_taken=turn + 1,
                        )

                    if contract_mode == "code_law":
                        coding_result = await self._run_coding_phase(
                            task_state=task_state,
                            nl_contract=nl_contract,
                            manager=manager,
                            conversations=conversations,
                            turn=turn,
                        )
                        conversations.extend(coding_result["conversations"])

                        if coding_result["contract"] is not None:
                            contract = coding_result["contract"]
                            manager.set_contract(contract.content, last_proposal["proposer"], conversations)

                            return NegotiationResult(
                                contract=contract,
                                conversations=conversations,
                                agreement_reached=True,
                                turns_taken=turn + 1,
                                metadata={
                                    "coding_phase_stats": coding_result.get("stats", {}),
                                },
                            )
                        else:
                            conversations.append({
                                "turn": turn,
                                "player": "system",
                                "raw_message": "<CODING_PHASE_FAILED> Could not generate a valid Python contract",
                                "action": "CODING_FAILED",
                                "contract_text": None,
                                "reasoning": coding_result.get("error", "coding phase failed"),
                                "html": "",
                            })
                            return NegotiationResult(
                                contract=None,
                                conversations=conversations,
                                agreement_reached=False,
                                turns_taken=len(conversations),
                            )
                    else:
                        # Invalid contract mode - fail explicitly
                        return NegotiationResult(
                            contract=None,
                            conversations=conversations,
                            agreement_reached=False,
                            turns_taken=turn + 1,
                            metadata={"error": f"invalid_contract_mode: {contract_mode}"},
                        )

            # Handle PROPOSE
            if parsed["action"] == "PROPOSE" and parsed.get("contract_text"):
                proposals.append({
                    "turn": turn,
                    "proposer": current_player,
                    "contract_text": parsed["contract_text"],
                    "reasoning": parsed.get("reasoning", ""),
                })

            # Switch players
            current_player = "column" if current_player == "row" else "row"

        # No agreement reached within max_turns
        return NegotiationResult(
            contract=None,
            conversations=conversations,
            agreement_reached=False,
            turns_taken=len(conversations),
        )

    async def _get_player_message(
        self,
        task_state: Any,
        player: str,
        conversations: list[dict],
        proposals: list[dict],
        turn: int,
    ) -> dict[str, str]:
        """Get a message from a player during negotiation."""
        model = get_model()

        metadata = task_state.metadata
        system_prompt = negotiation_system_prompt(
            player=player,
            formal_game=metadata.get("formal_game", "Unknown game"),
            contract_mode=metadata.get("contract_mode", "no_communication"),
            is_4x4=metadata.get("is_4x4", False),
            max_turns=self.max_turns,
            prompt_mode=self.prompt_mode,
        )
        user_prompt = negotiation_user_prompt(
            player=player,
            story=metadata.get(f"story_{player}", ""),
            actions=metadata.get(f"actions_{player}", []),
            is_4x4=metadata.get("is_4x4", False),
            turn=turn,
            conversations=conversations,
            proposals=proposals,
        )

        messages = [
            ChatMessageSystem(content=system_prompt),
            ChatMessageUser(content=user_prompt),
        ]

        started_at = time.perf_counter()
        try:
            response = await model.generate(messages)
            ended_at = time.perf_counter()
            telemetry = record_phase_call(
                phase="negotiation",
                role=player,
                output=response,
                started_at=started_at,
                ended_at=ended_at,
                model=model,
            )
            return {"message": response.completion, "html": "", "telemetry": telemetry}
        except Exception as e:
            ended_at = time.perf_counter()
            telemetry = record_phase_call(
                phase="negotiation",
                role=player,
                started_at=started_at,
                ended_at=ended_at,
                model=model,
                error=f"{type(e).__name__}: {e}",
            )
            # Return error response
            return {
                "message": f"<PROPOSE> Error <REASONING> {e}",
                "html": "",
                "telemetry": telemetry,
            }

    def _parse_message(self, message: str, player: str) -> dict[str, Any]:
        """Parse a player's message to extract action, contract text, and reasoning.

        Args:
            message: The player's message.
            player: Which player sent this message.

        Returns:
            Dict with "action", and optionally "contract_text" and "reasoning".
        """
        # Check for ACCEPT first (simple tag, no content)
        if re.search(r"<ACCEPT>", message, re.IGNORECASE):
            return {"action": "ACCEPT"}

        # Look for PROPOSE tag with REASONING
        propose_match = re.search(
            r"<PROPOSE>\s*(.*?)\s*<REASONING>\s*(.*?)(?:\s*$|<PROPOSE>|\Z)",
            message,
            re.DOTALL | re.IGNORECASE
        )

        if propose_match:
            contract_text = propose_match.group(1).strip()
            reasoning = propose_match.group(2).strip()

            # Clean up any trailing tags or noise
            reasoning = re.sub(r"<[^>]*>$", "", reasoning, flags=re.IGNORECASE).strip()

            return {
                "action": "PROPOSE",
                "contract_text": contract_text,
                "reasoning": reasoning,
            }

        # If no REASONING tag found but PROPOSE exists, try to parse without it
        propose_simple = re.search(r"<PROPOSE>\s*(.*?)\s*(?:$|<(?:ACCEPT|PROPOSE))", message, re.DOTALL | re.IGNORECASE)
        if propose_simple:
            contract_text = propose_simple.group(1).strip()
            return {
                "action": "PROPOSE",
                "contract_text": contract_text,
                "reasoning": "No reasoning provided",
            }

        # Default: treat as a counter-proposal (implicit PROPOSE)
        return {
            "action": "PROPOSE",
            "contract_text": message.strip(),
            "reasoning": "No reasoning provided",
        }

    async def _run_coding_phase(
        self,
        task_state: Any,
        nl_contract: str,
        manager: Any,
        conversations: list[dict[str, Any]],
        turn: int,
    ) -> dict[str, Any]:
        """Translate an agreed NL contract into Python with technical retries.

        Args:
            task_state: The inspect_ai TaskState with game metadata.
            nl_contract: The agreed natural language contract.
            manager: The contract manager.
            conversations: Existing conversation history.
            turn: Current negotiation turn number.

        Returns:
            Dict with:
            - contract: Contract object if succeeded, None if failed
            - conversations: List of coding phase conversation turns
            - stats: Statistics about the coding phase
            - error: Error message if failed
        """
        from .contract import GameContractState, Contract, ContractType
        from .coding_agent import (
            CodingAgent,
            DEFAULT_CODING_AGENT_MODEL,
            DEFAULT_CODING_AGENT_TEMPERATURE,
        )

        agent = CodingAgent(
            model=DEFAULT_CODING_AGENT_MODEL,
            temperature=DEFAULT_CODING_AGENT_TEMPERATURE,
        )

        # Build game state
        # payoff_matrix key differs between 2x2 and 4x4 datasets:
        #   4x4: 'payoff_matrix_4x4' (parsed JSON list-of-lists)
        #   2x2: individual '1_1_payoff', etc. keys; no single 'payoff_matrix' key
        is_4x4 = task_state.metadata.get("is_4x4", False)
        if is_4x4:
            import json as _json
            payoff_matrix = _json.loads(
                task_state.metadata.get("payoff_matrix_4x4", "[]")
            )
        else:
            # 2x2 payoff matrix is stored as separate keys
            payoff_matrix = {
                k: task_state.metadata[k]
                for k in ("1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff")
                if k in task_state.metadata
            }
        game_state = GameContractState(
            scenario_id=task_state.metadata.get("id", ""),
            formal_game=task_state.metadata.get("formal_game", ""),
            actions_row=task_state.metadata.get("actions_row", []),
            actions_column=task_state.metadata.get("actions_column", []),
            payoff_matrix=payoff_matrix,
            is_4x4=is_4x4,
        )

        coding_conversations: list[dict[str, Any]] = []
        stats = {
            "attempts": 0,
            "coding_agent_errors": 0,
            "validation_errors": 0,
            "last_error": None,
        }

        # Get any existing active contracts (for amendments)
        active_contract = manager.get_contract()
        active_nl_law = ""
        active_code_law = ""
        if active_contract is not None:
            active_nl_law = active_contract.metadata.get("nl_contract", active_contract.content)
            if active_contract.contract_type == ContractType.PYTHON_LAW:
                active_code_law = active_contract.content

        max_attempts = 3
        feedback = None

        for attempt in range(max_attempts):
            stats["attempts"] = attempt + 1
            try:
                processed_contract, response, html, telemetry = await agent.translate(
                    nl_contract=nl_contract,
                    mode=ContractMode.CODE_LAW,
                    state=game_state,
                    active_nl_law=active_nl_law,
                    active_code_law=active_code_law,
                    feedback=feedback,
                )

                if processed_contract is None:
                    stats["coding_agent_errors"] += 1
                    feedback = "No Python code block defining enforce() was produced."
                    stats["last_error"] = feedback
                    coding_conversations.append({
                        "turn": turn + attempt,
                        "player": "coding_agent",
                        "message": response,
                        "phase": "coding_agent_translation",
                        "html": html,
                        "error": True,
                        "error_message": feedback,
                        "telemetry": telemetry,
                    })
                    continue

                is_valid, validation_error = self._validate_python_contract(
                    processed_contract,
                    game_state,
                )
                coding_conversations.append({
                    "turn": turn + attempt,
                    "player": "coding_agent",
                    "message": response,
                    "phase": "coding_agent_translation",
                    "html": html,
                    "error": not is_valid,
                    "error_message": validation_error,
                    "telemetry": telemetry,
                })

                if is_valid:
                    contract = Contract(
                        contract_type=ContractType.PYTHON_LAW,
                        content=processed_contract,
                        proposer="coding_agent",
                        enforcement_status="active",
                        conversation_history=list(conversations) + coding_conversations,
                        agreement_round=turn,
                        metadata={
                            "nl_contract": nl_contract,
                            "coding_retries": attempt,
                            "coding_phase_stats": stats,
                            "coding_agent_config": {
                                "model": agent.model_name,
                                "temperature": agent.temperature,
                            },
                        },
                    )
                    return {
                        "contract": contract,
                        "conversations": coding_conversations,
                        "stats": stats,
                    }

                stats["validation_errors"] += 1
                stats["last_error"] = validation_error
                feedback = (
                    "The previous Python contract failed technical validation. "
                    f"Error: {validation_error}"
                )
                if attempt < max_attempts - 1:
                    coding_conversations.append({
                        "turn": turn + attempt,
                        "player": "system",
                        "message": f"<CODING_RETRY> {feedback}",
                        "phase": "coding_retry_feedback",
                        "reasoning": validation_error,
                    })
            except Exception as e:
                import traceback as _tb
                tb_text = _tb.format_exc()
                error_msg = f"{type(e).__name__}: {e}"
                stats["coding_agent_errors"] += 1
                stats["last_error"] = error_msg
                feedback = f"The previous attempt failed with an exception: {error_msg}"
                coding_conversations.append({
                    "turn": turn + attempt,
                    "player": "system",
                    "message": f"<CODING_RETRY> {feedback}",
                    "phase": "coding_retry_feedback",
                    "reasoning": error_msg,
                    "traceback": tb_text,
                    "error": True,
                })

        return {
            "contract": None,
            "conversations": coding_conversations,
            "stats": stats,
            "error": f"Invalid Python code after {max_attempts} attempts: {stats['last_error']}",
        }

    def _validate_python_contract(
        self,
        python_contract: str,
        game_state: Any,
    ) -> tuple[bool, str | None]:
        """Validate that generated Python can run through the law enforcer."""
        from .contract import Contract, ContractType
        from .enforcers import LawEnforcer

        enforcer = LawEnforcer()
        is_valid, ast_error = enforcer._validate_ast(python_contract)
        if not is_valid:
            return False, ast_error

        row_action = game_state.actions_row[0] if game_state.actions_row else ""
        column_action = game_state.actions_column[0] if game_state.actions_column else ""
        result = enforcer.enforce(
            contract=Contract(
                contract_type=ContractType.PYTHON_LAW,
                content=python_contract,
                proposer="coding_agent",
            ),
            actions={"row": row_action, "column": column_action},
            state=game_state,
            context="",
        )
        if not result.success:
            return False, result.reasoning
        return True, None

