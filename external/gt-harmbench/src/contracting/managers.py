"""Contract managers for GT-HarmBench contracting evaluation."""

import re
from abc import ABC, abstractmethod
from typing import Any

from .contract import (
    Contract,
    ContractMode,
    ContractType,
    EnforcementResult,
    GameContractState,
    NegotiationResult,
)


class ContractManager(ABC):
    """Abstract base class for contract managers.

    Manages the contract lifecycle: negotiation, storage, and enforcement.
    """

    def __init__(self) -> None:
        self._contract: Contract | None = None
        self._conversation: list[dict[str, Any]] = []
        self._execution_state: dict[str, Any] = {}

    def _extract_code_block(self, message: str) -> str | None:
        """Extract code block from a message.

        Looks for ```python or ``` code blocks and returns the content.
        """
        matches = re.findall(r"```(?:\w+)?\s*\n?(.*?)```", message, re.DOTALL)
        if matches:
            return matches[0].strip()
        return None

    def _validate_agree_tag(self, message: str, require_tag: bool) -> bool:
        """Check if message contains AGREE tag if required."""
        if not require_tag:
            return True
        return "<AGREE>" in message.upper()

    @abstractmethod
    def extract_from_message(
        self,
        message: str,
        require_agree_tag: bool = True,
    ) -> str | None:
        """Extract contract content from a message.

        Args:
            message: The message to extract from.
            require_agree_tag: Whether to require <AGREE> tag.

        Returns:
            The extracted contract content, or None if invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def set_contract(
        self,
        content: str,
        proposer: str,
        conversation_history: list[dict[str, Any]],
    ) -> None:
        """Set the active contract.

        Args:
            content: The contract content.
            proposer: Which player proposed the contract ("row" or "column").
            conversation_history: The conversation that led to the contract.
        """
        raise NotImplementedError

    @abstractmethod
    def get_enforcer(self) -> "ContractEnforcer":
        """Get the enforcer for this contract type."""
        raise NotImplementedError

    def get_enforcement_context(self) -> str:
        """Get the conversation context for enforcement."""
        if not self._conversation:
            return ""
        return "\n".join(
            f"{turn['player']}: {turn['message']}" for turn in self._conversation
        )

    def has_contract(self) -> bool:
        """Check if there is an active contract."""
        return self._contract is not None

    def get_contract(self) -> Contract | None:
        """Get the active contract."""
        return self._contract

    def add_conversation_turn(
        self,
        player: str,
        message: str,
        turn_number: int,
    ) -> None:
        """Add a conversation turn to history."""
        self._conversation.append(
            {
                "player": player,
                "message": message,
                "turn": turn_number,
            }
        )

    def clear_contract(self) -> None:
        """Clear the active contract and conversation."""
        self._contract = None
        self._conversation = []
        self._execution_state = {}

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self._conversation = []

    def get_execution_state(self) -> dict[str, Any]:
        """Get the execution state (for Python contracts)."""
        return self._execution_state

    def enforce(
        self,
        actions: dict[str, str],
        state: GameContractState,
    ) -> EnforcementResult:
        """Enforce the contract on the given actions.

        Args:
            actions: The chosen actions for each player.
            state: The game state.

        Returns:
            EnforcementResult with potentially modified actions.
        """
        if not self.has_contract():
            return EnforcementResult(
                success=True,
                modified_actions=actions,
                reasoning="No active contract.",
            )
        result = self.get_enforcer().enforce(
            contract=self._contract,
            actions=actions,
            state=state,
            context=self.get_enforcement_context(),
            execution_state=self._execution_state,
        )
        # Update execution state if provided
        next_state = result.metadata.get("execution_state")
        if isinstance(next_state, dict):
            self._execution_state = next_state
        return result


class NLContractManager(ContractManager):
    """Manager for natural language contracts.

    Natural-language contracts are carried into the decision prompt, but are not
    formally enforced after decisions.
    """

    def __init__(self, temperature: float = 0.3) -> None:
        """Initialize the NL contract manager.

        Args:
            temperature: Temperature for judgment generation (default 0.3).
        """
        super().__init__()
        self.temperature = temperature

    def extract_from_message(
        self,
        message: str,
        require_agree_tag: bool = True,
    ) -> str | None:
        """Extract contract from natural language message."""
        if not self._validate_agree_tag(message, require_agree_tag):
            return None
        # Remove <AGREE> tag and clean up
        cleaned = re.sub(r"<AGREE[^>]*>", "", message, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned or None

    def set_contract(
        self,
        content: str,
        proposer: str,
        conversation_history: list[dict[str, Any]],
    ) -> None:
        """Set a natural language contract."""
        self._execution_state = {}
        self._contract = Contract(
            contract_type=ContractType.NATURAL_LANGUAGE,
            content=content,
            proposer=proposer,
            enforcement_status="active",
            conversation_history=list(conversation_history),
        )

    def get_enforcer(self) -> "ContractEnforcer":
        """Get the no-op enforcer."""
        from .enforcers import NoOpEnforcer
        return NoOpEnforcer()


class LawContractManager(ContractManager):
    """Manager for Python code-law contracts.

    Executes Python code to enforce contracts.
    """

    def extract_from_message(
        self,
        message: str,
        require_agree_tag: bool = True,
    ) -> str | None:
        """Extract Python code from message."""
        if not self._validate_agree_tag(message, require_agree_tag):
            return None
        return self._extract_code_block(message)

    def set_contract(
        self,
        content: str,
        proposer: str,
        conversation_history: list[dict[str, Any]],
    ) -> None:
        """Set a Python code-law contract."""
        self._execution_state = {}
        self._contract = Contract(
            contract_type=ContractType.PYTHON_LAW,
            content=content,
            proposer=proposer,
            enforcement_status="active",
            conversation_history=list(conversation_history),
        )

    def get_enforcer(self) -> "ContractEnforcer":
        """Get the Python law enforcer."""
        from .enforcers import LawEnforcer
        return LawEnforcer()


class NoCommsManager(ContractManager):
    """Manager for no-communication baseline.

    Does not support contracting.
    """

    def extract_from_message(
        self,
        message: str,
        require_agree_tag: bool = True,
    ) -> str | None:
        """No contract extraction in no-communication mode."""
        return None

    def set_contract(
        self,
        content: str,
        proposer: str,
        conversation_history: list[dict[str, Any]],
    ) -> None:
        """No contract setting in no-communication mode."""
        return None

    def has_contract(self) -> bool:
        """Always returns False - no contracts allowed."""
        return False

    def get_contract(self) -> Contract | None:
        """Always returns None - no contracts allowed."""
        return None

    def get_enforcer(self) -> "ContractEnforcer":
        """Get the no-op enforcer."""
        from .enforcers import NoOpEnforcer
        return NoOpEnforcer()


class WelfareOptimalContractManager(ContractManager):
    """Manager for pre-written welfare-optimal contracts.

    Sets a pre-written contract without negotiation and uses monitoring-only
    enforcement (tracks violations without modifying actions).
    """

    def __init__(self) -> None:
        """Initialize the welfare-optimal contract manager."""
        super().__init__()

    def extract_from_message(
        self,
        message: str,
        require_agree_tag: bool = True,
    ) -> str | None:
        """No contract extraction - contracts are pre-written."""
        return None

    def set_contract(
        self,
        content: str,
        proposer: str,
        conversation_history: list[dict[str, Any]],
    ) -> None:
        """Set a pre-written welfare-optimal contract.

        Args:
            content: The contract content (natural language description).
            proposer: Which player proposed the contract (should be "system").
            conversation_history: Empty list for pre-written contracts.
        """
        self._execution_state = {}
        self._contract = Contract(
            contract_type=ContractType.PYTHON_LAW,  # Use PYTHON_LAW for monitoring enforcer
            content=content,
            proposer=proposer,
            enforcement_status="active",
            conversation_history=list(conversation_history),
        )

    def get_enforcer(self) -> "ContractEnforcer":
        """Get the monitoring-only enforcer."""
        from .enforcers import MonitoringEnforcer
        return MonitoringEnforcer()


def create_contract_manager(mode: ContractMode) -> ContractManager:
    """Factory function to create a contract manager.

    Args:
        mode: The contracting mode.

    Returns:
        A ContractManager instance appropriate for the mode.
    """
    if mode == ContractMode.NO_COMMUNICATION:
        return NoCommsManager()
    elif mode == ContractMode.CODE_NL:
        return NLContractManager()
    elif mode == ContractMode.CODE_LAW:
        return LawContractManager()
    elif mode in (ContractMode.WELFARE_OPTIMAL_ENFORCED, ContractMode.WELFARE_OPTIMAL_UNENFORCED):
        return WelfareOptimalContractManager()
    else:
        raise ValueError(f"Unknown contract mode: {mode}")
