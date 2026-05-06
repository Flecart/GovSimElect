"""Contract dataclasses for GT-HarmBench contracting evaluation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ContractMode(str, Enum):
    """Contracting mode for the evaluation."""
    NO_COMMUNICATION = "no_communication"  # No contracting, baseline
    CODE_NL = "code_nl"  # Natural language contracting
    CODE_LAW = "code_law"  # Python code-law contracting
    WELFARE_OPTIMAL_ENFORCED = "welfare_optimal_enforced"  # Monitoring effect
    WELFARE_OPTIMAL_UNENFORCED = "welfare_optimal_unenforced"  # Monitoring effect=


class ContractType(str, Enum):
    """Type of contract content."""
    NO_CONTRACT = "no_contract"
    NATURAL_LANGUAGE = "nl"
    PYTHON_LAW = "python_law"


@dataclass
class Contract:
    """A contract agreed upon by two players.

    Attributes:
        contract_type: The type of contract (NL or Python law).
        content: The contract text or code.
        proposer: Which player proposed the contract ("row" or "column").
        enforcement_status: Whether the contract is pending, active, or failed.
        conversation_history: List of conversation turns that led to the contract.
        agreement_round: Which round/turn the contract was agreed upon.
        metadata: Additional data about the contract.
    """
    contract_type: ContractType
    content: str
    proposer: str  # "row" or "column"
    enforcement_status: str = "pending"
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    agreement_round: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def contract_str(self) -> str:
        """Get the contract content as a string."""
        return self.content

    def to_dict(self) -> dict[str, Any]:
        """Convert contract to dictionary for serialization."""
        return {
            "contract_type": self.contract_type.value,
            "content": self.content,
            "proposer": self.proposer,
            "enforcement_status": self.enforcement_status,
            "conversation_history": self.conversation_history,
            "agreement_round": self.agreement_round,
            "metadata": self.metadata,
        }


@dataclass
class NegotiationResult:
    """Result of a negotiation phase.

    Attributes:
        contract: The agreed-upon contract, or None if no agreement.
        conversations: List of conversation turns during negotiation.
        agreement_reached: Whether the players reached agreement.
        turns_taken: Number of turns taken in negotiation.
        metadata: Optional metadata (e.g., coding phase statistics).
    """
    contract: Contract | None
    conversations: list[dict[str, Any]]
    agreement_reached: bool
    turns_taken: int
    metadata: dict[str, Any]

    def __init__(
        self,
        contract: Contract | None,
        conversations: list[dict[str, Any]],
        agreement_reached: bool,
        turns_taken: int,
        metadata: dict[str, Any] | None = None,
    ):
        self.contract = contract
        self.conversations = conversations
        self.agreement_reached = agreement_reached
        self.turns_taken = turns_taken
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "contract": self.contract.to_dict() if self.contract else None,
            "conversations": self.conversations,
            "agreement_reached": self.agreement_reached,
            "turns_taken": self.turns_taken,
            "metadata": self.metadata,
        }


@dataclass
class EnforcementResult:
    """Result of contract enforcement.

    Attributes:
        success: Whether enforcement was successful.
        modified_actions: The enforced actions for each player.
        reasoning: Explanation of the enforcement decision.
        violations_detected: List of violations found (if any).
        execution_log: Log of execution steps (for Python contracts).
        metadata: Additional data about the enforcement.
        payoff_adjustments: Payoff adjustments from fines and transfers.
            Structure: {"row": {"fines": [], "received": [], "sent": []},
                        "column": {"fines": [], "received": [], "sent": []}}
    """
    success: bool
    modified_actions: dict[str, str]  # {"row": "action1", "col": "action2"}
    reasoning: str
    violations_detected: list[str] = field(default_factory=list)
    execution_log: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    payoff_adjustments: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "modified_actions": self.modified_actions,
            "reasoning": self.reasoning,
            "violations_detected": self.violations_detected,
            "execution_log": self.execution_log,
            "metadata": self.metadata,
            "payoff_adjustments": self.payoff_adjustments,
        }


@dataclass
class GameContractState:
    """State information for contracting in a game scenario.

    Similar to FishingContractState in mech-design, but adapted for general 2-player games.

    Attributes:
        scenario_id: ID of the scenario being played.
        formal_game: Type of formal game (e.g., "Prisoner's Dilemma").
        actions_row: Available actions for row player.
        actions_column: Available actions for column player.
        payoff_matrix: Payoff matrix for the game.
        is_4x4: Whether this is a 4x4 moral hazard scenario (hidden effort unobservable).
    """
    scenario_id: str | None
    formal_game: str
    actions_row: list[str]
    actions_column: list[str]
    payoff_matrix: dict[str, Any]  # Contains 1_1_payoff, 1_2_payoff, etc. or 4x4 format
    is_4x4: bool = False  # Default False for backward compatibility
