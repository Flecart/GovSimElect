"""Contract dataclasses for fishing formal-contract experiments."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ContractMode(str, Enum):
  CODE_LAW = "code_law"
  CODE_NL = "code_nl"
  FREE_CHAT = "free_chat"
  GOVSIM = "govsim"
  NO_COMMUNICATION = "no_communication"


class NegotiationProtocol(str, Enum):
  MAYORAL_VOTING = "mayoral_voting"
  ROUND_ROBIN = "round_robin"


class ContractType(str, Enum):
  NATURAL_LANGUAGE = "nl"
  PYTHON_LAW = "python_law"
  NO_CONTRACT = "no_contract"


@dataclass
class FishingContractState:
  round_number: int
  fish_population: float
  sustainability_threshold: float
  num_agents: int
  max_rounds: int
  speaker_idx: int = 0
  regime: str | None = None  # "healthy", "degraded", or None for non-hysteresis
  setting_context: str = ""
  negotiation_max_turns: int = 10  # restaurant-phase speaking turns (round-robin / free-chat cap)


@dataclass
class FormalContract:
  """Stored negotiation/coding result (NL or Python source), distinct from deployed `Contract` law class."""

  contract_type: ContractType
  content: str
  proposer: str
  round_created: int
  enforcement_status: str = "pending"
  metadata: dict[str, Any] = field(default_factory=dict)
  voting_scheme: str | None = None
  votes: dict[str, bool] = field(default_factory=dict)
  agreements: dict[str, str] = field(default_factory=dict)
  passed: bool = False
  consensus_threshold: int | None = None
  agent_names: list[str] = field(default_factory=list)
  conversation_history: list[dict[str, Any]] = field(default_factory=list)

  @property
  def contract_str(self) -> str:
    return self.content

  def to_dict(self) -> dict[str, Any]:
    return {
        "contract_type": self.contract_type.value,
        "content": self.content,
        "proposer": self.proposer,
        "round_created": self.round_created,
        "enforcement_status": self.enforcement_status,
        "metadata": self.metadata,
        "voting_scheme": self.voting_scheme,
        "votes": self.votes,
        "agreements": self.agreements,
        "passed": self.passed,
        "consensus_threshold": self.consensus_threshold,
        "agent_names": self.agent_names,
        "conversation_history": self.conversation_history,
    }


@dataclass
class NLNegotiationResult:
  nl_contract: str | None
  conversations: list[dict[str, Any]]
  votes: dict[str, bool] | None = None
  agreements: dict[str, str] | None = None


@dataclass
class EnforcementResult:
  success: bool
  modified_catches: dict[str, float]
  reasoning: str
  reward_adjustments: dict[str, float] = field(default_factory=dict)
  execution_log: list[str] = field(default_factory=list)
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {
        "success": self.success,
        "modified_catches": self.modified_catches,
        "reasoning": self.reasoning,
        "reward_adjustments": self.reward_adjustments,
        "execution_log": self.execution_log,
        "metadata": self.metadata,
    }
