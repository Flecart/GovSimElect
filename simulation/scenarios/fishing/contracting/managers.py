"""Contract managers for fishing formal-contract experiments."""

import re
from abc import ABC, abstractmethod
from typing import Any

from .contract import (
    Contract,
    ContractType,
    EnforcementResult,
    FishingContractState,
)
from .enforcers import (
    ContractEnforcer,
    LawEnforcer,
    NoOpEnforcer,
)


class ContractManager(ABC):
  def __init__(self) -> None:
    self._contract: Contract | None = None
    self._conversation: list[dict[str, Any]] = []
    self._execution_state: dict[str, Any] = {}

  def _extract_code_block(self, message: str) -> str | None:
    matches = re.findall(r"```(?:\w+)?\s*\n?(.*?)```", message, re.DOTALL)
    if matches:
      return matches[0].strip()
    return None

  def _validate_agree_tag(self, message: str, require_tag: bool) -> bool:
    if not require_tag:
      return True
    return "<AGREE>" in message.upper()

  @abstractmethod
  def extract_from_message(
      self,
      message: str,
      require_agree_tag: bool = True,
  ) -> str | None:
    raise NotImplementedError

  @abstractmethod
  def set_contract(
      self,
      content: str,
      agreed_at_round: int,
      agent_names: list[str],
  ) -> None:
    raise NotImplementedError

  @abstractmethod
  def get_enforcer(self) -> ContractEnforcer:
    raise NotImplementedError

  def get_enforcement_context(self) -> str:
    if not self._conversation:
      return ""
    return "\n".join(
        f"{turn['agent']}: {turn['message']}" for turn in self._conversation
    )

  def has_contract(self) -> bool:
    return self._contract is not None

  def get_contract(self) -> Contract | None:
    return self._contract

  def set_voting_data(
      self,
      voting_scheme: str,
      votes: dict[str, bool] | None = None,
      agreements: dict[str, str] | None = None,
      passed: bool = False,
      consensus_threshold: int | None = None,
  ) -> None:
    if self._contract is None:
      return
    self._contract.voting_scheme = voting_scheme
    if votes is not None:
      self._contract.votes = votes
    if agreements is not None:
      self._contract.agreements = agreements
    self._contract.passed = passed
    self._contract.consensus_threshold = consensus_threshold

  def add_conversation_turn(
      self,
      agent_name: str,
      message: str,
      turn_number: int,
      phase: str,
      html: str = "",
  ) -> None:
    self._conversation.append(
        {
            "agent": agent_name,
            "message": message,
            "turn": turn_number,
            "phase": phase,
            "html": html,
        }
    )

  def clear_contract(self) -> None:
    self._contract = None
    self._conversation = []
    self._execution_state = {}

  def clear_conversation(self) -> None:
    self._conversation = []

  def get_execution_state(self) -> dict[str, Any]:
    return self._execution_state

  def enforce(
      self,
      decisions: dict[str, float],
      state: FishingContractState,
  ) -> EnforcementResult:
    if not self.has_contract():
      return EnforcementResult(
          success=True,
          modified_catches=decisions,
          reasoning="No active contract.",
      )
    result = self.get_enforcer().enforce(
        contract=self._contract,
        decisions=decisions,
        state=state,
        context=self.get_enforcement_context(),
        execution_state=self._execution_state,
    )
    next_state = result.metadata.get("execution_state")
    if isinstance(next_state, dict):
      self._execution_state = next_state
    return result


class NLContractManager(ContractManager):
  def __init__(self) -> None:
    super().__init__()

  def extract_from_message(
      self,
      message: str,
      require_agree_tag: bool = True,
  ) -> str | None:
    if not self._validate_agree_tag(message, require_agree_tag):
      return None
    cleaned = re.sub(r"<AGREE[^>]*>", "", message, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or None

  def set_contract(
      self,
      content: str,
      agreed_at_round: int,
      agent_names: list[str],
  ) -> None:
    self._execution_state = {}
    self._contract = Contract(
        contract_type=ContractType.NATURAL_LANGUAGE,
        content=content,
        proposer="nl_negotiation",
        round_created=agreed_at_round,
        enforcement_status="active",
        agent_names=agent_names,
        conversation_history=list(self._conversation),
    )

  def get_enforcer(self) -> ContractEnforcer:
    return NoOpEnforcer()


class LawContractManager(ContractManager):
  def extract_from_message(
      self,
      message: str,
      require_agree_tag: bool = True,
  ) -> str | None:
    if not self._validate_agree_tag(message, require_agree_tag):
      return None
    return self._extract_code_block(message)

  def set_contract(
      self,
      content: str,
      agreed_at_round: int,
      agent_names: list[str],
  ) -> None:
    self._execution_state = {}
    self._contract = Contract(
        contract_type=ContractType.PYTHON_LAW,
        content=content,
        proposer="coding_agent",
        round_created=agreed_at_round,
        enforcement_status="active",
        agent_names=agent_names,
        conversation_history=list(self._conversation),
    )

  def get_enforcer(self) -> ContractEnforcer:
    return LawEnforcer()


class NoCommsManager(ContractManager):
  def extract_from_message(
      self,
      message: str,
      require_agree_tag: bool = True,
  ) -> str | None:
    return None

  def set_contract(
      self,
      content: str,
      agreed_at_round: int,
      agent_names: list[str],
  ) -> None:
    return None

  def has_contract(self) -> bool:
    return False

  def get_contract(self) -> Contract | None:
    return None

  def get_enforcer(self) -> ContractEnforcer:
    return NoOpEnforcer()
