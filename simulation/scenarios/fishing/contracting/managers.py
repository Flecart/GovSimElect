"""Contract managers for fishing formal-contract experiments."""

import re
from abc import ABC, abstractmethod
from typing import Any

from .contract import (
    ContractType,
    EnforcementResult,
    FishingContractState,
    FormalContract,
)
from .enforcers import (
    ContractEnforcer,
    LawEnforcer,
    NoOpEnforcer,
)


class ContractManager(ABC):
  def __init__(self) -> None:
    self._contract: FormalContract | None = None
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
    # Allow explicit, targeted agreement forms like <AGREE Alice>.
    return re.search(r"<AGREE(?:\s[^>]*)?>", message, re.IGNORECASE) is not None

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

  def get_contract(self) -> FormalContract | None:
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
    # Preserve agreement signals in extracted text so downstream logic can
    # inspect whether assent was generic or targeted.
    cleaned = re.sub(r"\s+", " ", message).strip()
    return cleaned or None

  def set_contract(
      self,
      content: str,
      agreed_at_round: int,
      agent_names: list[str],
  ) -> None:
    self._execution_state = {}
    self._contract = FormalContract(
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
  def __init__(self) -> None:
    super().__init__()
    self._deployed_law: Any = None
    self._framework_store: Any = None  # FrameworkOwnedStore, set lazily

  def _ensure_framework(self) -> Any:
    from .enforcement.framework_store import FrameworkOwnedStore

    if self._framework_store is None:
      self._framework_store = FrameworkOwnedStore.from_dict(
          self._execution_state.get("framework")
          if isinstance(self._execution_state, dict)
          else None,
      )
    return self._framework_store

  def _deploy_python_law(self, source: str, agent_names: list[str]) -> None:
    from .enforcement.ast_check import validate_law_source
    from .enforcement.contract import Contract as LawContractBase
    from .enforcement.deploy import (
        discover_contract_subclass,
        exec_law_source,
        instantiate_contract,
    )

    validate_law_source(source)
    namespace = exec_law_source(source, LawContractBase)
    cls = discover_contract_subclass(namespace)
    old = self._deployed_law
    disc: list[str] = []
    prior = dict(old.state) if old is not None else None
    self._deployed_law = instantiate_contract(
        cls,
        num_agents=len(agent_names),
        agent_names=list(agent_names),
        prior_state=prior,
        old_instance=old,
        discontinuity_log=disc,
    )
    if self._contract is not None and disc:
      log = self._contract.metadata.setdefault("law_deploy_log", [])
      for msg in disc:
        log.append(msg)

  def set_contract(
      self,
      content: str,
      agreed_at_round: int,
      agent_names: list[str],
  ) -> None:
    self._execution_state = {}
    self._contract = FormalContract(
        contract_type=ContractType.PYTHON_LAW,
        content=content,
        proposer="coding_agent",
        round_created=agreed_at_round,
        enforcement_status="active",
        agent_names=agent_names,
        conversation_history=list(self._conversation),
    )
    self._deploy_python_law(content, agent_names)

  def clear_contract(self) -> None:
    super().clear_contract()
    self._deployed_law = None
    self._framework_store = None

  def on_post_regen(self, env: Any) -> None:
    """After fish reproduce: ``on_round_end`` on the deployed law."""
    if self._deployed_law is None or not self.has_contract():
      return
    from .enforcers import LawEnforcer

    LawEnforcer().run_on_round_end(
        law=self._deployed_law,
        framework_store=self._ensure_framework(),
        env=env,
        formal_contract=self._contract,
    )
    fs = self._ensure_framework()
    self._execution_state = {
        "framework": fs.to_dict(),
        "contract_bespoke_state": (
            dict(self._deployed_law.state) if self._deployed_law is not None else {}
        ),
    }

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
        deployed_instance=self._deployed_law,
        framework_store=self._ensure_framework(),
    )
    next_state = result.metadata.get("execution_state")
    if isinstance(next_state, dict):
      self._execution_state = next_state
    return result

  def extract_from_message(
      self,
      message: str,
      require_agree_tag: bool = True,
  ) -> str | None:
    if not self._validate_agree_tag(message, require_agree_tag):
      return None
    return self._extract_code_block(message)

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

  def get_contract(self) -> FormalContract | None:
    return None

  def get_enforcer(self) -> ContractEnforcer:
    return NoOpEnforcer()
