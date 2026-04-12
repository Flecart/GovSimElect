"""Enforcement for fishing formal contracts."""

import json
import random
import re
import traceback
from abc import ABC, abstractmethod
from typing import Any

from simulation.utils import ModelWandbWrapper

from .contract import EnforcementResult, FishingContractState, FormalContract
from .enforcement.context import EnforcementContext
from .enforcement.framework_store import FrameworkOwnedStore
from .prompts import nl_judge_prompt


class ContractEnforcer(ABC):
  def __init__(self, component: Any = None) -> None:
    self.component = component

  def _validate_contract(
      self,
      contract: FormalContract | None,
      decisions: dict[str, float],
      contract_type_name: str,
  ) -> EnforcementResult | None:
    if not contract or not contract.contract_str:
      return EnforcementResult(
          success=False,
          modified_catches=decisions,
          reasoning=f"No valid {contract_type_name} contract.",
      )
    return None

  def _handle_exception(
      self,
      exc: Exception,
      decisions: dict[str, float],
      prefix: str,
  ) -> EnforcementResult:
    return EnforcementResult(
        success=False,
        modified_catches=decisions,
        reasoning=f"{prefix}: {exc}",
        metadata={"error": str(exc)},
    )

  @abstractmethod
  def enforce(
      self,
      contract: FormalContract,
      decisions: dict[str, float],
      state: FishingContractState,
      context: str = "",
      execution_state: dict[str, Any] | None = None,
  ) -> EnforcementResult:
    raise NotImplementedError


class NoOpEnforcer(ContractEnforcer):
  def enforce(
      self,
      contract: FormalContract,
      decisions: dict[str, float],
      state: FishingContractState,
      context: str = "",
      execution_state: dict[str, Any] | None = None,
  ) -> EnforcementResult:
    return EnforcementResult(
        success=True,
        modified_catches=decisions,
        reasoning="No contract enforcement configured.",
    )


def _round_rng(store: FrameworkOwnedStore, month: int) -> random.Random:
  if store.law_random_seed is None:
    store.law_random_seed = 0x9E3779B1
  seed = (int(store.law_random_seed) + int(month) * 1_000_003) & 0xFFFFFFFF
  return random.Random(seed)


def _validate_catches_output(
    final: dict[str, float],
    agent_names: list[str],
    submissions: dict[str, float],
) -> tuple[bool, dict[str, float], str]:
  """Return (ok, normalized_or_fallback, error_message)."""
  expected = set(agent_names)
  if set(final.keys()) != expected:
    return (
        False,
        dict(submissions),
        f"resolve() keys {set(final.keys())!r} != expected {expected!r}",
    )
  out: dict[str, float] = {}
  for name in agent_names:
    try:
      v = float(final[name])
    except (TypeError, ValueError):
      return False, dict(submissions), f"Non-numeric catch for {name!r}"
    if v < 0 or v != v:  # nan
      return False, dict(submissions), f"Invalid catch for {name!r}: {final[name]!r}"
    out[name] = v
  return True, out, ""


class LawEnforcer(ContractEnforcer):
  """Runs a deployed ``Contract`` subclass: hooks + ``resolve`` + payoff context."""

  def enforce(
      self,
      contract: FormalContract,
      decisions: dict[str, float],
      state: FishingContractState,
      context: str = "",
      execution_state: dict[str, Any] | None = None,
      deployed_instance: Any = None,
      framework_store: FrameworkOwnedStore | None = None,
  ) -> EnforcementResult:
    del context  # reserved
    if error := self._validate_contract(contract, decisions, "python-law"):
      return error
    if deployed_instance is None or framework_store is None:
      return EnforcementResult(
          success=False,
          modified_catches=decisions,
          reasoning="Python law instance or framework store missing.",
      )

    law = deployed_instance
    agent_names = list(contract.agent_names) if contract.agent_names else sorted(decisions.keys())
    submissions = {n: float(decisions.get(n, 0.0)) for n in agent_names}
    month = state.round_number + 1
    fp = float(state.fish_population)
    rng = _round_rng(framework_store, month)

    reward_adjustments = {a: 0.0 for a in agent_names}
    execution_log: list[str] = []

    def _make_ctx() -> EnforcementContext:
      return EnforcementContext(
          month=month,
          fish_population=fp,
          num_agents=len(agent_names),
          agent_names=agent_names,
          store=framework_store,
          reward_adjustments=reward_adjustments,
          execution_log=execution_log,
          rng=rng,
      )

    try:
      law.on_round_start(month, fp, _make_ctx())
    except Exception as exc:
      tb = traceback.format_exc()
      execution_log.append(f"on_round_start failed: {exc}\n{tb}")

    resolve_ok = True
    reasoning = "Applied Python-law contract."
    try:
      final = law.resolve(month, fp, submissions, _make_ctx())
    except Exception as exc:
      tb = traceback.format_exc()
      contract.metadata["last_enforcement_failure"] = tb
      resolve_ok = False
      reasoning = f"resolve() raised; fail-open on submissions: {exc}"
      final = dict(submissions)
      for a in agent_names:
        reward_adjustments[a] = 0.0
      execution_log.clear()
      execution_log.append(f"RESOLVE EXCEPTION:\n{tb}")

    modified: dict[str, float]
    success = True
    if resolve_ok:
      ok, normalized, err_msg = _validate_catches_output(
          final if isinstance(final, dict) else {},
          agent_names,
          submissions,
      )
      if not ok:
        contract.metadata["last_enforcement_failure"] = err_msg
        reasoning = f"Malformed resolve() output; fail-open: {err_msg}"
        normalized = dict(submissions)
        for a in agent_names:
          reward_adjustments[a] = 0.0
        execution_log.clear()
        execution_log.append(f"MALFORMED RESOLVE OUTPUT: {err_msg}")
        success = False
      modified = normalized
    else:
      modified = final
      success = False

    meta_state = {
        "framework": framework_store.to_dict(),
        "contract_bespoke_state": dict(getattr(law, "state", {}) or {}),
    }
    return EnforcementResult(
        success=success,
        modified_catches=modified,
        reasoning=reasoning,
        reward_adjustments=reward_adjustments,
        execution_log=execution_log,
        metadata={
            "contract_type": "python_law",
            "execution_state": meta_state,
        },
    )

  def run_on_round_end(
      self,
      law: Any,
      framework_store: FrameworkOwnedStore,
      env: Any,
      formal_contract: FormalContract | None,
  ) -> None:
    """Invoke ``on_round_end`` after regeneration (optional hook)."""
    if formal_contract is None:
      return
    month = max(1, int(getattr(env, "num_round", 1)))
    fp_after = float(env.internal_global_state["resource_in_pool"])
    name_map = getattr(env, "agent_id_to_name", {})
    agents = getattr(env, "agents", [])
    agent_names = [str(name_map.get(aid, aid)) for aid in agents]
    final_catches = {
        str(name_map.get(aid, aid)): float(
            env.internal_global_state["last_collected_resource"].get(aid, 0.0)
        )
        for aid in agents
    }
    rng = _round_rng(framework_store, month + 17_000)
    reward_adjustments = {a: 0.0 for a in agent_names}
    execution_log: list[str] = []

    def _make_ctx() -> EnforcementContext:
      return EnforcementContext(
          month=month,
          fish_population=fp_after,
          num_agents=len(agent_names),
          agent_names=agent_names,
          store=framework_store,
          reward_adjustments=reward_adjustments,
          execution_log=execution_log,
          rng=rng,
      )

    try:
      law.on_round_end(month, fp_after, final_catches, _make_ctx())
    except Exception as exc:
      tb = traceback.format_exc()
      formal_contract.metadata.setdefault("round_end_hook_failures", []).append(tb)
      formal_contract.metadata["last_hook_failure"] = str(exc)
      return

    for aid in agents:
      name = str(name_map.get(aid, aid))
      delta = float(reward_adjustments.get(name, 0.0))
      env.rewards[aid] = env.rewards.get(aid, 0.0) + delta


# Legacy code, could be still used if we want to have LLM courts.
class CodeNLJudge:
  """LLM judge for natural-language contracts."""

  def __init__(
      self,
      model: ModelWandbWrapper,
      temperature: float = 0.3,
  ) -> None:
    self.model = model
    self.temperature = temperature

  def evaluate(
      self,
      contract: FormalContract,
      decisions: dict[str, float],
      state: FishingContractState,
      conversation_summary: str = "",
  ) -> EnforcementResult:
    system_prompt, user_prompt = nl_judge_prompt(
        contract_text=contract.contract_str,
        decisions=decisions,
        state=state,
        conversation_summary=conversation_summary,
    )
    session = self.model.start_prompt("ContractJudge", "contracting", "judge_contract")
    session.add_message("system", system_prompt)
    session.add_user(user_prompt)
    response = self.model.complete_prompt(
        session,
        temperature=self.temperature,
        default_value=json.dumps({
            "modified_catches": decisions,
            "reasoning": "The contract is too vague to alter the submitted catches.",
            "violations_detected": [],
        }),
    )
    parsed = self._parse_json_object(response)
    modified = parsed.get("modified_catches", decisions)
    if not isinstance(modified, dict):
      modified = decisions
    normalized = {}
    for agent, amount in decisions.items():
      value = modified.get(agent, amount)
      try:
        normalized[agent] = max(0.0, float(value))
      except (TypeError, ValueError):
        normalized[agent] = float(amount)
    return EnforcementResult(
        success=True,
        modified_catches=normalized,
        reasoning=str(parsed.get("reasoning", "Applied natural-language contract.")),
        metadata={
            "violations_detected": parsed.get("violations_detected", []),
            "raw_response": response,
        },
    )

  def _parse_json_object(self, response: str) -> dict[str, Any]:
    try:
      return json.loads(response)
    except json.JSONDecodeError:
      pass
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
      try:
        return json.loads(match.group(0))
      except json.JSONDecodeError:
        pass
    return {
        "modified_catches": {},
        "reasoning": "Judge returned invalid JSON; leaving catches unchanged.",
        "violations_detected": ["invalid_json"],
    }


class NLJudgeEnforcer(ContractEnforcer):
  def enforce(
      self,
      contract: FormalContract,
      decisions: dict[str, float],
      state: FishingContractState,
      context: str = "",
      execution_state: dict[str, Any] | None = None,
  ) -> EnforcementResult:
    if error := self._validate_contract(contract, decisions, "natural-language"):
      return error
    if self.component is None:
      return EnforcementResult(
          success=False,
          modified_catches=decisions,
          reasoning="No natural-language judge configured.",
      )
    try:
      return self.component.evaluate(
          contract=contract,
          decisions=decisions,
          state=state,
          conversation_summary=context,
      )
    except Exception as exc:
      return self._handle_exception(exc, decisions, "NL judge failed")
