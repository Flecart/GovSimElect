"""Enforcement for fishing formal contracts."""

import ast
import copy
import json
import re
from abc import ABC, abstractmethod
from typing import Any

from simulation.utils import ModelWandbWrapper

from .contract import Contract, EnforcementResult, FishingContractState
from .prompts import nl_judge_prompt


class ContractEnforcer(ABC):
  def __init__(self, component: Any = None) -> None:
    self.component = component

  def _validate_contract(
      self,
      contract: Contract | None,
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
      contract: Contract,
      decisions: dict[str, float],
      state: FishingContractState,
      context: str = "",
      execution_state: dict[str, Any] | None = None,
  ) -> EnforcementResult:
    raise NotImplementedError


class NoOpEnforcer(ContractEnforcer):
  def enforce(
      self,
      contract: Contract,
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


class LawExecutionRuntime:
  """Named helper API available to Python-law contracts."""

  def __init__(
      self,
      catches: dict[str, float],
      execution_state: dict[str, Any] | None = None,
  ) -> None:
    self.catches = catches
    self.reward_adjustments: dict[str, float] = {}
    self.execution_log: list[str] = []
    self.state = copy.deepcopy(execution_state or {})
    self.state.setdefault("escrow", {})
    self.state.setdefault("insurance_pools", {})
    self.state.setdefault("violation_counts", {})
    self.state.setdefault("participation_costs", {})
    self.state.setdefault("transfers", [])

  def _ensure_agent(self, name: str) -> str:
    agent = str(name)
    self.catches.setdefault(agent, 0.0)
    self.reward_adjustments.setdefault(agent, 0.0)
    return agent

  def _adjust_reward(self, name: str, delta: float, label: str) -> None:
    agent = self._ensure_agent(name)
    self.reward_adjustments[agent] = self.reward_adjustments.get(agent, 0.0) + float(delta)
    self.execution_log.append(f"{label}: {agent} {delta:+.2f}")

  def fish(self, name: str, amount: float) -> None:
    agent = self._ensure_agent(name)
    self.catches[agent] = max(0.0, float(amount))
    self.execution_log.append(f"fish({agent}, {float(amount):.2f})")

  def transfer(self, src: str, dst: str, amount: float, reason: str = "") -> None:
    value = max(0.0, float(amount))
    source = self._ensure_agent(src)
    target = self._ensure_agent(dst)
    if value <= 0:
      return
    self._adjust_reward(source, -value, f"transfer_out[{reason or 'unspecified'}]")
    self._adjust_reward(target, value, f"transfer_in[{reason or 'unspecified'}]")
    self.state["transfers"].append({
        "src": source,
        "dst": target,
        "amount": value,
        "reason": reason,
    })

  def escrow(
      self,
      name: str,
      amount: float,
      bucket: str = "default",
      reason: str = "",
  ) -> None:
    value = max(0.0, float(amount))
    agent = self._ensure_agent(name)
    if value <= 0:
      return
    bucket_key = f"{agent}:{bucket}"
    self.state["escrow"][bucket_key] = self.state["escrow"].get(bucket_key, 0.0) + value
    self._adjust_reward(agent, -value, f"escrow[{bucket}]")
    self.execution_log.append(f"escrow reason={reason or 'none'}")

  def release_escrow(
      self,
      name: str,
      amount: float | None = None,
      bucket: str = "default",
      recipient: str | None = None,
      reason: str = "",
  ) -> None:
    agent = self._ensure_agent(name)
    bucket_key = f"{agent}:{bucket}"
    available = float(self.state["escrow"].get(bucket_key, 0.0))
    if available <= 0:
      return
    value = available if amount is None else min(available, max(0.0, float(amount)))
    if value <= 0:
      return
    self.state["escrow"][bucket_key] = available - value
    payee = self._ensure_agent(recipient or agent)
    self._adjust_reward(payee, value, f"release_escrow[{bucket}]")
    self.execution_log.append(f"release_escrow reason={reason or 'none'}")

  def sanction(self, name: str, amount: float, reason: str = "") -> None:
    value = max(0.0, float(amount))
    if value <= 0:
      return
    self._adjust_reward(name, -value, f"sanction[{reason or 'unspecified'}]")

  def graduated_sanction(
      self,
      name: str,
      base_amount: float,
      key: str = "default",
      multiplier: float = 1.0,
      reason: str = "",
  ) -> float:
    agent = self._ensure_agent(name)
    counter_key = f"{agent}:{key}"
    count = int(self.state["violation_counts"].get(counter_key, 0)) + 1
    self.state["violation_counts"][counter_key] = count
    amount = max(0.0, float(base_amount)) * max(0.0, float(multiplier)) * count
    self.sanction(agent, amount, reason or key)
    self.execution_log.append(f"graduated_sanction count={count}")
    return amount

  def insurance(
      self,
      name: str,
      premium: float = 0.0,
      payout: float = 0.0,
      pool: str = "default",
      recipient: str | None = None,
      reason: str = "",
  ) -> None:
    member = self._ensure_agent(name)
    pool_key = str(pool)
    self.state["insurance_pools"].setdefault(pool_key, 0.0)
    premium_value = max(0.0, float(premium))
    payout_value = max(0.0, float(payout))
    if premium_value > 0:
      self.state["insurance_pools"][pool_key] += premium_value
      self._adjust_reward(member, -premium_value, f"insurance_premium[{pool_key}]")
    if payout_value > 0:
      available = float(self.state["insurance_pools"].get(pool_key, 0.0))
      paid = min(available, payout_value)
      if paid > 0:
        self.state["insurance_pools"][pool_key] = available - paid
        self._adjust_reward(recipient or member, paid, f"insurance_payout[{pool_key}]")
    self.execution_log.append(f"insurance reason={reason or 'none'}")

  def participation_cost(self, name: str, amount: float, reason: str = "") -> None:
    value = max(0.0, float(amount))
    if value <= 0:
      return
    agent = self._ensure_agent(name)
    self.state["participation_costs"][agent] = (
        self.state["participation_costs"].get(agent, 0.0) + value
    )
    self._adjust_reward(agent, -value, f"participation_cost[{reason or 'unspecified'}]")


class LawEnforcer(ContractEnforcer):
  def _sanitize_contract_code(self, contract_str: str) -> str:
    lines = contract_str.split("\n")
    sanitized_lines = []
    in_test_section = False
    for line in lines:
      lower = line.lower()
      if any(pattern in lower for pattern in ["# example", "# test", "# usage"]):
        in_test_section = True
        continue
      if in_test_section:
        if line.strip() and not line.strip().startswith("#"):
          in_test_section = False
        else:
          continue
      if re.match(r"^catches\s*=\s*{", line.strip()):
        continue
      sanitized_lines.append(line)

    sanitized = "\n".join(sanitized_lines)
    try:
      tree = ast.parse(sanitized)
      for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
            keyword in node.name.lower()
            for keyword in ["catch", "enforce", "apply", "modify", "regulate", "check"]
        ):
          if node.name == "catch":
            sanitized += (
                "\nfor _name, _amount in list(catches.items()):\n"
                f"    result = {node.name}(_name, _amount, month)\n"
                "    if isinstance(result, (int, float)):\n"
                "        fish(_name, result)\n"
            )
          else:
            sanitized += (
                f"\nresult = {node.name}(catches)\n"
                "if isinstance(result, dict):\n"
                "    catches.update(result)\n"
            )
          break
    except SyntaxError:
      return sanitized
    return sanitized

  def _resolve_catch_callable(
      self,
      exec_globals: dict[str, Any],
      exec_locals: dict[str, Any],
  ):
    catch_callable = exec_locals.get("catch") or exec_globals.get("catch")
    if callable(catch_callable):
      return catch_callable

    candidate_classes = [
        exec_locals.get("LawContract"),
        exec_globals.get("LawContract"),
        exec_locals.get("FishingLaw"),
        exec_globals.get("FishingLaw"),
        exec_locals.get("Contract"),
        exec_globals.get("Contract"),
    ]
    for candidate in candidate_classes:
      if not isinstance(candidate, type):
        continue
      try:
        instance = candidate(contract_state=exec_globals.get("contract_state"))
      except TypeError:
        try:
          instance = candidate()
        except TypeError:
          continue
      catch_method = getattr(instance, "catch", None)
      if callable(catch_method):
        return catch_method
    return None

  def _apply_catch_function(
      self,
      catch_callable,
      catches: dict[str, float],
      month: int,
      fish_callback,
  ) -> None:
    for name, amount in list(catches.items()):
      result = catch_callable(name, amount, month)
      if isinstance(result, (int, float)):
        fish_callback(name, result)

  def enforce(
      self,
      contract: Contract,
      decisions: dict[str, float],
      state: FishingContractState,
      context: str = "",
      execution_state: dict[str, Any] | None = None,
  ) -> EnforcementResult:
    if error := self._validate_contract(contract, decisions, "python-law"):
      return error

    runtime = LawExecutionRuntime(decisions.copy(), execution_state)
    exec_globals = {
        "__builtins__": {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "int": int,
            "float": float,
            "dict": dict,
        },
        "catches": runtime.catches,
        "fish_population": state.fish_population,
        "month": state.round_number + 1,
        "sustainability_threshold": state.sustainability_threshold,
        "num_agents": state.num_agents,
        "contract_state": runtime.state,
        "fish": runtime.fish,
        "transfer": runtime.transfer,
        "escrow": runtime.escrow,
        "release_escrow": runtime.release_escrow,
        "sanction": runtime.sanction,
        "graduated_sanction": runtime.graduated_sanction,
        "insurance": runtime.insurance,
        "participation_cost": runtime.participation_cost,
    }
    exec_locals: dict[str, Any] = {}
    should_apply_catch_callable = True
    try:
      exec(contract.contract_str, exec_globals, exec_locals)
    except Exception:
      should_apply_catch_callable = False
      exec_locals.clear()
      runtime = LawExecutionRuntime(decisions.copy(), execution_state)
      exec_globals["catches"] = runtime.catches
      exec_globals["month"] = state.round_number + 1
      exec_globals["contract_state"] = runtime.state
      exec_globals["fish"] = runtime.fish
      exec_globals["transfer"] = runtime.transfer
      exec_globals["escrow"] = runtime.escrow
      exec_globals["release_escrow"] = runtime.release_escrow
      exec_globals["sanction"] = runtime.sanction
      exec_globals["graduated_sanction"] = runtime.graduated_sanction
      exec_globals["insurance"] = runtime.insurance
      exec_globals["participation_cost"] = runtime.participation_cost
      exec(self._sanitize_contract_code(contract.contract_str), exec_globals, exec_locals)

    catch_callable = self._resolve_catch_callable(exec_globals, exec_locals)
    if should_apply_catch_callable and catch_callable is not None:
      self._apply_catch_function(
          catch_callable=catch_callable,
          catches=runtime.catches,
          month=state.round_number + 1,
          fish_callback=runtime.fish,
      )

    modified = exec_locals.get("modified_catches", exec_globals.get("catches", decisions))
    return EnforcementResult(
        success=True,
        modified_catches=modified,
        reasoning="Applied Python-law contract.",
        reward_adjustments=runtime.reward_adjustments,
        execution_log=runtime.execution_log,
        metadata={
            "contract_type": "python_law",
            "execution_state": runtime.state,
        },
    )


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
      contract: Contract,
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
      contract: Contract,
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
