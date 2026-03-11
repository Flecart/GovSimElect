"""Enforcement for fishing formal contracts."""

import ast
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
  ) -> EnforcementResult:
    raise NotImplementedError


class NoOpEnforcer(ContractEnforcer):
  def enforce(
      self,
      contract: Contract,
      decisions: dict[str, float],
      state: FishingContractState,
      context: str = "",
  ) -> EnforcementResult:
    return EnforcementResult(
        success=True,
        modified_catches=decisions,
        reasoning="No contract enforcement configured.",
    )


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
            for keyword in ["enforce", "apply", "modify", "regulate", "check"]
        ):
          sanitized += (
              f"\nresult = {node.name}(catches)\n"
              "if isinstance(result, dict):\n"
              "    catches.update(result)\n"
          )
          break
    except SyntaxError:
      return sanitized
    return sanitized

  def enforce(
      self,
      contract: Contract,
      decisions: dict[str, float],
      state: FishingContractState,
      context: str = "",
  ) -> EnforcementResult:
    if error := self._validate_contract(contract, decisions, "python-law"):
      return error

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
        "catches": decisions.copy(),
        "fish_population": state.fish_population,
        "sustainability_threshold": state.sustainability_threshold,
        "num_agents": state.num_agents,
    }
    exec_locals: dict[str, Any] = {}
    try:
      exec(contract.contract_str, exec_globals, exec_locals)
    except Exception:
      exec_locals.clear()
      exec_globals["catches"] = decisions.copy()
      exec(self._sanitize_contract_code(contract.contract_str), exec_globals, exec_locals)

    modified = exec_locals.get("modified_catches", exec_globals.get("catches", decisions))
    return EnforcementResult(
        success=True,
        modified_catches=modified,
        reasoning="Applied Python-law contract.",
        metadata={"contract_type": "python_law"},
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
