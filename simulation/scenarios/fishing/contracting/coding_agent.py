"""Coding agent for phase-2 translation into Python-law contracts."""

import re

from simulation.utils import ModelWandbWrapper

from .contract import ContractMode, FishingContractState
from .prompts import coding_agent_prompt


class CodingAgent:
  """Translate agreed NL contracts into Python enforcement code."""

  def __init__(
      self,
      model: ModelWandbWrapper,
      temperature: float = 0.3,
  ) -> None:
    self.model = model
    self.temperature = temperature

  async def translate(
      self,
      nl_contract: str,
      mode: ContractMode,
      state: FishingContractState,
      active_nl_law: str = "",
      active_code_law: str = "",
      feedback: str | None = None,
  ) -> tuple[str | None, str, str]:
    if mode == ContractMode.CODE_NL:
      return nl_contract, "CODE_NL mode: using the natural-language contract directly.", ""
    if mode != ContractMode.CODE_LAW:
      return None, f"Unsupported coding mode: {mode.value}", ""

    system_prompt, user_prompt = coding_agent_prompt(
        nl_contract=nl_contract,
        state=state,
        active_nl_law=active_nl_law,
        active_code_law=active_code_law,
        feedback=feedback,
    )
    session = self.model.start_prompt(
        "CodingAgent",
        "contracting",
        "translate_python_law",
    )
    session.add_message("system", system_prompt)
    session.add_user(user_prompt)
    response = await self.model.acomplete_prompt(
        session,
        temperature=self.temperature,
        default_value=(
            "```python\n"
            "class DefaultLaw(Contract):\n"
            "    VERSION = 1\n"
            "    def resolve(self, month, fish_population, submissions, ctx):\n"
            "        return dict(submissions)\n"
            "```"
        ),
    )
    code = self._extract_python_code(response)
    return code, response, session.html()

  def _extract_python_code(self, response: str) -> str | None:
    patterns = [
        r"```python\s*\n?(.*?)```",
        r"```\s*\n?(.*?)```",
    ]
    for pattern in patterns:
      matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
      if matches:
        code = matches[0].strip()
        if (
            "class " in code
            and "Contract" in code
            and "def resolve(" in code
        ):
          return code
    return None
