"""Tool access for contract negotiation and drafting."""

import json
import re
from dataclasses import dataclass
from typing import Any

from .contract import ContractType


TOOL_CALL_PATTERN = re.compile(
    r"<TOOL_CALL>\s*(\{.*?\})\s*</TOOL_CALL>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class ToolCall:
  name: str
  args: dict[str, Any]


class ContractToolbox:
  """Expose prior contract history as explicit tools."""

  def __init__(self, history: list[dict[str, Any]]) -> None:
    self.history = history

  def parse_tool_call(self, response: str) -> ToolCall | None:
    match = TOOL_CALL_PATTERN.search(response or "")
    if not match:
      return None
    try:
      payload = json.loads(match.group(1))
    except json.JSONDecodeError:
      return None
    name = str(payload.get("tool", "")).strip()
    if not name:
      return None
    args = payload.get("args", {})
    if not isinstance(args, dict):
      args = {}
    return ToolCall(name=name, args=args)

  def render_tool_instructions(self) -> str:
    return (
        "AVAILABLE TOOLS:\n"
        "- `get_latest_nl_law`: fetch the most recent adopted natural-language law.\n"
        "- `get_latest_code_law`: fetch the most recent adopted Python law, if one exists.\n"
        "- `get_recent_contracts`: fetch a short list of the most recent adopted contracts.\n"
        "If you want to use a tool, your entire response should be exactly one tool call in this format:\n"
        '<TOOL_CALL>{"tool":"get_latest_nl_law"}</TOOL_CALL>\n'
        "or\n"
        '<TOOL_CALL>{"tool":"get_recent_contracts","args":{"limit":3}}</TOOL_CALL>\n'
        "After the tool result is returned, you may answer normally or call another tool."
    )

  def execute(self, tool_call: ToolCall) -> str:
    if tool_call.name == "get_latest_nl_law":
      return self._get_latest_nl_law()
    if tool_call.name == "get_latest_code_law":
      return self._get_latest_code_law()
    if tool_call.name == "get_recent_contracts":
      return self._get_recent_contracts(tool_call.args.get("limit", 3))
    return f"Unknown tool `{tool_call.name}`."

  def _get_latest_nl_law(self) -> str:
    for entry in reversed(self.history):
      nl_contract = str(entry.get("nl_contract", "")).strip()
      if nl_contract:
        return f"LATEST_NL_LAW:\n{nl_contract}"
    return "LATEST_NL_LAW:\nNone."

  def _get_latest_code_law(self) -> str:
    for entry in reversed(self.history):
      if entry.get("contract_type") != ContractType.PYTHON_LAW.value:
        continue
      coded_contract = str(entry.get("content", "")).strip()
      if coded_contract:
        return f"LATEST_CODE_LAW:\n```python\n{coded_contract}\n```"
    return "LATEST_CODE_LAW:\nNone."

  def _get_recent_contracts(self, limit: Any) -> str:
    try:
      limit_value = max(1, min(int(limit), 10))
    except (TypeError, ValueError):
      limit_value = 3
    recent = self.history[-limit_value:]
    if not recent:
      return "RECENT_CONTRACTS:\nNone."
    lines = []
    for entry in reversed(recent):
      round_created = entry.get("round_created", "?")
      contract_type = entry.get("contract_type", "unknown")
      nl_contract = str(entry.get("nl_contract", "")).strip() or "(none)"
      lines.append(
          f"- round {round_created} [{contract_type}] nl={nl_contract}"
      )
    return "RECENT_CONTRACTS:\n" + "\n".join(lines)
