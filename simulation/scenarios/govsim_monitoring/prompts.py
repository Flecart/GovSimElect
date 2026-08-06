"""GovSim-style prompts for decentralized monitoring and norm formation."""

from __future__ import annotations

from simulation.persona.common import PersonaIdentity


def system_prompt(
    identity: PersonaIdentity,
    *,
    peer_names: list[str],
    capacity: float,
    alpha: float,
    maintenance_requirement: float,
    consumption: float,
    penalty: float,
) -> str:
  """Revised GovSim v3 context for the monitoring experiment."""
  peers = ", ".join(peer_names)
  text = (
      f"You're a fisherman named {identity.name} who, along with four others "
      f"({peers}), fishes in a lake every month. The lake has a carrying "
      f"capacity of {capacity:g} tons of fish. At the beginning of each month, "
      "each fisherman chooses a harvesting effort between 0 and 1 and a "
      "maintenance contribution between 0 and 1. Your harvest is the lake "
      f"stock multiplied by your effort and the current efficiency. Efficiency "
      f"starts at {alpha:g} and cannot exceed 1. Your maintenance contribution "
      "is deducted from your wealth and changes efficiency for the next month. "
      f"Collective contributions around {maintenance_requirement:g} maintain "
      "efficiency; contributing less makes it deteriorate, while contributing "
      "more improves it with diminishing returns. You observe current efficiency "
      "each month. "
      f"Every month you must pay a survival consumption cost of {consumption:g}. "
      "If your wealth becomes negative, you die and the community collapses. "
      "After harvesting, you choose a monitoring propensity between 0 and 1. "
      "One other fisherman is selected randomly, and you privately observe that "
      "person's effort, contribution, and harvest with the probability you chose. "
      f"You may punish an observed fisherman. Punishment deducts {penalty:g} from "
      "the target once even if several people punish them, and suspends the target "
      "from harvesting and contributing next month. A suspended fisherman still "
      "pays survival consumption and may monitor, discuss, propose, and vote. "
      "Monitoring observations are private unless a fisherman chooses to mention "
      "them during discussion. After monitoring, the fishermen discuss the "
      "situation, propose natural-language community norms, and vote on which norm "
      "to adopt. At the end of the month, the remaining fish reproduce and double "
      f"up to the carrying capacity of {capacity:g}. If fewer than 5 tons remain "
      "after harvesting, the lake collapses."
  )
  if identity.goals:
    text += identity.goals
  return text


HARVEST_PROMPT = """{state}
{memory}
Task: Choose your harvesting effort and maintenance contribution for this month.
Both values must be numbers between 0 and 1. Contribution is paid from your wealth
and does not reduce your harvesting effort.
Return exactly:
Effort: NUMBER
Contribution: NUMBER"""


MONITOR_PROMPT = """{state}
{memory}
Task: Choose your monitoring propensity for this month. One peer will be selected
uniformly at random and you will privately observe them with this probability.
Return only one number between 0 and 1."""


PUNISH_PROMPT = """{state}
{memory}
Your private monitoring succeeded:
{observation}
Task: Decide whether to punish this observed fisherman. A punishment deducts the
binary penalty from the target and suspends them from harvesting and contributing
next month. Return only `yes` or `no`."""


DISCUSSION_PROMPT = """{location_time}{memory}
Scenario: {participants} are engaged in a group chat.

Conversation so far:
{conversation}

Task: What would you say next in the group chat? Ensure the conversation flows naturally and avoids repetition.
Determine whether your response concludes the conversation. If not, identify the next speaker.
Return exactly in this format:
Response: ...
Conversation conclusion by me: yes/no
Next speaker: NAME or N/A"""


PROPOSAL_PROMPT = """{state}
{memory}
Public discussion:
{conversation}

Task:
1. Update your personal natural-language norm about harvesting, maintenance,
monitoring, punishment, or community governance.
2. Propose a natural-language norm for the whole community. It may contain numeric
caps or any other relevant rule.
Return exactly:
Personal: ...
Community: ..."""


VOTE_PROMPT = """{state}
{memory}
Public discussion:
{conversation}

Proposed community norms:
{proposals}

Task: Vote for the proposal that should become the shared community norm.
Return only its proposal number, with no other text."""
