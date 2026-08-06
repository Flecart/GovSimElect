"""Prompts reproduced from Appendix A of arXiv:2510.14401v2.

The paper is licensed CC BY 4.0. Bracketed placeholders are populated with the
agent's private observation and the experiment configuration at runtime.
"""

from __future__ import annotations


BASE_CONTEXT = """You are a villager who fishes from a shared lake together with others in your
community. Each villager needs to consume {consumption} units of fish
daily to survive. When your payoff becomes negative, you die.
Each villager holds a personal strategy about what they should do,
and the community has also a shared policy.
Your personal strategy: “{agent_norm}”
Shared community policy: “{group_norm}”"""


PUNISHMENT_BASE_CONTEXT = """You are a villager who fishes from a shared lake together with others in your
community. Each villager needs to consume {consumption} units of fish
daily to survive. When your payoff becomes negative, you die.
Each villager holds a personal strategy about what they should do,
and the community also has a shared policy.
Your personal strategy: “{agent_norm}”
Shared community policy: “{group_norm}”"""


EFFORT_PROMPT = """{base_context}
{observation}
You observe each villager’s fishing effort and total payoff. Based on
both your personal belief and the community policy, decide how much effort
you want to put into fishing today.
Respond with a single number between 0.0 and 1.0. No explanation."""


PUNISHMENT_PROMPT = """{base_context}
{observation}
You observe each villager’s fishing effort and total payoff. Based on
both your personal belief and the community policy, decide whether to punish
others who violate these standards by deducting their payoff by {penalty},
while you need to pay {punish_cost} yourself.
Return only the ID number of the villager to punish, or “N/A” if no
punishment is needed. No explanation."""


NORM_PROMPT = """{base_context}
{observation}
You observe each villager’s fishing effort and total payoff. Based on
your observations:
1. Update your personal strategy about what you should do
2. Propose what the others should do in the community
Respond in exactly this format:
Personal: [Your updated personal belief]
Community: [Your proposed community policy]
No additional explanation."""


VOTE_PROMPT = """{base_context}
{observation}
Proposed community policies:
{proposals}
Based on your personal strategy and the current state of the lake,
vote for which proposed policy you think should become the new shared
policy.
Respond with only the exact text of your chosen policy (copy it exactly as shown above). No explanation."""


def render_base(
    consumption: float,
    agent_norm: str,
    group_norm: str,
    *,
    punishment: bool = False,
) -> str:
  template = PUNISHMENT_BASE_CONTEXT if punishment else BASE_CONTEXT
  return template.format(
      consumption=consumption,
      agent_norm=agent_norm,
      group_norm=group_norm,
  )
