"""Prompt helpers for fishing formal contracts."""

from simulation.persona import PersonaAgent
from simulation.scenarios.fishing.agents.persona_v3.cognition.utils import (
    conversation_to_string_with_dash,
    get_sytem_prompt,
    list_to_comma_string,
    memory_prompt,
)

from .contract import ContractMode, FishingContractState


def _state_lines(state: FishingContractState) -> str:
  return (
      f"Round: {state.round_number + 1}/{state.max_rounds}\n"
      f"Fish currently in the lake: {state.fish_population:.1f}\n"
      f"Number of fishers: {state.num_agents}\n"
  )


def _resource_report(agent_resource_num: dict[str, int]) -> str:
  if not agent_resource_num:
    return "No harvest report is available."
  items = [f"- {agent}: {amount} tons" for agent, amount in agent_resource_num.items()]
  return "Latest observed harvests:\n" + "\n".join(items)


def negotiation_system_prompt(
    persona: PersonaAgent,
    target_personas: list[PersonaAgent],
    retrieved_memories: list[tuple],
    state: FishingContractState,
    mode: ContractMode,
) -> str:
  counterpart_names = [p.identity.name for p in target_personas]
  contract_mode = {
      ContractMode.CODE_LAW: "a natural-language agreement that may later be translated into Python enforcement code",
      ContractMode.CODE_NL: "a natural-language agreement that will be enforced directly in natural language",
      ContractMode.GOVSIM: "an informal conversation with no binding contract",
      ContractMode.NO_COMMUNICATION: "no communication",
  }[mode]
  return (
      f"{get_sytem_prompt(persona.identity)}\n"
      f"{memory_prompt(persona.identity, retrieved_memories)}\n"
      "You are in the monthly post-harvest discussion with "
      f"{list_to_comma_string(counterpart_names)}.\n"
      f"Your goal is to negotiate {contract_mode} for the next round.\n"
      f"{_state_lines(state)}"
  )


def mayor_proposal_prompt(
    state: FishingContractState,
    agent_resource_num: dict[str, int],
) -> str:
  return (
      f"{_resource_report(agent_resource_num)}\n\n"
      "Propose one clear contract for the next round in plain English.\n"
      "Keep it concrete and enforceable. Do not write Python code.\n"
      "If helpful, specify numeric catch limits, sanctions, or redistribution rules.\n"
      "Return only the proposal text."
  )


def voter_prompt(nl_contract: str) -> str:
  return (
      "Evaluate the following proposed contract for the next round.\n"
      f"Contract:\n{nl_contract}\n\n"
      "If you approve it, start your response with <VOTE_YES>.\n"
      "If you reject it, start your response with <VOTE_NO> and briefly explain why."
  )


def round_robin_prompt(
    history: list[dict[str, str]],
    state: FishingContractState,
    agent_resource_num: dict[str, int],
) -> str:
  history_text = conversation_to_string_with_dash(
      [(turn["agent"], turn["message"]) for turn in history]
  ) if history else "-No previous messages.\n"
  return (
      f"{_resource_report(agent_resource_num)}\n\n"
      f"{_state_lines(state)}\n"
      "Conversation so far:\n"
      f"{history_text}\n"
      "Respond with your next message in the contract negotiation.\n"
      "If you propose a contract, state it plainly in English.\n"
      "Use <AGREE> if you agree with the most recent acceptable proposal.\n"
      "Use <AGREE to=Name> if you are explicitly agreeing to a specific fisher's proposal.\n"
      "Use <END_CONVERSATION> only if further negotiation is pointless."
  )


def coding_agent_prompt(
    nl_contract: str,
    state: FishingContractState,
    feedback: str | None = None,
) -> tuple[str, str]:
  feedback_section = ""
  if feedback:
    feedback_section = (
        "\nThe last draft was rejected. Address this feedback exactly:\n"
        f"{feedback}\n"
    )
  system_prompt = (
      "You are a coding agent. Translate an agreed fishing contract from plain English"
      " into short Python code that modifies the `catches` dictionary in place."
  )
  user_prompt = (
      f"{_state_lines(state)}\n"
      "Write Python code only inside a ```python fenced block.\n"
      "Available variables:\n"
      "- catches: dict[str, float]\n"
      "- fish_population: float\n"
      "- sustainability_threshold: float\n"
      "- num_agents: int\n"
      "Avoid using `sustainability_threshold` unless the natural-language contract explicitly refers to it.\n"
      "The code may edit `catches` directly or assign `modified_catches`.\n"
      "Do not include example usage or tests.\n"
      f"{feedback_section}\n"
      "Natural-language contract:\n"
      f"{nl_contract}\n"
  )
  return system_prompt, user_prompt


def coded_contract_vote_prompt(
    nl_contract: str,
    coded_contract: str,
    retry_attempt: int,
) -> tuple[str, str]:
  retry_note = ""
  if retry_attempt:
    retry_note = (
        f"\nThis is retry attempt #{retry_attempt}; the previous code draft was rejected.\n"
    )
  system_prompt = (
      "You are reviewing whether generated Python contract code faithfully represents"
      " the agreed natural-language contract."
  )
  user_prompt = (
      f"Natural-language contract:\n{nl_contract}\n\n"
      "Generated Python contract:\n"
      f"```python\n{coded_contract}\n```\n"
      f"{retry_note}\n"
      "If the Python code faithfully represents the agreement, start your response with <VOTE_YES>.\n"
      "Otherwise start with <VOTE_NO> and explain the mismatch briefly."
  )
  return system_prompt, user_prompt


def coded_contract_feedback_prompt(
    nl_contract: str,
    coded_contract: str,
) -> tuple[str, str]:
  system_prompt = (
      "You are explaining why a generated Python contract did not match the negotiated agreement."
  )
  user_prompt = (
      f"Natural-language contract:\n{nl_contract}\n\n"
      "Generated Python contract:\n"
      f"```python\n{coded_contract}\n```\n\n"
      "Explain the mismatch in at most three short sentences."
  )
  return system_prompt, user_prompt


def nl_judge_prompt(
    contract_text: str,
    decisions: dict[str, float],
    state: FishingContractState,
    conversation_summary: str,
) -> tuple[str, str]:
  system_prompt = (
      "You are a strict contract judge for a fishing commons simulation."
      " Decide what each fisher's enforced catch should be under the negotiated"
      " natural-language contract."
  )
  catch_lines = "\n".join(
      f"- {agent}: {amount}" for agent, amount in decisions.items()
  )
  user_prompt = (
      f"{_state_lines(state)}\n"
      f"Contract:\n{contract_text}\n\n"
      f"Conversation context:\n{conversation_summary or '(none)'}\n\n"
      f"Submitted catches:\n{catch_lines}\n\n"
      "Return valid JSON only with this shape:\n"
      '{"modified_catches":{"Agent":1},"reasoning":"...","violations_detected":["..."]}\n'
      "Use the submitted catches unchanged if the contract is too vague to alter them."
  )
  return system_prompt, user_prompt
