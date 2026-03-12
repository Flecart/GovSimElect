"""Prompt helpers for fishing formal contracts."""

from simulation.persona import PersonaAgent
from simulation.scenarios.fishing.agents.persona_v3.cognition import leaders as leaders_lib
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


def _persona_role_context(persona: PersonaAgent) -> str:
  svo_prompt, disinfo_prompt, leader_prompt = leaders_lib.get_leader_persona_prompts(
      persona
  )
  parts = [
      prompt.strip()
      for prompt in (svo_prompt, disinfo_prompt, leader_prompt)
      if prompt and prompt.strip()
  ]
  return "\n".join(parts)


def _persona_harvest_context(
    persona: PersonaAgent,
    agent_resource_num: dict[str, int],
) -> str:
  if not agent_resource_num:
    return ""
  own_amount = agent_resource_num.get(persona.identity.name)
  if own_amount is None:
    return ""
  return f"Your latest observed harvest: {own_amount} tons.\n"


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
  role_context = _persona_role_context(persona)
  return (
      f"{get_sytem_prompt(persona.identity)}\n"
      f"{memory_prompt(persona.identity, retrieved_memories)}\n"
      f"{role_context + chr(10) if role_context else ''}"
      "You are in the monthly post-harvest discussion with "
      f"{list_to_comma_string(counterpart_names)}.\n"
      f"Your goal is to negotiate {contract_mode} for the next round.\n"
      f"{_state_lines(state)}"
  )


def mayor_proposal_prompt(
    persona: PersonaAgent,
    state: FishingContractState,
    agent_resource_num: dict[str, int],
) -> str:
  return (
      f"{_persona_harvest_context(persona, agent_resource_num)}"
      f"{_resource_report(agent_resource_num)}\n\n"
      f"{_state_lines(state)}\n"
      "Propose one clear contract for the next round in plain English.\n"
      "Keep it concrete and enforceable. Do not write Python code.\n"
      "If helpful, specify numeric catch limits, sanctions, redistribution rules,"
      " or a required amount of fish to leave in the lake.\n"
      "Write the contract so that another system could translate it into explicit enforcement logic.\n"
      "Avoid vague phrases like 'be sustainable' unless you also state the concrete rule.\n"
      "If you want a numeric cap, state the exact tons per fisher.\n"
      "If you want the rule to depend on fish left in the lake, state that dependency explicitly.\n"
      "Return only the proposal text."
  )


def voter_prompt(
    persona: PersonaAgent,
    state: FishingContractState,
    nl_contract: str,
    agent_resource_num: dict[str, int],
) -> tuple[str, str]:
  role_context = _persona_role_context(persona)
  system_prompt = (
      f"{get_sytem_prompt(persona.identity)}\n"
      f"{role_context + chr(10) if role_context else ''}"
      "You are deciding whether to approve a proposed natural-language fishing contract.\n"
      "Judge whether it is a workable agreement for the next round, not whether it is perfect.\n"
      f"{_state_lines(state)}"
  )
  user_prompt = (
      f"{_persona_harvest_context(persona, agent_resource_num)}"
      f"{_resource_report(agent_resource_num)}\n\n"
      "Evaluate the following proposed contract for the next round.\n"
      f"Contract:\n{nl_contract}\n\n"
      "Vote YES if you can accept this as the contract for the next round.\n"
      "Vote NO only if you materially disagree with the rule or think it is too vague to follow.\n"
      "If you approve it, start your response with <VOTE_YES>.\n"
      "If you reject it, start your response with <VOTE_NO> and briefly explain why."
  )
  return system_prompt, user_prompt


def round_robin_prompt(
    persona: PersonaAgent,
    history: list[dict[str, str]],
    state: FishingContractState,
    agent_resource_num: dict[str, int],
) -> str:
  history_text = conversation_to_string_with_dash(
      [(turn["agent"], turn["message"]) for turn in history]
  ) if history else "-No previous messages.\n"
  return (
      f"{_persona_harvest_context(persona, agent_resource_num)}"
      f"{_resource_report(agent_resource_num)}\n\n"
      f"{_state_lines(state)}\n"
      "Conversation so far:\n"
      f"{history_text}\n"
      "Respond with your next message in the contract negotiation.\n"
      "If you propose a contract, state it plainly in English.\n"
      "If you agree, you must include either <AGREE> or <AGREE to=Name>.\n"
      "If you disagree or reject the current proposal, you must include <DISAGREE>.\n"
      "Use <AGREE> if you agree with the most recent acceptable proposal.\n"
      "Use <AGREE to=Name> if you are explicitly agreeing to a specific fisher's proposal.\n"
      "Use <END_CONVERSATION> only if further negotiation is pointless."
  )


def round_robin_tag_clarification_prompt(
    original_response: str,
) -> tuple[str, str]:
  system_prompt = (
      "You are fixing a contract-negotiation response to follow a strict tag protocol."
  )
  user_prompt = (
      "Your last response did not follow the required agreement protocol.\n"
      "Rewrite it so that it includes exactly one of these tags:\n"
      "- <AGREE>\n"
      "- <AGREE to=Name>\n"
      "- <DISAGREE>\n"
      "Preserve your original meaning. Keep the response short.\n"
      f"Original response:\n{original_response}\n"
  )
  return system_prompt, user_prompt


def coding_agent_prompt(
    nl_contract: str,
    state: FishingContractState,
    feedback: str | None = None,
) -> tuple[str, str]:

    feedback_section = ""
    if feedback:
        feedback_section = (
            "# PREVIOUS REJECTION FEEDBACK\n"
            "Your last draft was voted down. Fix EXACTLY what the voters asked for.\n"
            "Do not re-introduce logic they already rejected.\n"
            f"{feedback}\n\n"
        )

    system_prompt = (
        "You are a contract-to-code translator for a fishing commons simulation.\n"
        "Your ONLY job: convert an agreed natural-language fishing contract into\n"
        "minimal Python code that sets each fisher's catch amount.\n"
        "Be literal. Be minimal. Do not invent constraints the contract does not state."
    )

    user_prompt = (
        f"{_state_lines(state)}\n"
        "\n"
        "# YOUR TASK\n"
        "Translate the natural-language contract below into Python code.\n"
        "\n"
        "# AVAILABLE VARIABLES (pre-defined — do NOT redeclare them)\n"
        "  catches: dict[str, float]\n"
        "    Maps each fisher's name to their catch in tons.\n"
        "    All values start at the one defined by their choice. This is already initialized to the choice of the agents.\n"
        "  fish_population: float\n"
        "    Current tons of fish in the lake BEFORE catches are deducted.\n"
        "  num_agents: int\n"
        "    Number of fishers.\n"
        "  sustainability_threshold: float\n"
        "    A legacy variable that may be present in the runtime.\n"
        "    IGNORE IT unless the contract explicitly refers to it by name.\n"
        "\n"
        "# EXECUTION MODEL\n"
        "Your code runs ONCE per round. After it executes, the framework will:\n"
        "  1. Deduct each fisher's catch from fish_population.\n"
        "  2. Apply the reproduction rule (remaining fish double).\n"
        "Do NOT simulate deduction or reproduction — the framework handles both.\n"
        "Your ONLY responsibility is setting values in `catches`.\n"
        "\n"
        "# TRANSLATION RULES\n"
        "- Be LITERAL: encode only constraints the contract states.\n"
        "- If the contract says 'each fisher catches at most X tons' → set each catch to X.\n"
        "- If the contract says 'leave at least Y tons in the lake' →\n"
        "    cap total catches at max(0, fish_population - Y), then divide fairly.\n"
        "    But ONLY if the contract explicitly states this constraint.\n"
        "- Do NOT add formulas, safety margins, or dynamic calculations the contract does not specify.\n"
        "- Do NOT model reproduction or compute post-catch population.\n"
        "- Do NOT use `sustainability_threshold` unless the contract explicitly names it.\n"
        "- If the contract gives a fixed per-fisher limit, implement that fixed limit directly.\n"
        "\n"
        "# OUTPUT FORMAT\n"
        "Write a single ```python fenced block. Nothing else.\n"
        "- Edit `catches` in place.\n"
        "- No imports, no print(), no tests, no example usage.\n"
        "\n"
        f"{feedback_section}"
        "# NATURAL-LANGUAGE CONTRACT\n"
        f"{nl_contract}\n"
    )

    return system_prompt, user_prompt


def coded_contract_vote_prompt(
    persona: PersonaAgent,
    state: FishingContractState,
    nl_contract: str,
    coded_contract: str,
    retry_attempt: int,
) -> tuple[str, str]:
  return (
      _coded_contract_vote_system_prompt(persona, state),
      _coded_contract_vote_user_prompt(nl_contract, coded_contract, retry_attempt),
  )


def _coded_contract_vote_system_prompt(
    persona: PersonaAgent,
    state: FishingContractState,
) -> str:
  role_context = _persona_role_context(persona)
  return (
      f"You are {persona.identity.name}, a fisherman in the village.\n\n"
      "CURRENT SITUATION:\n"
      f"- Fish in the lake: {state.fish_population:.1f} tons\n"
      f"- Round: {state.round_number + 1} of {state.max_rounds}\n\n"
      f"{role_context + chr(10) if role_context else ''}"
      "A natural-language contract was agreed upon, and a coding agent has translated it into Python.\n"
      "You must judge whether the Python faithfully represents the agreed contract.\n"
      "Judge faithfulness to the agreement, not whether you personally prefer a different deal."
  )


def _coded_contract_vote_user_prompt(
    nl_contract: str,
    coded_contract: str,
    retry_attempt: int,
) -> str:
  retry_note = ""
  if retry_attempt:
    retry_note = (
        f"\nThis is retry attempt #{retry_attempt}; the previous code draft was rejected.\n"
    )
  return (
      "ORIGINAL NATURAL-LANGUAGE AGREEMENT:\n"
      f"\"\"\"\n{nl_contract}\n\"\"\"\n\n"
      "GENERATED PYTHON CONTRACT:\n"
      f"```python\n{coded_contract}\n```\n"
      f"{retry_note}\n"
      "VOTING GUIDELINES:\n"
      "- Vote YES if the code captures the core intent of the agreement, even if formatting or implementation details differ.\n"
      "- Vote NO only for substantive mismatches: missing rules, wrong calculations, or logic that changes the agreement's effect.\n"
      "- The goal is a workable faithful contract, not perfect style.\n\n"
      "If the Python code faithfully represents the agreement, start your response with <VOTE_YES>.\n"
      "Otherwise start with <VOTE_NO> and explain the substantive mismatch briefly."
  )


def coded_contract_feedback_prompt(
    persona: PersonaAgent,
    state: FishingContractState,
    nl_contract: str,
    coded_contract: str,
) -> tuple[str, str]:
  role_context = _persona_role_context(persona)
  system_prompt = (
      f"You are {persona.identity.name}, a fisherman who voted NO on a generated Python contract.\n"
      f"{role_context + chr(10) if role_context else ''}"
      f"{_state_lines(state)}"
      "Explain why the code failed to match the negotiated agreement."
  )
  user_prompt = (
      f"Natural-language contract:\n{nl_contract}\n\n"
      "Generated Python contract:\n"
      f"```python\n{coded_contract}\n```\n\n"
      "Explain the mismatch in at most three short sentences.\n"
      "Focus on substantive errors: missing rules, wrong calculations, or logic that changes the contract's effect."
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
