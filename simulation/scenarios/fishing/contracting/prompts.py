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
  lines = (
      f"Round: {state.round_number + 1}/{state.max_rounds}\n"
      f"Fish currently in the lake: {state.fish_population:.1f}\n"
      f"Number of fishers: {state.num_agents}\n"
  )
  if state.regime is not None:
    lines += f"Lake condition: {state.regime}\n"
  if state.setting_context.strip():
    lines += f"Environment details: {state.setting_context.strip()}\n"
  return lines


def _resource_report(agent_resource_num: dict[str, int]) -> str:
  if not agent_resource_num:
    return "No harvest report is available."
  items = [f"- {agent}: {amount} tons" for agent, amount in agent_resource_num.items()]
  return "Latest observed harvests:\n" + "\n".join(items)


def _active_law_block(active_law_text: str = "") -> str:
  if not active_law_text:
    return "There is currently no active formal law.\n"
  return (
      "CURRENT ACTIVE LAW (remains in force unless replaced by a new vote):\n"
      f"{active_law_text}\n"
  )


def _coding_interface_limitations_block(mode: ContractMode) -> str:
  if mode != ContractMode.CODE_LAW:
    return ""
  return (
      "PYTHON-LAW INTERFACE LIMITS:\n"
      "- Any approved agreement must be translated into a persistent Python law.\n"
      "- The deployed law can control catches through `catch(name, amount, month)` and the provided primitives only.\n"
      "- It may use persistent `contract_state`, but it cannot invent hidden world state beyond that interface.\n"
      "- The framework, not the law, handles fish deduction and reproduction after catches are set.\n"
      "- Prefer concrete, durable rules. Reject proposals that depend on vague future discretion or mechanics the interface cannot represent.\n"
  )


def _coding_primitives_block(mode: ContractMode) -> str:
  if mode != ContractMode.CODE_LAW:
    return ""
  return _python_law_interface_spec_block() + "\n"


def _python_law_interface_spec_block() -> str:
  return (
      "# AVAILABLE VARIABLES (pre-defined — do NOT redeclare them)\n"
      "  catches: dict[str, float]\n"
      "    Maps each fisher's name to their catch in tons.\n"
      "    All values start at the one defined by their choice. This is already initialized to the choice of the agents.\n"
      "  fish_population: float\n"
      "    Current tons of fish in the lake BEFORE catches are deducted.\n"
      "  num_agents: int\n"
      "    Number of fishers.\n"
      "  contract_state: dict[str, Any]\n"
      "    Persistent contract state carried across rounds for this law.\n"
      "    Use it for ledgers, escrow balances, sanctions, insurance pools, or violation counts.\n"
      "  sustainability_threshold: float\n"
      "    A legacy variable that may be present in the runtime.\n"
      "    IGNORE IT unless the contract explicitly refers to it by name.\n"
      "\n"
      "# AVAILABLE PRIMITIVES\n"
      "  fish(name, amount)\n"
      "    Set a fisher's desired catch by name.\n"
      "  transfer(src, dst, amount, reason='')\n"
      "    Move payoff from one fisher to another.\n"
      "  escrow(name, amount, bucket='default', reason='')\n"
      "    Lock payoff into persistent escrow tracked in contract_state.\n"
      "  release_escrow(name, amount=None, bucket='default', recipient=None, reason='')\n"
      "    Release some or all escrow back to the fisher or to a recipient.\n"
      "  sanction(name, amount, reason='')\n"
      "    Apply a direct monetary penalty.\n"
      "  graduated_sanction(name, base_amount, key='default', multiplier=1.0, reason='')\n"
      "    Increase sanctions based on prior violation counts stored in contract_state.\n"
      "  insurance(name, premium=0.0, payout=0.0, pool='default', recipient=None, reason='')\n"
      "    Collect premiums and/or pay out from an insurance pool tracked in contract_state.\n"
      "  participation_cost(name, amount, reason='')\n"
      "    Charge a monetary participation fee.\n"
      "\n"
      "# REQUIRED INTERFACE\n"
      "Define a top-level function with this exact signature:\n"
      "```python\n"
      "def catch(name, amount, month):\n"
      "    ...\n"
      "```\n"
      "- `name` is the fisher's human-readable name.\n"
      "- `amount` is that fisher's submitted catch choice for the month.\n"
      "- `month` is the current month index / round number.\n"
      "- The function may return a replacement catch amount, or `None` if it directly calls `fish(name, ...)`.\n"
      "- Use `contract_state` for permanence across months, like a deployed blockchain contract.\n"
      "\n"
      "# EXECUTION MODEL\n"
      "Your code is deployed once and then reused every round until replaced.\n"
      "At enforcement time, the framework will execute your law and call `catch(name, amount, month)` once for each fisher.\n"
      "After contract execution, the framework will:\n"
      "  1. Deduct each fisher's catch from fish_population.\n"
      "  2. Apply the reproduction rule (remaining fish double).\n"
      "  3. Apply any payoff adjustments produced by transfer / escrow / sanction / insurance / participation_cost.\n"
      "Do NOT simulate deduction or reproduction — the framework handles both.\n"
      "Your responsibility is setting catches and any explicit contractual payoff mechanics.\n"
  )


def _active_code_law_block(active_code_law: str = "") -> str:
  if not active_code_law:
    return "There is currently no deployed Python law.\n"
  return (
      "CURRENT DEPLOYED PYTHON LAW (remains active unless replaced):\n"
      f"```python\n{active_code_law}\n```\n"
  )


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
    tool_instructions: str = "",
    active_law_text: str = "",
) -> str:
  counterpart_names = [p.identity.name for p in target_personas]
  contract_mode = {
      ContractMode.CODE_LAW: "a natural-language agreement that may later be translated into Python enforcement code",
      ContractMode.CODE_NL: "a natural-language common constitution that outlines rules and guidelines for the community",
      ContractMode.FREE_CHAT: "an informal free chat",
      ContractMode.GOVSIM: "an informal conversation",
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
      f"{_active_law_block(active_law_text)}"
      f"{_coding_interface_limitations_block(mode)}"
      f"{tool_instructions + chr(10) if tool_instructions else ''}"
      f"{_state_lines(state)}"
  )


def free_chat_turn_prompt(
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
      "This is an open chat opportunity before starting to fish, you can discuss anything you want.\n"
      "Conversation so far:\n"
      f"{history_text}\n"
      "Respond with one short, natural message.\n"
      "Use <END_CONVERSATION> only if there is clearly nothing else to add."
  )


def mayor_proposal_prompt(
    persona: PersonaAgent,
    state: FishingContractState,
    agent_resource_num: dict[str, int],
    mode: ContractMode,
    tool_instructions: str = "",
    active_law_text: str = "",
) -> str:
  return (
      f"{_persona_harvest_context(persona, agent_resource_num)}"
      f"{_resource_report(agent_resource_num)}\n\n"
      f"{_active_law_block(active_law_text)}"
      f"{_coding_interface_limitations_block(mode)}"
      f"{_state_lines(state)}\n"
      f"{tool_instructions + chr(10) if tool_instructions else ''}"
      "Propose one clear contract for the next round in plain English.\n"
      "If there is already an active law and you want to keep it exactly as it is without any changes,\n"
      "you must output exactly the word <UNCHANGED> and nothing else.\n"
      "Otherwise, if you want a new law or to change the current one, keep it concrete and enforceable. Do not write Python code.\n"
      "When proposing a new law, include exactly one <constitution>...</constitution> block.\n"
      "Only the text inside that block will be voted on and adopted.\n"
      "Put only the final rule text inside <constitution>...</constitution>.\n"
      "If helpful, specify numeric catch limits, sanctions, redistribution rules,"
      " or a required amount of fish to leave in the lake.\n"
      "Write the contract so that another system could translate it into explicit enforcement logic.\n"
      "Avoid vague phrases like 'be sustainable' unless you also state the concrete rule.\n"
      "If you want a numeric cap, state the exact tons per fisher.\n"
      "If you want the rule to depend on fish left in the lake, state that dependency explicitly.\n"
      "Return only your proposal text (or <UNCHANGED>), following the tagging rule above."
  )


def voter_prompt(
    persona: PersonaAgent,
    state: FishingContractState,
    nl_contract: str,
    agent_resource_num: dict[str, int],
    mode: ContractMode,
    active_law_text: str = "",
) -> tuple[str, str]:
  role_context = _persona_role_context(persona)
  interface_vote_guidance = ""
  if mode == ContractMode.CODE_LAW:
    interface_vote_guidance = (
        "Vote NO if you would rather keep the current law than approve a proposal the Python-law interface cannot represent faithfully.\n"
    )
  system_prompt = (
      f"{get_sytem_prompt(persona.identity)}\n"
      f"{role_context + chr(10) if role_context else ''}"
      "You are deciding whether to approve a proposed natural-language fishing contract.\n"
      "Judge whether it is a workable agreement for the next round, not whether it is perfect.\n"
      f"{_coding_interface_limitations_block(mode)}"
      f"{_coding_primitives_block(mode)}"
      f"{_state_lines(state)}"
  )
  user_prompt = (
      f"{_persona_harvest_context(persona, agent_resource_num)}"
      f"{_resource_report(agent_resource_num)}\n\n"
      f"{_active_law_block(active_law_text)}"
      f"{_coding_primitives_block(mode)}"
      "Evaluate the following proposed contract for the next round.\n"
      "If the proposed Contract is <UNCHANGED>, it means keeping the current active law in force.\n"
      f"Contract:\n{nl_contract}\n\n"
      "Vote YES if you can accept this as the contract for the next round.\n"
      "Vote NO if you materially disagree with the rule or think it is too vague to follow.\n"
      f"{interface_vote_guidance}"
      "If you approve it, start your response with <VOTE_YES>.\n"
      "If you reject it, start your response with <VOTE_NO> and briefly explain why."
  )
  return system_prompt, user_prompt


def round_robin_prompt(
    persona: PersonaAgent,
    history: list[dict[str, str]],
    state: FishingContractState,
    agent_resource_num: dict[str, int],
    mode: ContractMode,
    tool_instructions: str = "",
    active_law_text: str = "",
) -> str:
  history_text = conversation_to_string_with_dash(
      [(turn["agent"], turn["message"]) for turn in history]
  ) if history else "-No previous messages.\n"
  return (
      f"{_persona_harvest_context(persona, agent_resource_num)}"
      f"{_resource_report(agent_resource_num)}\n\n"
      f"{_active_law_block(active_law_text)}"
      f"{_coding_interface_limitations_block(mode)}"
      f"{_state_lines(state)}\n"
      f"{tool_instructions + chr(10) if tool_instructions else ''}"
      "Conversation so far:\n"
      f"{history_text}\n"
      "Respond with your next message in the contract negotiation.\n"
      "If there is an active law and you propose to keep it exactly as it is, you can propose the exact word <UNCHANGED>.\n"
      "Otherwise, if you propose a new contract, include exactly one <constitution>...</constitution> block.\n"
      "Only the text inside <constitution>...</constitution> will be voted on as the contract.\n"
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
    active_nl_law: str = "",
    active_code_law: str = "",
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
        "a persistent Python law that stays in force across future months until replaced.\n"
        "Be literal. Be minimal. Do not invent constraints the contract does not state."
    )

    user_prompt = (
        f"{_state_lines(state)}\n"
        "\n"
        "# YOUR TASK\n"
        "Translate the natural-language contract below into a deployed-law style Python contract.\n"
        "The law should be stable across months and use persistent `contract_state` for memory.\n"
        "Once adopted, the law remains active until a later vote replaces it.\n"
        "If the new agreement is an amendment to the current law rather than a full replacement,\n"
        "preserve the current law's unchanged obligations in the new code.\n"
        "If the new agreement clearly replaces the current law, you may discard the old logic.\n"
        "\n"
        f"{_python_law_interface_spec_block()}\n"
        "# TRANSLATION RULES\n"
        "- Be LITERAL: encode only constraints the contract states.\n"
        "- Prefer the named primitives above, especially inside `catch(name, amount, month)`, over raw dict manipulation.\n"
        "- If the contract says 'each fisher catches at most X tons' → set each catch to X.\n"
        "- If the contract says 'leave at least Y tons in the lake' →\n"
        "    cap total catches at max(0, fish_population - Y), then divide fairly.\n"
        "    But ONLY if the contract explicitly states this constraint.\n"
        "- If the contract specifies escrow, sanctions, insurance, participation fees, or transfers, encode them using the provided primitives.\n"
        "- Use `contract_state` only for persistent contract state; do not invent unrelated state.\n"
        "- The law must keep working in future months, not just the current one.\n"
        "- Do NOT add formulas, safety margins, or dynamic calculations the contract does not specify.\n"
        "- Do NOT model reproduction or compute post-catch population.\n"
        "- Do NOT use `sustainability_threshold` unless the contract explicitly names it.\n"
        "- If the contract gives a fixed per-fisher limit, implement that fixed limit directly.\n"
        "\n"
        "# OUTPUT FORMAT\n"
        "Write a single ```python fenced block. Nothing else.\n"
        "- The code must define `catch(name, amount, month)`.\n"
        "- You may call the named primitives and edit `catches` in place if needed.\n"
        "- No imports, no print(), no tests, no example usage.\n"
        "\n"
        "# CURRENT ACTIVE LAW\n"
        f"{_active_law_block(active_nl_law)}"
        f"{_active_code_law_block(active_code_law)}"
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
    active_nl_law: str = "",
    active_code_law: str = "",
) -> tuple[str, str]:
  return (
      _coded_contract_vote_system_prompt(persona, state, active_nl_law, active_code_law),
      _coded_contract_vote_user_prompt(
          nl_contract,
          coded_contract,
          retry_attempt,
          active_nl_law,
      ),
  )


def _coded_contract_vote_system_prompt(
    persona: PersonaAgent,
    state: FishingContractState,
    active_nl_law: str = "",
    active_code_law: str = "",
) -> str:
  role_context = _persona_role_context(persona)
  return (
      f"You are {persona.identity.name}, a fisherman in the village.\n\n"
      "CURRENT SITUATION:\n"
      f"- Fish in the lake: {state.fish_population:.1f} tons\n"
      f"- Round: {state.round_number + 1} of {state.max_rounds}\n\n"
      f"{role_context + chr(10) if role_context else ''}"
      f"{_active_law_block(active_nl_law)}"
      f"{_active_code_law_block(active_code_law)}"
      f"{_coding_interface_limitations_block(ContractMode.CODE_LAW)}"
      f"{_coding_primitives_block(ContractMode.CODE_LAW)}"
      "A natural-language contract was agreed upon, and a coding agent has translated it into Python.\n"
      "You must judge whether the Python faithfully represents the agreed contract and whether the proposed replacement is preferable to keeping the current deployed law.\n"
      "Judge faithfulness to the agreement and to the coding interface, not whether you personally prefer a different deal."
  )


def _coded_contract_vote_user_prompt(
    nl_contract: str,
    coded_contract: str,
    retry_attempt: int,
    active_nl_law: str = "",
) -> str:
  retry_note = ""
  if retry_attempt:
    retry_note = (
        f"\nThis is retry attempt #{retry_attempt}; the previous code draft was rejected.\n"
    )
  return (
      "ORIGINAL NATURAL-LANGUAGE AGREEMENT:\n"
      f"\"\"\"\n{nl_contract}\n\"\"\"\n\n"
      f"{_coding_primitives_block(ContractMode.CODE_LAW)}\n"
      "GENERATED PYTHON CONTRACT:\n"
      f"```python\n{coded_contract}\n```\n"
      f"{retry_note}\n"
      "VOTING GUIDELINES:\n"
      "- Vote YES if the code captures the core intent of the agreement, even if formatting or implementation details differ.\n"
      "- Vote NO for substantive mismatches: missing rules, wrong calculations, logic that changes the agreement's effect, or an amendment that drops important parts of the current law without authorization.\n"
      "- Vote NO if the proposal depends on mechanics this coding interface cannot faithfully implement.\n"
      f"- A NO vote keeps the current law in force{': ' + active_nl_law if active_nl_law else '.'}\n"
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
  # deprecated
  raise NotImplementedError("nl_judge_prompt is deprecated")
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


