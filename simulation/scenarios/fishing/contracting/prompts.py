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
      f"Round: {state.round_number + 1}\n"
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


def _negotiation_rounds_notice(state: FishingContractState) -> str:
  """How many restaurant-phase speaking turns exist (so fishers can pace themselves)."""
  n = state.negotiation_max_turns
  agents = max(1, state.num_agents)
  per = n / agents
  per_s = str(int(per)) if float(per).is_integer() else f"{per:.1f}"
  return (
      f"This discussion is limited to {n} speaking turns total (round-robin: one fisher speaks per turn). "
      f"With {state.num_agents} fishers, each person may speak about {per_s} times on average before the discussion ends.\n"
      "If no agreement is reached with majority vote before the end of the discussion, the current law remains in force."
  )


def _deliberation_scope_block(mode: ContractMode) -> str:
  """NL deliberation: community rules, not 'can this be translated to Python'."""
  if mode != ContractMode.CODE_LAW:
    return ""
  return (
      "DELIBERATION SCOPE:\n"
      "- Negotiate concrete rules for the fishery for the upcoming round: catches, fairness, penalties, and mutual obligations.\n"
      "- Catches are whole numbers of tons; the minimum any fisher can catch is 1 ton.\n"
      "- Prefer specific numbers and clear conditions over vague slogans.\n"
      "- Judge proposals on clarity, fairness, and workability for fishers.\n"
  )


def _nl_enforcement_affordance_sheet() -> str:
  return (
      "WHAT THE ENFORCEMENT LAYER CAN DO (natural-language constitution scope):\n"
      "- Set, cap, or reallocate catches among fishers for the month.\n"
      "- Move payoff between fishers; lock payoff in escrow that persists across rounds.\n"
      "- Apply fines, including escalating fines for repeat violations (framework-tracked counts).\n"
      "- Run an insurance pool with premiums and payouts.\n"
      "- Charge participation fees.\n"
      "- Keep persistent memory across rounds (constitution-specific ledgers in the law; escrow/insurance/violations via the framework).\n"
      "WHAT IT CANNOT DO:\n"
      "- Observe anything beyond submitted catches and fish in the lake for that round.\n"
      "- Punish intentions or private thoughts.\n"
      "- Coordinate with external systems or the open internet.\n"
      "- Take actions between rounds (only hooks the framework invokes at defined points).\n"
  )


def _coding_primitives_block(mode: ContractMode) -> str:
  """NL-facing prompts: affordance sheet only (no low-level API list)."""
  if mode != ContractMode.CODE_LAW:
    return ""
  return _nl_enforcement_affordance_sheet() + "\n"


def _python_law_interface_spec_block() -> str:
  return (
      "# DEPLOYED LAW SHAPE\n"
      "Subclass the framework base class `Contract` (VERSION class attribute; bump if you change `self.state` schema).\n"
      "```python\n"
      "class MyLaw(Contract):\n"
      "    VERSION = 1\n"
      "    def __init__(self, num_agents, agent_names, *, prior_state=None):\n"
      "        super().__init__(num_agents, agent_names, prior_state=prior_state)\n"
      "    def on_round_start(self, month, fish_population, ctx) -> None: ...\n"
      "    def resolve(self, month, fish_population, submissions, ctx) -> dict[str, float]: ...\n"
      "    def on_round_end(self, month, fish_population_after, final_catches, ctx) -> None: ...\n"
      "```\n"
      "# DESIGN RULES\n"
      "1. `resolve()` is pure on catches, imperative on payoffs: the returned dict is the only source of truth for tons deducted from the lake; "
      "all transfers, escrow, sanctions, insurance, and fees go through `ctx` methods.\n"
      "2. State: bespoke ledgers in `self.state`. Escrow balances, insurance pools, and violation counts are framework-owned — read with "
      "`ctx.escrow_balance`, `ctx.insurance_pool`, `ctx.violation_count` — do not store those in `self.state`.\n"
      "3. Resolution is simultaneous: `submissions` has every fisher's choice at once; return final catches for everyone at once.\n"
      "4. If a clause cannot be expressed here, put a comment at the top: `# UNIMPLEMENTABLE: <clause>` and implement the rest faithfully.\n"
      "\n"
      "# EnforcementContext `ctx` (payoff primitives; same semantics as before, scoped to this call)\n"
      "  ctx.transfer(src, dst, amount, reason='')\n"
      "  ctx.escrow(name, amount, bucket='default', reason='')\n"
      "  ctx.release_escrow(name, amount=None, bucket='default', recipient=None, reason='')\n"
      "  ctx.sanction(name, amount, reason='')\n"
      "  ctx.graduated_sanction(name, base_amount, key='default', multiplier=1.0, reason='')\n"
      "  ctx.insurance(name, premium=0.0, payout=0.0, pool='default', recipient=None, reason='')\n"
      "  ctx.participation_cost(name, amount, reason='')\n"
      "  ctx.escrow_balance(name, bucket='default') -> float\n"
      "  ctx.insurance_pool(pool='default') -> float\n"
      "  ctx.violation_count(name, key='default') -> int\n"
      "  ctx.random() -> float   # seeded RNG — do not import `random` in your law\n"
      "\n"
      "# READ-ONLY CONTEXT FIELDS\n"
      "  ctx.month, ctx.fish_population, ctx.num_agents, ctx.agent_names\n"
      "\n"
      "# EXECUTION MODEL\n"
      "The framework deploys your class once and reuses the instance until replaced.\n"
      "Each harvest month, in order: `on_round_start` → `resolve` → framework deducts catches from the lake → reproduction → `on_round_end` "
      "(after regen, with `fish_population_after` and realized catches).\n"
      "Do NOT mutate fish stock yourself or simulate reproduction.\n"
      "\n"
      "# NOT AVAILABLE IN GENERATED LAW CODE\n"
      "- No imports except the small allowlist validated at deploy time (stdlib subsets only; no `os`, `subprocess`, `socket`, etc.).\n"
      "- No filesystem, network, subprocess, or `os` access.\n"
      "- Do not use Python's `random` module — use `ctx.random()` only.\n"
      "- No `eval`, `exec`, `open`, or `__import__`.\n"
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
    state: FishingContractState,
    mode: ContractMode,
    tool_instructions: str = "",
    active_law_text: str = "",
) -> str:
  counterpart_names = [p.identity.name for p in target_personas]
  contract_mode = {
      ContractMode.CODE_LAW: "a natural-language constitution for the next round, this will be enforced.",
      ContractMode.CODE_NL: "a natural-language common constitution that outlines rules and guidelines for the community",
      ContractMode.FREE_CHAT: "an informal free chat",
      ContractMode.GOVSIM: "an informal conversation",
      ContractMode.NO_COMMUNICATION: "no communication",
  }[mode]
  
  mode_text = f"You will be negotiating {contract_mode}"
  return (
      f"{get_sytem_prompt(persona.identity)}\n"
      f"{mode_text}\n"
      "You are in the monthly post-harvest discussion with "
      f"{list_to_comma_string(counterpart_names)}.\n"
      f"{_negotiation_rounds_notice(state)}"
      f"{_active_law_block(active_law_text)}"
      f"{_deliberation_scope_block(mode)}"
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
      f"{_negotiation_rounds_notice(state)}"
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
  retrieved_memories = persona.retrieve.retrieve([], 8)
  memory_prompt_text = memory_prompt(persona.identity, retrieved_memories)
  return (
      f"{_persona_harvest_context(persona, agent_resource_num)}"
      f"{_resource_report(agent_resource_num)}\n\n"
      f"{memory_prompt_text}\n"
      "Propose one clear contract for the next round in plain English.\n"
      "When proposing a new law, include exactly one <constitution>...</constitution> block with the declared new constitution."
      "If you agree with the current one, add <constitution> <UNCHANGED> </constitution>."
      " The important part is that you add the <constitution> tags and the word <UNCHANGED> in between.\n"
      "Only the text inside that block will be voted on and adopted."
      " When writing the law, use only specific legal language. <constitution> XYZ shall ... </constitution>."
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
  system_prompt = (
      f"{get_sytem_prompt(persona.identity)}\n"
      f"{role_context + chr(10) if role_context else ''}"
      "You are deciding whether to approve a proposed natural-language fishing contract.\n"
      "Judge whether it is a workable agreement for the next round, not whether it is perfect.\n"
      f"{_negotiation_rounds_notice(state)}"
      f"{_deliberation_scope_block(mode)}"
      f"{_state_lines(state)}"
  )
  user_prompt = (
      f"{_persona_harvest_context(persona, agent_resource_num)}"
      f"{_resource_report(agent_resource_num)}\n\n"
      f"{_active_law_block(active_law_text)}"
      "Evaluate the following proposed contract for the next round.\n"
      "If the proposed Contract is <UNCHANGED>, it means keeping the current active law in force.\n"
      f"Contract:\n{nl_contract}\n\n"
      "Vote YES if you can accept this as the contract for the next round.\n"
      "Vote NO if you materially disagree with the rule or think it is too vague to follow.\n"
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
  
  retrieved_memories = persona.retrieve.retrieve([], 8)
  memory_prompt_text = memory_prompt(persona.identity, retrieved_memories)
  return (
      f"{_persona_harvest_context(persona, agent_resource_num)}\n"
      f"{memory_prompt_text}\n"
      f"{_active_law_block(active_law_text)}\n"
      f"{_negotiation_rounds_notice(state)}\n"
      f"{_deliberation_scope_block(mode)}\n"
      f"{_state_lines(state)}\n"
      f"{tool_instructions + chr(10) if tool_instructions else ''}"
      "Conversation so far:\n"
      f"{history_text}\n"
      "Respond with your next message in the contract negotiation.\n"
      "If there is an active law and you propose to keep it exactly as it is, you can propose the exact word <UNCHANGED>.\n"
      "Otherwise, if you propose a new contract, include exactly one <constitution>...</constitution> block. Don't add <constitution> if you agree with any present or proposed one.\n"
      "Only the text inside <constitution>...</constitution> will be voted on as the contract.\n"
      "If you agree with the most recent contract proposal in this thread, you must include <AGREE>.\n"
      "You can also agree with a specific proposer by including <AGREE to=Alice>.\n"
      "If you disagree or reject the current proposal, you must include <DISAGREE>.\n"
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
      "- <AGREE> or <AGREE to=Alice>\n"
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
            "Your last draft failed validation or could not be used. Fix EXACTLY what is described below.\n"
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
        "The law should be stable across months; use `self.state` only for constitution-specific memory.\n"
        "Once adopted, the law remains active until a later vote replaces it.\n"
        "If the new agreement is an amendment to the current law rather than a full replacement,\n"
        "preserve the current law's unchanged obligations in the new code.\n"
        "If the new agreement clearly replaces the current law, you may discard the old logic.\n"
        "\n"
        f"{_python_law_interface_spec_block()}\n"
        "# TRANSLATION RULES\n"
        "- Be LITERAL: encode only constraints the contract states.\n"
        "- Implement logic in `resolve()`; use `ctx` for all payoff primitives (not globals).\n"
        "- If the contract says 'each fisher catches at most X tons' → return a dict capping each submission at X.\n"
        "- If the contract says 'leave at least Y tons in the lake' →\n"
        "    cap total catches at max(0, fish_population - Y), then divide fairly.\n"
        "    But ONLY if the contract explicitly states this constraint.\n"
        "- If the contract specifies escrow, sanctions, insurance, participation fees, or transfers, encode them using `ctx` methods inside `resolve` (or hooks).\n"
        "- Use `self.state` only for bespoke ledgers; do not store framework escrow/violations there.\n"
        "- If your law uses `self.state`, override `__init__` with `prior_state=None` and call `super().__init__(..., prior_state=prior_state)` so state survives contract replacement.\n"
        "- The law must keep working in future months, not just the current one.\n"
        "- Do NOT add formulas, safety margins, or dynamic calculations the contract does not specify.\n"
        "- Do NOT model reproduction or compute post-catch population.\n"
        "- If the contract gives a fixed per-fisher limit, implement that fixed limit directly.\n"
        "\n"
        "# OUTPUT FORMAT\n"
        "Write a single ```python fenced block. Nothing else.\n"
        "- The code must define exactly one subclass of `Contract` with `resolve(self, month, fish_population, submissions, ctx)`.\n"
        "- No print(), no tests, no example usage.\n"
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


def constitution_vote_prompt(
    persona: PersonaAgent,
    state: FishingContractState,
    nl_contract: str,
    active_nl_law: str = "",
    active_code_law: str = "",
) -> tuple[str, str]:
  """Ratify the negotiated natural-language text before coding (no 'Python translatability' criterion)."""
  role_context = _persona_role_context(persona)
  system_prompt = (
      f"You are {persona.identity.name}, a fisherman in the village.\n\n"
      "CURRENT SITUATION:\n"
      f"- Fish in the lake: {state.fish_population:.1f} tons\n"
      f"- Round: {state.round_number + 1}\n\n"
      f"{role_context + chr(10) if role_context else ''}"
      f"{_active_law_block(active_nl_law)}"
      f"{_active_code_law_block(active_code_law)}"
      f"{_negotiation_rounds_notice(state)}"
      f"{_deliberation_scope_block(ContractMode.CODE_LAW)}"
      "Fishers finished negotiating; this is a final ratification vote on the proposed constitution.\n"
      "Judge whether this text is acceptable as the governing agreement for the next round: clarity, fairness, "
      "and fit for your community — not whether it could be turned into a particular kind of computer program.\n"
  )
  user_prompt = (
      "PROPOSED CONSTITUTION (natural language):\n"
      f"\"\"\"\n{nl_contract}\n\"\"\"\n\n"
      "VOTING GUIDELINES:\n"
      "- Vote <VOTE_YES> if you accept this text as the formal agreement for the next round.\n"
      "- Vote <VOTE_NO> if the text is vague, unfair, or worse than keeping the current law.\n"
      f"- A NO vote leaves the current law in force{': ' + active_nl_law if active_nl_law else '.'}\n\n"
      "If you ratify the constitution, start your response with <VOTE_YES>.\n"
      "Otherwise start with <VOTE_NO> and explain briefly."
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
      f"- Round: {state.round_number + 1}\n\n"
      f"{role_context + chr(10) if role_context else ''}"
      f"{_active_law_block(active_nl_law)}"
      f"{_active_code_law_block(active_code_law)}"
      f"{_python_law_interface_spec_block()}"
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
      f"{_python_law_interface_spec_block()}\n"
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


