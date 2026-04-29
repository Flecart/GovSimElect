"""Negotiation managers for the fishing formal-contract flow."""

import re

from simulation.persona import PersonaAgent

from .contract import ContractMode, FishingContractState, NLNegotiationResult
from .prompts import (
    coded_contract_feedback_prompt,
    coded_contract_vote_prompt,
    constitution_vote_prompt,
    free_chat_turn_prompt,
    mayor_proposal_prompt,
    negotiation_system_prompt,
    round_robin_prompt,
    round_robin_tag_clarification_prompt,
    voter_prompt,
)
from .tooling import ContractToolbox


async def _aprompt_persona(
    persona: PersonaAgent,
    phase_name: str,
    query_name: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float | None = None,
    default_value: str = "",
) -> tuple[str, str]:
  session = persona.act.model.start_prompt(persona.identity.name, phase_name, query_name)
  session.add_message("system", system_prompt)
  session.add_user(user_prompt)
  response = await persona.act.model.acomplete_prompt(
      session,
      temperature=temperature,
      default_value=default_value,
  )
  return response, session.html()


async def _aprompt_persona_with_tools(
    persona: PersonaAgent,
    phase_name: str,
    query_name: str,
    system_prompt: str,
    user_prompt: str,
    *,
    toolbox: ContractToolbox | None,
    max_tool_calls: int,
    temperature: float | None = None,
    default_value: str = "",
) -> tuple[str, str, int]:
  if toolbox is None or max_tool_calls <= 0:
    response, html = await _aprompt_persona(
        persona,
        phase_name,
        query_name,
        system_prompt,
        user_prompt,
        temperature=temperature,
        default_value=default_value,
    )
    return response, html, 0

  session = persona.act.model.start_prompt(persona.identity.name, phase_name, query_name)
  session.add_message("system", system_prompt)
  session.add_user(user_prompt)
  tool_calls_used = 0

  while True:
    response = await persona.act.model.acomplete_prompt(
        session,
        temperature=temperature,
        default_value=default_value,
    )
    tool_call = toolbox.parse_tool_call(response)
    if tool_call is None or tool_calls_used >= max_tool_calls:
      return response, session.html(), tool_calls_used
    tool_result = toolbox.execute(tool_call)
    tool_calls_used += 1
    session.add_user(
        "TOOL RESULT:\n"
        f"{tool_result}\n\n"
        "Continue the negotiation. If you need another tool, you may call one. Otherwise answer normally."
    )


def _parse_vote_tag(message: str) -> bool:
  return "<VOTE_YES>" in message.upper()


def _extract_constitution_text(message: str) -> str | None:
  tagged_block = re.search(
      r"<constitution>\s*(.*?)\s*</constitution>",
      message,
      flags=re.IGNORECASE | re.DOTALL,
  )
  if tagged_block:
    content = re.sub(r"\s+", " ", tagged_block.group(1)).strip()
    return content if len(content) > 5 else None
  open_only_block = re.search(
      r"<constitution>\s*(.*)",
      message,
      flags=re.IGNORECASE | re.DOTALL,
  )
  if open_only_block:
    content = re.sub(r"\s+", " ", open_only_block.group(1)).strip()
    return content if len(content) > 5 else None
  return None


class MayoralVotingNegotiationManager:
  def __init__(
      self,
      agents: list[PersonaAgent],
      max_turns: int = 10,
      toolbox: ContractToolbox | None = None,
      max_tool_calls: int = 0,
  ) -> None:
    self.agents = agents
    self.max_turns = max_turns
    self.toolbox = toolbox
    self.max_tool_calls = max_tool_calls

  async def run_nl_negotiation(
      self,
      state: FishingContractState,
      mode: ContractMode,
      agent_resource_num: dict[str, int],
      active_law_text: str = "",
  ) -> NLNegotiationResult:
    mayor_idx = state.speaker_idx % len(self.agents)
    mayor = self.agents[mayor_idx]
    others = [agent for agent in self.agents if agent != mayor]
    tool_instructions = (
        self.toolbox.render_tool_instructions() if self.toolbox is not None else ""
    )
    system_prompt = negotiation_system_prompt(
        mayor,
        self.agents,
        state,
        mode,
        tool_instructions=tool_instructions,
        active_law_text=active_law_text,
    )
    proposal, proposal_html, tool_calls_used = await _aprompt_persona_with_tools(
        mayor,
        "contracting",
        "mayor_proposal",
        system_prompt,
        mayor_proposal_prompt(
            mayor,
            state,
            agent_resource_num,
            mode,
            tool_instructions=tool_instructions,
            active_law_text=active_law_text,
        ),
        toolbox=self.toolbox,
        max_tool_calls=self.max_tool_calls,
        default_value=(
            "No contract at all"
        ),
    )
    cleaned = self._clean_message(proposal)
    conversations = [{
        "turn": 0,
        "agent": mayor.identity.name,
        "message": proposal,
        "phase": "nl_negotiation",
        "html": proposal_html,
        "tool_calls_used": tool_calls_used,
    }]
    if not cleaned:
      return NLNegotiationResult(nl_contract=None, conversations=conversations)

    votes = {mayor.identity.name: True}
    for idx, agent in enumerate(others, start=1):
      retrieved = agent.retrieve.retrieve(["contract", cleaned, "lake"], 8)
      system_prompt, user_prompt = voter_prompt(
          agent,
          state,
          cleaned,
          agent_resource_num,
          mode,
          active_law_text=active_law_text,
      )
      vote_response, vote_html = await _aprompt_persona(
          agent,
          "contracting",
          "mayor_vote",
          system_prompt,
          user_prompt,
          default_value="<VOTE_YES>",
      )
      voted_yes = self._parse_vote(vote_response)
      votes[agent.identity.name] = voted_yes
      conversations.append({
          "turn": idx,
          "agent": agent.identity.name,
          "message": vote_response,
          "phase": "nl_negotiation",
          "vote": voted_yes,
          "html": vote_html,
      })

    yes_votes = sum(1 for voted_yes in votes.values() if voted_yes)
    if yes_votes > len(self.agents) / 2:
      return NLNegotiationResult(
          nl_contract=cleaned,
          conversations=conversations,
          votes=votes,
      )
    return NLNegotiationResult(nl_contract=None, conversations=conversations, votes=votes)

  def _clean_message(self, message: str) -> str | None:
    message = re.sub(r"```\w*\n?.*?```", "", message, flags=re.DOTALL)
    constitution = _extract_constitution_text(message)
    if constitution:
      return constitution
    message = re.sub(r"\s+", " ", message).strip()
    return message if len(message) > 5 else None

  def _parse_vote(self, message: str) -> bool:
    return "<VOTE_YES>" in message.upper()


class RoundRobinNegotiationManager:
  def __init__(
      self,
      agents: list[PersonaAgent],
      max_turns: int = 10,
      min_agree_agents: int | None = None,
      toolbox: ContractToolbox | None = None,
      max_tool_calls: int = 0,
  ) -> None:
    self.agents = agents
    self.max_turns = max_turns
    self.min_agree_agents = min_agree_agents
    self.toolbox = toolbox
    self.max_tool_calls = max_tool_calls

  async def run_nl_negotiation(
      self,
      state: FishingContractState,
      mode: ContractMode,
      agent_resource_num: dict[str, int],
      active_law_text: str = "",
  ) -> NLNegotiationResult:
    conversations: list[dict[str, str]] = []
    proposals: list[tuple[int, str, str]] = []
    proposal_by_proposer: dict[str, str] = {}
    agreements: dict[str, str] = {}

    for turn in range(self.max_turns):
      speaker_idx = (state.speaker_idx + turn) % len(self.agents)
      speaker = self.agents[speaker_idx]
      tool_instructions = (
          self.toolbox.render_tool_instructions() if self.toolbox is not None else ""
      )
      system_prompt = negotiation_system_prompt(
          speaker,
          self.agents,
          state,
          mode,
          tool_instructions=tool_instructions,
          active_law_text=active_law_text,
      )
      response, html, tool_calls_used = await _aprompt_persona_with_tools(
          speaker,
          "contracting",
          "round_robin_contract",
          system_prompt,
          round_robin_prompt(
              speaker,
              conversations[-5:],
              state,
              agent_resource_num,
              mode,
              tool_instructions=tool_instructions,
              active_law_text=active_law_text,
          ),
          toolbox=self.toolbox,
          max_tool_calls=self.max_tool_calls,
          default_value="No proposal at this time.",
      )
      response, clarification_html = await self._ensure_tagged_round_robin_response(
          speaker,
          response,
      )
      if clarification_html:
        html = f"{html}{clarification_html}"
      speaker_name = speaker.identity.name
      conversations.append({
          "turn": turn,
          "agent": speaker_name,
          "message": response,
          "phase": "nl_negotiation",
          "html": html,
          "tool_calls_used": tool_calls_used,
      })

      has_agree, has_disagree = self._parse_agreement_signal(response)
      nl_contract = self._clean_message(response)
      is_own_proposal = self._is_own_proposal_message(
          speaker_name=speaker_name,
          response=response,
          nl_contract=nl_contract,
          has_agree=has_agree,
          has_disagree=has_disagree,
      )
      if is_own_proposal:
        proposals.append((turn, speaker_name, nl_contract))
        proposal_by_proposer[speaker_name.lower()] = nl_contract
        agreements[speaker_name] = nl_contract
      elif has_agree and proposals:
        # Only resolve AGREE if speaker didn't also submit their own constitution.
        # A constitution block always takes precedence over a co-occurring AGREE tag.
        target_proposer = self._extract_agree_target(response)
        target_contract = self._resolve_agreement_target_contract(
            target_proposer=target_proposer,
            proposal_by_proposer=proposal_by_proposer,
            proposals=proposals,
            agreements=agreements,
        )
        if target_contract is not None:
          agreements[speaker_name] = target_contract
    
      consensus = self._check_consensus(agreements)
      if consensus:
        return NLNegotiationResult(
            nl_contract=consensus[0],
            conversations=conversations,
            agreements=agreements,
        )
      if "<END_CONVERSATION>" in response.upper():
        break

    return NLNegotiationResult(
        nl_contract=None,
        conversations=conversations,
        agreements=agreements,
    )

  async def _ensure_tagged_round_robin_response(
      self,
      speaker: PersonaAgent,
      response: str,
  ) -> tuple[str, str]:
    has_agree, has_disagree = self._parse_agreement_signal(response)
    if has_agree or has_disagree or "<END_CONVERSATION>" in response.upper():
      return response, ""
    # A response with a <constitution> block is a new proposal — no agreement tag required.
    if _extract_constitution_text(response) is not None:
      return response, ""
    system_prompt, user_prompt = round_robin_tag_clarification_prompt(response)
    clarified_response, clarified_html = await _aprompt_persona(
        speaker,
        "contracting",
        "round_robin_tag_clarification",
        system_prompt,
        user_prompt,
        default_value=f"<DISAGREE> {response.strip()}",
    )
    clarified_has_agree, clarified_has_disagree = self._parse_agreement_signal(
        clarified_response
    )
    if clarified_has_agree or clarified_has_disagree:
      return clarified_response, clarified_html
    return f"<DISAGREE> {response.strip()}", clarified_html

  def _parse_agreement_signal(self, response: str) -> tuple[bool, bool]:
    """Detect agreement/disagreement tags.

    Supported agree forms include:
      - <AGREE>
      - <AGREE Alice>
      - <AGREE to=Alice>
      - <AGREE to="Alice">
    """
    has_agree = re.search(r"<AGREE(?:\s[^>]*)?>", response, re.IGNORECASE) is not None
    has_disagree = "<DISAGREE>" in response.upper()
    return has_agree, has_disagree

  def _check_consensus(
      self,
      agreements: dict[str, str],
  ) -> tuple[str, list[str], int] | None:
    threshold = self.min_agree_agents or len(self.agents)
    grouped: dict[str, list[str]] = {}
    originals: dict[str, str] = {}
    for agent_name, content in agreements.items():
      normalized = re.sub(r"\s+", " ", content.strip().lower())
      grouped.setdefault(normalized, []).append(agent_name)
      originals.setdefault(normalized, content)
    for normalized, agent_names in grouped.items():
      if len(agent_names) >= threshold:
        return originals[normalized], agent_names, len(agent_names)
    return None

  def _clean_message(self, message: str) -> str | None:
    message = re.sub(r"<AGREE[^>]*>", "", message, flags=re.IGNORECASE)
    message = re.sub(r"<DISAGREE>", "", message, flags=re.IGNORECASE)
    message = re.sub(r"```\w*\n?.*?```", "", message, flags=re.DOTALL)
    message = re.sub(r"<END_CONVERSATION>", "", message, flags=re.IGNORECASE)
    constitution = _extract_constitution_text(message)
    if constitution:
      return constitution
    cleaned = re.sub(r"\s+", " ", message).strip()
    return cleaned if len(cleaned) > 5 else None

  def _is_own_proposal_message(
      self,
      speaker_name: str,
      response: str,
      nl_contract: str | None,
      has_agree: bool,
      has_disagree: bool,
  ) -> bool:
    if not nl_contract:
      return False
    has_constitution = _extract_constitution_text(response) is not None
    if has_agree and not has_disagree and not has_constitution:
      return False
    if has_disagree:
      return has_constitution
    if not has_agree:
      return True
    return True

  def _extract_agree_target(self, response: str) -> str | None:
    match = re.search(r"<AGREE(?:\s+([^>]*))?>", response, re.IGNORECASE)
    if not match:
      return None
    raw_args = (match.group(1) or "").strip()
    if not raw_args:
      return None

    keyed_match = re.search(
        r"\bto\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^\s>]+))",
        raw_args,
        re.IGNORECASE,
    )
    if keyed_match:
      return next((group for group in keyed_match.groups() if group), None)

    # Backward-compatible support: <AGREE Alice>
    return raw_args

  def _resolve_agreement_target_contract(
      self,
      target_proposer: str | None,
      proposal_by_proposer: dict[str, str],
      proposals: list[tuple[int, str, str]],
      agreements: dict[str, str] | None = None,
  ) -> str | None:
    if target_proposer:
      key = target_proposer.strip().lower()
      # Direct proposer lookup first.
      if key in proposal_by_proposer:
        return proposal_by_proposer[key]
      # Transitive: named target agreed to someone else's proposal.
      if agreements:
        for agent_name, contract in agreements.items():
          if agent_name.lower() == key:
            return contract
      return None
    # Bare <AGREE> with no target → latest proposal.
    if not proposals:
      return None
    return proposals[-1][2]

class CodedContractVotingManager:
  def __init__(
      self,
      agents: list[PersonaAgent],
  ) -> None:
    self.agents = agents

  async def vote_on_constitution(
      self,
      nl_contract: str,
      state: FishingContractState,
      active_nl_law: str = "",
      active_code_law: str = "",
  ) -> tuple[bool, list[dict[str, str]], dict[str, bool]]:
    """Majority vote on the natural-language constitution before coding (CODE_LAW)."""
    votes: dict[str, bool] = {}
    conversations: list[dict[str, str]] = []
    for agent in self.agents:
      system_prompt, user_prompt = constitution_vote_prompt(
          persona=agent,
          state=state,
          nl_contract=nl_contract,
          active_nl_law=active_nl_law,
          active_code_law=active_code_law,
      )
      response, html = await _aprompt_persona(
          agent,
          "contracting",
          "constitution_vote",
          system_prompt,
          user_prompt,
          default_value="<VOTE_YES>",
      )
      voted_yes = _parse_vote_tag(response)
      votes[agent.identity.name] = voted_yes
      conversations.append({
          "agent": agent.identity.name,
          "message": response,
          "phase": "constitution_vote",
          "vote": voted_yes,
          "html": html,
      })

    yes_votes = sum(1 for voted_yes in votes.values() if voted_yes)
    passed = yes_votes > len(self.agents) / 2
    return passed, conversations, votes

  async def vote_on_coded_contract(
      self,
      coded_contract: str,
      nl_contract: str,
      state: FishingContractState,
      retry_attempt: int = 0,
      active_nl_law: str = "",
      active_code_law: str = "",
  ) -> tuple[bool, str | None, list[dict[str, str]], dict[str, bool]]:
    raise NotImplementedError("CodedContractVotingManager.vote_on_coded_contract is not implemented")
    votes: dict[str, bool] = {}
    conversations: list[dict[str, str]] = []
    for agent in self.agents:
      system_prompt, user_prompt = coded_contract_vote_prompt(
          persona=agent,
          state=state,
          nl_contract=nl_contract,
          coded_contract=coded_contract,
          retry_attempt=retry_attempt,
          active_nl_law=active_nl_law,
          active_code_law=active_code_law,
      )
      response, html = await _aprompt_persona(
          agent,
          "contracting",
          "coded_contract_vote",
          system_prompt,
          user_prompt,
          default_value="<VOTE_YES>",
      )
      voted_yes = _parse_vote_tag(response)
      votes[agent.identity.name] = voted_yes
      conversations.append({
          "agent": agent.identity.name,
          "message": response,
          "phase": "coded_contract_vote",
          "vote": voted_yes,
          "html": html,
      })

    yes_votes = sum(1 for voted_yes in votes.values() if voted_yes)
    if yes_votes > len(self.agents) / 2:
      return True, coded_contract, conversations, votes
    feedback = await self._gather_feedback(votes, coded_contract, nl_contract, state)
    return False, feedback, conversations, votes

  async def _gather_feedback(
      self,
      votes: dict[str, bool],
      coded_contract: str,
      nl_contract: str,
      state: FishingContractState,
  ) -> str:
    feedback = []
    for agent in self.agents:
      if votes.get(agent.identity.name, False):
        continue
      system_prompt, user_prompt = coded_contract_feedback_prompt(
          persona=agent,
          state=state,
          nl_contract=nl_contract,
          coded_contract=coded_contract,
      )
      response, _ = await _aprompt_persona(
          agent,
          "contracting",
          "coded_contract_feedback",
          system_prompt,
          user_prompt,
          default_value="The code does not faithfully encode the negotiated rule.",
      )
      sentences = [s.strip() for s in response.split(".") if s.strip()]
      concise_response = ". ".join(sentences[:3]) if sentences else response.strip()
      feedback.append(f"{agent.identity.name}: {concise_response}")
    return "\n".join(feedback) if feedback else "No specific feedback was provided."


class FreeChatNegotiationManager:
  def __init__(
      self,
      agents: list[PersonaAgent],
      max_turns: int = 10,
  ) -> None:
    self.agents = agents
    self.max_turns = max_turns

  async def run_nl_negotiation(
      self,
      state: FishingContractState,
      mode: ContractMode,
      agent_resource_num: dict[str, int],
      active_law_text: str = "",
  ) -> NLNegotiationResult:
    del active_law_text
    conversations: list[dict[str, str]] = []
    for turn in range(self.max_turns):
      speaker_idx = (state.speaker_idx + turn) % len(self.agents)
      speaker = self.agents[speaker_idx]
      retrieved = speaker.retrieve.retrieve([], 8)
      system_prompt = negotiation_system_prompt(
          speaker,
          self.agents,
          retrieved,
          state,
          mode,
      )
      response, html = await _aprompt_persona(
          speaker,
          "contracting",
          "free_chat_turn",
          system_prompt,
          free_chat_turn_prompt(
              conversations[-5:],
              state,
              agent_resource_num,
          ),
          default_value="I am glad we can talk openly this month.",
      )
      conversations.append({
          "turn": turn,
          "agent": speaker.identity.name,
          "message": response,
          "phase": "free_chat",
          "html": html,
      })
      if "<END_CONVERSATION>" in response.upper():
        break
    return NLNegotiationResult(
        nl_contract=None,
        conversations=conversations,
        agreements={},
    )
