"""Negotiation managers for the fishing formal-contract flow."""

import re

from simulation.persona import PersonaAgent

from .contract import ContractMode, FishingContractState, NLNegotiationResult
from .prompts import (
    coded_contract_feedback_prompt,
    coded_contract_vote_prompt,
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
  upper = message.upper()
  return "<VOTE_YES>" in upper or "YES" in upper.split()[:5]


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
    retrieved_memories = mayor.retrieve.retrieve(["contract", "agreement", "lake"], 8)
    tool_instructions = (
        self.toolbox.render_tool_instructions() if self.toolbox is not None else ""
    )
    system_prompt = negotiation_system_prompt(
        mayor,
        self.agents,
        retrieved_memories,
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
    message = re.sub(r"\s+", " ", message).strip()
    return message if len(message) > 5 else None

  def _parse_vote(self, message: str) -> bool:
    upper = message.upper()
    return "<VOTE_YES>" in upper or "YES" in upper.split()[:5]


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
    agreements: dict[str, str] = {}

    for turn in range(self.max_turns):
      speaker_idx = (state.speaker_idx + turn) % len(self.agents)
      speaker = self.agents[speaker_idx]
      retrieved = speaker.retrieve.retrieve(["contract", "agreement", "lake"], 8)
      tool_instructions = (
          self.toolbox.render_tool_instructions() if self.toolbox is not None else ""
      )
      system_prompt = negotiation_system_prompt(
          speaker,
          self.agents,
          retrieved,
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

      has_agree, target_agent, has_disagree = self._parse_agreement_signal(response)
      nl_contract = self._clean_message(response)
      is_own_proposal = self._is_own_proposal_message(
          speaker_name=speaker_name,
          nl_contract=nl_contract,
          has_agree=has_agree,
          target_agent=target_agent,
          has_disagree=has_disagree,
      )
      if is_own_proposal:
        proposals.append((turn, speaker_name, nl_contract))
        agreements[speaker_name] = nl_contract
      if has_agree and proposals:
        proposal_to_agree = self._find_most_recent_proposal(
            proposals,
            target_agent,
        )
        if proposal_to_agree and proposal_to_agree[1] != speaker_name:
          agreements[speaker_name] = proposal_to_agree[2]

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
    has_agree, _, has_disagree = self._parse_agreement_signal(response)
    if has_agree or has_disagree or "<END_CONVERSATION>" in response.upper():
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
    clarified_has_agree, _, clarified_has_disagree = self._parse_agreement_signal(
        clarified_response
    )
    if clarified_has_agree or clarified_has_disagree:
      return clarified_response, clarified_html
    return f"<DISAGREE> {response.strip()}", clarified_html

  def _parse_agreement_signal(
      self,
      response: str,
  ) -> tuple[bool, str | None, bool]:
    match = re.search(r"<AGREE(?:\s+to\s*=\s*([^>]+))?>", response, re.IGNORECASE)
    if match:
      target = match.group(1)
      if target:
        return True, target.strip().strip("\"'"), False
      return True, None, False
    if "<DISAGREE>" in response.upper():
      return False, None, True
    return False, None, False

  def _find_most_recent_proposal(
      self,
      proposals: list[tuple[int, str, str]],
      target_agent: str | None,
  ) -> tuple[int, str, str] | None:
    if target_agent is None:
      return proposals[-1]
    for proposal in reversed(proposals):
      if proposal[1] == target_agent:
        return proposal
    return None

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
    cleaned = re.sub(r"\s+", " ", message).strip()
    return cleaned if len(cleaned) > 5 else None

  def _is_own_proposal_message(
      self,
      speaker_name: str,
      nl_contract: str | None,
      has_agree: bool,
      target_agent: str | None,
      has_disagree: bool,
  ) -> bool:
    if not nl_contract:
      return False
    if has_disagree:
      return self._looks_like_counterproposal(nl_contract)
    if not has_agree:
      return True
    if target_agent is None:
      return True
    return target_agent == speaker_name

  def _looks_like_counterproposal(self, message: str) -> bool:
    lowered = message.lower()
    proposal_markers = [
        "i propose",
        "my proposal",
        "counterproposal",
        "instead,",
        "rather,",
        "no more than",
        "at most",
        "each of us should",
    ]
    return any(marker in lowered for marker in proposal_markers)


class CodedContractVotingManager:
  def __init__(
      self,
      agents: list[PersonaAgent],
  ) -> None:
    self.agents = agents

  async def vote_on_coded_contract(
      self,
      coded_contract: str,
      nl_contract: str,
      state: FishingContractState,
      retry_attempt: int = 0,
  ) -> tuple[bool, str | None, list[dict[str, str]], dict[str, bool]]:
    votes: dict[str, bool] = {}
    conversations: list[dict[str, str]] = []
    for agent in self.agents:
      system_prompt, user_prompt = coded_contract_vote_prompt(
          persona=agent,
          state=state,
          nl_contract=nl_contract,
          coded_contract=coded_contract,
          retry_attempt=retry_attempt,
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
