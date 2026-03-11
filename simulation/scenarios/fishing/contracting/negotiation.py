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
    voter_prompt,
)


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


class MayoralVotingNegotiationManager:
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
  ) -> NLNegotiationResult:
    mayor_idx = state.speaker_idx % len(self.agents)
    mayor = self.agents[mayor_idx]
    others = [agent for agent in self.agents if agent != mayor]
    retrieved_memories = mayor.retrieve.retrieve(["contract", "agreement", "lake"], 8)
    system_prompt = negotiation_system_prompt(
        mayor,
        self.agents,
        retrieved_memories,
        state,
        mode,
    )
    proposal, proposal_html = await _aprompt_persona(
        mayor,
        "contracting",
        "mayor_proposal",
        system_prompt,
        mayor_proposal_prompt(state, agent_resource_num),
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
    }]
    if not cleaned:
      return NLNegotiationResult(nl_contract=None, conversations=conversations)

    votes = {mayor.identity.name: True}
    for idx, agent in enumerate(others, start=1):
      retrieved = agent.retrieve.retrieve(["contract", cleaned, "lake"], 8)
      system_prompt = negotiation_system_prompt(
          agent,
          self.agents,
          retrieved,
          state,
          mode,
      )
      vote_response, vote_html = await _aprompt_persona(
          agent,
          "contracting",
          "mayor_vote",
          system_prompt,
          voter_prompt(cleaned),
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
  ) -> None:
    self.agents = agents
    self.max_turns = max_turns
    self.min_agree_agents = min_agree_agents

  async def run_nl_negotiation(
      self,
      state: FishingContractState,
      mode: ContractMode,
      agent_resource_num: dict[str, int],
  ) -> NLNegotiationResult:
    conversations: list[dict[str, str]] = []
    proposals: list[tuple[int, str, str]] = []
    agreements: dict[str, str] = {}

    for turn in range(self.max_turns):
      speaker_idx = (state.speaker_idx + turn) % len(self.agents)
      speaker = self.agents[speaker_idx]
      retrieved = speaker.retrieve.retrieve(["contract", "agreement", "lake"], 8)
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
          "round_robin_contract",
          system_prompt,
          round_robin_prompt(conversations[-5:], state, agent_resource_num),
          default_value="No proposal at this time.",
      )
      speaker_name = speaker.identity.name
      conversations.append({
          "turn": turn,
          "agent": speaker_name,
          "message": response,
          "phase": "nl_negotiation",
          "html": html,
      })

      has_agree, target_agent = self._parse_agreement_signal(
          response,
          [agent.identity.name for agent in self.agents],
      )
      nl_contract = self._clean_message(response)
      if nl_contract and not self._looks_like_agreement_only(response):
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

  def _parse_agreement_signal(
      self,
      response: str,
      agent_names: list[str],
  ) -> tuple[bool, str | None]:
    match = re.search(r"<AGREE(?:\s+to\s*=\s*([^>]+))?>", response, re.IGNORECASE)
    if match:
      target = match.group(1)
      if target:
        return True, target.strip().strip("\"'")
      return True, None

    lowered = response.lower()
    semantic_agree_patterns = [
        r"\bi agree with ([a-zA-Z]+)",
        r"\bi support ([a-zA-Z]+)'?s proposal",
        r"\bi back ([a-zA-Z]+)'?s proposal",
        r"\bi accept ([a-zA-Z]+)'?s proposal",
    ]
    for pattern in semantic_agree_patterns:
      semantic_match = re.search(pattern, lowered)
      if semantic_match:
        target = semantic_match.group(1)
        for agent_name in agent_names:
          if agent_name.lower() == target.lower():
            return True, agent_name
    if self._looks_like_agreement_only(response):
      return True, None
    return False, None

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
    message = re.sub(r"```\w*\n?.*?```", "", message, flags=re.DOTALL)
    message = re.sub(r"<END_CONVERSATION>", "", message, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", message).strip()
    return cleaned if len(cleaned) > 5 else None

  def _looks_like_agreement_only(self, message: str) -> bool:
    cleaned = re.sub(r"<[^>]+>", "", message).strip().lower()
    patterns = [
        r"^i agree\b",
        r"^i support\b",
        r"^i accept\b",
        r"^yes,? i agree\b",
        r"^i'm on board\b",
        r"^i am on board\b",
        r"^that sounds good\b",
    ]
    return any(re.search(pattern, cleaned) for pattern in patterns)


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
      retry_attempt: int = 0,
  ) -> tuple[bool, str | None, list[dict[str, str]], dict[str, bool]]:
    votes: dict[str, bool] = {}
    conversations: list[dict[str, str]] = []
    for agent in self.agents:
      system_prompt, user_prompt = coded_contract_vote_prompt(
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
      voted_yes = "<VOTE_YES>" in response.upper() or "YES" in response.upper().split()[:5]
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
    feedback = await self._gather_feedback(votes, coded_contract, nl_contract)
    return False, feedback, conversations, votes

  async def _gather_feedback(
      self,
      votes: dict[str, bool],
      coded_contract: str,
      nl_contract: str,
  ) -> str:
    feedback = []
    for agent in self.agents:
      if votes.get(agent.identity.name, False):
        continue
      system_prompt, user_prompt = coded_contract_feedback_prompt(
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
      feedback.append(f"{agent.identity.name}: {response.strip()}")
    return "\n".join(feedback) if feedback else "No specific feedback was provided."
