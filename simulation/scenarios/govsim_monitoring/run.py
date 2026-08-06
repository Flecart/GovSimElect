"""Run the five-agent GovSim monitoring and natural-language governance cycle."""

from __future__ import annotations

import asyncio
import collections
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from simulation.persona.common import PersonaIdentity
from simulation.persona.memory import MemoryRecord, MemoryType, build_memory
from simulation.scenarios.fishing.agents.persona_v3.cognition.utils import (
    conversation_to_string_with_dash,
)
from simulation.utils import ModelWandbWrapper

from .prompts import (
    DISCUSSION_PROMPT,
    HARVEST_PROMPT,
    MONITOR_PROMPT,
    PROPOSAL_PROMPT,
    PUNISH_PROMPT,
    VOTE_PROMPT,
    system_prompt,
)


INITIAL_NORM = "No norm is currently choosen"


def clamp_unit(value: float) -> float:
  return min(1.0, max(0.0, value))


def parse_unit(response: str, default: float = 0.5) -> float:
  match = re.search(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", response)
  return clamp_unit(float(match.group(0))) if match else default


def parse_harvest_choice(response: str) -> tuple[float, float]:
  effort = re.search(
      r"^Effort:\s*((?:0(?:\.\d+)?|1(?:\.0+)?))\s*$",
      response,
      flags=re.IGNORECASE | re.MULTILINE,
  )
  contribution = re.search(
      r"^Contribution:\s*((?:0(?:\.\d+)?|1(?:\.0+)?))\s*$",
      response,
      flags=re.IGNORECASE | re.MULTILINE,
  )
  return (
      clamp_unit(float(effort.group(1))) if effort else 0.5,
      clamp_unit(float(contribution.group(1))) if contribution else 0.0,
  )


def parse_yes_no(response: str) -> bool:
  match = re.search(r"\b(yes|no)\b", response, flags=re.IGNORECASE)
  return bool(match and match.group(1).lower() == "yes")


def parse_norms(response: str, current: str, group: str) -> tuple[str, str]:
  personal = re.search(r"^Personal:\s*(.+)$", response, flags=re.MULTILINE)
  community = re.search(r"^Community:\s*(.+)$", response, flags=re.MULTILINE)
  return (
      personal.group(1).strip() if personal else current,
      community.group(1).strip() if community else group,
  )


def parse_proposal_vote(response: str, proposal_count: int, default: int = 0) -> int:
  """Parse a one-based proposal number into a bounded zero-based index."""
  match = re.search(r"\b(\d+)\b", response)
  if not match:
    return default
  selected = int(match.group(1)) - 1
  return selected if 0 <= selected < proposal_count else default


def choose_winner(votes: dict[str, str], proposals: list[str], rng: random.Random) -> str:
  counts = collections.Counter(v for v in votes.values() if v in proposals)
  if not counts:
    return rng.choice(proposals)
  highest = max(counts.values())
  tied = [proposal for proposal in proposals if counts[proposal] == highest]
  return rng.choice(tied)


def update_efficiency(alpha: float, total_contribution: float, requirement: float) -> float:
  """Centered sigmoid: contribution C maintains alpha, less lowers, more raises."""
  if requirement <= 0:
    raise ValueError("maintenance_requirement must be positive")
  centered = total_contribution / requirement - 1.0
  sigmoid = 1.0 / (1.0 + math.exp(-centered))
  return min(1.0, max(0.0, alpha * (0.5 + sigmoid)))


@dataclass
class Fisher:
  agent_id: str
  name: str
  identity: PersonaIdentity
  memory: Any
  wealth: float
  norm: str = INITIAL_NORM
  alive: bool = True
  effort: float = 0.0
  contribution: float = 0.0
  harvest: float = 0.0
  monitoring_propensity: float = 0.0
  suspended_round: int | None = None
  has_memory: bool = False


@dataclass
class RoundResult:
  round: int
  stock_before: float
  stock_after_harvest: float
  stock_after_regeneration: float
  alpha_before: float
  alpha_after: float
  group_norm: str
  efforts: dict[str, float]
  contributions: dict[str, float]
  harvests: dict[str, float]
  wealth: dict[str, float]
  monitoring: dict[str, dict[str, Any]]
  punished: list[str]
  suspended: list[str]
  discussion: list[dict[str, str]]
  personal_norms: dict[str, str]
  proposals: dict[str, str]
  votes: dict[str, str]
  collapsed: bool
  collapse_reason: str | None


class GovSimMonitoringExperiment:
  def __init__(
      self,
      cfg: DictConfig,
      wrappers: list[ModelWandbWrapper],
      experiment_storage: str,
      *,
      memory_factory=build_memory,
  ) -> None:
    self.cfg = cfg
    self.wrappers = wrappers
    self.storage = Path(experiment_storage)
    self.rng = random.Random(int(cfg.seed))
    self.stock = float(cfg.environment.initial_stock)
    self.alpha = float(cfg.environment.initial_alpha)
    self.group_norm = str(OmegaConf.select(cfg, "initial_group_norm", default=INITIAL_NORM))
    self.message_path = self.storage / "messages.jsonl"
    self.message_lock = asyncio.Lock()
    self.message_sequence = 0
    self.fishers: list[Fisher] = []

    for index in range(int(cfg.personas.num)):
      agent_id = f"persona_{index}"
      persona_cfg = cfg.personas[agent_id]
      identity = PersonaIdentity(agent_id=agent_id, **persona_cfg)
      memory_path = self.storage / agent_id
      os.makedirs(memory_path, exist_ok=True)
      self.fishers.append(Fisher(
          agent_id=agent_id,
          name=identity.name,
          identity=identity,
          memory=memory_factory(str(memory_path), cfg.memory),
          wealth=float(cfg.environment.initial_wealth),
      ))

  def _system(self, fisher: Fisher) -> str:
    env = self.cfg.environment
    return system_prompt(
        fisher.identity,
        peer_names=[f.name for f in self.fishers if f.agent_id != fisher.agent_id],
        capacity=float(env.carrying_capacity),
        alpha=float(env.initial_alpha),
        maintenance_requirement=float(env.maintenance_requirement),
        consumption=float(env.consumption),
        penalty=float(env.punishment_penalty),
    )

  def _state(self, fisher: Fisher) -> str:
    return (
        f"Current lake stock: {self.stock:.6g}\n"
        f"Current harvesting efficiency: {self.alpha:.6g}\n"
        f"Your current wealth: {fisher.wealth:.6g}\n"
        f"Your previous monitoring propensity: {fisher.monitoring_propensity:.6g}\n"
        f"Your personal norm: {fisher.norm}\n"
        f"Shared community norm: {self.group_norm}"
    )

  def _private_history(self, fisher: Fisher, round_id: int, query: str) -> str:
    if not fisher.has_memory:
      return f"Private historical context for {fisher.name}:\n- No prior history."
    current_time = datetime(2025, 1, 1) + timedelta(days=31 * round_id)
    records = fisher.memory.search(
        query,
        int(self.cfg.memory.top_k),
        current_time,
    )
    lines = "\n".join(f"- {created:%Y-%m-%d}: {text}" for created, text in records)
    return f"Private historical context for {fisher.name}:\n{lines or '- No relevant history.'}"

  async def _ask(
      self,
      fisher: Fisher,
      round_id: int,
      phase: str,
      prompt: str,
      *,
      max_tokens: int,
      default: str,
  ) -> str:
    model = self.wrappers[int(fisher.agent_id.split("_")[-1])]
    session = model.start_prompt(fisher.name, phase, phase)
    session.add_message("system", self._system(fisher))
    session.add_user(prompt)
    response = await model.acomplete_prompt(
        session,
        max_tokens=max_tokens,
        temperature=float(self.cfg.llm.temperature),
        top_p=float(self.cfg.llm.top_p),
        default_value=default,
    )
    messages = list(session.messages)
    model.end_prompt(session)
    async with self.message_lock:
      self.message_sequence += 1
      with self.message_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "sequence": self.message_sequence,
            "round": round_id,
            "agent_id": fisher.agent_id,
            "agent_name": fisher.name,
            "phase": phase,
            "messages": messages[:-1],
            "request": prompt,
            "response": response.strip(),
        }, ensure_ascii=False) + "\n")
        handle.flush()
    return response.strip()

  def _store(self, fisher: Fisher, round_id: int, description: str) -> None:
    created = datetime(2025, 1, 1) + timedelta(days=31 * round_id)
    fisher.memory.store(MemoryRecord(
        type=MemoryType.EVENT,
        description=description,
        created=created,
        expiration=created + timedelta(days=31 * (int(self.cfg.environment.max_rounds) + 1)),
        importance_score=10,
    ))
    fisher.has_memory = True

  async def _harvest_and_contribute(self, round_id: int) -> None:
    async def choose(fisher: Fisher):
      if fisher.suspended_round == round_id:
        return fisher.agent_id, 0.0, 0.0
      prompt = HARVEST_PROMPT.format(
          state=self._state(fisher),
          memory=self._private_history(
              fisher, round_id, "harvesting maintenance wealth norms and sanctions"
          ),
      )
      response = await self._ask(
          fisher,
          round_id,
          "harvest_contribution",
          prompt,
          max_tokens=80,
          default="Effort: 0.5\nContribution: 0",
      )
      effort, contribution = parse_harvest_choice(response)
      return fisher.agent_id, effort, contribution

    choices = await asyncio.gather(*(choose(fisher) for fisher in self.fishers))
    by_id = {agent_id: (effort, contribution) for agent_id, effort, contribution in choices}
    stock_before = self.stock
    for fisher in self.fishers:
      fisher.effort, fisher.contribution = by_id[fisher.agent_id]
      fisher.harvest = self.alpha * fisher.effort * stock_before
    self.stock = max(0.0, stock_before - sum(f.harvest for f in self.fishers))
    consumption = float(self.cfg.environment.consumption)
    for fisher in self.fishers:
      fisher.wealth += fisher.harvest - consumption - fisher.contribution

  async def _monitor_and_punish(
      self, round_id: int
  ) -> tuple[dict[str, dict[str, Any]], set[str]]:
    async def choose_propensity(fisher: Fisher):
      prompt = MONITOR_PROMPT.format(
          state=self._state(fisher),
          memory=self._private_history(fisher, round_id, "monitoring trust punishment peers"),
      )
      response = await self._ask(
          fisher, round_id, "monitoring", prompt, max_tokens=20, default="0.5"
      )
      return fisher.agent_id, parse_unit(response)

    propensities = dict(await asyncio.gather(
        *(choose_propensity(fisher) for fisher in self.fishers)
    ))
    records: dict[str, dict[str, Any]] = {}
    successful: list[tuple[Fisher, Fisher, str]] = []
    for fisher in self.fishers:
      fisher.monitoring_propensity = propensities[fisher.agent_id]
      candidates = [candidate for candidate in self.fishers if candidate is not fisher]
      target = self.rng.choice(candidates)
      success = self.rng.random() < fisher.monitoring_propensity
      record: dict[str, Any] = {
          "propensity": fisher.monitoring_propensity,
          "target": target.name,
          "success": success,
          "observed": None,
          "punished": False,
      }
      if success:
        observed = {
            "effort": target.effort,
            "contribution": target.contribution,
            "harvest": target.harvest,
        }
        record["observed"] = observed
        observation = (
            f"{target.name}: effort {target.effort:.6g}, contribution "
            f"{target.contribution:.6g}, realized harvest {target.harvest:.6g}."
        )
        self._store(
            fisher,
            round_id,
            f"Private monitoring in round {round_id}: I observed {observation}",
        )
        successful.append((fisher, target, observation))
      records[fisher.name] = record

    async def decide(actor: Fisher, target: Fisher, observation: str):
      prompt = PUNISH_PROMPT.format(
          state=self._state(actor),
          memory=self._private_history(actor, round_id, "observed behavior punishment norm"),
          observation=observation,
      )
      response = await self._ask(
          actor, round_id, "punishment", prompt, max_tokens=20, default="no"
      )
      return actor, target, parse_yes_no(response)

    decisions = await asyncio.gather(*(decide(*item) for item in successful))
    punished: set[str] = set()
    for actor, target, should_punish in decisions:
      records[actor.name]["punished"] = should_punish
      if should_punish:
        punished.add(target.agent_id)

    penalty = float(self.cfg.environment.punishment_penalty)
    for fisher in self.fishers:
      if fisher.agent_id in punished:
        fisher.wealth -= penalty
        fisher.suspended_round = round_id + 1
        self._store(
            fisher,
            round_id,
            (
                f"Punishment received in round {round_id}: one or more peers "
                f"imposed the community's binary penalty of {penalty:.6g} on me. "
                f"My wealth fell to {fisher.wealth:.6g}, and I am suspended from "
                f"harvesting and contributing in round {round_id + 1}."
            ),
        )
    return records, punished

  async def _discuss(self, round_id: int) -> list[dict[str, str]]:
    conversation: list[tuple[str, str]] = []
    current = self.rng.choice(self.fishers)
    max_steps = int(self.cfg.discussion.max_steps)
    for _ in range(max_steps):
      rendered = conversation_to_string_with_dash(conversation)
      prompt = DISCUSSION_PROMPT.format(
          location_time=(
              "Location: restaurant\n"
              f"Date: {(datetime(2025, 1, 1) + timedelta(days=31 * round_id)):%Y-%m-%d}\n\n"
          ),
          memory=self._private_history(
              current, round_id, "current lake monitoring contribution punishment discussion"
          ),
          participants=", ".join(f.name for f in self.fishers),
          conversation=rendered,
      )
      default = (
          "Response: We should preserve the lake and agree on responsible behavior.\n"
          "Conversation conclusion by me: yes\nNext speaker: N/A"
      )
      response = await self._ask(
          current, round_id, "discussion", prompt, max_tokens=500, default=default
      )
      utterance_match = re.search(
          r"Response:\s*(.*?)(?:\nConversation conclusion by me:|\Z)",
          response,
          flags=re.IGNORECASE | re.DOTALL,
      )
      utterance = utterance_match.group(1).strip() if utterance_match else response.strip()
      conversation.append((current.name, utterance))
      conclusion = re.search(
          r"Conversation conclusion by me:\s*(yes|no)",
          response,
          flags=re.IGNORECASE,
      )
      ended = parse_yes_no(conclusion.group(0) if conclusion else "yes")
      if ended:
        break
      next_match = re.search(r"Next speaker:\s*(.+)", response, flags=re.IGNORECASE)
      next_name = next_match.group(1).strip() if next_match else ""
      current = next((f for f in self.fishers if f.name.lower() == next_name.lower()), self.rng.choice(self.fishers))

    transcript = conversation_to_string_with_dash(conversation)
    for fisher in self.fishers:
      self._store(
          fisher,
          round_id,
          f"Public discussion transcript in round {round_id}:\n{transcript}",
      )
    return [{"speaker": speaker, "utterance": utterance} for speaker, utterance in conversation]

  async def _propose_and_vote(
      self, round_id: int, discussion: list[dict[str, str]]
  ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    conversation = conversation_to_string_with_dash([
        (item["speaker"], item["utterance"]) for item in discussion
    ])

    async def propose(fisher: Fisher):
      prompt = PROPOSAL_PROMPT.format(
          state=self._state(fisher),
          memory=self._private_history(fisher, round_id, "norm proposal lake community"),
          conversation=conversation,
      )
      default = f"Personal: {fisher.norm}\nCommunity: {self.group_norm}"
      response = await self._ask(
          fisher, round_id, "norm_proposal", prompt, max_tokens=1200, default=default
      )
      personal, community = parse_norms(response, fisher.norm, self.group_norm)
      return fisher, personal, community

    proposal_results = await asyncio.gather(*(propose(fisher) for fisher in self.fishers))
    proposals: dict[str, str] = {}
    personal_norms: dict[str, str] = {}
    for fisher, personal, community in proposal_results:
      fisher.norm = personal
      personal_norms[fisher.name] = personal
      proposals[fisher.name] = community

    unique = list(dict.fromkeys(proposals.values()))
    proposal_block = "\n".join(
        f"{index}. {proposal}" for index, proposal in enumerate(unique, start=1)
    )

    async def vote(fisher: Fisher):
      prompt = VOTE_PROMPT.format(
          state=self._state(fisher),
          memory=self._private_history(fisher, round_id, "vote proposal community norm"),
          conversation=conversation,
          proposals=proposal_block,
      )
      response = await self._ask(
          fisher, round_id, "community_vote", prompt, max_tokens=32, default="1"
      )
      selected = unique[parse_proposal_vote(response, len(unique))]
      return fisher.name, selected

    votes = dict(await asyncio.gather(*(vote(fisher) for fisher in self.fishers)))
    self.group_norm = choose_winner(votes, unique, self.rng)
    return personal_norms, proposals, votes

  def _remember_round(self, fisher: Fisher, result: RoundResult) -> None:
    monitor = result.monitoring[fisher.name]
    monitor_text = (
        f"Monitoring targeted {monitor['target']}; success={monitor['success']}; "
        f"punished={monitor['punished']}."
    )
    self._store(
        fisher,
        result.round,
        (
            f"Round {result.round}: I used effort {fisher.effort:.6g}, contributed "
            f"{fisher.contribution:.6g}, harvested {fisher.harvest:.6g}, and ended "
            f"with wealth {fisher.wealth:.6g}. {monitor_text} The lake regenerated "
            f"to {result.stock_after_regeneration:.6g}. The shared norm became: "
            f"{result.group_norm}"
        ),
    )

  async def step(self, round_id: int) -> RoundResult:
    stock_before = self.stock
    alpha_before = self.alpha
    await self._harvest_and_contribute(round_id)
    stock_after_harvest = self.stock
    monitoring, punished_ids = await self._monitor_and_punish(round_id)

    for fisher in self.fishers:
      fisher.alive = fisher.wealth >= 0
    first_death = any(not fisher.alive for fisher in self.fishers)
    lake_depleted = stock_after_harvest < float(self.cfg.environment.collapse_threshold)

    discussion: list[dict[str, str]] = []
    personal_norms = {f.name: f.norm for f in self.fishers}
    proposals: dict[str, str] = {}
    votes: dict[str, str] = {}
    if not first_death:
      discussion = await self._discuss(round_id)
      personal_norms, proposals, votes = await self._propose_and_vote(round_id, discussion)

    self.alpha = update_efficiency(
        self.alpha,
        sum(fisher.contribution for fisher in self.fishers),
        float(self.cfg.environment.maintenance_requirement),
    )
    self.stock = min(
        float(self.cfg.environment.carrying_capacity),
        stock_after_harvest * 2,
    )
    collapse_reason = "first_death" if first_death else "lake_depletion" if lake_depleted else None
    result = RoundResult(
        round=round_id,
        stock_before=stock_before,
        stock_after_harvest=stock_after_harvest,
        stock_after_regeneration=self.stock,
        alpha_before=alpha_before,
        alpha_after=self.alpha,
        group_norm=self.group_norm,
        efforts={f.name: f.effort for f in self.fishers},
        contributions={f.name: f.contribution for f in self.fishers},
        harvests={f.name: f.harvest for f in self.fishers},
        wealth={f.name: f.wealth for f in self.fishers},
        monitoring=monitoring,
        punished=[f.name for f in self.fishers if f.agent_id in punished_ids],
        suspended=[f.name for f in self.fishers if f.suspended_round == round_id],
        discussion=discussion,
        personal_norms=personal_norms,
        proposals=proposals,
        votes=votes,
        collapsed=collapse_reason is not None,
        collapse_reason=collapse_reason,
    )
    for fisher in self.fishers:
      self._remember_round(fisher, result)
    return result

  async def run(self) -> dict[str, Any]:
    self.storage.mkdir(parents=True, exist_ok=True)
    results_path = self.storage / "consolidated_results.json"
    rounds: list[RoundResult] = []
    with results_path.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps({
          "type": "initialization",
          "data": {
              **OmegaConf.to_container(self.cfg, resolve=True),
              "scenario": "govsim_monitoring",
          },
      }) + "\n")
      for round_id in range(1, int(self.cfg.environment.max_rounds) + 1):
        result = await self.step(round_id)
        rounds.append(result)
        handle.write(json.dumps({"type": "round", "data": asdict(result)}, ensure_ascii=False) + "\n")
        handle.flush()
        if result.collapsed:
          break

    summary = {
        "survival_time": len(rounds),
        "collapsed": rounds[-1].collapsed if rounds else False,
        "collapse_reason": rounds[-1].collapse_reason if rounds else None,
        "final_stock": self.stock,
        "final_alpha": self.alpha,
        "final_group_norm": self.group_norm,
        "final_wealth": {f.name: f.wealth for f in self.fishers},
        "total_contributions": {
            f.name: sum(result.contributions[f.name] for result in rounds)
            for f in self.fishers
        },
    }
    with (self.storage / "summary.json").open("w", encoding="utf-8") as handle:
      json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


async def run(
    cfg: DictConfig,
    logger,
    wrappers: list[ModelWandbWrapper],
    framework_wrapper: ModelWandbWrapper,
    coding_wrapper: ModelWandbWrapper,
    experiment_storage: str,
):
  del framework_wrapper, coding_wrapper
  summary = await GovSimMonitoringExperiment(cfg, wrappers, experiment_storage).run()
  logger.log_game(summary, last_log=True)
