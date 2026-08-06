"""Run one LLM common-pool-resource norm-formation experiment."""

from __future__ import annotations

import asyncio
import collections
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from simulation.persona.memory import MemoryRecord, MemoryType, build_memory
from simulation.utils import ModelWandbWrapper

from .prompts import (
    EFFORT_PROMPT,
    NORM_PROMPT,
    PUNISHMENT_PROMPT,
    VOTE_PROMPT,
    render_base,
)


SELFISH_NORMS = (
    "Maximize your catch while the fish are abundant",
    "Take as much as you can before others do",
    "Your family’s needs come first, fish as much as possible",
    "The lake is there to be used, take your maximum share",
    "Fish aggressively to ensure your own survival",
    "Get the most value from your fishing effort",
    "Compete to harvest more than others",
    "Focus on your immediate gains from the lake",
)

INITIAL_GROUP_NORM = "No norm is currently choosen"


@dataclass
class Villager:
  agent_id: int
  norm: str
  wealth: float
  memory: Any
  alive: bool = True
  effort: float = 0.0
  harvest: float = 0.0
  punished: bool = False
  punished_target: int | None = None
  has_memory: bool = False


@dataclass
class RoundResult:
  round: int
  stock_before: float
  stock_after_harvest: float
  stock_after_regeneration: float
  group_norm: str
  efforts: dict[int, float]
  harvests: dict[int, float]
  wealth: dict[int, float]
  punishments: dict[int, int | None]
  proposals: dict[int, str]
  votes: dict[int, str]
  collapsed: bool
  efficiency: float


def regenerate_stock(
    stock: float,
    capacity: float,
    *,
    mode: str = "doubling",
    growth_rate: float = 0.2,
) -> float:
  """Regenerate post-harvest stock; doubling is the general default."""
  if mode == "doubling":
    return min(capacity, stock * 2)
  if mode == "logistic":
    return min(capacity, stock + growth_rate * stock * (1 - stock / capacity))
  raise ValueError("regeneration.mode must be 'doubling' or 'logistic'")


def parse_effort(response: str, default: float = 0.5) -> float:
  match = re.search(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", response)
  if match is None:
    return default
  return min(1.0, max(0.0, float(match.group(0))))


def parse_punishment(response: str, actor_id: int, valid_ids: set[int]) -> int | None:
  if re.search(r"\bN\s*/?\s*A\b", response, flags=re.IGNORECASE):
    return None
  match = re.search(r"\b\d+\b", response)
  if match is None:
    return None
  target = int(match.group(0))
  return target if target in valid_ids and target != actor_id else None


def parse_norms(response: str, current_norm: str, group_norm: str) -> tuple[str, str]:
  personal = re.search(r"^Personal:\s*(.+)$", response, flags=re.MULTILINE)
  community = re.search(r"^Community:\s*(.+)$", response, flags=re.MULTILINE)
  return (
      personal.group(1).strip() if personal else current_norm,
      community.group(1).strip() if community else group_norm,
  )


def choose_winning_policy(
    votes: dict[int, str], proposals: list[str], rng: random.Random
) -> str:
  counts = collections.Counter(vote for vote in votes.values() if vote in proposals)
  if not counts:
    return rng.choice(proposals)
  highest = max(counts.values())
  tied = [proposal for proposal in proposals if counts[proposal] == highest]
  return rng.choice(tied)


class PaperCPRExperiment:
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
    self.experiment_storage = Path(experiment_storage)
    self.rng = random.Random(int(cfg.seed))
    self.stock = float(cfg.environment.initial_stock)
    self.group_norm = str(OmegaConf.select(
        cfg, "initial_group_norm", default=INITIAL_GROUP_NORM
    ))
    self.villagers: list[Villager] = []

    population = str(cfg.population).lower()
    if population != "selfish":
      raise ValueError("This single experiment supports population=selfish")
    if str(cfg.alignment).lower() != "full":
      raise ValueError("This single experiment supports alignment=full")
    if str(cfg.observation.peer_visibility).lower() != "all":
      raise ValueError("This Appendix-prompt experiment supports peer_visibility=all")
    if str(cfg.group_decision.tie_breaking).lower() != "random":
      raise ValueError("This single experiment supports tie_breaking=random")

    for agent_id in range(int(cfg.personas.num)):
      memory_path = self.experiment_storage / f"persona_{agent_id}"
      os.makedirs(memory_path, exist_ok=True)
      memory = memory_factory(str(memory_path), cfg.memory)
      self.villagers.append(Villager(
          agent_id=agent_id,
          norm=self.rng.choice(SELFISH_NORMS),
          wealth=float(cfg.environment.initial_wealth),
          memory=memory,
      ))

  def _visible_outcomes(self) -> str:
    rows = [f"Current lake stock: {self.stock:.6g}"]
    rows.extend(
        f"Villager {villager.agent_id}: fishing effort {villager.effort:.6g}, "
        f"total payoff {villager.wealth:.6g}"
        for villager in self.villagers
    )
    return "\n".join(rows)

  def _private_history(self, villager: Villager, round_id: int) -> str:
    if not villager.has_memory:
      return "Your private historical context:\nNo prior personal history."
    current_time = datetime(2000, 1, 1) + timedelta(days=round_id)
    memories = villager.memory.search(
        "My fishing decisions, outcomes, punishments, and community norms",
        int(self.cfg.memory.top_k),
        current_time,
    )
    text = "\n".join(f"- {description}" for _, description in memories)
    return f"Your private historical context:\n{text or 'No relevant personal history.'}"

  def _observation(self, villager: Villager, round_id: int) -> str:
    return f"{self._visible_outcomes()}\n{self._private_history(villager, round_id)}"

  async def _ask(
      self,
      villager: Villager,
      phase: str,
      prompt: str,
      *,
      max_tokens: int,
      default: str,
  ) -> str:
    model = self.wrappers[villager.agent_id]
    session = model.start_prompt(str(villager.agent_id), phase, phase)
    session.add_user(prompt)
    response = await model.acomplete_prompt(
        session,
        max_tokens=max_tokens,
        temperature=float(self.cfg.llm.temperature),
        top_p=float(self.cfg.llm.top_p),
        default_value=default,
    )
    model.end_prompt(session)
    return response.strip()

  def _base_context(self, villager: Villager, *, punishment: bool = False) -> str:
    return render_base(
        float(self.cfg.environment.consumption),
        villager.norm,
        self.group_norm,
        punishment=punishment,
    )

  async def _choose_efforts(self, round_id: int) -> dict[int, float]:
    async def choose(villager: Villager):
      prompt = EFFORT_PROMPT.format(
          base_context=self._base_context(villager),
          observation=self._observation(villager, round_id),
      )
      response = await self._ask(
          villager, "effort", prompt, max_tokens=20, default="0.5"
      )
      return villager.agent_id, parse_effort(response)

    return dict(await asyncio.gather(*(choose(v) for v in self.villagers if v.alive)))

  async def _choose_punishments(self, round_id: int) -> dict[int, int | None]:
    valid_ids = {villager.agent_id for villager in self.villagers if villager.alive}

    async def choose(villager: Villager):
      prompt = PUNISHMENT_PROMPT.format(
          base_context=self._base_context(villager, punishment=True),
          observation=self._observation(villager, round_id),
          penalty=float(self.cfg.environment.punishment_penalty),
          punish_cost=float(self.cfg.environment.punishment_cost),
      )
      response = await self._ask(
          villager, "punishment", prompt, max_tokens=20, default="N/A"
      )
      return villager.agent_id, parse_punishment(
          response, villager.agent_id, valid_ids
      )

    return dict(await asyncio.gather(*(choose(v) for v in self.villagers if v.alive)))

  async def _propose_norms(
      self, round_id: int
  ) -> tuple[dict[int, str], dict[int, str]]:
    async def propose(villager: Villager):
      prompt = NORM_PROMPT.format(
          base_context=self._base_context(villager),
          observation=self._observation(villager, round_id),
      )
      default = f"Personal: {villager.norm}\nCommunity: {self.group_norm}"
      response = await self._ask(
          villager, "norm_proposal", prompt, max_tokens=250, default=default
      )
      personal, community = parse_norms(response, villager.norm, self.group_norm)
      return villager.agent_id, personal, community

    results = await asyncio.gather(*(propose(v) for v in self.villagers if v.alive))
    personal = {agent_id: norm for agent_id, norm, _ in results}
    proposals = {agent_id: proposal for agent_id, _, proposal in results}
    return personal, proposals

  async def _vote(
      self, round_id: int, proposal_by_agent: dict[int, str]
  ) -> tuple[dict[int, str], str]:
    proposals = list(dict.fromkeys(proposal_by_agent.values()))
    proposal_text = "\n".join(f"- {proposal}" for proposal in proposals)

    async def vote(villager: Villager):
      prompt = VOTE_PROMPT.format(
          base_context=self._base_context(villager),
          observation=self._observation(villager, round_id),
          proposals=proposal_text,
      )
      response = await self._ask(
          villager, "community_vote", prompt, max_tokens=250,
          default=proposals[0],
      )
      selected = next(
          (proposal for proposal in proposals if response.strip(" \"“”") == proposal),
          proposals[0],
      )
      return villager.agent_id, selected

    votes = dict(await asyncio.gather(*(vote(v) for v in self.villagers if v.alive)))
    return votes, choose_winning_policy(votes, proposals, self.rng)

  def _remember(self, villager: Villager, result: RoundResult) -> None:
    punish_text = (
        f"I punished villager {villager.punished_target}."
        if villager.punished_target is not None
        else "I did not punish anyone."
    )
    description = (
        f"Round {result.round}: I used effort {villager.effort:.6g}, harvested "
        f"{villager.harvest:.6g}, and ended with payoff {villager.wealth:.6g}. "
        f"{punish_text} The lake ended at {result.stock_after_regeneration:.6g}. "
        f'The shared policy became “{result.group_norm}”.'
    )
    created = datetime(2000, 1, 1) + timedelta(days=result.round)
    villager.memory.store(MemoryRecord(
        type=MemoryType.EVENT,
        description=description,
        created=created,
        expiration=created + timedelta(days=int(self.cfg.environment.max_rounds) + 1),
        importance_score=10,
    ))
    villager.has_memory = True

  async def step(self, round_id: int) -> RoundResult:
    stock_before = self.stock
    efforts = await self._choose_efforts(round_id)
    productivity = float(self.cfg.environment.productivity)
    harvests = {
        agent_id: productivity * effort * stock_before
        for agent_id, effort in efforts.items()
    }
    stock_after_harvest = max(0.0, stock_before - sum(harvests.values()))
    self.stock = stock_after_harvest

    consumption = float(self.cfg.environment.consumption)
    for villager in self.villagers:
      villager.effort = efforts.get(villager.agent_id, 0.0)
      villager.harvest = harvests.get(villager.agent_id, 0.0)
      villager.wealth += villager.harvest - consumption

    punishments = await self._choose_punishments(round_id)
    penalty = float(self.cfg.environment.punishment_penalty)
    cost = float(self.cfg.environment.punishment_cost)
    punished_ids = {target for target in punishments.values() if target is not None}
    for villager in self.villagers:
      villager.punished_target = punishments.get(villager.agent_id)
      villager.punished = villager.agent_id in punished_ids
      if villager.punished_target is not None:
        villager.wealth -= cost
      if villager.punished:
        villager.wealth -= penalty
      villager.alive = villager.wealth >= 0

    if any(not villager.alive for villager in self.villagers):
      proposals: dict[int, str] = {}
      votes: dict[int, str] = {}
    else:
      personal_norms, proposals = await self._propose_norms(round_id)
      for villager in self.villagers:
        villager.norm = personal_norms[villager.agent_id]
      votes, self.group_norm = await self._vote(round_id, proposals)

    regeneration_mode = str(OmegaConf.select(
        self.cfg, "environment.regeneration.mode", default="doubling"
    ))
    growth_rate = float(OmegaConf.select(
        self.cfg, "environment.regeneration.rate", default=0.2
    ))
    self.stock = regenerate_stock(
        stock_after_harvest,
        float(self.cfg.environment.carrying_capacity),
        mode=regeneration_mode,
        growth_rate=growth_rate,
    )
    collapse_threshold = float(self.cfg.environment.collapse_threshold)
    collapsed = self.stock <= collapse_threshold or any(
        not villager.alive for villager in self.villagers
    )
    optimal_harvest = (
        growth_rate * float(self.cfg.environment.carrying_capacity) / 4
        if regeneration_mode == "logistic"
        else float(self.cfg.environment.carrying_capacity) / 2
    )
    result = RoundResult(
        round=round_id,
        stock_before=stock_before,
        stock_after_harvest=stock_after_harvest,
        stock_after_regeneration=self.stock,
        group_norm=self.group_norm,
        efforts=efforts,
        harvests=harvests,
        wealth={v.agent_id: v.wealth for v in self.villagers},
        punishments=punishments,
        proposals=proposals,
        votes=votes,
        collapsed=collapsed,
        efficiency=sum(harvests.values()) / optimal_harvest,
    )
    for villager in self.villagers:
      self._remember(villager, result)
    return result

  async def run(self) -> dict[str, Any]:
    self.experiment_storage.mkdir(parents=True, exist_ok=True)
    log_path = self.experiment_storage / "consolidated_results.json"
    results: list[RoundResult] = []
    with log_path.open("a") as log_file:
      log_file.write(json.dumps({
          "type": "initialization",
          "data": OmegaConf.to_container(self.cfg, resolve=True),
      }) + "\n")
      for round_id in range(1, int(self.cfg.environment.max_rounds) + 1):
        result = await self.step(round_id)
        results.append(result)
        log_file.write(json.dumps({"type": "round", "data": asdict(result)}) + "\n")
        log_file.flush()
        if result.collapsed:
          break

    summary = {
        "survival_time": len(results),
        "collapsed": results[-1].collapsed if results else False,
        "mean_efficiency": (
            sum(result.efficiency for result in results) / len(results)
            if results else 0.0
        ),
        "final_stock": self.stock,
        "final_group_norm": self.group_norm,
    }
    with (self.experiment_storage / "summary.json").open("w") as summary_file:
      json.dump(summary, summary_file, indent=2)
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
  experiment = PaperCPRExperiment(cfg, wrappers, experiment_storage)
  summary = await experiment.run()
  logger.log_game(summary, last_log=True)
