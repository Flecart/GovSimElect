import asyncio
import json
import random
from datetime import datetime
import pytest
from omegaconf import OmegaConf

from simulation.scenarios.cpr_norms.prompts import (
    EFFORT_PROMPT,
    NORM_PROMPT,
    PUNISHMENT_PROMPT,
    VOTE_PROMPT,
)
from simulation.scenarios.cpr_norms.run import (
    PaperCPRExperiment,
    choose_winning_policy,
    parse_effort,
    parse_norms,
    parse_punishment,
    regenerate_stock,
)


class FakeMemory:
  def __init__(self):
    self.records = []
    self.search_results = []

  def store(self, record):
    self.records.append(record)

  def search(self, query, top_k, current_time):
    del query, current_time
    return self.search_results[:top_k]


class FakeSession:
  def __init__(self, phase):
    self.phase = phase
    self.messages = []

  def add_user(self, content):
    self.messages.append(content)


class FakeWrapper:
  def start_prompt(self, agent_name, phase_name, query_name):
    del agent_name, query_name
    return FakeSession(phase_name)

  async def acomplete_prompt(self, session, **kwargs):
    del kwargs
    return {
        "effort": "0.5",
        "punishment": "N/A",
        "norm_proposal": (
            "Personal: Use effort 0.5\n"
            "Community: Everyone should use effort 0.5"
        ),
        "community_vote": "Everyone should use effort 0.5",
    }[session.phase]

  def end_prompt(self, session):
    del session


def make_cfg():
  return OmegaConf.create({
      "seed": 7,
      "population": "selfish",
      "alignment": "Full",
      "initial_group_norm": "No norm is currently choosen",
      "observation": {"peer_visibility": "all"},
      "group_decision": {"tie_breaking": "random"},
      "personas": {"num": 2},
      "environment": {
          "max_rounds": 50,
          "carrying_capacity": 300,
          "initial_stock": 100,
          "initial_wealth": 0,
          "collapse_threshold": 5,
          "consumption": 2,
          "productivity": 0.1,
          "punishment_penalty": 10,
          "punishment_cost": 0,
          "regeneration": {"mode": "logistic", "rate": 0.2},
      },
      "llm": {"temperature": 0, "top_p": 1},
      "memory": {"top_k": 5},
  })


def test_appendix_prompt_wording_is_preserved():
  assert "Respond with a single number between 0.0 and 1.0. No explanation." in EFFORT_PROMPT
  assert "Return only the ID number of the villager to punish" in PUNISHMENT_PROMPT
  assert "Personal: [Your updated personal belief]" in NORM_PROMPT
  assert "copy it exactly as shown above" in VOTE_PROMPT


def test_appendix_punishment_prompt_preserves_its_word_order(tmp_path):
  experiment = PaperCPRExperiment(
      make_cfg(), [FakeWrapper(), FakeWrapper()], str(tmp_path),
      memory_factory=lambda base_path, cfg: FakeMemory(),
  )
  context = experiment._base_context(experiment.villagers[0], punishment=True)
  assert "and the community also has a shared policy." in context


def test_parsers_constrain_model_outputs():
  assert parse_effort("0.75") == 0.75
  assert parse_effort("effort 4", default=0.25) == 0.25
  assert parse_punishment("2", actor_id=0, valid_ids={0, 1, 2}) == 2
  assert parse_punishment("0", actor_id=0, valid_ids={0, 1}) is None
  assert parse_punishment("N/A", actor_id=0, valid_ids={0, 1}) is None
  assert parse_norms(
      "Personal: Be careful\nCommunity: Limit effort", "old", "shared"
  ) == ("Be careful", "Limit effort")


def test_regeneration_modes_and_seeded_tie_breaking():
  assert regenerate_stock(60, 100) == 100
  assert regenerate_stock(50, 100, mode="logistic", growth_rate=0.2) == 55
  first = choose_winning_policy(
      {0: "A", 1: "B"}, ["A", "B"], random.Random(4)
  )
  second = choose_winning_policy(
      {0: "A", 1: "B"}, ["A", "B"], random.Random(4)
  )
  assert first == second
  assert first in {"A", "B"}


def test_one_round_matches_configured_dynamics_and_keeps_memories_private(tmp_path):
  memories = []

  def memory_factory(base_path, cfg):
    del base_path, cfg
    memory = FakeMemory()
    memories.append(memory)
    return memory

  experiment = PaperCPRExperiment(
      make_cfg(), [FakeWrapper(), FakeWrapper()], str(tmp_path),
      memory_factory=memory_factory,
  )
  memories[0].search_results = [(datetime(2000, 1, 1), "agent zero secret")]
  memories[1].search_results = [(datetime(2000, 1, 1), "agent one secret")]
  experiment.villagers[0].has_memory = True
  experiment.villagers[1].has_memory = True

  history = experiment._private_history(experiment.villagers[0], round_id=1)
  assert "agent zero secret" in history
  assert "agent one secret" not in history

  result = asyncio.run(experiment.step(1))

  assert result.efforts == {0: 0.5, 1: 0.5}
  assert result.harvests == {0: 5.0, 1: 5.0}
  assert result.wealth == {0: 3.0, 1: 3.0}
  assert result.stock_after_harvest == 90.0
  assert result.stock_after_regeneration == pytest.approx(102.6)
  assert result.group_norm == "Everyone should use effort 0.5"
  assert result.collapsed is False
  assert len(memories[0].records) == 1
  assert len(memories[1].records) == 1
  exchanges = [
      json.loads(line)
      for line in (tmp_path / "messages.jsonl").read_text().splitlines()
  ]
  assert len(exchanges) == 8
  assert {exchange["agent_id"] for exchange in exchanges} == {0, 1}
  assert {exchange["phase"] for exchange in exchanges} == {
      "effort", "punishment", "norm_proposal", "community_vote"
  }
  assert all(exchange["request"] and exchange["response"] for exchange in exchanges)


def test_single_experiment_config_selects_requested_condition():
  cfg = OmegaConf.load(
      "simulation/scenarios/fishing/conf/experiment/"
      "paper_cpr_harsh_selfish_full.yaml"
  )
  assert cfg.scenario == "cpr_norms"
  assert cfg.population == "selfish"
  assert cfg.alignment == "Full"
  assert cfg.initial_group_norm == "No norm is currently choosen"
  assert cfg.observation.peer_visibility == "all"
  assert cfg.group_decision.tie_breaking == "random"
  assert cfg.environment.regeneration.mode == "logistic"
  assert cfg.environment.regeneration.rate == 0.2
  assert cfg.environment.initial_stock == 100
  assert cfg.environment.consumption == 2
  assert cfg.environment.productivity == 0.1
  assert cfg.environment.initial_wealth == 0
  assert cfg.environment.collapse_threshold == 5
