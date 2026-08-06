import random
import asyncio

import pytest
from omegaconf import OmegaConf

from simulation.scenarios.govsim_monitoring.run import (
    GovSimMonitoringExperiment,
    choose_winner,
    parse_harvest_choice,
    parse_unit,
    parse_yes_no,
    update_efficiency,
)
from simulation.utils.models import PromptSession


class FakeMemory:
  def __init__(self):
    self.records = []

  def store(self, record):
    self.records.append(record)

  def search(self, query, top_k, current_time):
    del query
    return [
        (record.created, record.description)
        for record in self.records
        if record.expiration > current_time
    ][-top_k:]

  def save(self):
    pass


class FakeWrapper:
  def start_prompt(self, agent_name, phase_name, query_name):
    return PromptSession(agent_name, phase_name, query_name)

  async def acomplete_prompt(self, session, **kwargs):
    del kwargs
    responses = {
        "harvest_contribution": "Effort: 0.2\nContribution: 0.2",
        "monitoring": "1.0",
        "punishment": "no",
        "discussion": (
            "Response: Let us maintain the lake together.\n"
            "Conversation conclusion by me: yes\nNext speaker: N/A"
        ),
        "norm_proposal": (
            "Personal: Contribute fairly and harvest modestly.\n"
            "Community: Everyone contributes 0.2 and uses effort 0.2."
        ),
        "community_vote": "Everyone contributes 0.2 and uses effort 0.2.",
    }
    response = responses[session.phase_name]
    session.add_assistant(response)
    return response

  def end_prompt(self, session):
    del session


def test_centered_sigmoid_lowers_maintains_and_raises_efficiency():
  alpha = 0.1
  assert update_efficiency(alpha, 0, 1) < alpha
  assert update_efficiency(alpha, 1, 1) == pytest.approx(alpha)
  assert update_efficiency(alpha, 2, 1) > alpha
  assert update_efficiency(0.9, 100, 1) == 1


def test_centered_sigmoid_rejects_non_positive_requirement():
  with pytest.raises(ValueError, match="maintenance_requirement"):
    update_efficiency(0.1, 1, 0)


def test_action_parsers_are_bounded_and_have_safe_defaults():
  assert parse_harvest_choice("Effort: 0.75\nContribution: 0.2") == (0.75, 0.2)
  assert parse_harvest_choice("unstructured") == (0.5, 0.0)
  assert parse_unit("1.0") == 1
  assert parse_yes_no("yes") is True
  assert parse_yes_no("no") is False


def test_seeded_tie_breaking_is_reproducible():
  votes = {"John": "A", "Kate": "B"}
  assert choose_winner(votes, ["A", "B"], random.Random(4)) == choose_winner(
      votes, ["A", "B"], random.Random(4)
  )


def test_one_round_applies_payoff_monitoring_governance_and_regeneration(tmp_path):
  personas = {
      "num": 5,
      **{
          f"persona_{index}": {"name": name, "goals": "Optimize long-run income."}
          for index, name in enumerate(["John", "Kate", "Jack", "Emma", "Luke"])
      },
  }
  cfg = OmegaConf.create({
      "seed": 0,
      "personas": personas,
      "initial_group_norm": "No norm is currently choosen",
      "environment": {
          "max_rounds": 12,
          "carrying_capacity": 100,
          "initial_stock": 100,
          "initial_wealth": 1,
          "collapse_threshold": 5,
          "consumption": 1,
          "initial_alpha": 0.1,
          "maintenance_requirement": 1,
          "punishment_penalty": 1,
      },
      "discussion": {"max_steps": 10},
      "llm": {"temperature": 0, "top_p": 1},
      "memory": {"top_k": 8},
  })
  memories = []

  def memory_factory(base_path, memory_cfg):
    del base_path, memory_cfg
    memory = FakeMemory()
    memories.append(memory)
    return memory

  experiment = GovSimMonitoringExperiment(
      cfg,
      [FakeWrapper() for _ in range(5)],
      str(tmp_path),
      memory_factory=memory_factory,
  )
  result = asyncio.run(experiment.step(1))

  assert result.stock_after_harvest == pytest.approx(90)
  assert result.stock_after_regeneration == pytest.approx(100)
  assert result.alpha_before == pytest.approx(0.1)
  assert result.alpha_after == pytest.approx(0.1)
  assert all(value == pytest.approx(1.8) for value in result.wealth.values())
  assert all(record["success"] for record in result.monitoring.values())
  assert result.punished == []
  assert result.group_norm == "Everyone contributes 0.2 and uses effort 0.2."
  assert all(memory.records for memory in memories)
