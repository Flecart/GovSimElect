import pytest

from simulation.scenarios.govsim_monitoring.audit import (
    build_compliance_cases,
    cross_tabulate,
    validate_verdicts,
)


def test_cases_use_norm_active_before_round_choices():
  initialization = {"initial_group_norm": "No norm is currently choosen"}
  rounds = [
      {
          "round": 1,
          "efforts": {"John": 0.5},
          "contributions": {"John": 0},
          "suspended": [],
          "group_norm": "Effort must not exceed 0.3.",
      },
      {
          "round": 2,
          "efforts": {"John": 0.5},
          "contributions": {"John": 0},
          "suspended": [],
          "group_norm": "Effort must be zero.",
      },
  ]
  cases = build_compliance_cases(initialization, rounds)
  assert cases[0]["active_norm"] == "No norm is currently choosen"
  assert cases[1]["active_norm"] == "Effort must not exceed 0.3."


def test_cross_tabulation_distinguishes_detection_and_punishment():
  rounds = [{
      "round": 1,
      "efforts": {"John": 0.5, "Kate": 0.5, "Jack": 0.5, "Emma": 0.5},
      "contributions": {"John": 0, "Kate": 0, "Jack": 0, "Emma": 0},
      "monitoring": {
          "Luke": {"success": True, "target": "John"},
          "Emma": {"success": True, "target": "Kate"},
      },
      "punished": ["John", "Jack", "Emma"],
  }]
  verdicts = {
      f"r1-{agent}": {"id": f"r1-{agent}", "violation": agent != "Emma"}
      for agent in ["John", "Kate", "Jack", "Emma"]
  }
  result = cross_tabulate(rounds, verdicts)
  assert result["counts"] == {
      "undetected_unpunished": 0,
      "detected_unpunished": 1,
      "detected_punished": 1,
      "undetected_punished": 1,
  }
  assert [item["id"] for item in result["punished_nonviolations"]] == ["r1-Emma"]


def test_verdict_validation_requires_exact_boolean_results():
  cases = [{"id": "r1-John"}]
  assert validate_verdicts(
      cases, [{"id": "r1-John", "violation": True}]
  )["r1-John"]["violation"] is True
  with pytest.raises(ValueError, match="non-boolean"):
    validate_verdicts(cases, [{"id": "r1-John", "violation": "yes"}])
  with pytest.raises(ValueError, match="differ"):
    validate_verdicts(cases, [])
