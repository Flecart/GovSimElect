import json

from simulation.scenarios.fishing.compliance_audit import (
    balanced_sample_per_model,
    build_cases_from_run,
    classify_objective_frame,
    summarize_results,
    wilson_interval,
)


def _write_jsonl(path, records):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      "\n".join(json.dumps(record) for record in records) + "\n",
      encoding="utf-8",
  )


def test_objective_frame_uses_authored_goal_text():
  assert classify_objective_frame("Maximize your own catch of fish") == "I"
  assert classify_objective_frame("Maximize your group overall catch of fish") == "We"


def test_build_cases_uses_contract_active_for_same_harvest_round(tmp_path):
  root = tmp_path / "std"
  run = (
      root
      / "gpt-4o-p1-2026-04-29-17:03"
      / "2-nl-p1"
      / "2-nl-p1-0"
  )
  consolidated = [
      {
          "type": "initialization",
          "data": {
              "config": repr({
                  "personas": {
                      "persona_0": {
                          "name": "John",
                          "goals": "Maximize your group overall catch of fish",
                      },
                      "persona_1": {
                          "name": "Kate",
                          "goals": "Maximize your own catch of fish",
                      },
                  }
              })
          },
      },
      {
          "type": "harvest",
          "data": {
              "0": {"John": 10, "Kate": 10},
              "1": {"John": 8, "Kate": 12},
          },
      },
  ]
  contract_records = [
      {
          "type": "enforcement",
          "data": {
              "round": 0,
              "state": {"fish_population": 100, "sustainability_threshold": 10},
              "contract": {
                  "contract_type": "nl",
                  "content": "Everyone is free to choose.",
                  "round_created": -1,
                  "enforcement_status": "active",
                  "metadata": {"default_contract": True},
              },
              "result": {"modified_catches": {"persona_0": 10, "persona_1": 10}},
          },
      },
      {
          "type": "enforcement",
          "data": {
              "round": 1,
              "state": {
                  "fish_population": 100,
                  "sustainability_threshold": 10,
                  "setting_context": "Fixed regeneration 2.0",
              },
              "contract": {
                  "contract_type": "nl",
                  "content": "Each fisher shall catch at most 10 tons.",
                  "round_created": 0,
                  "enforcement_status": "active",
                  "metadata": {},
              },
              "result": {"modified_catches": {"persona_0": 8, "persona_1": 12}},
          },
      },
  ]
  _write_jsonl(run / "consolidated_results.json", consolidated)
  _write_jsonl(run / "contracting_results.jsonl", contract_records)

  cases = build_cases_from_run(run / "contracting_results.jsonl", [root])

  assert len(cases) == 2
  assert {case["round"] for case in cases} == {1}
  assert {case["agent_name"]: case["frame"] for case in cases} == {
      "John": "We",
      "Kate": "I",
  }
  assert {case["agent_name"]: case["submitted_catch"] for case in cases} == {
      "John": 8,
      "Kate": 12,
  }
  assert cases[0]["contract_round_created"] == 0


def test_balanced_sample_is_deterministic_and_balances_strata():
  candidates = []
  for model in ("A", "B"):
    for regime in ("std", "sto"):
      for prosocial, frames in ((0, ("I",)), (1, ("I", "We")), (5, ("We",))):
        for frame in frames:
          for repetition in range(5):
            candidates.append({
                "case_id": f"{model}-{regime}-{prosocial}-{frame}-{repetition}",
                "model": model,
                "regime": regime,
                "prosocial_count": prosocial,
                "frame": frame,
            })

  first = balanced_sample_per_model(candidates, cases_per_model=16, seed=7)
  second = balanced_sample_per_model(candidates, cases_per_model=16, seed=7)

  assert [case["case_id"] for case in first] == [case["case_id"] for case in second]
  assert sum(case["model"] == "A" for case in first) == 16
  assert sum(case["model"] == "B" for case in first) == 16
  counts = {}
  for case in first:
    key = (case["model"], case["regime"], case["prosocial_count"], case["frame"])
    counts[key] = counts.get(key, 0) + 1
  assert set(counts.values()) == {2}


def test_summary_excludes_ambiguous_cases_from_rate():
  results = [
      {
          "model": "A",
          "regime": "std",
          "prosocial_count": 1,
          "frame": "I",
          "agent_status": "followed",
          "confidence": 0.9,
      },
      {
          "model": "A",
          "regime": "std",
          "prosocial_count": 1,
          "frame": "I",
          "agent_status": "violated",
          "confidence": 0.8,
      },
      {
          "model": "A",
          "regime": "std",
          "prosocial_count": 1,
          "frame": "I",
          "agent_status": "ambiguous",
          "confidence": 0.5,
      },
  ]

  overall = next(
      row for row in summarize_results(results) if row["group_type"] == "overall"
  )

  assert overall["n"] == 3
  assert overall["assessable_n"] == 2
  assert overall["followed_rate"] == 0.5
  assert overall["ambiguous_n"] == 1


def test_wilson_interval_is_bounded_for_perfect_rate():
  lower, upper = wilson_interval(50, 50)
  assert 0.92 < lower < 1.0
  assert upper == 1.0
