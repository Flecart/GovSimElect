"""Pure helpers for auditing natural-language fishing-contract compliance.

The audit unit is one agent's harvest decision in one round.  Cases use the
contract attached to that round's ``enforcement`` event, which is the contract
that was active before the harvest decisions were applied.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal, TypedDict


Frame = Literal["I", "We"]
ComplianceStatus = Literal["followed", "violated", "ambiguous"]
GroupStatus = Literal["followed", "violated", "ambiguous", "not_applicable"]

DEFAULT_LAW_MARKER = "default_contract"
RUN_DIRECTORY_RE = re.compile(
    r"^(?P<model>.+)-p(?P<prosocial>[0-5])-(?P<date>\d{4}-\d{2}-\d{2})-"
)
SUMMARY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("overall", ()),
    ("model", ("model",)),
    ("regime", ("regime",)),
    ("frame", ("frame",)),
    ("prosocial_count", ("prosocial_count",)),
    ("model_regime", ("model", "regime")),
    ("model_frame", ("model", "frame")),
    ("regime_frame", ("regime", "frame")),
    ("regime_prosocial", ("regime", "prosocial_count")),
    (
        "regime_prosocial_frame",
        ("regime", "prosocial_count", "frame"),
    ),
    (
        "model_regime_prosocial",
        ("model", "regime", "prosocial_count"),
    ),
    ("model_regime_frame", ("model", "regime", "frame")),
    (
        "full_stratum",
        ("model", "regime", "prosocial_count", "frame"),
    ),
)


class ComplianceCase(TypedDict):
  """Serializable input supplied to the independent compliance judge."""

  case_id: str
  model: str
  regime: str
  prosocial_count: int
  frame: Frame
  run_id: str
  round: int
  agent_id: str
  agent_name: str
  objective: str
  active_contract: str
  contract_round_created: int
  fish_before_harvest: float
  sustainability_threshold: float
  regeneration_context: str
  submitted_catch: float
  realized_catch: float | None
  group_submitted_catches: dict[str, float]
  group_realized_catches: dict[str, float]
  source_contract_log: str
  source_results_log: str


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
  """Read non-empty JSON objects from a JSONL file."""
  return [
      json.loads(line)
      for line in Path(path).read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]


def parse_initialization_config(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
  """Parse the single initialization config stored as a Python literal."""
  configs = [
      record.get("data", {}).get("config")
      for record in records
      if record.get("type") == "initialization"
  ]
  if len(configs) != 1:
    raise ValueError(f"Expected one initialization record, found {len(configs)}")
  raw = configs[0]
  if isinstance(raw, dict):
    return raw
  if not isinstance(raw, str):
    raise ValueError("Initialization config is neither a mapping nor a string")
  parsed = ast.literal_eval(raw)
  if not isinstance(parsed, dict):
    raise ValueError("Initialization config did not parse to a mapping")
  return parsed


def classify_objective_frame(objective: str) -> Frame:
  """Classify an authored fishing objective as an I- or We-frame."""
  normalized = " ".join(objective.lower().split())
  if "group overall catch" in normalized or "overall number of your fishes" in normalized:
    return "We"
  if "own catch" in normalized or "only get" in normalized:
    return "I"
  raise ValueError(f"Unrecognized persona objective: {objective!r}")


def persona_metadata(config: dict[str, Any]) -> dict[str, dict[str, str]]:
  """Return agent name, objective, and frame keyed by persona id."""
  personas = config.get("personas", {})
  metadata: dict[str, dict[str, str]] = {}
  for agent_id, persona in personas.items():
    if not str(agent_id).startswith("persona_") or not isinstance(persona, dict):
      continue
    objective = str(persona.get("goals", ""))
    metadata[str(agent_id)] = {
        "name": str(persona.get("name", agent_id)),
        "objective": objective,
        "frame": classify_objective_frame(objective),
    }
  if not metadata:
    raise ValueError("No persona metadata found in initialization config")
  return metadata


def harvests_by_round(records: Iterable[dict[str, Any]]) -> dict[int, dict[str, float]]:
  """Return realized named catches keyed by zero-based round."""
  harvest_records = [
      record.get("data", {}) for record in records if record.get("type") == "harvest"
  ]
  if len(harvest_records) != 1:
    raise ValueError(f"Expected one harvest record, found {len(harvest_records)}")
  return {
      int(round_id): {str(name): float(value) for name, value in catches.items()}
      for round_id, catches in harvest_records[0].items()
  }


def run_metadata(contract_log: Path, results_roots: Iterable[Path]) -> dict[str, Any]:
  """Infer model, regime, composition, and run id from the result path."""
  resolved = contract_log.resolve()
  matching_root: Path | None = None
  for root in results_roots:
    root_resolved = root.resolve()
    if resolved.is_relative_to(root_resolved):
      matching_root = root_resolved
      break
  if matching_root is None:
    raise ValueError(f"{contract_log} is outside the configured results roots")

  relative = resolved.relative_to(matching_root)
  if len(relative.parts) < 4:
    raise ValueError(f"Unexpected result path layout: {contract_log}")
  experiment_directory = relative.parts[0]
  match = RUN_DIRECTORY_RE.match(experiment_directory)
  if not match:
    raise ValueError(f"Cannot parse experiment directory: {experiment_directory}")
  return {
      "model": match.group("model"),
      "regime": matching_root.name,
      "prosocial_count": int(match.group("prosocial")),
      "run_id": contract_log.parent.name,
  }


def stable_case_id(source: Path, round_id: int, agent_id: str) -> str:
  """Return a deterministic short identifier for an agent-round case."""
  raw = f"{source.resolve()}|{round_id}|{agent_id}"
  return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def is_adopted_contract(contract: Any) -> bool:
  """Return whether a contract is a non-default, active NL agreement."""
  if not isinstance(contract, dict):
    return False
  metadata = contract.get("metadata", {})
  return bool(
      contract.get("contract_type") == "nl"
      and str(contract.get("content", "")).strip()
      and contract.get("enforcement_status") == "active"
      and not (isinstance(metadata, dict) and metadata.get(DEFAULT_LAW_MARKER))
  )


def build_cases_from_run(
    contract_log: str | Path,
    results_roots: Iterable[str | Path],
) -> list[ComplianceCase]:
  """Build all eligible agent-round cases from one NL-contract run."""
  contract_path = Path(contract_log)
  consolidated_path = contract_path.with_name("consolidated_results.json")
  if not consolidated_path.exists():
    raise FileNotFoundError(consolidated_path)

  roots = [Path(root) for root in results_roots]
  metadata = run_metadata(contract_path, roots)
  consolidated = read_jsonl(consolidated_path)
  config = parse_initialization_config(consolidated)
  personas = persona_metadata(config)
  realized = harvests_by_round(consolidated)
  cases: list[ComplianceCase] = []

  for record in read_jsonl(contract_path):
    if record.get("type") != "enforcement":
      continue
    data = record.get("data", {})
    contract = data.get("contract")
    if not is_adopted_contract(contract):
      continue
    round_id = int(data["round"])
    state = data.get("state", {})
    result = data.get("result", {})
    submitted_by_id = {
        str(agent_id): float(value)
        for agent_id, value in result.get("modified_catches", {}).items()
    }
    group_submitted = {
        personas[agent_id]["name"]: value
        for agent_id, value in submitted_by_id.items()
        if agent_id in personas
    }
    group_realized = realized.get(round_id, {})
    if set(submitted_by_id) != set(personas):
      raise ValueError(
          f"Round {round_id} agent ids differ from personas in {contract_path}"
      )

    for agent_id, submitted_catch in submitted_by_id.items():
      persona = personas[agent_id]
      agent_name = persona["name"]
      cases.append({
          "case_id": stable_case_id(contract_path, round_id, agent_id),
          "model": metadata["model"],
          "regime": metadata["regime"],
          "prosocial_count": metadata["prosocial_count"],
          "frame": persona["frame"],  # type: ignore[typeddict-item]
          "run_id": metadata["run_id"],
          "round": round_id,
          "agent_id": agent_id,
          "agent_name": agent_name,
          "objective": persona["objective"],
          "active_contract": str(contract["content"]),
          "contract_round_created": int(contract.get("round_created", -1)),
          "fish_before_harvest": float(state.get("fish_population", 0.0)),
          "sustainability_threshold": float(
              state.get("sustainability_threshold", 0.0)
          ),
          "regeneration_context": str(state.get("setting_context", "")),
          "submitted_catch": submitted_catch,
          "realized_catch": (
              float(group_realized[agent_name])
              if agent_name in group_realized
              else None
          ),
          "group_submitted_catches": group_submitted,
          "group_realized_catches": group_realized,
          "source_contract_log": str(contract_path),
          "source_results_log": str(consolidated_path),
      })
  return cases


def discover_cases(results_roots: Iterable[str | Path]) -> list[ComplianceCase]:
  """Load candidate cases from every retained ``2-nl-p*`` run."""
  roots = [Path(root) for root in results_roots]
  paths: list[Path] = []
  for root in roots:
    paths.extend(sorted(root.glob("**/2-nl-p*/**/contracting_results.jsonl")))
  if not paths:
    raise ValueError(f"No NL contracting logs found under {roots}")
  cases: list[ComplianceCase] = []
  for path in sorted(set(paths)):
    cases.extend(build_cases_from_run(path, roots))
  if not cases:
    raise ValueError("NL logs were found, but none contained an adopted contract")
  return cases


def balanced_sample_per_model(
    candidates: list[ComplianceCase], cases_per_model: int, seed: int
) -> list[ComplianceCase]:
  """Balance models and regimes, then cycle over composition/frame strata."""
  if cases_per_model <= 0:
    raise ValueError("cases_per_model must be positive")
  by_model: dict[str, list[ComplianceCase]] = defaultdict(list)
  for case in candidates:
    by_model[case["model"]].append(case)

  selected: list[ComplianceCase] = []
  for model_index, model in enumerate(sorted(by_model)):
    pool = by_model[model]
    if len(pool) < cases_per_model:
      raise ValueError(
          f"{model} has only {len(pool)} eligible cases; "
          f"cannot sample {cases_per_model}"
      )
    rng = random.Random(seed + model_index)
    by_regime: dict[str, list[ComplianceCase]] = defaultdict(list)
    for case in pool:
      by_regime[case["regime"]].append(case)
    regimes = sorted(by_regime)
    quotas = {
        regime: cases_per_model // len(regimes)
        + (index < cases_per_model % len(regimes))
        for index, regime in enumerate(regimes)
    }
    deficit = 0
    for regime in regimes:
      available = len(by_regime[regime])
      if quotas[regime] > available:
        deficit += quotas[regime] - available
        quotas[regime] = available
    while deficit:
      recipients = [
          regime
          for regime in regimes
          if quotas[regime] < len(by_regime[regime])
      ]
      if not recipients:
        raise ValueError(f"Cannot allocate {cases_per_model} cases for {model}")
      for regime in recipients:
        if not deficit:
          break
        quotas[regime] += 1
        deficit -= 1

    model_sample: list[ComplianceCase] = []
    for regime in regimes:
      strata: dict[tuple[int, str], list[ComplianceCase]] = defaultdict(list)
      for case in by_regime[regime]:
        strata[(case["prosocial_count"], case["frame"])].append(case)
      for cases in strata.values():
        rng.shuffle(cases)
      keys = sorted(strata)
      rng.shuffle(keys)
      cursor = 0
      regime_sample: list[ComplianceCase] = []
      while len(regime_sample) < quotas[regime]:
        available = [key for key in keys if strata[key]]
        if not available:
          raise ValueError(
              f"Candidate pool unexpectedly exhausted for {model}/{regime}"
          )
        key = available[cursor % len(available)]
        regime_sample.append(strata[key].pop())
        cursor += 1
      model_sample.extend(regime_sample)
    selected.extend(model_sample)
  return sorted(
      selected,
      key=lambda case: (
          case["model"],
          case["regime"],
          case["prosocial_count"],
          case["frame"],
          case["case_id"],
      ),
  )


def wilson_interval(
    successes: int, observations: int, z: float = 1.959963984540054
) -> tuple[float, float]:
  """Return a two-sided Wilson interval for a binomial proportion."""
  if observations <= 0:
    return math.nan, math.nan
  proportion = successes / observations
  denominator = 1 + z**2 / observations
  center = (proportion + z**2 / (2 * observations)) / denominator
  radius = (
      z
      * math.sqrt(
          proportion * (1 - proportion) / observations
          + z**2 / (4 * observations**2)
      )
      / denominator
  )
  return center - radius, center + radius


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Summarize followed/violated rates with Wilson 95% confidence intervals."""
  rows: list[dict[str, Any]] = []

  def add_groups(
      group_type: str,
      fields: tuple[str, ...],
      subset: list[dict[str, Any]],
  ) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for result in subset:
      grouped[tuple(result[field] for field in fields)].append(result)
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
      identifiers = dict(zip(fields, key))
      assessable = [row for row in group if row["agent_status"] != "ambiguous"]
      followed = sum(row["agent_status"] == "followed" for row in assessable)
      violated = sum(row["agent_status"] == "violated" for row in assessable)
      ambiguous = len(group) - len(assessable)
      lower, upper = wilson_interval(followed, len(assessable))
      rows.append({
          "group_type": group_type,
          "model": identifiers.get("model", "ALL"),
          "regime": identifiers.get("regime", "ALL"),
          "prosocial_count": identifiers.get("prosocial_count", "ALL"),
          "frame": identifiers.get("frame", "ALL"),
          "n": len(group),
          "assessable_n": len(assessable),
          "followed_n": followed,
          "violated_n": violated,
          "ambiguous_n": ambiguous,
          "followed_rate": followed / len(assessable) if assessable else math.nan,
          "followed_ci_lower": lower,
          "followed_ci_upper": upper,
          "violation_rate": violated / len(assessable) if assessable else math.nan,
          "mean_confidence": (
              sum(float(row["confidence"]) for row in group) / len(group)
          ),
      })

  for group_type, fields in SUMMARY_GROUPS:
    add_groups(group_type, fields, results)
  mixed = [
      result for result in results if 1 <= int(result["prosocial_count"]) <= 4
  ]
  add_groups("mixed_frame", ("frame",), mixed)
  add_groups("mixed_model_frame", ("model", "frame"), mixed)
  add_groups("mixed_regime_frame", ("regime", "frame"), mixed)
  add_groups(
      "mixed_model_regime_frame",
      ("model", "regime", "frame"),
      mixed,
  )
  return rows


def sampling_summary(
    candidates: list[ComplianceCase], sampled: list[ComplianceCase]
) -> list[dict[str, Any]]:
  """Return population and sample counts for every balancing stratum."""
  fields = ("model", "regime", "prosocial_count", "frame")
  population: dict[tuple[Any, ...], int] = defaultdict(int)
  sample_counts: dict[tuple[Any, ...], int] = defaultdict(int)
  for case in candidates:
    population[tuple(case[field] for field in fields)] += 1
  for case in sampled:
    sample_counts[tuple(case[field] for field in fields)] += 1
  return [
      {
          **dict(zip(fields, key)),
          "eligible_n": population[key],
          "sampled_n": sample_counts[key],
      }
      for key in sorted(population)
  ]


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
  """Write dictionaries to CSV, serializing nested values as JSON."""
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  if not rows:
    raise ValueError(f"Cannot write an empty CSV: {destination}")
  fieldnames = list(rows[0])
  with destination.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({
          key: (
              json.dumps(value, ensure_ascii=False, sort_keys=True)
              if isinstance(value, (dict, list))
              else value
          )
          for key, value in row.items()
      })


def read_csv(path: str | Path) -> list[dict[str, str]]:
  """Read a CSV checkpoint as dictionaries."""
  with Path(path).open(encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))
