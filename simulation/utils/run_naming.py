"""Helpers for deterministic run and storage naming."""

import re
from pathlib import Path
from typing import Any


def sanitize_run_name(value: str) -> str:
  """Normalize a config-derived name into a safe path component."""
  value = value.strip()
  value = value.replace("\\", "/")
  value = value.split("/")[-1]
  value = re.sub(r"\s+", "-", value)
  value = re.sub(r"[^A-Za-z0-9._-]", "-", value)
  value = re.sub(r"-{2,}", "-", value).strip("-")
  return value or "run"


def resolve_config_run_name(cfg: Any) -> str:
  """Pick a base run name from config, preferring explicit fields."""
  experiment = getattr(cfg, "experiment", None)
  run_name = getattr(experiment, "run_name", "") if experiment else ""
  if run_name:
    return sanitize_run_name(str(run_name))

  explicit = getattr(experiment, "storage_name", "") if experiment else ""
  if explicit:
    return sanitize_run_name(str(explicit))

  group_name = getattr(cfg, "group_name", "")
  if group_name:
    return sanitize_run_name(str(group_name))

  env = getattr(experiment, "env", None) if experiment else None
  env_name = getattr(env, "name", "") if env else ""
  if env_name:
    return sanitize_run_name(str(env_name))

  experiment_name = getattr(experiment, "name", "run") if experiment else "run"
  return sanitize_run_name(str(experiment_name))


def _resolve_explicit_group_name(cfg: Any) -> str | None:
  experiment = getattr(cfg, "experiment", None)
  run_name = getattr(experiment, "run_name", "") if experiment else ""
  if run_name:
    return sanitize_run_name(str(run_name))

  explicit = getattr(experiment, "storage_name", "") if experiment else ""
  if explicit:
    return sanitize_run_name(str(explicit))
  return None


def _resolve_child_run_base_name(cfg: Any) -> str:
  experiment = getattr(cfg, "experiment", None)
  env = getattr(experiment, "env", None) if experiment else None
  env_name = getattr(env, "name", "") if env else ""
  if env_name:
    return sanitize_run_name(str(env_name))
  return "run"


def make_unique_run_name(parent_dir: str | Path, base_name: str) -> str:
  """Return `base_name` or a suffixed `base_name-N` that does not exist yet."""
  parent = Path(parent_dir)
  candidate = sanitize_run_name(base_name)
  if not (parent / candidate).exists():
    return candidate

  suffix = 0
  while (parent / f"{candidate}-{suffix}").exists():
    suffix += 1
  return f"{candidate}-{suffix}"


def reserve_run_storage_path(parent_dir: str | Path, cfg: Any) -> str:
  """Create and reserve a unique run directory, returning its relative path."""
  parent = Path(parent_dir)
  explicit_group = _resolve_explicit_group_name(cfg)

  if explicit_group:
    group_dir = parent / explicit_group
    group_dir.mkdir(parents=True, exist_ok=True)
    child_base = _resolve_child_run_base_name(cfg)
    suffix = 0
    while True:
      relative = Path(explicit_group) / f"{child_base}-{suffix}"
      try:
        (parent / relative).mkdir(parents=False, exist_ok=False)
        return relative.as_posix()
      except FileExistsError:
        suffix += 1

  base_name = resolve_config_run_name(cfg)
  candidate = sanitize_run_name(base_name)
  try:
    (parent / candidate).mkdir(parents=False, exist_ok=False)
    return candidate
  except FileExistsError:
    suffix = 0
    while True:
      relative = f"{candidate}-{suffix}"
      try:
        (parent / relative).mkdir(parents=False, exist_ok=False)
        return relative
      except FileExistsError:
        suffix += 1
