"""Helpers for deterministic run and storage naming."""

import os
import re
from pathlib import Path
from typing import Any


def sanitize_run_name(value: str) -> str:
  """Normalize a config-derived run name into a safe path component."""
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
