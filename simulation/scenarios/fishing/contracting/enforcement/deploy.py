"""Load a `Contract` subclass from exec'd law source and instantiate."""

from __future__ import annotations

import builtins
import inspect
import textwrap
from typing import Any

from .contract import Contract


def discover_contract_subclass(namespace: dict[str, Any]) -> type[Contract]:
  """Return the unique leaf subclass of ``Contract`` in the namespace."""
  candidates: list[type[Contract]] = []
  for obj in namespace.values():
    if isinstance(obj, type) and issubclass(obj, Contract) and obj is not Contract:
      candidates.append(obj)  # type: ignore[arg-type]
  if not candidates:
    raise ValueError(
        "No subclass of Contract found. Define exactly one class that inherits from Contract.",
    )
  leaves: list[type[Contract]] = []
  for c in candidates:
    if any(issubclass(other, c) and other is not c for other in candidates):
      continue
    leaves.append(c)
  if len(leaves) != 1:
    raise ValueError(
        f"Expected exactly one concrete Contract subclass; found {len(leaves)}: "
        f"{[x.__name__ for x in leaves]}",
    )
  return leaves[0]


def instantiate_contract(
    cls: type[Contract],
    *,
    num_agents: int,
    agent_names: list[str],
    prior_state: dict[str, Any] | None,
    old_instance: Contract | None,
    discontinuity_log: list[str],
) -> Contract:
  """Opt-in ``prior_state`` only if ``__init__`` accepts it; log version skips."""
  sig = inspect.signature(cls.__init__)
  params = sig.parameters
  kwargs: dict[str, Any] = {
      "num_agents": num_agents,
      "agent_names": list(agent_names),
  }
  if "prior_state" in params:
    migrated = prior_state
    if (
        old_instance is not None
        and cls.VERSION != old_instance.__class__.VERSION
        and migrated
    ):
      discontinuity_log.append(
          f"Contract VERSION {cls.VERSION} vs prior {old_instance.__class__.VERSION}; "
          "prior_state passed because __init__ accepts it.",
      )
    kwargs["prior_state"] = migrated
  else:
    if old_instance is not None and prior_state:
      discontinuity_log.append(
          f"State reset: new law class {cls.__name__} VERSION={cls.VERSION} "
          f"does not accept prior_state; prior state keys {list(prior_state.keys())} dropped "
          f"(prior VERSION={old_instance.__class__.VERSION}).",
      )
  return cls(**kwargs)  # type: ignore[misc]


def build_safe_exec_globals(base_contract: type[Contract]) -> dict[str, Any]:
  """Minimal builtins + base ``Contract`` for law module execution."""
  return {
      # Module metadata expected during class creation / some std patterns.
      "__name__": "<law>",
      "__doc__": None,
      "__package__": None,
      "__builtins__": {
          # Required for `class` statements when __builtins__ is a restricted dict.
          "__build_class__": builtins.__build_class__,
          "super": super,
          "abs": abs,
          "min": min,
          "max": max,
          "sum": sum,
          "len": len,
          "int": int,
          "float": float,
          "bool": bool,
          "str": str,
          "dict": dict,
          "list": list,
          "tuple": tuple,
          "set": set,
          "range": range,
          "enumerate": enumerate,
          "zip": zip,
          "round": round,
          "sorted": sorted,
          "isinstance": isinstance,
          "issubclass": issubclass,
          "Exception": Exception,
          "ValueError": ValueError,
          "TypeError": TypeError,
          "KeyError": KeyError,
      },
      "Contract": base_contract,
  }


def exec_law_source(source: str, base_contract: type[Contract]) -> dict[str, Any]:
  """Execute law source and return the merged globals/locals namespace."""
  cleaned = textwrap.dedent(source).strip()
  g = build_safe_exec_globals(base_contract)
  loc: dict[str, Any] = {}
  code = compile(cleaned, "<law>", "exec")
  exec(code, g, loc)  # noqa: S102 — validated static law only
  merged = {**g, **loc}
  return merged
