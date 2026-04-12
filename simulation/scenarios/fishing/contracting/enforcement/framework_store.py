"""Framework-owned persistent state (escrow, insurance, violations) — not `Contract.state`."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FrameworkOwnedStore:
  """Serializes to/from the manager `execution_state` under the key ``framework``."""

  escrow: dict[str, float] = field(default_factory=dict)
  insurance_pools: dict[str, float] = field(default_factory=dict)
  violation_counts: dict[str, int] = field(default_factory=dict)
  participation_costs: dict[str, float] = field(default_factory=dict)
  transfers: list[dict[str, Any]] = field(default_factory=list)
  law_random_seed: int | None = None

  def clone(self) -> FrameworkOwnedStore:
    return copy.deepcopy(self)

  def to_dict(self) -> dict[str, Any]:
    return {
        "escrow": dict(self.escrow),
        "insurance_pools": dict(self.insurance_pools),
        "violation_counts": dict(self.violation_counts),
        "participation_costs": dict(self.participation_costs),
        "transfers": list(self.transfers),
        "law_random_seed": self.law_random_seed,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any] | None) -> FrameworkOwnedStore:
    if not data:
      return cls()
    return cls(
        escrow=dict(data.get("escrow") or {}),
        insurance_pools=dict(data.get("insurance_pools") or {}),
        violation_counts=dict(data.get("violation_counts") or {}),
        participation_costs=dict(data.get("participation_costs") or {}),
        transfers=list(data.get("transfers") or []),
        law_random_seed=data.get("law_random_seed"),
    )
