"""Per-round enforcement context: payoff primitives + read-only views into framework store."""

from __future__ import annotations

import random
from typing import Any

from .framework_store import FrameworkOwnedStore


class EnforcementContext:
  """Bound to one enforcement phase; mutates shared reward adjustments and framework store."""

  def __init__(
      self,
      *,
      month: int,
      fish_population: float,
      num_agents: int,
      agent_names: list[str],
      store: FrameworkOwnedStore,
      reward_adjustments: dict[str, float],
      execution_log: list[str],
      rng: random.Random,
  ) -> None:
    self._month = month
    self._fish_population = fish_population
    self._num_agents = num_agents
    self._agent_names = list(agent_names)
    self._store = store
    self._reward_adjustments = reward_adjustments
    self._execution_log = execution_log
    self._rng = rng

  @property
  def month(self) -> int:
    return self._month

  @property
  def fish_population(self) -> float:
    return self._fish_population

  @property
  def num_agents(self) -> int:
    return self._num_agents

  @property
  def agent_names(self) -> list[str]:
    return list(self._agent_names)

  def random(self) -> float:
    return self._rng.random()

  def _normalize_name(self, name: str) -> str:
    return str(name)

  def _ensure_agent(self, name: str) -> str:
    agent = self._normalize_name(name)
    self._reward_adjustments.setdefault(agent, 0.0)
    return agent

  def _adjust_reward(self, name: str, delta: float, label: str) -> None:
    agent = self._ensure_agent(name)
    self._reward_adjustments[agent] = self._reward_adjustments.get(agent, 0.0) + float(delta)
    self._execution_log.append(f"{label}: {agent} {delta:+.2f}")

  def transfer(self, src: str, dst: str, amount: float, reason: str = "") -> None:
    value = max(0.0, float(amount))
    source = self._ensure_agent(src)
    target = self._ensure_agent(dst)
    if value <= 0:
      return
    self._adjust_reward(source, -value, f"transfer_out[{reason or 'unspecified'}]")
    self._adjust_reward(target, value, f"transfer_in[{reason or 'unspecified'}]")
    self._store.transfers.append({
        "src": source,
        "dst": target,
        "amount": value,
        "reason": reason,
    })

  def escrow(
      self,
      name: str,
      amount: float,
      bucket: str = "default",
      reason: str = "",
  ) -> None:
    value = max(0.0, float(amount))
    agent = self._ensure_agent(name)
    if value <= 0:
      return
    bucket_key = f"{agent}:{bucket}"
    self._store.escrow[bucket_key] = self._store.escrow.get(bucket_key, 0.0) + value
    self._adjust_reward(agent, -value, f"escrow[{bucket}]")
    self._execution_log.append(f"escrow reason={reason or 'none'}")

  def release_escrow(
      self,
      name: str,
      amount: float | None = None,
      bucket: str = "default",
      recipient: str | None = None,
      reason: str = "",
  ) -> None:
    agent = self._ensure_agent(name)
    bucket_key = f"{agent}:{bucket}"
    available = float(self._store.escrow.get(bucket_key, 0.0))
    if available <= 0:
      return
    value = available if amount is None else min(available, max(0.0, float(amount)))
    if value <= 0:
      return
    self._store.escrow[bucket_key] = available - value
    payee = self._ensure_agent(recipient or agent)
    self._adjust_reward(payee, value, f"release_escrow[{bucket}]")
    self._execution_log.append(f"release_escrow reason={reason or 'none'}")

  def sanction(self, name: str, amount: float, reason: str = "") -> None:
    value = max(0.0, float(amount))
    if value <= 0:
      return
    self._adjust_reward(name, -value, f"sanction[{reason or 'unspecified'}]")

  def graduated_sanction(
      self,
      name: str,
      base_amount: float,
      key: str = "default",
      multiplier: float = 1.0,
      reason: str = "",
  ) -> float:
    agent = self._ensure_agent(name)
    counter_key = f"{agent}:{key}"
    count = int(self._store.violation_counts.get(counter_key, 0)) + 1
    self._store.violation_counts[counter_key] = count
    g_amount = max(0.0, float(base_amount)) * max(0.0, float(multiplier)) * count
    self.sanction(agent, g_amount, reason or key)
    self._execution_log.append(f"graduated_sanction count={count}")
    return g_amount

  def insurance(
      self,
      name: str,
      premium: float = 0.0,
      payout: float = 0.0,
      pool: str = "default",
      recipient: str | None = None,
      reason: str = "",
  ) -> None:
    member = self._ensure_agent(name)
    pool_key = str(pool)
    self._store.insurance_pools.setdefault(pool_key, 0.0)
    premium_value = max(0.0, float(premium))
    payout_value = max(0.0, float(payout))
    if premium_value > 0:
      self._store.insurance_pools[pool_key] += premium_value
      self._adjust_reward(member, -premium_value, f"insurance_premium[{pool_key}]")
    if payout_value > 0:
      available = float(self._store.insurance_pools.get(pool_key, 0.0))
      paid = min(available, payout_value)
      if paid > 0:
        self._store.insurance_pools[pool_key] = available - paid
        self._adjust_reward(recipient or member, paid, f"insurance_payout[{pool_key}]")
    self._execution_log.append(f"insurance reason={reason or 'none'}")

  def participation_cost(self, name: str, amount: float, reason: str = "") -> None:
    value = max(0.0, float(amount))
    if value <= 0:
      return
    agent = self._ensure_agent(name)
    self._store.participation_costs[agent] = (
        self._store.participation_costs.get(agent, 0.0) + value
    )
    self._adjust_reward(agent, -value, f"participation_cost[{reason or 'unspecified'}]")

  def escrow_balance(self, name: str, bucket: str = "default") -> float:
    bucket_key = f"{self._normalize_name(name)}:{bucket}"
    return float(self._store.escrow.get(bucket_key, 0.0))

  def insurance_pool(self, pool: str = "default") -> float:
    return float(self._store.insurance_pools.get(str(pool), 0.0))

  def violation_count(self, name: str, key: str = "default") -> int:
    counter_key = f"{self._normalize_name(name)}:{key}"
    return int(self._store.violation_counts.get(counter_key, 0))
