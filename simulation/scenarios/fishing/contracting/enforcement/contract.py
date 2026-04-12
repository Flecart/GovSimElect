"""Base class for coder-generated deployed laws (subclass of `Contract`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from .context import EnforcementContext


class Contract:
  """Deployed once. Reused every round until replaced."""

  VERSION = 1

  def __init__(
      self,
      num_agents: int,
      agent_names: list[str],
      *,
      prior_state: dict[str, Any] | None = None,
  ) -> None:
    self.num_agents = int(num_agents)
    self.agent_names = list(agent_names)
    self.state: dict[str, Any] = dict(prior_state) if prior_state else {}

  def on_round_start(
      self,
      month: int,
      fish_population: float,
      ctx: EnforcementContext,
  ) -> None:
    del month, fish_population, ctx

  def resolve(
      self,
      month: int,
      fish_population: float,
      submissions: dict[str, float],
      ctx: EnforcementContext,
  ) -> dict[str, float]:
    del month, fish_population, ctx
    return dict(submissions)

  def on_round_end(
      self,
      month: int,
      fish_population_after: float,
      final_catches: dict[str, float],
      ctx: EnforcementContext,
  ) -> None:
    del month, fish_population_after, final_catches, ctx
