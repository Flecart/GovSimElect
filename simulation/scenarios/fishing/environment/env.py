import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ...common import ConcurrentEnv, PerturbationEnv
from simulation.persona.common import PersonaEvent
from simulation.scenarios.common.environment.concurrent_env import (
    get_expiration_next_month,
)
from .regen import RegenManager, RegenMode, RegenState

ADVERSE_TRIGGER_ROUND = 6
ADVERSE_FISH_DROP = 50
ADVERSE_MIN_FISH = 5
ADVERSE_EVENT_LABEL = "drought"


def tons_in_lake(num):
  return f"Before everyone fishes, there are {num} tons of fish in the lake."


def tons_caught(agent_name, wanted, caught):
  return (
      f"{agent_name} wanted to catch {wanted} tons of fish, and caught {caught}"
      " tons."
  )


def univ(sustainability_threshold):
  return (
      "Given the current situation, if everyone fishes more than"
      f" {sustainability_threshold} tons, the lake population will shrink next"
      " month."
  )


def tons_caught_home(agent_name, caught):
  return f"This month, {agent_name} caught {caught} tonnes of fish."


class FishingConcurrentEnv(ConcurrentEnv):
  """Fishing concurrent environment."""

  def __init__(
      self,
      cfg: DictConfig,
      experiment_storage: str,
      map_id_to_name: dict[str, str],
      regen_factor_range: tuple[float, float] = (2.0, 2.0),
  ) -> None:
    super().__init__(
        cfg,
        experiment_storage,
        map_id_to_name,
        regen_factor_range=regen_factor_range,
    )
    self.POOL_LOCATION = "lake"

    regen_mode_str = OmegaConf.select(cfg, "regen_mode", default="deterministic") or "deterministic"
    self._regen_mode = RegenMode(regen_mode_str)
    regen_seed = int(OmegaConf.select(cfg, "regen_seed", default=42) or 42)
    rfr = OmegaConf.select(cfg, "regen_factor_range", default=[1.5, 2.5])
    iid_values = (float(rfr[0]), float(rfr[1]))
    self._regen_manager = RegenManager(
        mode=self._regen_mode, seed=regen_seed, iid_values=iid_values
    )
    self._last_regen_state: RegenState | None = None
    self._adverse_mode_enabled = bool(
        OmegaConf.select(cfg, "adverse_scenario", default=False)
    )
    self._adverse_triggered = False

  def _prompt_pool_amount_of_resource(self):
    if self.cfg.harvesting_order == "concurrent":
      num = self.internal_global_state["resource_in_pool"]
    else:
      raise ValueError(f"Unknown fishing order: {self.cfg.harvesting_order}")
    return tons_in_lake(num)

  def _prompt_pool_amount_of_resource_after_harvesting(self, agent):
    wanted = self.internal_global_state["wanted_resource"][agent]
    caught = self.internal_global_state["last_collected_resource"][agent]
    agent_name = self.agent_id_to_name[agent]
    return tons_caught(agent_name, wanted, caught)

  def _prompt_universalization(self, sustainability_threshold):
    return univ(sustainability_threshold)

  def _apply_regeneration(self):
    """Apply regeneration using the configured regen mode."""
    total_extraction = sum(
        self.internal_global_state["last_collected_resource"].values()
    )
    regen_state = self._regen_manager.sample_regen(total_extraction)
    self._last_regen_state = regen_state

    self.internal_global_state["resource_in_pool"] = int(min(
        self.cfg.initial_resource_in_pool,
        self.internal_global_state["resource_in_pool"] * regen_state.regen_factor,
    ))
    self.internal_global_state["resource_before_harvesting"] = (
        self.internal_global_state["resource_in_pool"]
    )
    # Keep regen_factor in state for downstream use (sustainability threshold).
    self.internal_global_state["regen_factor"] = regen_state.regen_factor
    # Store regen state for observation and logging.
    self.internal_global_state["regen_state"] = regen_state
    self.internal_global_state["recent_adverse_event"] = None
    if (
        self._adverse_mode_enabled
        and not self._adverse_triggered
        and self.num_round == ADVERSE_TRIGGER_ROUND
    ):
      before_shock = int(self.internal_global_state["resource_in_pool"])
      after_shock = int(max(before_shock - ADVERSE_FISH_DROP, ADVERSE_MIN_FISH))
      self.internal_global_state["resource_in_pool"] = after_shock
      self.internal_global_state["resource_before_harvesting"] = after_shock
      self._adverse_triggered = True
      self.internal_global_state["recent_adverse_event"] = {
          "type": ADVERSE_EVENT_LABEL,
          "trigger_round": ADVERSE_TRIGGER_ROUND,
          "stock_before": before_shock,
          "stock_after": after_shock,
          "fish_drop": ADVERSE_FISH_DROP,
      }
    self.log_step_regen(regen_state)

  def _observe_pool(self, agent):
    obs = super()._observe_pool(agent)
    adverse_event = self.internal_global_state.get("recent_adverse_event")
    if adverse_event is not None:
      msg = (
          "A drought has hit the lake. Fish stock dropped suddenly since last month."
      )
      obs.events.append(
          PersonaEvent(
              msg,
              created=self.internal_global_state["next_time"][agent],
              expiration=get_expiration_next_month(
                  self.internal_global_state["next_time"][agent]
              ),
              always_include=True,
          )
      )
    # For hysteresis mode, inject regime observation event.
    if self._regen_mode == RegenMode.ENDOGENOUS_HYSTERESIS and self._last_regen_state is not None:
      regime = self._last_regen_state.regime
      if regime == "degraded":
        msg = "The lake appears polluted. Fish are reproducing more slowly than normal."
      else:
        msg = "The lake appears healthy."
      obs.events.append(
          PersonaEvent(
              msg,
              created=self.internal_global_state["next_time"][agent],
              expiration=get_expiration_next_month(
                  self.internal_global_state["next_time"][agent]
              ),
              always_include=True,
          )
      )
    return obs

  def log_step_regen(self, regen_state: RegenState):
    """Log per-round regeneration state."""
    total_extraction = sum(
        self.internal_global_state["last_collected_resource"].values()
    )
    tmp = {
        "agent_id": ["framework"],
        "round": [self.num_round],
        "action": ["regen"],
        "stock_before_extraction": [
            self.internal_global_state.get("resource_before_harvesting", None)
        ],
        "total_extraction": [total_extraction],
        "realized_r_t": [regen_state.regen_factor],
        "regen_mode": [regen_state.mode.value],
    }
    adverse_event = self.internal_global_state.get("recent_adverse_event")
    if adverse_event is not None:
      tmp["adverse_event_type"] = [adverse_event.get("type")]
      tmp["adverse_event_trigger_round"] = [adverse_event.get("trigger_round")]
      tmp["adverse_event_stock_before"] = [adverse_event.get("stock_before")]
      tmp["adverse_event_stock_after"] = [adverse_event.get("stock_after")]
      tmp["adverse_event_fish_drop"] = [adverse_event.get("fish_drop")]
    if regen_state.regime is not None:
      tmp["regime"] = [regen_state.regime]
      tmp["m_t"] = [regen_state.m_t]
      tmp["p_shift"] = [regen_state.p_shift]
      tmp["p_recover"] = [regen_state.p_recover]
      tmp["transitioned"] = [regen_state.transitioned]
    df_log = pd.DataFrame(tmp, index=[len(self.df_acc)])
    self.df_acc.append(df_log)


class FishingPerturbationEnv(PerturbationEnv):

  def __init__(
      self,
      cfg: DictConfig,
      experiment_storage: str,
      map_id_to_name: dict[str, str],
  ) -> None:
    super().__init__(cfg, experiment_storage, map_id_to_name)
    self.POOL_LOCATION = "lake"

  def _prompt_pool_amount_of_resource(self):
    if self.cfg.harvesting_order == "concurrent":
      num = self.internal_global_state["resource_in_pool"]
    else:
      raise ValueError(f"Unknown fishing order: {self.cgf.harvesting_order}")
    return tons_in_lake(num)

  def _prompt_pool_amount_of_resource_after_harvesting(self, agent):
    wanted = self.internal_global_state["wanted_resource"][agent]
    caught = self.internal_global_state["last_collected_resource"][agent]
    agent_name = self.agent_id_to_name[agent]
    return tons_caught(agent_name, wanted, caught)

  def _prompt_universalization(self, sustainability_threshold):
    return univ(sustainability_threshold)

  def _prompt_home_observe_agent_resource(self, agent):
    caught = self.internal_global_state["last_collected_resource"][agent]
    agent_name = self.agent_id_to_name[agent]
    return tons_caught_home(agent_name, caught)

