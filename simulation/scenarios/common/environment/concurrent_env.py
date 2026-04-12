"""Concurrent environment."""

import os
from datetime import datetime, timedelta
from enum import StrEnum, auto

import numpy as np
import omegaconf
import pandas as pd
from pettingzoo.utils import agent_selector
from simulation.persona.common import (
    PersonaAction,
    PersonaActionChat,
    PersonaActionHarvesting,
    PersonaEnvironment,
    PersonaEvent,
    PersonaIdentity,
)

from .common import HarvestingObs


def get_reflection_day(current_date):
  next_month = current_date.replace(day=28) + timedelta(days=4)
  last_day_of_current_month = next_month - timedelta(days=next_month.day)
  return last_day_of_current_month


def get_discussion_day(current_date):
  reflection = get_reflection_day(current_date)
  return reflection - timedelta(days=1)


def get_expiration_next_month(current_date):
  return get_reflection_day(current_date)


class Phase(StrEnum):
  HARVEST = "lake"
  POOL_AFTER_HARVESTING = "pool_after_harvesting"
  RESTAURANT = "restaurant"
  HOME = "home"


class ConcurrentEnv:
  """Base class for concurrent environments."""

  def __init__(
      self,
      cfg: omegaconf.DictConfig,
      experiment_storage: str,
      map_id_to_name: dict[str, str],
      # Default is to double.
      regen_factor_range: tuple[float, float] = (2.0, 2.0),
  ) -> None:
    self.cfg = cfg
    self.experiment_storage = experiment_storage

    self.possible_agents = [f"persona_{i}" for i in range(cfg.num_agents)]
    self.agent_name_mapping = dict(
        zip(self.possible_agents, list(range(len(self.possible_agents))))
    )
    self.agent_id_to_name = map_id_to_name

    self.POOL_LOCATION = "pool"
    self._regen_factor_range = regen_factor_range
    self._harvest_enforcer = None
    self._post_harvest_observer = None
    self._post_regen_hook = None
    self._contracting_enabled = False

  ### Prompt text

  def _prompt_pool_amount_of_resource(self):
    raise NotImplementedError

  def _prompt_pool_amount_of_resource_after_harvesting(self, agent):
    raise NotImplementedError

  def _prompt_universalization(self, sustainability_threshold):
    raise NotImplementedError

  def _observe_pool(self, agent) -> HarvestingObs:
    sustainability_threshold = self.internal_global_state[
        "sustainability_threshold"
    ]
    events = [
        PersonaEvent(
            self._prompt_pool_amount_of_resource(),
            created=self.internal_global_state["next_time"][agent],
            expiration=get_expiration_next_month(
                self.internal_global_state["next_time"][agent]
            ),
            always_include=True,
        )
    ]
    if self.cfg.inject_universalization:
      events.append(
          PersonaEvent(
              self._prompt_universalization(sustainability_threshold),
              created=self.internal_global_state["next_time"][agent],
              expiration=get_expiration_next_month(
                  self.internal_global_state["next_time"][agent]
              ),
              always_include=True,
          )
      )
    obs = HarvestingObs(
        phase=self._phase_to_label(self.phase),
        current_location=self.internal_global_state["next_location"][agent],
        current_location_agents=self.internal_global_state["next_location"],
        current_time=self.internal_global_state["next_time"][agent],
        events=events,
        chat=None,
        current_resource_num=self.internal_global_state["resource_in_pool"],
        agent_resource_num={agent: 0 for agent in self.agents},
        before_harvesting_sustainability_threshold=sustainability_threshold,
    )
    return obs

  def _observe_pool_after_harvesting(self, agent) -> HarvestingObs:
    obs = HarvestingObs(
        phase=self._phase_to_label(self.phase),
        current_location=self.internal_global_state["next_location"][agent],
        current_location_agents=self.internal_global_state["next_location"],
        current_time=self.internal_global_state["next_time"][agent],
        events=[
            PersonaEvent(
                self._prompt_pool_amount_of_resource_after_harvesting(agent),
                created=get_discussion_day(
                    self.internal_global_state["next_time"][agent]
                )
                - timedelta(minutes=1),  # hack to sort the events
                expiration=get_expiration_next_month(
                    self.internal_global_state["next_time"][agent]
                ),
                always_include=True,
            )
        ],
        chat=None,
        current_resource_num=self.internal_global_state["resource_in_pool"],
        agent_resource_num={agent: 0 for agent in self.agents},
        before_harvesting_sustainability_threshold=self.internal_global_state[
            "sustainability_threshold"
        ],
    )
    return obs

  def _observe_restaurant(self, agent) -> HarvestingObs:
    events = []
    state = HarvestingObs(
        phase=self._phase_to_label(self.phase),
        current_location=self.internal_global_state["next_location"][agent],
        current_location_agents=self.internal_global_state["next_location"],
        current_time=self.internal_global_state["next_time"][agent],
        events=events,
        chat=None,
        current_resource_num=self.internal_global_state["resource_in_pool"],
        agent_resource_num=self.internal_global_state[
            "last_collected_resource"
        ],
        before_harvesting_sustainability_threshold=self.internal_global_state[
            "sustainability_threshold"
        ],
    )
    return state

  def _observe_home(self, agent) -> HarvestingObs:
    state = HarvestingObs(
        phase=self._phase_to_label(self.phase),
        current_location=self.internal_global_state["next_location"][agent],
        current_location_agents=self.internal_global_state["next_location"],
        current_time=self.internal_global_state["next_time"][agent],
        events=[],
        chat=None,
        current_resource_num=self.internal_global_state["resource_in_pool"],
        agent_resource_num={agent: 0 for agent in self.agents},
        before_harvesting_sustainability_threshold=self.internal_global_state[
            "sustainability_threshold"
        ],
    )
    return state

  def _observe(self, agent) -> HarvestingObs:
    """Observe should return the observation of the specified agent.

    Depending on the current phase, the observation will be different
    """

    match self.phase:
      case Phase.HARVEST:
        state = self._observe_pool(agent)
      case Phase.POOL_AFTER_HARVESTING:
        state = self._observe_pool_after_harvesting(agent)
      case Phase.RESTAURANT:
        state = self._observe_restaurant(agent)
      case Phase.HOME:
        state = self._observe_home(agent)
      case _:
        raise ValueError(f"Unknown phase: {self.phase!r}")
    return state

  def _phase_to_label(self, phase: Phase) -> str:
    match phase:
      case Phase.HARVEST:
        return self.POOL_LOCATION
      case Phase.POOL_AFTER_HARVESTING:
        return "pool_after_harvesting"
      case Phase.RESTAURANT:
        return "restaurant"
      case Phase.HOME:
        return "home"
      case _:
        raise ValueError(f"Unknown phase: {phase!r}")

  def close(self):
    """Close should release any graphical displays, subprocesses, network connections

    or any other environment data which should not be kept around after the
    user is no longer using the environment.
    """
    pass

  def set_harvest_enforcer(self, callback):
    self._harvest_enforcer = callback

  def set_post_regen_hook(self, callback):
    """Optional callback ``env -> None`` after fish regeneration each round (e.g. contract ``on_round_end``)."""
    self._post_regen_hook = callback

  def set_post_harvest_observer(self, callback):
    self._post_harvest_observer = callback

  def set_contracting_enabled(self, enabled: bool):
    self._contracting_enabled = bool(enabled)

  def _first_location(self) -> str:
    """Location after home / start of cycle: harvest always begins at the pool."""
    return self.POOL_LOCATION

  def _init_agent(self, agent):
    self.internal_global_state["collected_resource"][agent] = 0
    self.internal_global_state["wanted_resource"][agent] = 0
    self.internal_global_state["last_collected_resource"][agent] = 0
    self.internal_global_state["next_location"][agent] = self._first_location()
    self.internal_global_state["next_time"][agent] = datetime(
        2025, 1, 1, 1, 0, 0
    )

    self.rewards[agent] = 0.0
    self.terminations[agent] = False

  def _set_sustainability_threshold(self):
    regen_factor = self.internal_global_state["regen_factor"]
    sustainability_threshold = (
        (regen_factor - 1)
        * self.internal_global_state["resource_in_pool"]
        / (self.internal_global_state["num_agents"] * regen_factor)
    )
    sustainability_threshold = max(sustainability_threshold, 0)
    sustainability_threshold = int(sustainability_threshold)
    self.internal_global_state["sustainability_threshold"] = (
        sustainability_threshold
    )

  def reset(
      self,
      seed=None,
      options=None,
  ) -> tuple[str, HarvestingObs]:
    self.random = np.random.RandomState(seed)

    self.agents = self.possible_agents[: self.cfg.num_agents]

    self.num_round = 0
    self.df_acc = []

    # RL specific (for pettingzoo)
    self.rewards = {}
    self.terminations = {}

    # Initialise the regen factor.
    init_regen_factor = 2.0

    # Environment specific
    self.internal_global_state = {
        "num_agents": float(self.cfg.num_agents),
        "resource_in_pool": self.cfg.initial_resource_in_pool,
        "resource_before_harvesting": self.cfg.initial_resource_in_pool,
        "sustainability_threshold": None,
        "collected_resource": {},
        "wanted_resource": {},
        "last_collected_resource": {},
        "next_location": {},
        "next_time": {},
        "action": {},
        "regen_factor": init_regen_factor,  # Initialise to 2.0 (doubling).
        "contract_enforcement": None,
        "contract_reward_adjustments": {},
    }
    self._set_sustainability_threshold()

    for agent in self.agents:
      self._init_agent(agent)

    self._agent_selector = agent_selector(self.agents)
    self.agent_selection = self._agent_selector.next()
    phases = [
        Phase.HARVEST,
        Phase.POOL_AFTER_HARVESTING,
        Phase.RESTAURANT,
        Phase.HOME,
    ]
    self._phase_selector = agent_selector(phases)
    self.phase = self._phase_selector.next()

    return self.agent_selection, self._observe(self.agent_selection)

  def save_log(self):
    path = f"{self.experiment_storage}/log_env.json"
    tmp_path = f"{path}.tmp"
    if not self.df_acc:
      df = pd.DataFrame()
    else:
      # Exclude empty frames: pd.concat warns (and future dtypes may differ) when
      # mixing empty / all-NA pieces; empty frames add no rows anyway.
      frames = [d for d in self.df_acc if not d.empty]
      df = pd.concat(frames, sort=False) if frames else pd.DataFrame()
    df.to_json(tmp_path, orient="records")
    os.replace(tmp_path, path)

  def _assign_stochastic(self):
    resource_per_agent = {agent: 0 for agent in self.agents}

    wanted = self.internal_global_state["wanted_resource"].copy()
    remaining = self.internal_global_state["resource_in_pool"]
    while sum(wanted.values()) > 0 and remaining > 0:
      # filter agents which want more fish
      agents_to_assign = [agent for agent in self.agents if wanted[agent] > 0]
      if len(agents_to_assign) == 0:
        break
      # pick random agent
      agent = self.random.choice(agents_to_assign)
      wanted[agent] -= 1
      resource_per_agent[agent] += 1
      remaining -= 1

    self.internal_global_state["resource_in_pool"] = int(remaining)

    for agent in self.agents:
      # convert to int
      resource_per_agent[agent] = int(resource_per_agent[agent])

    return resource_per_agent

  def _assign_proportional(self):
    resource_per_agent = {agent: 0 for agent in self.agents}
    was_rounded_down = {agent: False for agent in self.agents}

    wanted = self.internal_global_state["wanted_resource"].copy()
    remaining = self.internal_global_state["resource_in_pool"]
    while sum(wanted.values()) > 0 and remaining > 0:
      total_wanted = sum(wanted.values())
      if total_wanted > remaining:
        remaining_res_now = remaining
        for agent in wanted.keys():
          tmp = remaining_res_now * (wanted[agent] / total_wanted)
          tmp = min(
              tmp, wanted[agent]
          )  # we cannot assign more than the agent wanted
          if tmp == 0:
            continue
          if tmp == int(tmp):
            resource_per_agent[agent] += int(tmp)
            was_rounded_down[agent] = False
          else:
            tmp = np.floor(tmp)
            was_rounded_down[agent] = True
            resource_per_agent[agent] += tmp
          remaining -= tmp
          wanted[agent] -= tmp

        if remaining > 0:
          # assign the remaining fish to the agents that were rounded down, randomly

          if remaining > len([w for w in wanted.values() if w > 0]):
            continue
          else:
            for _ in range(int(remaining)):
              agents_to_assign = [
                  agent
                  for agent in self.agents
                  if was_rounded_down[agent] and wanted[agent] > 0
              ]
              if len(agents_to_assign) == 0:
                break
              total = sum(wanted[agent] for agent in agents_to_assign)
              p = [wanted[agent] / total for agent in agents_to_assign]
              agent = self.random.choice(agents_to_assign, p=p)
              resource_per_agent[agent] += 1
              wanted[agent] -= 1
              agents_to_assign.remove(agent)
              remaining -= 1

      else:
        for agent in self.agents:
          resource_per_agent[agent] = wanted[agent]
          remaining -= wanted[agent]
          wanted[agent] = 0

    self.internal_global_state["resource_in_pool"] = int(remaining)

    for agent in self.agents:
      # convert to int
      resource_per_agent[agent] = int(resource_per_agent[agent])

    return resource_per_agent

  def _assign_resource(self):
    if self.cfg.assign_resource_strategy == "stochastic":
      resource_per_agent = self._assign_stochastic()
    elif self.cfg.assign_resource_strategy == "proportional":
      raise DeprecationWarning("ConcurrentEnv._assign_proportional is deprecated. Use aassign_resource instead.")
      resource_per_agent = self._assign_proportional()
    else:
      raise ValueError(
          "Unknown assign resource strategy:"
          f" {self.cfg.assign_resource_strategy}"
      )

    for agent in self.agents:
      res = resource_per_agent[agent]
      action = self.internal_global_state["action"][agent]
      self.log_step_harvest(action, res)

    for agent in self.agents:
      res = resource_per_agent[agent]
      self.internal_global_state["collected_resource"][agent] += res
      self.internal_global_state["last_collected_resource"][agent] = res

      self.rewards[agent] += res

    self._apply_contract_reward_adjustments()

  def _apply_contract_reward_adjustments(self):
    enforcement_result = self.internal_global_state.get("contract_enforcement")
    adjustments = getattr(enforcement_result, "reward_adjustments", {}) or {}
    normalized = {}
    for agent in self.agents:
      delta = adjustments.get(agent, 0.0)
      try:
        delta = float(delta)
      except (TypeError, ValueError):
        delta = 0.0
      self.rewards[agent] += delta
      normalized[agent] = delta
    self.internal_global_state["contract_reward_adjustments"] = normalized

  def _step_lake_bet(self, action: PersonaActionHarvesting):
    res = action.quantity
    self.internal_global_state["wanted_resource"][self.agent_selection] = res
    self.internal_global_state["action"][self.agent_selection] = action
    self.internal_global_state["next_location"][
        self.agent_selection
    ] = self.POOL_LOCATION
    if self._agent_selector.is_last():
      if self._harvest_enforcer is not None:
        enforcement_result = self._harvest_enforcer(
            self.internal_global_state["wanted_resource"].copy(),
            self,
        )
        self.internal_global_state["contract_enforcement"] = enforcement_result
        if enforcement_result and hasattr(enforcement_result, "modified_catches"):
          self.internal_global_state["wanted_resource"].update(
              enforcement_result.modified_catches
          )
      self._assign_resource()
      if self._post_harvest_observer is not None:
        self._post_harvest_observer(
            self.internal_global_state["last_collected_resource"].copy(),
            self,
            self.internal_global_state.get("contract_enforcement"),
        )
      self.phase = self._phase_selector.next()
    self.agent_selection = self._agent_selector.next()

  def _step_pool_after_harvesting(self, action: PersonaActionHarvesting):
    # We have no interaction with other agents at the lake
    self.internal_global_state["next_location"][
        self.agent_selection
    ] = "restaurant"
    self.internal_global_state["next_time"][self.agent_selection] = (
        get_discussion_day(
            self.internal_global_state["next_time"][self.agent_selection]
        )
    )
    # We do nothing, we need to only ensure each of the agents has observe how much it has harvested
    if self._agent_selector.is_last():
      self.phase = self._phase_selector.next()
    self.agent_selection = self._agent_selector.next()

  def _step_restaurant(self, action: PersonaActionChat):
    # Post-harvest group chat at the restaurant (phase label remains "restaurant").
    if type(action) == PersonaActionChat:
      self.log_step_conversation(action)
      # Advance to the next phase
      for a in self.agents:
        self.internal_global_state["next_location"][a] = "home"
        self.internal_global_state["next_time"][a] = get_reflection_day(
            self.internal_global_state["next_time"][a]
        )
      self.phase = self._phase_selector.next()
      self.agent_selection = self._agent_selector.reset()

  def _step_home(self, action: PersonaAction):
    self.internal_global_state["next_location"][
        self.agent_selection
    ] = self._first_location()
    self.internal_global_state["next_time"][self.agent_selection] += timedelta(
        days=1
    )
    # A possible idea here is to probe the agent for some reflection / thoughts
    # NOTE do we need to register something here?

  @property
  def regen_factor(self) -> int:
    return self.internal_global_state["regen_factor"]

  def _apply_regeneration(self):
    """Apply regeneration to the pool. Override in subclasses for custom modes."""
    self.internal_global_state["resource_in_pool"] = int(min(
        self.cfg.initial_resource_in_pool,
        self.internal_global_state["resource_in_pool"]
        * self.internal_global_state["regen_factor"],
    ))
    self.internal_global_state["resource_before_harvesting"] = (
        self.internal_global_state["resource_in_pool"]
    )
    # Sample the next regeneration factor.
    self.internal_global_state["regen_factor"] = self.random.uniform(
        self._regen_factor_range[0], self._regen_factor_range[1]
    )

  def step(
      self, action: PersonaAction
  ) -> tuple[str, HarvestingObs, dict, dict]:
    if self.terminations[self.agent_selection]:
      self.save_log()
      return

    assert action.agent_id == self.agent_selection

    match self.phase:
      case Phase.HARVEST:
        assert action.location == self.POOL_LOCATION
        assert type(action) == PersonaActionHarvesting
        self._step_lake_bet(action)
      case Phase.POOL_AFTER_HARVESTING:
        assert action.location == self.POOL_LOCATION
        self._step_pool_after_harvesting(action)
      case Phase.RESTAURANT:
        assert action.location == "restaurant"
        self._step_restaurant(action)
      case Phase.HOME:
        assert action.location == "home"
        self._step_home(action)
        if self._agent_selector.is_last():
          self.num_round += 1
          self.phase = self._phase_selector.next()

          # We want to see also the discussion in case no fish remain
          self.terminations = {
              agent: (
                  self.internal_global_state["resource_in_pool"]
                  < 5  # less than 5 fish remain, so we collapse
                  or self.num_round >= self.cfg.max_num_rounds
              )
              for agent in self.agents
          }
          self._apply_regeneration()
          if self._post_regen_hook is not None:
            self._post_regen_hook(self)
          self._set_sustainability_threshold()
          if self.cfg.harvesting_order == "random-sequential":
            agents = list(np.random.permutation(self.agents))
            self._agent_selector = agent_selector(agents)
        self.agent_selection = self._agent_selector.next()
      case _:
        raise ValueError(f"Unknown phase: {self.phase!r}")

    self.save_log()
    return (
        self.agent_selection,
        self._observe(self.agent_selection),
        self.rewards,
        self.terminations,
    )

  ########################################
  # Logging
  ########################################

  def log_step_harvest(
      self,
      action: PersonaActionHarvesting,
      resource_collected: int,
  ):
    tmp = {
        "agent_id": [action.agent_id],
        "round": [self.num_round],
        "action": ["harvesting"],
        "resource_in_pool_before_harvesting": [
            self.internal_global_state["resource_before_harvesting"]
        ],
        "resource_in_pool_after_harvesting": [
            self.internal_global_state["resource_in_pool"]
        ],
        "concurrent_harvesting": [True],
        "resource_collected": [resource_collected],
        "wanted_resource": [action.quantity],
        "commands": [getattr(action, "commands", [])],
        "contract_reward_adjustment": [
            self.internal_global_state.get("contract_reward_adjustments", {}).get(
                action.agent_id, 0.0
            )
        ],
        "html_interactions": [action.html_interactions],
    }
    if "sustainable_intention" in action.stats:
      tmp["sustainable_intention"] = [action.stats["sustainable_intention"]]
    df_log = pd.DataFrame(tmp, index=[len(self.df_acc)])
    self.df_acc.append(df_log)

  def log_step_regen(self, regen_state):
    """Log per-round regeneration state. Called from _apply_regeneration overrides."""
    pass

  def log_step_conversation(self, chat: PersonaActionChat):
    for i, (p, u) in enumerate(chat.conversation):
      df_log = pd.DataFrame(
          {
              "agent_id": p.agent_id,
              "agent_name": p.name,
              "round": self.num_round,
              "action": "utterance",
              "resource_limit": chat.conversation_resource_limit,
              "utterance": u,
              "html_interactions": [chat.html_interactions[i]],
          },
          index=[len(self.df_acc)],
      )
      self.df_acc.append(df_log)
    df_log = pd.DataFrame(
        {
            "agent_id": "framework",
            "agent_name": "framework",
            "round": self.num_round,
            "action": "conversation_summary",
            "resource_limit": chat.conversation_resource_limit,
            "html_interactions": [chat.html_interactions[-2]],
        },
        index=[len(self.df_acc)],
    )
    self.df_acc.append(df_log)

    df_log = pd.DataFrame(
        {
            "agent_id": "framework",
            "agent_name": "framework",
            "round": self.num_round,
            "action": "conversation_resource_limit",
            "resource_limit": chat.conversation_resource_limit,
            "html_interactions": [chat.html_interactions[-1]],
        },
        index=[len(self.df_acc)],
    )
    self.df_acc.append(df_log)
