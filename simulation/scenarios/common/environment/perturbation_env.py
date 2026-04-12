import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from pettingzoo.utils import agent_selector

from simulation.persona.common import (
    PersonaAction,
    PersonaActionChat,
    PersonaActionHarvesting,
    PersonaEvent,
    PersonaIdentity,
)

from .common import HarvestingObs
from .concurrent_env import (
    ConcurrentEnv,
    Phase,
    get_discussion_day,
    get_expiration_next_month,
    get_reflection_day,
)

"""
Uses the env.perturbations settins

Perturbation:
name: NAME
round: ROUND_AT_WHICH_TO_APPLY


"""


class PerturbationEnv(ConcurrentEnv):
    def __init__(
        self, cfg: DictConfig, experiment_storage: str, map_id_to_name: dict[str, str]
    ) -> None:
        raise DeprecationWarning("Use ConcurrentEnv instead.")
        super().__init__(cfg, experiment_storage, map_id_to_name)

        assert len(cfg.perturbations) == 1
        self.perturbation = cfg.perturbations[0].perturbation

    def _prompt_home_observe_agent_resource(self, agent):
        raise NotImplementedError

    def _observe_home(self, agent) -> HarvestingObs:
        if (
            self.cfg.language_nature == "none"
            or self.cfg.language_nature == "none_and_no_obs"
        ):
            # Inject what each person has fished
            events = []
            if self.cfg.language_nature == "none":
                for agent in self.agents:
                    events.append(
                        PersonaEvent(
                            self._prompt_home_observe_agent_resource(agent),
                            created=self.internal_global_state["next_time"][agent],
                            expiration=get_expiration_next_month(
                                self.internal_global_state["next_time"][agent]
                            ),
                        )
                    )

            state = HarvestingObs(
                phase=self._phase_to_label(self.phase),
                current_location=self.internal_global_state["next_location"][agent],
                current_location_agents=self.internal_global_state["next_location"],
                current_time=self.internal_global_state["next_time"][agent],
                events=events,
                chat=None,
                current_resource_num=self.internal_global_state["resource_in_pool"],
                agent_resource_num={agent: 0 for agent in self.agents},
                before_harvesting_sustainability_threshold=self.internal_global_state[
                    "sustainability_threshold"
                ],
            )
        else:
            state = super()._observe_home(agent)
        return state

    def _step_pool_after_harvesting(self, action: PersonaActionHarvesting):
        # We have no interaction with other agents at the lake
        self.internal_global_state["next_location"][self.agent_selection] = "restaurant"
        self.internal_global_state["next_time"][self.agent_selection] = (
            get_discussion_day(
                self.internal_global_state["next_time"][self.agent_selection]
            )
        )

        # Apply perturbations
        if (
            self.cfg.language_nature == "none"
            or self.cfg.language_nature == "none_and_no_obs"
        ):
            self.internal_global_state["next_location"][self.agent_selection] = "home"
            self.internal_global_state["next_time"][self.agent_selection] = (
                get_reflection_day(
                    self.internal_global_state["next_time"][self.agent_selection]
                )
            )

        # Next phase / next agent
        if self._agent_selector.is_last():
            self.phase = self._phase_selector.next()
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        raise DeprecationWarning("Use ConcurrentEnv instead.")

    def reset(self):
        raise DeprecationWarning("Use ConcurrentEnv instead.")
