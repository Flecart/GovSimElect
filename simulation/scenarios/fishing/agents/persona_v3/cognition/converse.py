"""Handling conversation among the personas."""

import asyncio
import random

from typing import Optional
from datetime import datetime

from simulation.persona import PersonaAgent
from simulation.persona.cognition.converse import ConverseComponent
from simulation.persona.cognition.retrieve import RetrieveComponent
from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper

from .converse_prompts import (
    aprompt_converse_utterance_in_group,
    aprompt_summarize_conversation_in_one_sentence,
    prompt_converse_utterance_in_group,
    prompt_summarize_conversation_in_one_sentence,
)
from .reflect_prompts import aprompt_find_harvesting_limit_from_conversation
from .reflect_prompts import prompt_find_harvesting_limit_from_conversation


class FishingConverseComponent(ConverseComponent):
  """Fishing converse component."""

  def conduct_contracting_phase(
      self,
      target_personas: list[PersonaAgent],
      current_time: datetime,
      current_resource_num: int,
      agent_resource_num: dict[str, int],
      sustainability_threshold: int,
  ) -> tuple[list[tuple[PersonaIdentity, str]], str, int | None, list[str]]:
    contract_runtime = getattr(self.persona, "contracting_runtime", None)
    if contract_runtime is None or not contract_runtime.enabled:
      raise RuntimeError("Contracting phase called without enabled contracting runtime.")
    return asyncio.run(
        contract_runtime.aconduct_negotiation(
            personas=target_personas,
            current_time=current_time,
            current_resource_num=current_resource_num,
            sustainability_threshold=sustainability_threshold,
            agent_resource_num=agent_resource_num,
        )
    )

  def conduct_restaurant_phase(
      self,
      target_personas: list[PersonaAgent],
      current_location: str,
      current_time: datetime,
      current_resource_num: int,
      agent_resource_num: dict[str, int],
      sustainability_threshold: int,
      mayoral_agenda: str | None = None,
      harvest_report: str | None = None,
      leader_persona: PersonaAgent | None = None,
      debug: bool = False,
  ):
    raise DeprecationWarning("FishingConverseComponent.converse_restaurant_phase is deprecated. Use aconverse_restaurant_phase instead.")

  def converse_group(
      self,
      target_personas: list[PersonaAgent],
      current_location: str,
      current_time: datetime,
      current_resource_num: int,
      agent_resource_num: dict[str, int],
      sustainability_threshold: int,
      mayoral_agenda: str | None = None,
      harvest_report: str | None = None,
      leader_persona: PersonaAgent | None = None,
      debug: bool = False,
  ) -> tuple[list[tuple[str, str]], str]:
    raise DeprecationWarning("FishingConverseComponent.converse_group is deprecated. Use aconverse_group instead.")

  async def aconduct_contracting_phase(
      self,
      target_personas: list[PersonaAgent],
      current_time: datetime,
      current_resource_num: int,
      agent_resource_num: dict[str, int],
      sustainability_threshold: int,
  ) -> tuple[list[tuple[PersonaIdentity, str]], str, int | None, list[str]]:
    contract_runtime = getattr(self.persona, "contracting_runtime", None)
    if contract_runtime is None or not contract_runtime.enabled:
      raise RuntimeError("Contracting phase called without enabled contracting runtime.")
    return await contract_runtime.aconduct_negotiation(
        personas=target_personas,
        current_time=current_time,
        current_resource_num=current_resource_num,
        sustainability_threshold=sustainability_threshold,
        agent_resource_num=agent_resource_num,
    )

  async def aconduct_restaurant_phase(
      self,
      target_personas: list[PersonaAgent],
      current_location: str,
      current_time: datetime,
      current_resource_num: int,
      agent_resource_num: dict[str, int],
      sustainability_threshold: int,
      mayoral_agenda: str | None = None,
      harvest_report: str | None = None,
      leader_persona: PersonaAgent | None = None,
      debug: bool = False,
  ):
    """Async restaurant step; contracting vs free chat is decided inside aconverse_group."""
    return await self.aconverse_group(
        target_personas,
        current_location,
        current_time,
        current_resource_num,
        agent_resource_num,
        sustainability_threshold,
        mayoral_agenda=mayoral_agenda,
        harvest_report=harvest_report,
        leader_persona=leader_persona,
        debug=debug,
    )

  def _build_restaurant_preamble(
      self,
      target_personas: list[PersonaAgent],
      current_leader_id: PersonaIdentity,
      mayoral_agenda: str | None,
      harvest_report: str | None,
      leader_persona: PersonaAgent | None,
      agent_resource_num: dict[str, int],
  ) -> tuple[list[tuple[PersonaIdentity, str]], list[str]]:
    """Inject harvest / leader context before the main exchange (contracting or chat)."""
    current_conversation: list[tuple[PersonaIdentity, str]] = []
    html_interactions: list[str] = []

    if (
        self.cfg.inject_resource_observation
        and self.cfg.inject_resource_observation_strategy == "individual"
    ):
      for persona in target_personas:
        p = self.other_personas[persona.identity.name]
        current_conversation.append(
            (
                p.identity,
                (
                    "This month, I caught"
                    f" {agent_resource_num[p.agent_id]} tons of fish!"
                ),
            ),
        )
        html_interactions.append(
            "<strong>Framework</strong>:  This month, I caught"
            f" {agent_resource_num[p.agent_id]} tons of fish!"
        )
    elif (
        self.cfg.inject_resource_observation
        and self.cfg.inject_resource_observation_strategy == "manager"
    ):
      current_conversation.append(
          (
              current_leader_id,  # always anonymous leader for my experiemnts, so changed the prompt
              (
                  f"The monthly fishing report:\n{harvest_report}"
              ),
          ),
      )
      if mayoral_agenda and leader_persona:
        raise ValueError("Mayoral agenda is not supported for prosociality experiments.")
        # current_conversation.append(
        #     (
        #         current_leader_id,
        #         (
        #             "I'd also like to share my policy agenda to help guide our "
        #             f"collective action: {mayoral_agenda}"
        #         ),
        #     ),
        # )

    return current_conversation, html_interactions

  async def _phase_main_restaurant_exchange(
      self,
      current_conversation: list[tuple[PersonaIdentity, str]],
      html_interactions: list[str],
      target_personas: list[PersonaAgent],
      current_location: str,
      current_time: datetime,
      current_resource_num: int,
      agent_resource_num: dict[str, int],
      sustainability_threshold: int,
      leader_persona: PersonaAgent | None,
      debug: bool,
  ) -> None:
    """Contracting negotiation or free-form turn-taking. Mutates conversation and html lists."""
    from simulation.scenarios.fishing.contracting import ContractingOrchestrator
      
    rt: ContractingOrchestrator | None = getattr(self.persona, "contracting_runtime", None)
    if rt is not None and rt.enabled:
      print(f"CONTRACTING: {self.persona.identity.name}")
      neg_transcript, _, _, html_neg = (
          await rt.aconduct_negotiation(
              personas=target_personas,
              current_time=current_time,
              current_resource_num=current_resource_num,
              sustainability_threshold=sustainability_threshold,
              agent_resource_num=agent_resource_num,
          )
      )
      current_conversation.extend(neg_transcript)
      html_interactions.extend(html_neg)
      return

    max_conversation_steps = self.cfg.max_conversation_steps
    # ANG: we don't have a leader persona in this version.
    # if leader_persona:
    #   current_persona = leader_persona
    # else:
    current_persona = random.choice(target_personas)

    while True:
    # ANG: we never have in the new version so many memories that we need to search
    # next versions can use fancier memory systems
    
    #   focal_points: list[str] = []
    #   if current_conversation:
    #     for _, utterance in current_conversation[-4:]:
    #       focal_points.append(utterance)

    #     if len(current_conversation) == 4:
    #       focal_points += current_conversation[0:1]
    #     elif len(current_conversation) > 5:
    #       focal_points += current_conversation[0:2]
      focal_points = []
      focal_points = self.other_personas[
          current_persona.identity.name
      ].retrieve.retrieve(focal_points, top_k=8)

      if self.cfg.prompt_utterance != "one_shot":
        raise NotImplementedError(
            f"prompt_utterance={self.cfg.prompt_utterance}"
        )

      utterance, end_conversation, next_name, h = (
          await aprompt_converse_utterance_in_group(
              self.model,
              current_persona,
              target_personas,
              focal_points,
              current_location,
              current_time,
              self.conversation_render(current_conversation),
              debug,
          )
      )
      html_interactions.append(h)

      current_conversation.append((current_persona.identity, utterance))

      if (
          end_conversation
          or len(current_conversation) >= max_conversation_steps
      ):
        break
      current_persona = self.other_personas[next_name.capitalize()]

  async def aconverse_group(
      self,
      target_personas: list[PersonaAgent],
      current_location: str,
      current_time: datetime,
      current_resource_num: int,
      agent_resource_num: dict[str, int],
      sustainability_threshold: int,
      mayoral_agenda: str | None = None,
      harvest_report: str | None = None,
      leader_persona: PersonaAgent | None = None,
      debug: bool = False,
  ) -> tuple[
      list[tuple[PersonaIdentity, str]],
      str,
      int | None,
      list[str],
  ]:
    if leader_persona:
      current_leader_id = leader_persona.identity
    else:
      current_leader_id = PersonaIdentity("framework", "Anonymous Leader")

    current_conversation, html_interactions = self._build_restaurant_preamble(
        target_personas=target_personas,
        current_leader_id=current_leader_id,
        mayoral_agenda=mayoral_agenda,
        harvest_report=harvest_report,
        leader_persona=leader_persona,
        agent_resource_num=agent_resource_num,
    )

    await self._phase_main_restaurant_exchange(
        current_conversation,
        html_interactions,
        target_personas,
        current_location,
        current_time,
        current_resource_num,
        agent_resource_num,
        sustainability_threshold,
        leader_persona,
        debug,
    )

    summary_conversation, h = (
        await aprompt_summarize_conversation_in_one_sentence(
            self.model_framework,
            self.conversation_render(current_conversation),
        )
    )
    html_interactions.append(h)

    resource_limit, h = await aprompt_find_harvesting_limit_from_conversation(
        self.model_framework, self.conversation_render(current_conversation)
    )
    html_interactions.append(h)

    async def finalize_persona_memory(persona):
      p = self.other_personas[persona.identity.name]
      tasks = [
          p.store.astore_chat(
              summary_conversation,
              self.conversation_render(current_conversation),
              self.persona.current_time,
          ),
          p.reflect.areflect_on_convesation(
              self.conversation_render(current_conversation)
          ),
      ]
      if resource_limit is not None:
        tasks.append(
            p.store.astore_thought(
                (
                    "The community agreed on a maximum limit of"
                    f" {resource_limit} tons of fish per person."
                ),
                self.persona.current_time,
                always_include=True,
            )
        )
      await asyncio.gather(*tasks)

    await asyncio.gather(
        *(finalize_persona_memory(persona) for persona in target_personas)
    )
    if debug:
      print(
          "CONVERSATION TRANSCRIPT:"
          f" {self.conversation_render(current_conversation)}."
      )
    return (
        current_conversation,
        summary_conversation,
        resource_limit,
        html_interactions,
    )

  def conversation_render(
      self, conversation: list[tuple[PersonaIdentity, str]]
  ):
    return [(p.name, u) for p, u in conversation]
