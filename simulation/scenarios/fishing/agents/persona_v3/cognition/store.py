from simulation.persona.cognition import StoreComponent
from simulation.persona.common import ChatObservation, PersonaIdentity
from simulation.persona.memory import MemoryBackend
from simulation.utils import ModelWandbWrapper

from .store_prompts import (
    aprompt_importance_action,
    aprompt_importance_chat,
    aprompt_importance_event,
    aprompt_importance_thought,
    prompt_importance_action,
    prompt_importance_chat,
    prompt_importance_event,
    prompt_importance_thought,
    prompt_text_to_triple,
)


class FishingStoreComponent(StoreComponent):

    def __init__(
        self,
        model: ModelWandbWrapper,
        model_framework: ModelWandbWrapper,
        memory_backend: MemoryBackend,
        cfg,
    ) -> None:
        super().__init__(model, model_framework, memory_backend, cfg)
        self.prompt_importance_thought = prompt_importance_thought
        self.prompt_importance_chat = prompt_importance_chat
        self.prompt_importance_event = prompt_importance_event
        self.prompt_importance_action = prompt_importance_action
        self.aprompt_importance_thought = aprompt_importance_thought
        self.aprompt_importance_chat = aprompt_importance_chat
        self.aprompt_importance_event = aprompt_importance_event
        self.aprompt_importance_action = aprompt_importance_action
