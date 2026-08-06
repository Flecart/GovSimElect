from simulation.utils import ModelWandbWrapper

from ..memory import MemoryBackend
from .component import Component


class RetrieveComponent(Component):
    """Retrieve relevant private memories from the configured backend."""

    def __init__(
        self,
        model: ModelWandbWrapper,
        model_framework: ModelWandbWrapper,
        memory_backend: MemoryBackend,
    ):
        super().__init__(model, model_framework)
        self.memory_backend = memory_backend

    def retrieve(
        self, focal_points: list[str], top_k: int
    ):
        query = "\n".join(focal_points) if focal_points else "current situation"
        return self.memory_backend.search(query, top_k, self.persona.current_time)
