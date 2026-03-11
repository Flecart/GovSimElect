import asyncio

from simulation.utils import ModelWandbWrapper

from ..common import ChatObservation, PersonaIdentity
from .component import Component


class ReflectComponent(Component):

    prompt_insight_and_evidence: callable
    prompt_planning_thought_on_conversation: callable
    prompt_memorize_from_conversation: callable

    def __init__(
        self,
        model: ModelWandbWrapper,
        model_framework: ModelWandbWrapper,
    ):
        super().__init__(model, model_framework)

    def run(self, focal_points: list[str]):
        acc = []
        for focal_point in focal_points:
            retireved_memory = self.persona.retrieve.retrieve([focal_point], 10)

            insights = self.prompt_insight_and_evidence(
                self.model, self.persona.identity, retireved_memory
            )
            for insight in insights:
                self.persona.store.store_thought(insight, self.persona.current_time)
                acc.append(insight)

    async def arun(self, focal_points: list[str]):
        acc = []
        for focal_point in focal_points:
            retireved_memory = self.persona.retrieve.retrieve([focal_point], 10)
            insights = await self.aprompt_insight_and_evidence(
                self.model, self.persona.identity, retireved_memory
            )
            tasks = [
                self.persona.store.astore_thought(insight, self.persona.current_time)
                for insight in insights
            ]
            if tasks:
                await asyncio.gather(*tasks)
            acc.extend(insights)
        return acc

    def reflect_on_convesation(self, conversation: list[tuple[str, str]]):
        planning = self.prompt_planning_thought_on_conversation(
            self.model, self.persona.identity, conversation
        )  # TODO should be this be store in scratch for planning?
        self.persona.store.store_thought(planning, self.persona.current_time)
        memo = self.prompt_memorize_from_conversation(
            self.model, self.persona.identity, conversation
        )
        self.persona.store.store_thought(memo, self.persona.current_time)

    async def areflect_on_convesation(self, conversation: list[tuple[str, str]]):
        planning, memo = await asyncio.gather(
            self.aprompt_planning_thought_on_conversation(
                self.model, self.persona.identity, conversation
            ),
            self.aprompt_memorize_from_conversation(
                self.model, self.persona.identity, conversation
            ),
        )
        await asyncio.gather(
            self.persona.store.astore_thought(planning, self.persona.current_time),
            self.persona.store.astore_thought(memo, self.persona.current_time),
        )
