import asyncio
import re

from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper

from .utils import conversation_to_string_with_dash
from .utils import extract_all_matches
from .utils import extract_first_match
from .utils import get_sytem_prompt
from .utils import numbered_memory_prompt
from .utils import reasoning_steps_prompt


async def aprompt_insight_and_evidence(
    model: ModelWandbWrapper, persona: PersonaIdentity, statements: list[str]
):
    session = model.start_prompt(
        persona.name, "cognition_retrieve", "prompt_insight_and_evidence"
    )
    session.add_user(
        f"{get_sytem_prompt(persona)}\n"
        f"{numbered_memory_prompt(persona, statements)}\n"
        "What high-level insights can you infer from the statements above?\n"
        "Return one insight per line, prefixed with '- '. Keep only the insight text."
    )
    response = await model.acomplete_prompt(
        session,
        max_tokens=600,
        default_value="",
    )
    model.end_prompt(session)
    insights = [line[2:].strip() for line in response.splitlines() if line.strip().startswith("- ")]
    if insights:
        return insights
    fallback = [line.strip(" -*0123456789.)") for line in response.splitlines() if line.strip()]
    return [line for line in fallback if line]


def prompt_insight_and_evidence(
    model: ModelWandbWrapper, persona: PersonaIdentity, statements: list[str]
):
    return asyncio.run(aprompt_insight_and_evidence(model, persona, statements))


async def aprompt_planning_thought_on_conversation(
    model: ModelWandbWrapper,
    persona: PersonaIdentity,
    conversation: list[tuple[str, str]],
) -> str:
    session = model.start_prompt(
        persona.name, "cognition_retrieve", "prompt_planning_thought_on_conversation"
    )
    session.add_user(
        f"{get_sytem_prompt(persona)}\n"
        "Conversation:\n"
        f"{conversation_to_string_with_dash(conversation)}\n"
        "Write one full sentence describing anything from the conversation that you need to remember for planning."
    )
    response = await model.acomplete_prompt(
        session,
        max_tokens=200,
        default_value="There is nothing specific I need to remember for planning.",
    )
    model.end_prompt(session)
    return response.strip().splitlines()[0].strip()


def prompt_planning_thought_on_conversation(
    model: ModelWandbWrapper,
    persona: PersonaIdentity,
    conversation: list[tuple[str, str]],
) -> str:
    return asyncio.run(
        aprompt_planning_thought_on_conversation(model, persona, conversation)
    )


async def aprompt_memorize_from_conversation(
    model: ModelWandbWrapper,
    persona: PersonaIdentity,
    conversation: list[tuple[str, str]],
) -> str:
    session = model.start_prompt(
        persona.name, "cognition_retrieve", "prompt_memorize_from_conversation"
    )
    session.add_user(
        f"{get_sytem_prompt(persona)}\n"
        "Conversation:\n"
        f"{conversation_to_string_with_dash(conversation)}\n"
        "Write one full sentence describing anything interesting from the conversation from your own perspective."
    )
    response = await model.acomplete_prompt(
        session,
        max_tokens=200,
        default_value="There was nothing especially memorable in the conversation.",
    )
    model.end_prompt(session)
    return response.strip().splitlines()[0].strip()


def prompt_memorize_from_conversation(
    model: ModelWandbWrapper,
    persona: PersonaIdentity,
    conversation: list[tuple[str, str]],
) -> str:
    return asyncio.run(aprompt_memorize_from_conversation(model, persona, conversation))


async def aprompt_find_harvesting_limit_from_conversation(
    model: ModelWandbWrapper,
    conversation: list[tuple[str, str]],
) -> tuple[int, str]:
    session = model.start_prompt(
        "framework",
        "cognition_reflect",
        "prompt_find_harvesting_limit_from_conversation",
    )
    session.add_user(
        "In the following conversation, determine whether there was an explicit agreement on a concrete fishing limit.\n"
        "If there was a limit, return exactly `Answer: N` where N is the numeric limit per person.\n"
        "If there was no explicit limit, return exactly `Answer: N/A`.\n"
        f"Conversation:\n{conversation_to_string_with_dash(conversation)}\n"
        f"{reasoning_steps_prompt()}"
    )
    response = await model.acomplete_prompt(
        session,
        max_tokens=300,
        default_value="Answer: N/A",
    )
    model.end_prompt(session)
    parsed = extract_first_match(r"Answer:\s*(N/A|\d+)", response, "N/A", re.IGNORECASE)
    if parsed is None or parsed.upper() == "N/A":
        return None, session.html()
    return int(parsed), session.html()


def prompt_find_harvesting_limit_from_conversation(
    model: ModelWandbWrapper,
    conversation: list[tuple[str, str]],
) -> tuple[int, str]:
    return asyncio.run(aprompt_find_harvesting_limit_from_conversation(model, conversation))
