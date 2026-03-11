import asyncio

from simulation.persona.common import PersonaIdentity
from simulation.persona.memory.associative_memory import Action, Chat, Event, Thought
from simulation.utils import ModelWandbWrapper

from .utils import extract_first_match
from .utils import get_sytem_prompt


def _parse_significance(response: str, default: int = 5) -> int:
    parsed = extract_first_match(r"\b(10|[1-9])\b", response, str(default))
    return int(parsed)


async def aprompt_importance_chat(
    model: ModelWandbWrapper, persona: PersonaIdentity, chat: Chat
):
    session = model.start_prompt(
        persona.name, "cognition_retrieve", "prompt_importance_chat"
    )
    session.add_user(
        f"{get_sytem_prompt(persona)}\n"
        "Task: Rate the significance of a conversation.\n"
        "Return only one integer from 1 to 10, where 1 is mundane and 10 is highly impactful.\n"
        f"Conversation to rate:\n{chat.description}\n"
    )
    response = await model.acomplete_prompt(
        session,
        max_tokens=20,
        temperature=0.0,
        top_p=1.0,
        default_value="5",
    )
    model.end_prompt(session)
    return _parse_significance(response)


def prompt_importance_chat(
    model: ModelWandbWrapper, persona: PersonaIdentity, chat: Chat
):
    return asyncio.run(aprompt_importance_chat(model, persona, chat))


async def aprompt_importance_event(
    model: ModelWandbWrapper, persona: PersonaIdentity, event: Event
):
    session = model.start_prompt(persona.name, "cognition_perceive", "relevancy_event")
    session.add_user(
        f"{get_sytem_prompt(persona)}\n"
        "Task: Rate the significance of an event.\n"
        "Return only one integer from 1 to 10, where 1 is mundane and 10 is extremely significant.\n"
        f"Event to rate:\n{event.description}\n"
    )
    response = await model.acomplete_prompt(
        session,
        max_tokens=20,
        temperature=0.0,
        top_p=1.0,
        default_value="5",
    )
    model.end_prompt(session)
    return _parse_significance(response)


def prompt_importance_event(
    model: ModelWandbWrapper, persona: PersonaIdentity, event: Event
):
    return asyncio.run(aprompt_importance_event(model, persona, event))


async def aprompt_importance_thought(
    model: ModelWandbWrapper, persona: PersonaIdentity, thought: Thought
):
    session = model.start_prompt(
        persona.name, "cognition_retrieve", "prompt_importance_thought"
    )
    session.add_user(
        f"{get_sytem_prompt(persona)}\n"
        "Task: Rate the significance of a thought.\n"
        "Return only one integer from 1 to 10, where 1 is routine and 10 is highly significant.\n"
        f"Thought to rate:\n{thought.description}\n"
    )
    response = await model.acomplete_prompt(
        session,
        max_tokens=20,
        temperature=0.0,
        top_p=1.0,
        default_value="5",
    )
    model.end_prompt(session)
    return _parse_significance(response)


def prompt_importance_thought(
    model: ModelWandbWrapper, persona: PersonaIdentity, thought: Thought
):
    return asyncio.run(aprompt_importance_thought(model, persona, thought))


async def aprompt_importance_action(
    model: ModelWandbWrapper, persona: PersonaIdentity, action: Action
):
    session = model.start_prompt(
        persona.name, "cognition_retrieve", "prompt_importance_action"
    )
    session.add_user(
        f"{get_sytem_prompt(persona)}\n"
        "Task: Rate the significance of an action.\n"
        "Return only one integer from 1 to 10, where 1 is routine and 10 is highly significant.\n"
        f"Action to rate:\n{action.description}\n"
    )
    response = await model.acomplete_prompt(
        session,
        max_tokens=20,
        temperature=0.0,
        top_p=1.0,
        default_value="5",
    )
    model.end_prompt(session)
    return _parse_significance(response)


def prompt_importance_action(
    model: ModelWandbWrapper, persona: PersonaIdentity, action: Action
):
    return asyncio.run(aprompt_importance_action(model, persona, action))


async def aprompt_text_to_triple(model: ModelWandbWrapper, text: str):
    session = model.start_prompt("framework", "cognition_retrieve", "prompt_text_to_triple")
    session.add_user(
        "Split the phrase below into subject, predicate, and object.\n"
        "Return exactly three lines in the format:\n"
        "Subject: ...\nPredicate: ...\nObject: ...\n"
        f"Phrase: {text}\n"
    )
    response = await model.acomplete_prompt(
        session,
        max_tokens=200,
        temperature=0.0,
        top_p=1.0,
        default_value="Subject: \nPredicate: \nObject: ",
    )
    model.end_prompt(session)
    subject = extract_first_match(r"Subject:\s*(.*)", response, "", flags=0) or ""
    predicate = extract_first_match(r"Predicate:\s*(.*)", response, "", flags=0) or ""
    obj = extract_first_match(r"Object:\s*(.*)", response, "", flags=0) or ""
    return subject.strip(), predicate.strip(), obj.strip()


def prompt_text_to_triple(model: ModelWandbWrapper, text: str):
    return asyncio.run(aprompt_text_to_triple(model, text))
