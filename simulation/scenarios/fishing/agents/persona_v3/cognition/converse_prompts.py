"""Conversation prompts and responses for the fishing personas."""

import asyncio
from datetime import datetime
import os
import re

from simulation.persona import PersonaAgent
from simulation.scenarios.fishing.agents.persona_v3.cognition import leaders as leaders_lib
from simulation.utils import ModelWandbWrapper

from .utils import COGNITION_RESPONSES_JSON
from .utils import conversation_to_string_with_dash
from .utils import extract_first_match
from .utils import get_sytem_prompt
from .utils import list_to_comma_string
from .utils import location_time_info
from .utils import log_to_file
from .utils import memory_prompt


async def aprompt_converse_utterance_in_group(
    model: ModelWandbWrapper,
    init_persona: PersonaAgent,
    target_personas: list[PersonaAgent],
    init_retrieved_memory: list[str],
    current_location: str,
    current_time: datetime,
    current_conversation: list[tuple[str, str]],
    debug: bool = False,
) -> tuple[str, bool, str]:
  session = model.start_prompt(
      init_persona.identity.name, "cognition_converse", "converse_utterance"
  )
  svo_prompt, disinfo_prompt, leader_prompt = leaders_lib.get_leader_persona_prompts(
      init_persona
  )
  session.add_message("system", get_sytem_prompt(init_persona.identity))
  session.add_user(
      f"{location_time_info(current_location, current_time)}"
      f"{memory_prompt(init_persona.identity, init_retrieved_memory)}\n"
      "Scenario:"
      f" {list_to_comma_string([t.identity.name for t in target_personas])}"
      " are engaged in a group chat.\n"
      f"{svo_prompt + chr(10) if svo_prompt else ''}"
      f"{disinfo_prompt}\n"
      f"{leader_prompt}\n"
      "\nConversation so far:\n"
      f"{conversation_to_string_with_dash(current_conversation)}\n\n"
      "Task: What would you say next in the group chat? Ensure the conversation flows naturally and avoids repetition.\n"
      "Determine whether your response concludes the conversation. If not, identify the next speaker.\n"
      "Return exactly in this format:\n"
      "Response: ...\n"
      "Conversation conclusion by me: yes/no\n"
      "Next speaker: NAME or N/A\n"
  )
  if debug:
    print(f"\n\nCONVERSE PROMPT:\n\n{session._current_prompt()}\n")
  response = await model.acomplete_prompt(
      session,
      default_value=(
          "Response: I think we should be careful and keep the lake sustainable.\n"
          "Conversation conclusion by me: yes\n"
          "Next speaker: N/A"
      ),
  )
  utterance = extract_first_match(
      r"Response:\s*(.*?)(?:\nConversation conclusion by me:|\Z)",
      response,
      "",
      re.DOTALL | re.IGNORECASE,
  )
  utterance_ended = (
      extract_first_match(
          r"Conversation conclusion by me:\s*(yes|no)",
          response,
          "yes",
          re.IGNORECASE,
      ).lower()
      == "yes"
  )
  next_speaker = None
  if not utterance_ended:
      options = [t.identity.name for t in target_personas]
      next_speaker = extract_first_match(
          rf"Next speaker:\s*({'|'.join(re.escape(option) for option in options)})",
          response,
          options[0],
          re.IGNORECASE,
      )

  response_log_path = os.path.join(
      init_persona.experiment_storage, COGNITION_RESPONSES_JSON
  )
  log_to_file(
      log_type="converse_response",
      data={
          "speaker": init_persona.identity.name,
          "svo": init_persona.svo_type.value,
          "utterance": utterance,
          "utterance_ended": utterance_ended,
          "next_speaker": next_speaker,
      },
      log_path=response_log_path,
  )
  if debug:
    print(
        f"\n\nCONVERSE RESPONSE:\n\n{utterance}\nIS ENDED?"
        f" {utterance_ended}\nNEXT SPEAKER: {next_speaker}\n"
    )

  model.end_prompt(session)
  return utterance, utterance_ended, next_speaker, session.html()


def prompt_converse_utterance_in_group(*args, **kwargs):
  return asyncio.run(aprompt_converse_utterance_in_group(*args, **kwargs))


async def aprompt_summarize_conversation_in_one_sentence(
    model: ModelWandbWrapper,
    conversation: list[tuple[str, str]],
):
  session = model.start_prompt(
      "framework",
      "cognition_converse",
      "prompt_summarize_conversation_in_one_sentence",
  )
  session.add_user(
      "Conversation:\n"
      f"{conversation_to_string_with_dash(conversation)}\n\n"
      "Summarize the conversation above in one sentence."
  )
  response = await model.acomplete_prompt(
      session,
      max_tokens=120,
      default_value="The group discussed sustainable fishing and possible catch limits.",
  )
  summary = response.strip().splitlines()[0].strip()
  if not summary.endswith("."):
      summary += "."
  model.end_prompt(session)
  return summary, session.html()


def prompt_summarize_conversation_in_one_sentence(*args, **kwargs):
  return asyncio.run(aprompt_summarize_conversation_in_one_sentence(*args, **kwargs))
