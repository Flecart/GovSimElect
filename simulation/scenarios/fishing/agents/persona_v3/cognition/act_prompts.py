"""Acting prompts and responses for the fishing personas."""

import asyncio
from datetime import datetime
import os
import re

from simulation.persona import PersonaAgent
from simulation.scenarios.fishing.agents.persona_v3.cognition import leaders as leaders_lib
from simulation.utils import ModelWandbWrapper

from .utils import COGNITION_RESPONSES_JSON
from .utils import extract_first_match
from .utils import get_sytem_prompt
from .utils import location_time_info
from .utils import log_to_file
from .utils import memory_prompt
from .utils import reasoning_steps_prompt


async def aprompt_action_choose_amount_of_fish_to_catch(
    model: ModelWandbWrapper,
    agent: PersonaAgent,
    memories: list[str],
    current_location: str,
    current_time: datetime,
    context: str,
    interval: list[int],
    consider_identity_persona: bool = True,
    leader_agenda: str = "",
    debug: bool = False,
):
  del consider_identity_persona
  session = model.start_prompt(
      agent.identity.name, "fishing_cognition_act", "choose_act_options"
  )
  svo_prompt, _, leader_prompt = leaders_lib.get_leader_persona_prompts(agent)
  session.add_user(
      f"{get_sytem_prompt(agent.identity)}\n"
      f"{location_time_info(current_location, current_time)}"
      f"Current context: {context}\n"
      "\nThe current policy following the mayor's agenda is the following:"
      f" {leader_agenda}\n"
      f"{memory_prompt(agent.identity, memories)}\n"
      f"{svo_prompt + chr(10) if svo_prompt else ''}"
      f"{leader_prompt + chr(10) if leader_prompt else ''}"
      f"Task: With a fishing range set between {interval[0]}-{interval[-1]}, how many tons of fish would you catch this month?\n"
      f"{reasoning_steps_prompt()}\n"
      'Return your answer in this format:\nReasoning: ...\nAnswer: N tons'
  )
  if debug:
    print(f"\n\nCHOOSE AMOUNT PROMPT:\n\n{session._current_prompt()}\n")
  response = await model.acomplete_prompt(
      session,
      default_value="Reasoning: No reasoning available.\nAnswer: 0 tons",
  )
  option = int(extract_first_match(r"Answer:\s*(\d+)", response, "0", re.IGNORECASE))

  response_log_path = os.path.join(agent.experiment_storage, COGNITION_RESPONSES_JSON)
  log_to_file(
      log_type="action_response",
      data={
          "speaker": agent.identity.name,
          "svo": agent.svo_type.value,
          "reasoning": response,
          "option": option,
      },
      log_path=response_log_path,
  )
  if debug:
    print(f"\n\nCHOOSE AMOUNT RESPONSE:\n\n{response}\nCATCH: {option}")

  model.end_prompt(session)
  return option, session.html()


def prompt_action_choose_amount_of_fish_to_catch(*args, **kwargs):
  return asyncio.run(aprompt_action_choose_amount_of_fish_to_catch(*args, **kwargs))


async def aprompt_election_vote(
    model: ModelWandbWrapper,
    agent: PersonaAgent,
    memories: list[str],
    current_location: str,
    current_time: str,
    candidates: list[str],
    issues: dict[str, str],
    debug: bool = False,
) -> tuple[str, str]:
  del current_location, current_time
  session = model.start_prompt(
      agent.identity.name, "fishing_election", "vote_decision"
  )
  _, _, leader_prompt = leaders_lib.get_leader_persona_prompts(agent)
  candidate_block = "".join(f"- {candidate}: {issues[candidate]}\n" for candidate in candidates)
  extra = ""
  if leader_prompt:
    extra += f"Coming into the election, {leader_prompt}\n"
    extra += f"Leader agenda from last election: {agent.agenda}\n"
    extra += f"Harvest report from last cycle: {agent.harvest_report}\n"
  else:
    extra += "There is no current leader.\n"
  session.add_user(
      f"{get_sytem_prompt(agent.identity)}\n"
      f"{memory_prompt(agent.identity, memories)}"
      f"{extra}"
      "Task: Select a mayor for the fisheries union among the listed candidates and provide your rationale.\n"
      "Candidate positions:\n"
      f"{candidate_block}"
      f"{reasoning_steps_prompt()}\n"
      f"Return your answer in this format:\nReasoning: ...\nVote: one of {', '.join(candidates)}"
  )
  if debug:
    print(f"\n\nVOTE PROMPT:\n\n{session._current_prompt()}\n")
  response = await model.acomplete_prompt(
      session,
      default_value=f"Reasoning: No clear preference.\nVote: {candidates[0]}",
  )
  vote = extract_first_match(
      rf"Vote:\s*({'|'.join(re.escape(candidate) for candidate in candidates)})",
      response,
      "none",
      re.IGNORECASE,
  )
  if vote is None:
    vote = "none"

  response_log_path = os.path.join(agent.experiment_storage, COGNITION_RESPONSES_JSON)
  log_to_file(
      log_type="vote_response",
      data={
          "speaker": agent.identity.name,
          "svo": agent.svo_type.value,
          "reasoning": response,
          "option": vote,
      },
      log_path=response_log_path,
  )
  if debug:
    print(f"\n\nVOTE RESPONSE:\n\n{response}\n")

  model.end_prompt(session)
  return vote, session.html()


def prompt_election_vote(*args, **kwargs):
  return asyncio.run(aprompt_election_vote(*args, **kwargs))
