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


def get_contract_possibilities(contract_mode: str) -> str:
    if contract_mode == "code_law":
        return "The contract is a natural language contract with an enforcement mechanism in Python code smart contracts."
    elif contract_mode == "code_nl":
        return "The contract is a natural language contract with no enforcement mechanism."
    else:
        raise ValueError(f"Invalid contract mode: {contract_mode}")

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
    formal_contract: str = "",
    contract_mode: str = "",
    debug: bool = False,
):
  del consider_identity_persona
  session = model.start_prompt(
      agent.identity.name, "fishing_cognition_act", "choose_act_options"
  )
  #   contract_mode = {
  mode_text = {
      "code_law": "The constitution describes specific rules for the community. It will be enforced at its best effort.",
      "code_nl": "The constitution describes general rules of the community. It won't be enforced.",
  }
#   svo_prompt, _, leader_prompt = leaders_lib.get_leader_persona_prompts(agent)
  if formal_contract:
    contract_context = (
        "Contracts are renegotiated every month and allow for caps, penalties, and mutual obligations. The constitution is the current agreement between the fishers."
        f"{mode_text[contract_mode]}\n"
        "\n<constitution>\n"
        f" {formal_contract}\n"
        "</constitution>\n"
    )
  else:
    contract_context = ""
#   leader_prompt = ""
#   if leader_agenda == "":
#     leader_prompt = f"\nThe current policy following the mayor's agenda is the following: \n {leader_agenda}\n"
    
  session.add_message("system", get_sytem_prompt(agent.identity))
  session.add_user(
      f"{location_time_info(current_location, current_time)}"
      f"Current constitution: {context}\n"
      f"{contract_context}"
      f"{memory_prompt(agent.identity, memories)}\n"
    #   f"{svo_prompt + chr(10) if svo_prompt else ''}"
    #   f"{leader_prompt + chr(10) if leader_prompt else ''}"
      f"Task: With a fishing range set between {interval[0]}-{interval[-1]}, how many tons of fish would you catch this month? The minimum catchable amount is 1 ton.\n"
      f"You may express the action explicitly as fish({agent.identity.name}, N) where N is a whole number of tons.\n"
      f"{reasoning_steps_prompt()}\n"
      "Return your answer in one of these formats:\n"
      f"Reasoning: ...\nAction: fish({agent.identity.name}, N)\n"
      "or\n"
      "Reasoning: ...\nAnswer: N tons"
  )
  if debug:
    print(f"\n\nCHOOSE AMOUNT PROMPT:\n\n{session._current_prompt()}\n")
  response = await model.acomplete_prompt(
      session,
      default_value="Reasoning: No reasoning available.\nAnswer: 0 tons",
  )
  action_match = extract_first_match(
      rf"Action:\s*fish\(\s*{re.escape(agent.identity.name)}\s*,\s*(\d+)\s*\)",
      response,
      None,
      re.IGNORECASE,
  )
  answer_match = extract_first_match(r"Answer:\s*(\d+)", response, "0", re.IGNORECASE)
  option = int(action_match or answer_match or "0")

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
