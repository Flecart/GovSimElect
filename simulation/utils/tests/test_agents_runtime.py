import asyncio
import sys
from types import SimpleNamespace

from simulation.utils.models import ModelWandbWrapper, RemoteModelSpec


class FakeRunner:
  call = None

  @classmethod
  async def run(cls, agent, input, max_turns):
    cls.call = (agent, input, max_turns)
    return SimpleNamespace(
        final_output="agent response STOP ignored",
        context_wrapper=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=11, output_tokens=3)
        ),
    )


class FakeAgent:
  def __init__(self, **kwargs):
    self.kwargs = kwargs


class FakeModelSettings:
  def __init__(self, **kwargs):
    self.kwargs = kwargs


def test_openai_agents_runtime_uses_sdk(monkeypatch):
  monkeypatch.setitem(
      sys.modules,
      "agents",
      SimpleNamespace(
          Agent=FakeAgent,
          ModelSettings=FakeModelSettings,
          Runner=FakeRunner,
      ),
  )
  wrapper = ModelWandbWrapper(
      RemoteModelSpec("gpt-test", "OpenAI"),
      render=False,
      wanbd_logger=SimpleNamespace(),
      temperature=0.2,
      top_p=0.9,
      seed=1,
      runtime="openai_agents",
  )

  text, input_tokens, output_tokens = asyncio.run(
      wrapper._achat_completion(
          [
              {"role": "system", "content": "simulation rules"},
              {"role": "user", "content": "choose a catch"},
          ],
          max_tokens=50,
          temperature=0.2,
          top_p=0.9,
          stop="STOP",
      )
  )

  assert text == "agent response "
  assert (input_tokens, output_tokens) == (11, 3)
  agent, agent_input, max_turns = FakeRunner.call
  assert agent.kwargs["instructions"] == "simulation rules"
  assert agent.kwargs["model"] == "gpt-test"
  assert agent_input == [{"role": "user", "content": "choose a catch"}]
  assert max_turns == 1
