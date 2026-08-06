import asyncio
import sys
import warnings
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


class FakeCompletions:
  def __init__(self, fail_provider=False):
    self.fail_provider = fail_provider
    self.calls = []

  async def create(self, **kwargs):
    self.calls.append(kwargs)
    if self.fail_provider and "extra_body" in kwargs:
      raise RuntimeError("provider unavailable")
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
    )


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


def test_openrouter_uses_requested_provider_route():
  completions = FakeCompletions()
  wrapper = ModelWandbWrapper(
      RemoteModelSpec("anthropic/claude-sonnet-4", "OpenRouter"),
      render=False,
      wanbd_logger=SimpleNamespace(),
      temperature=0,
      top_p=1,
      seed=1,
      provider_routing={"order": ["amazon-bedrock"], "allow_fallbacks": False},
  )
  wrapper._get_async_client = lambda: SimpleNamespace(
      chat=SimpleNamespace(completions=completions)
  )

  text, _, _ = asyncio.run(wrapper._achat_completion(
      [{"role": "user", "content": "test"}],
      max_tokens=64,
      temperature=0,
      top_p=1,
  ))

  assert text == "ok"
  assert completions.calls[0]["extra_body"] == {
      "provider": {"order": ["amazon-bedrock"], "allow_fallbacks": False}
  }


def test_openrouter_warns_and_falls_back_without_provider_route():
  completions = FakeCompletions(fail_provider=True)
  wrapper = ModelWandbWrapper(
      RemoteModelSpec("anthropic/claude-sonnet-4", "OpenRouter"),
      render=False,
      wanbd_logger=SimpleNamespace(),
      temperature=0,
      top_p=1,
      seed=1,
      provider_routing={"order": ["amazon-bedrock"], "allow_fallbacks": False},
  )
  wrapper._get_async_client = lambda: SimpleNamespace(
      chat=SimpleNamespace(completions=completions)
  )

  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    text, _, _ = asyncio.run(wrapper._achat_completion(
        [{"role": "user", "content": "test"}],
        max_tokens=64,
        temperature=0,
        top_p=1,
    ))

  assert text == "ok"
  assert len(completions.calls) == 2
  assert "extra_body" in completions.calls[0]
  assert "extra_body" not in completions.calls[1]
  assert wrapper._provider_routing_failed is True
  assert "standard OpenRouter routing" in str(caught[0].message)
