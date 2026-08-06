from datetime import datetime, timedelta

import pytest

from simulation.persona.memory import Mem0Memory, MemoryType


class FakeMem0Client:
  def __init__(self, results=None):
    self.add_calls = []
    self.search_calls = []
    self.results = results or []

  def add(self, messages, **kwargs):
    self.add_calls.append((messages, kwargs))

  def search(self, query, **kwargs):
    self.search_calls.append((query, kwargs))
    return {"results": self.results}


def _memory(tmp_path, client, mode="platform", persona="persona_0"):
  return Mem0Memory(
      str(tmp_path / persona),
      {"mode": mode, "user_id_prefix": "test", "config": {}},
      client_factory=lambda _mode, _config: client,
  )


def test_personas_use_distinct_private_user_ids(tmp_path):
  first_client = FakeMem0Client()
  second_client = FakeMem0Client()
  first = _memory(tmp_path, first_client, persona="persona_0")
  second = _memory(tmp_path, second_client, persona="persona_1")

  assert first.user_id != second.user_id

  now = datetime(2026, 1, 1)
  record = first.add_event(None, None, None, "private catch", now, now + timedelta(days=1))
  record.importance_score = 8
  first.store(record)

  _, kwargs = first_client.add_calls[0]
  assert kwargs["user_id"] == first.user_id
  assert kwargs["metadata"]["type"] == MemoryType.EVENT.value
  assert kwargs["infer"] is False


def test_platform_search_is_filtered_and_drops_chat_and_expired_memory(tmp_path):
  now = datetime(2026, 1, 10)
  client = FakeMem0Client([
      {
          "memory": "relevant private event",
          "metadata": {
              "type": "event",
              "created": (now - timedelta(days=1)).isoformat(),
              "expiration": (now + timedelta(days=1)).isoformat(),
          },
      },
      {
          "memory": "chat summary",
          "metadata": {
              "type": "chat",
              "expiration": (now + timedelta(days=1)).isoformat(),
          },
      },
      {
          "memory": "expired event",
          "metadata": {
              "type": "event",
              "expiration": (now - timedelta(days=1)).isoformat(),
          },
      },
  ])
  memory = _memory(tmp_path, client)

  result = memory.search("lake", top_k=3, current_time=now)

  assert [description for _, description in result] == ["relevant private event"]
  _, kwargs = client.search_calls[0]
  assert kwargs["filters"] == {"user_id": memory.user_id}


def test_unknown_mem0_mode_is_rejected(tmp_path):
  with pytest.raises(ValueError, match="local.*platform"):
    Mem0Memory(str(tmp_path), {"mode": "unknown"})


def test_search_keeps_mem0_relevance_selection_and_prioritizes_required_memory(tmp_path):
  now = datetime(2026, 2, 1)
  future = (now + timedelta(days=1)).isoformat()
  client = FakeMem0Client([
      {
          "memory": "most relevant",
          "metadata": {
              "type": "event",
              "created": (now - timedelta(days=2)).isoformat(),
              "expiration": future,
          },
      },
      {
          "memory": "required context",
          "metadata": {
              "type": "event",
              "created": (now - timedelta(days=3)).isoformat(),
              "expiration": future,
              "always_include": True,
          },
      },
      {
          "memory": "less relevant but newer",
          "metadata": {
              "type": "event",
              "created": (now - timedelta(days=1)).isoformat(),
              "expiration": future,
          },
      },
  ])
  memory = _memory(tmp_path, client)

  result = memory.search("catch", top_k=2, current_time=now)

  assert [description for _, description in result] == [
      "required context",
      "most relevant",
  ]
