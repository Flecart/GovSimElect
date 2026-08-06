"""Memory records and the backend contract used by persona cognition."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol


class MemoryType(str, Enum):
  CHAT = "chat"
  THOUGHT = "thought"
  EVENT = "event"
  ACTION = "action"


@dataclass
class MemoryRecord:
  """A memory awaiting persistence in the configured backend."""

  type: MemoryType
  description: str
  created: datetime
  expiration: datetime
  conversation: list[tuple[str, str]] = field(default_factory=list)
  importance_score: float = 0.0
  always_include: bool = False

  def __str__(self) -> str:
    return self.description


class MemoryBackend(Protocol):
  """Minimal backend surface required by store and retrieval components."""

  def add_chat(
      self,
      subject,
      predicate,
      obj,
      description: str,
      conversation: list[tuple[str, str]],
      created: datetime,
      expiration: datetime,
  ) -> MemoryRecord: ...

  def add_thought(
      self, subject, predicate, obj, description: str, created: datetime,
      expiration: datetime,
  ) -> MemoryRecord: ...

  def add_event(
      self, subject, predicate, obj, description: str, created: datetime,
      expiration: datetime,
  ) -> MemoryRecord: ...

  def add_action(
      self, subject, predicate, obj, description: str, created: datetime,
      expiration: datetime,
  ) -> MemoryRecord: ...

  def store(self, record: MemoryRecord) -> None: ...

  def search(
      self, query: str, top_k: int, current_time: datetime,
  ) -> list[tuple[datetime, str]]: ...

  def save(self) -> None: ...
