"""Mem0-backed, per-persona long-term memory."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .base import MemoryRecord, MemoryType


def _plain_config(cfg: Any) -> dict[str, Any]:
  if cfg is None:
    return {}
  try:
    from omegaconf import OmegaConf

    if OmegaConf.is_config(cfg):
      return OmegaConf.to_container(cfg, resolve=True) or {}
  except ImportError:
    pass
  return dict(cfg)


class Mem0Memory:
  """Persist and retrieve memories through Mem0.

  A unique, run-scoped ``user_id`` is derived from each persona's storage path.
  This is the privacy boundary: every Mem0 operation is filtered to that id.
  """

  def __init__(
      self,
      base_path: str,
      cfg: Any,
      client_factory: Callable[[str, dict[str, Any]], Any] | None = None,
  ) -> None:
    config = _plain_config(cfg)
    self.mode = str(config.get("mode", "local")).lower()
    self.infer = bool(config.get("infer", False))
    self.search_limit_multiplier = int(config.get("search_limit_multiplier", 4))
    prefix = str(config.get("user_id_prefix", "govsim"))
    path_digest = hashlib.sha256(
        str(Path(base_path).resolve()).encode("utf-8")
    ).hexdigest()[:16]
    self.user_id = f"{prefix}:{Path(base_path).name}:{path_digest}"
    self._always_include: list[MemoryRecord] = []

    factory = client_factory or self._create_client
    self.client = factory(self.mode, _plain_config(config.get("config")))

  @staticmethod
  def _create_client(mode: str, config: dict[str, Any]):
    try:
      if mode == "platform":
        from mem0 import MemoryClient

        return MemoryClient(**config)
      if mode == "local":
        from mem0 import Memory

        return Memory.from_config(config) if config else Memory()
    except ImportError as exc:
      raise RuntimeError(
          "Mem0 memory requires the 'mem0ai' package. "
          "Install dependencies from requirements.txt."
      ) from exc
    raise ValueError("agent.memory.mode must be 'local' or 'platform'")

  @staticmethod
  def _record(
      memory_type: MemoryType,
      description: str,
      created: datetime,
      expiration: datetime,
      conversation: list[tuple[str, str]] | None = None,
  ) -> MemoryRecord:
    return MemoryRecord(
        type=memory_type,
        description=description,
        created=created,
        expiration=expiration,
        conversation=conversation or [],
    )

  def add_chat(
      self, subject, predicate, obj, description, conversation, created, expiration
  ) -> MemoryRecord:
    return self._record(
        MemoryType.CHAT, description, created, expiration, conversation
    )

  def add_thought(
      self, subject, predicate, obj, description, created, expiration
  ) -> MemoryRecord:
    return self._record(MemoryType.THOUGHT, description, created, expiration)

  def add_event(
      self, subject, predicate, obj, description, created, expiration
  ) -> MemoryRecord:
    return self._record(MemoryType.EVENT, description, created, expiration)

  def add_action(
      self, subject, predicate, obj, description, created, expiration
  ) -> MemoryRecord:
    return self._record(MemoryType.ACTION, description, created, expiration)

  def store(self, record: MemoryRecord) -> None:
    metadata = {
        "type": record.type.value,
        "created": record.created.isoformat(),
        "expiration": record.expiration.isoformat(),
        "importance": record.importance_score,
        "always_include": record.always_include,
    }
    self.client.add(
        [{"role": "user", "content": record.description}],
        user_id=self.user_id,
        metadata=metadata,
        infer=self.infer,
    )
    if record.always_include:
      self._always_include.append(record)

  @staticmethod
  def _results(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, dict):
      response = response.get("results", response.get("memories", []))
    return response if isinstance(response, list) else []

  @staticmethod
  def _parse_datetime(value: Any, fallback: datetime) -> datetime:
    if not value:
      return fallback
    try:
      return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(
          tzinfo=None
      )
    except ValueError:
      return fallback

  def search(
      self, query: str, top_k: int, current_time: datetime
  ) -> list[tuple[datetime, str]]:
    search_limit = max(top_k * self.search_limit_multiplier, top_k)
    response = self.client.search(
        query, filters={"user_id": self.user_id}, top_k=search_limit
    )

    always_include: list[tuple[datetime, str]] = []
    ranked: list[tuple[datetime, str]] = []
    for item in self._results(response):
      metadata = item.get("metadata") or {}
      if metadata.get("type") == MemoryType.CHAT.value:
        continue
      expiration = self._parse_datetime(metadata.get("expiration"), current_time)
      if expiration <= current_time:
        continue
      created = self._parse_datetime(metadata.get("created"), current_time)
      content = item.get("memory") or item.get("text")
      if content:
        value = (created, str(content))
        target = always_include if metadata.get("always_include") else ranked
        if value not in target:
          target.append(value)

    for record in self._always_include:
      value = (record.created, record.description)
      if record.expiration > current_time and value not in always_include:
        always_include.append(value)

    selected = always_include[:top_k]
    selected.extend(
        value for value in ranked if value not in selected
    )
    selected = selected[:top_k]
    return sorted(selected, key=lambda item: item[0])

  def save(self) -> None:
    """Mem0 persists on each add; retained for the simulation lifecycle API."""


def build_memory(base_path: str, cfg: Any) -> Mem0Memory:
  config = _plain_config(cfg)
  provider = str(config.get("provider", "mem0")).lower()
  if provider != "mem0":
    raise ValueError("agent.memory.provider must be 'mem0'")
  return Mem0Memory(base_path, config)
