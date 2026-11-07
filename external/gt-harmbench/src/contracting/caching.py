"""Provider prompt-caching hints.

OpenAI, DeepSeek, etc. cache shared prompt prefixes automatically (>=1024 tokens). For
OpenAI-family models we also send ``prompt_cache_key`` so requests sharing a prefix (same
scenario + player across negotiation turns / epochs) get routed to the same cache shard.
Measured through the LiteLLM proxy: the key is accepted for GPT models but *reduced* hits
for DeepSeek, so it is only sent to GPT models. This never changes prompt content.

Disable with PROMPT_CACHE_KEY=0.
"""

import os
import re

from inspect_ai.model import GenerateConfig

_GPT_RE = re.compile(r"(^|/)(gpt-|o\d)", re.IGNORECASE)


def _model_name(model) -> str:
    if isinstance(model, str):
        return model
    name = getattr(model, "name", "") or ""
    api = getattr(getattr(model, "api", None), "model_name", "") or ""
    return name or api


def prompt_cache_config(model, key: str, base: GenerateConfig | None = None) -> GenerateConfig:
    """Return a GenerateConfig carrying a prompt_cache_key for GPT models (else `base` unchanged)."""
    base = base or GenerateConfig()
    if os.environ.get("PROMPT_CACHE_KEY", "1") == "0":
        return base
    if not _GPT_RE.search(_model_name(model)):
        return base
    extra = dict(base.extra_body or {})
    extra["prompt_cache_key"] = re.sub(r"[^A-Za-z0-9_.:-]", "_", key)[:64]
    return base.merge(GenerateConfig(extra_body=extra))
