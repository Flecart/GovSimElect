"""Telemetry helpers for contracting model calls.

The inspect_ai model output shape can vary by provider, so these helpers avoid
depending on provider-specific usage objects while keeping the logged schema
stable for downstream analysis.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
)


def extract_model_name(model: Any) -> str | None:
    """Best-effort extraction of a model identifier from an inspect model object."""
    if model is None:
        return None

    for attr in ("name", "model_name", "model"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value

    return str(model) if model else None


def extract_model_usage(output: Any) -> dict[str, Any]:
    """Extract token usage from a model output using common provider aliases."""
    usage = _get_value(output, "usage")
    result: dict[str, Any] = {
        "input_tokens": _coerce_int(
            _first_present(usage, ("input_tokens", "prompt_tokens"))
        ),
        "output_tokens": _coerce_int(
            _first_present(usage, ("output_tokens", "completion_tokens"))
        ),
        "total_tokens": _coerce_int(_first_present(usage, ("total_tokens",))),
        "reasoning_tokens": _coerce_int(_extract_reasoning_tokens(usage)),
    }

    if result["total_tokens"] is None:
        input_tokens = result["input_tokens"] or 0
        output_tokens = result["output_tokens"] or 0
        if input_tokens or output_tokens:
            result["total_tokens"] = input_tokens + output_tokens

    result["has_usage"] = any(result[field] is not None for field in USAGE_FIELDS)
    return result


def record_phase_call(
    *,
    phase: str,
    role: str,
    output: Any = None,
    started_at: float | None = None,
    ended_at: float | None = None,
    model: Any = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a serializable telemetry record for one model call."""
    elapsed_seconds = None
    if started_at is not None and ended_at is not None:
        elapsed_seconds = max(0.0, ended_at - started_at)

    record: dict[str, Any] = {
        "phase": phase,
        "role": role,
        "model": extract_model_name(model) or _get_value(output, "model"),
        "elapsed_seconds": elapsed_seconds,
        "usage": extract_model_usage(output),
        "config": config or {},
        "success": error is None,
    }
    if error is not None:
        record["error"] = error
    if extra:
        record.update(extra)

    return _drop_none(record)


def summarize_phase_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate telemetry calls into per-phase and overall usage totals."""
    by_phase: dict[str, dict[str, Any]] = defaultdict(_empty_summary)
    total = _empty_summary()

    for call in calls:
        phase = str(call.get("phase") or "unknown")
        _add_call(by_phase[phase], call)
        _add_call(total, call)

    return {
        "calls": calls,
        "by_phase": dict(by_phase),
        "total": total,
    }


def _empty_summary() -> dict[str, Any]:
    return {
        "call_count": 0,
        "elapsed_seconds": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "calls_with_usage": 0,
        "errors": 0,
    }


def _add_call(summary: dict[str, Any], call: dict[str, Any]) -> None:
    summary["call_count"] += 1
    summary["elapsed_seconds"] += float(call.get("elapsed_seconds") or 0.0)
    if call.get("success") is False:
        summary["errors"] += 1

    usage = call.get("usage") or {}
    if usage.get("has_usage"):
        summary["calls_with_usage"] += 1

    for field in USAGE_FIELDS:
        summary[field] += int(usage.get(field) or 0)


def _get_value(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _first_present(obj: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _get_value(obj, key)
        if value is not None:
            return value
    return None


def _extract_reasoning_tokens(usage: Any) -> Any:
    direct = _first_present(usage, ("reasoning_tokens",))
    if direct is not None:
        return direct

    details = _first_present(
        usage,
        ("completion_tokens_details", "output_tokens_details"),
    )
    return _first_present(details, ("reasoning_tokens",))


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}
