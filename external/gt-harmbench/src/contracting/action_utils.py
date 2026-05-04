"""Shared action-parsing utilities for 4x4 moral hazard scenarios."""

import re

# Matches effort-suffix patterns like "Stay silent (High Effort)"
_EFFORT_SUFFIX_RE = re.compile(r'^(.+?)\s*\(([^)]+)\)\s*$')


def extract_base_action(action: str) -> str:
    """Strip effort-level suffix from a 4x4 action, e.g. 'Stay silent (High Effort)' -> 'Stay silent'."""
    m = _EFFORT_SUFFIX_RE.match(action.strip())
    return m.group(1).strip() if m else action.strip()


def extract_base_actions(actions: list[str]) -> list[str]:
    """Unique base actions preserving order, e.g. ['A (High)', 'A (Low)', 'B (High)'] -> ['A', 'B']."""
    return list(dict.fromkeys(extract_base_action(a) for a in actions))


def map_base_to_full_action(base_action: str, reference_action: str) -> str:
    """Re-attach the effort level from *reference_action* onto *base_action*."""
    m = _EFFORT_SUFFIX_RE.match(reference_action.strip())
    return f"{base_action} ({m.group(2).strip()})" if m else base_action
