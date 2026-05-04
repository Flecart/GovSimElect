"""Parsing helpers for model responses and action normalization."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def parse_response(answer: str) -> tuple[Optional[str], Optional[str]]:
    """Parse a single-line model answer into raw row/column choices.

    Expected format (from `AGENTS.md`):
    "Row choice: <row_action>, Column choice: <col_action> ..."
    """
    if not answer:
        return None, None

    # Everything that comes after "Row choice:" is the answer section
    row_answer = answer.split("Row choice:")[-1].strip()
    column_split = row_answer.split(", Column choice:")

    row_choice = column_split[0].split("Row choice:")[-1].strip().split("\n")[0]
    column_choice = column_split[-1].strip().split("\n")[0]

    # Remove trailing punctuation / decoration like ".", ";", "*"
    row_choice = row_choice.rstrip(" .;*")
    column_choice = column_choice.rstrip(" .;*")

    return row_choice or None, column_choice or None


def parse_responses(answer: str) -> List[Tuple[Optional[str], Optional[str]]]:
    """Parse possibly multi-line answer into a list of (row_choice, column_choice)."""
    if not answer:
        return []

    choices: List[Tuple[Optional[str], Optional[str]]] = []
    # Each line should look like: "Row choice: X, Column choice: Y"
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        if "Row choice:" not in line:
            continue
        # Reuse the single-line parser for robustness
        row, col = parse_response(line)
        if row or col:
            choices.append((row, col))

    return choices


def parse_actions_from_answer(
    answer_text: str,
    actions_row: Sequence[str],
    actions_column: Sequence[str],
) -> tuple[Optional[str], Optional[str]]:
    """Extract row and column actions from the answer text using `parse_response`.

    This first parses the free-form answer into raw row/column choices
    (as in `AGENTS.md`), then matches them to the canonical entries in
    `actions_row` / `actions_column` in a case-insensitive way.
    """
    row_action: Optional[str] = None
    col_action: Optional[str] = None

    if not answer_text:
        return row_action, col_action

    parsed_row, parsed_col = parse_response(answer_text)

    # Match parsed row choice to canonical row action
    if parsed_row:
        parsed_row_lower = parsed_row.lower()
        # Prefer exact (case-insensitive) matches
        for action in actions_row:
            if action.lower() == parsed_row_lower:
                row_action = action
                break
        # Fallback: allow substring matches if no exact match was found
        if row_action is None:
            for action in actions_row:
                a_low = action.lower()
                if parsed_row_lower in a_low or a_low in parsed_row_lower:
                    row_action = action
                    break

    # Match parsed column choice to canonical column action
    if parsed_col:
        parsed_col_lower = parsed_col.lower()
        # Prefer exact (case-insensitive) matches
        for action in actions_column:
            if action.lower() == parsed_col_lower:
                col_action = action
                break
        # Fallback: allow substring matches if no exact match was found
        if col_action is None:
            for action in actions_column:
                a_low = action.lower()
                if parsed_col_lower in a_low or a_low in parsed_col_lower:
                    col_action = action
                    break

    return row_action, col_action


def standardize_action(
    action: Optional[str],
    action_list: Sequence[str],
    *,
    is_row: bool = True,
) -> Optional[str]:
    """Map action to standardized label: UP/DOWN for row, LEFT/RIGHT for column."""
    if action is None or not action_list:
        return None

    try:
        idx = action_list.index(action)
    except ValueError:
        return None

    if is_row:
        return "UP" if idx == 0 else "DOWN"
    return "LEFT" if idx == 0 else "RIGHT"



