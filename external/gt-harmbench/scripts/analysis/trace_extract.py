"""Extract GT-HarmBench contracting traces as structured JSON.

This utility is intended for LLM agents that need to inspect contracting traces
without hand-parsing .eval archives or dumping full logs into context.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.analysis.contracting import get_contracting_score_metadata

PROMPT_MODES = {"base", "selfish", "cooperative"}
DEFAULT_FIELDS = ("source", "metadata", "contracts", "enforcement")
VALID_PHASES = {"negotiation", "coding", "system", "unknown", "all"}
NO_ENFORCEMENT_PLACEHOLDER: dict[str, Any] = {
    "summary": "No enforcement enacted.",
    "detail": (
        "Contract enforcement did not run for this sample (typically no binding "
        "contract was formed, or enforcement_result is absent or empty)."
    ),
}


VALID_FIELDS = {
    "source",
    "metadata",
    "game_context",
    "contracts",
    "proposals",
    "system_feedback",
    "enforcement",
    "payoffs",
    "usage",
    "reasoning",
    "raw",
    "all",
}


class TraceExtractError(Exception):
    """User-facing extraction error."""


@dataclass(frozen=True)
class TraceRecord:
    """A single trace instance from one sample file inside one .eval archive."""

    trace_id: str
    experiment_dir: Path
    prompt_mode: str
    contract_mode: str
    eval_file: Path
    sample_file: str
    sample: dict[str, Any]
    sample_index: int


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "list":
            output = command_list(args)
        elif args.command == "show":
            output = command_show(args)
        else:
            raise TraceExtractError(f"Unknown command: {args.command}")
    except TraceExtractError as exc:
        print_json({"ok": False, "error": str(exc)})
        return 2

    print_json(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract GT-HarmBench contracting traces as JSON.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List individual trace instances.")
    list_parser.add_argument("experiment_dir", type=Path)
    list_parser.add_argument("--prompt-mode")
    list_parser.add_argument("--contract-mode")
    list_parser.add_argument("--sample-id")
    list_parser.add_argument("--search")
    list_parser.add_argument("--limit", type=int, default=50)

    show_parser = subparsers.add_parser("show", help="Show one trace instance.")
    show_parser.add_argument("experiment_dir", type=Path)
    show_parser.add_argument("--trace-id", required=True)
    show_parser.add_argument("--phases")
    show_parser.add_argument("--fields")
    show_parser.add_argument("--include-raw", action="store_true")
    show_parser.add_argument("--include-reasoning", action="store_true", help="Include model reasoning traces (if available)")
    show_parser.add_argument("--search")

    return parser


def print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def command_list(args: argparse.Namespace) -> dict[str, Any]:
    pattern = compile_search(args.search)
    if args.limit is not None and args.limit < 1:
        raise TraceExtractError("--limit must be positive")

    records = []
    for record in iter_trace_records(
        args.experiment_dir,
        prompt_mode=args.prompt_mode,
        contract_mode=args.contract_mode,
        sample_id=args.sample_id,
    ):
        trace = build_trace(record, include_raw=False)
        matches = find_matches(trace, pattern) if pattern else []
        if pattern and not matches:
            continue

        item = build_list_item(trace)
        if matches:
            item["matches"] = matches
        records.append(item)

        if args.limit and len(records) >= args.limit:
            break

    return {
        "ok": True,
        "command": "list",
        "experiment_dir": str(args.experiment_dir),
        "count": len(records),
        "limit": args.limit,
        "traces": records,
    }


def command_show(args: argparse.Namespace) -> dict[str, Any]:
    if args.fields and args.phases:
        raise TraceExtractError("--fields and --phases are mutually exclusive")

    fields = parse_csv_arg(args.fields, VALID_FIELDS, "--fields") if args.fields else list(DEFAULT_FIELDS)
    phases = parse_csv_arg(args.phases, VALID_PHASES, "--phases") if args.phases else None
    pattern = compile_search(args.search)

    if fields and "events" in fields:
        raise TraceExtractError("events is not a valid --fields value; use --phases")
    if args.include_raw and not phases and "raw" not in fields:
        fields.append("raw")
    if args.include_reasoning and not phases and "reasoning" not in fields:
        fields.append("reasoning")

    records = [
        record
        for record in iter_trace_records(args.experiment_dir)
        if record.trace_id == args.trace_id
    ]
    if not records:
        raise TraceExtractError(f"No trace found for trace_id: {args.trace_id}")
    if len(records) > 1:
        raise TraceExtractError(f"Multiple traces matched trace_id: {args.trace_id}")

    trace = build_trace(records[0], include_raw=args.include_raw or "raw" in fields)
    matches = find_matches(trace, pattern) if pattern else []
    if pattern and not matches:
        return {
            "ok": True,
            "command": "show",
            "trace_id": args.trace_id,
            "matched": False,
            "source": trace["source"],
            "metadata": trace["metadata"],
            "matches": [],
        }

    if phases:
        result = select_phases(trace, phases, pattern)
    else:
        result = select_fields(trace, fields, pattern)

    if pattern:
        result["matches"] = matches
        result["matched"] = bool(matches)

    return {
        "ok": True,
        "command": "show",
        "trace": result,
    }


def iter_trace_records(
    experiment_dir: Path,
    *,
    prompt_mode: str | None = None,
    contract_mode: str | None = None,
    sample_id: str | None = None,
) -> Iterable[TraceRecord]:
    experiment_dir = experiment_dir.expanduser()
    if not experiment_dir.exists() or not experiment_dir.is_dir():
        raise TraceExtractError(f"Experiment directory does not exist: {experiment_dir}")

    for prompt_dir in sorted(p for p in experiment_dir.iterdir() if p.is_dir()):
        if prompt_dir.name not in PROMPT_MODES:
            continue
        if prompt_mode and prompt_dir.name != prompt_mode:
            continue

        for mode_dir in sorted(p for p in prompt_dir.iterdir() if p.is_dir()):
            if contract_mode and mode_dir.name != contract_mode:
                continue

            for eval_file in sorted(mode_dir.glob("*.eval")):
                yield from iter_eval_samples(
                    experiment_dir=experiment_dir,
                    prompt_mode=prompt_dir.name,
                    contract_mode=mode_dir.name,
                    eval_file=eval_file,
                    sample_id=sample_id,
                )


def iter_eval_samples(
    *,
    experiment_dir: Path,
    prompt_mode: str,
    contract_mode: str,
    eval_file: Path,
    sample_id: str | None,
) -> Iterable[TraceRecord]:
    try:
        with zipfile.ZipFile(eval_file, "r") as zf:
            sample_files = sorted(
                name
                for name in zf.namelist()
                if name.startswith("samples/") and name.endswith(".json")
            )
            for sample_index, sample_file in enumerate(sample_files):
                with zf.open(sample_file) as handle:
                    sample = json.load(handle)

                input_meta = sample.get("metadata", {})
                current_sample_id = str(input_meta.get("id", sample.get("id", "")))
                if sample_id and current_sample_id != str(sample_id):
                    continue

                trace_id = make_trace_id(
                    prompt_mode=prompt_mode,
                    contract_mode=contract_mode,
                    sample_id=current_sample_id,
                    sample_file=sample_file,
                    eval_file=eval_file,
                )
                yield TraceRecord(
                    trace_id=trace_id,
                    experiment_dir=experiment_dir,
                    prompt_mode=prompt_mode,
                    contract_mode=contract_mode,
                    eval_file=eval_file,
                    sample_file=sample_file,
                    sample=sample,
                    sample_index=sample_index,
                )
    except zipfile.BadZipFile as exc:
        raise TraceExtractError(f"Invalid .eval archive: {eval_file}") from exc
    except json.JSONDecodeError as exc:
        raise TraceExtractError(f"Invalid JSON in {eval_file}") from exc


def make_trace_id(
    *,
    prompt_mode: str,
    contract_mode: str,
    sample_id: str,
    sample_file: str,
    eval_file: Path,
) -> str:
    sample_stem = Path(sample_file).stem
    eval_stem = eval_file.stem
    parts = [prompt_mode, contract_mode, safe_id(sample_id or "unknown"), sample_stem, eval_stem]
    return "/".join(safe_id(part) for part in parts)


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value).strip())
    return cleaned.strip("_") or "unknown"


def extract_reasoning_from_output(output: dict[str, Any]) -> dict[str, Any]:
    """Extract reasoning content from model output.

    Args:
        output: Sample output dict with choices containing message content

    Returns:
        Dict with 'row_player' and 'col_player' reasoning info
        Each has: 'text' (decoded reasoning or None), 'summary', 'redacted',
                  'encoded_length', 'has_reasoning' (bool)
    """
    result = {
        "row_player": {"has_reasoning": False, "text": None, "summary": None, "redacted": None, "encoded_length": 0},
        "col_player": {"has_reasoning": False, "text": None, "summary": None, "redacted": None, "encoded_length": 0},
    }

    choices = output.get("choices", [])
    if not choices:
        return result

    for i, choice in enumerate(choices):
        if i >= 2:
            break
        player_key = "row_player" if i == 0 else "col_player"
        message = choice.get("message", {})
        content = message.get("content")

        if not content:
            continue

        reasoning_info = result[player_key]

        # Handle list format (ContentReasoning + ContentText)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "reasoning":
                    reasoning_info["has_reasoning"] = True
                    reasoning_text = item.get("reasoning", "")
                    reasoning_info["encoded_length"] = len(reasoning_text)
                    reasoning_info["summary"] = item.get("summary")
                    reasoning_info["redacted"] = item.get("redacted", False)

                    # Try to extract reasoning text (may be base64 encoded)
                    if reasoning_text and not item.get("redacted", False):
                        reasoning_info["text"] = reasoning_text
                    # If redacted, use summary if available
                    elif item.get("summary"):
                        reasoning_info["text"] = f"[Summary]: {item.get('summary')}"

                # Also handle object format (from inspect_ai)
                elif hasattr(item, "type") and item.type == "reasoning":
                    reasoning_info["has_reasoning"] = True
                    if hasattr(item, "reasoning") and item.reasoning:
                        reasoning_info["encoded_length"] = len(item.reasoning)
                        reasoning_info["text"] = item.reasoning
                    if hasattr(item, "summary"):
                        reasoning_info["summary"] = item.summary
                    if hasattr(item, "redacted"):
                        reasoning_info["redacted"] = item.redacted

    return result


def build_trace(record: TraceRecord, *, include_raw: bool) -> dict[str, Any]:
    sample = record.sample
    input_meta = sample.get("metadata", {})
    score_meta = get_contracting_score_metadata(sample)
    negotiation_result = input_meta.get("negotiation_result") or {}
    enforcement_result = input_meta.get("enforcement_result") or {}
    contract_data = negotiation_result.get("contract") or {}
    conversations = negotiation_result.get("conversations") or []
    events = [
        normalize_event(event, include_raw=include_raw)
        for event in conversations
        if isinstance(event, dict)
    ]

    metadata = build_metadata(input_meta, score_meta, negotiation_result)
    final_contract = (
        score_meta.get("contract_text")
        or contract_data.get("content")
        or negotiation_result.get("contract_text")
        or ""
    )

    # Extract reasoning from model output (if available)
    reasoning = extract_reasoning_from_output(sample.get("output", {}))

    trace = {
        "trace_id": record.trace_id,
        "source": {
            "experiment": record.experiment_dir.name,
            "prompt_mode": record.prompt_mode,
            "contract_mode": record.contract_mode,
            "eval_file": str(record.eval_file.relative_to(record.experiment_dir)),
            "sample_file": record.sample_file,
            "sample_index": record.sample_index,
        },
        "metadata": metadata,
        "game_context": {
            "formal_game": input_meta.get("formal_game", ""),
            "story_row": input_meta.get("story_row", ""),
            "story_col": input_meta.get("story_col", ""),
            "actions_row": input_meta.get("actions_row", []),
            "actions_column": input_meta.get("actions_column", []),
            "rewards_matrix": input_meta.get("rewards_matrix", []),
        },
        "contracts": {
            "final_contract": final_contract,
            "natural_language_contract": (contract_data.get("metadata") or {}).get("nl_contract"),
            "contract_type": contract_data.get("contract_type"),
            "proposer": contract_data.get("proposer"),
            "agreement_round": contract_data.get("agreement_round"),
            "proposals": build_proposals(events),
        },
        "events": events,
        "enforcement": build_enforcement(enforcement_result, score_meta),
        "payoffs": build_payoffs(score_meta),
        "usage": build_usage(input_meta, score_meta),
        "reasoning": reasoning,
    }

    if include_raw:
        trace["raw"] = {
            "sample": sample,
            "negotiation_result": negotiation_result,
            "enforcement_result": enforcement_result,
            "score_metadata": score_meta,
        }

    return trace


def build_metadata(
    input_meta: dict[str, Any],
    score_meta: dict[str, Any],
    negotiation_result: dict[str, Any],
) -> dict[str, Any]:
    contract_complied = score_meta.get("contract_complied")
    contract_activated = score_meta.get(
        "contract_activated",
        None if contract_complied is None else not contract_complied,
    )
    return {
        "sample_id": str(input_meta.get("id", "")),
        "base_scenario_id": str(
            input_meta.get("base_scenario_id") or input_meta.get("id", "")
        ),
        "formal_game": input_meta.get("formal_game", ""),
        "is_4x4": input_meta.get("is_4x4"),
        "contract_formed": score_meta.get(
            "contract_formed",
            negotiation_result.get("agreement_reached", False),
        ),
        "contract_complied": contract_complied,
        "contract_activated": contract_activated,
        "turns_to_agreement": score_meta.get(
            "turns_to_agreement",
            negotiation_result.get("turns_taken"),
        ),
        "formation_failure_reason": score_meta.get("formation_failure_reason"),
        "compliance_failure_reason": score_meta.get("compliance_failure_reason"),
        "row_action": score_meta.get("row_action"),
        "column_action": score_meta.get("column_action"),
        "row_effort_level": score_meta.get("row_effort_level"),
        "col_effort_level": score_meta.get("col_effort_level"),
        "row_action_category": score_meta.get("row_action_category"),
        "col_action_category": score_meta.get("col_action_category"),
    }


def normalize_event(event: dict[str, Any], *, include_raw: bool) -> dict[str, Any]:
    event_type = classify_event(event)
    speaker = event.get("player") or event.get("agent")
    normalized = {
        "phase": event_type,
        "turn": event.get("turn"),
        "speaker": speaker,
        "action": event.get("action"),
        "subphase": event.get("phase"),
        "contract_text": event.get("contract_text"),
        "reasoning": event.get("reasoning"),
        "message": event_message(event),
        "error": event.get("error"),
        "error_message": event.get("error_message"),
        "telemetry": event.get("telemetry"),
    }
    if include_raw:
        normalized["raw"] = event
    return drop_empty(normalized)


def classify_event(event: dict[str, Any]) -> str:
    phase = event.get("phase") or ""
    action = event.get("action") or ""
    player = event.get("player") or ""

    if player == "system" or action == "CODING_FAILED" or "feedback" in phase:
        return "system"
    if phase == "coding_agent_translation" or player == "coding_agent":
        return "coding"
    if action in ("PROPOSE", "ACCEPT"):
        return "negotiation"
    return "unknown"


def event_message(event: dict[str, Any]) -> str:
    return (
        event.get("message")
        or event.get("raw_message")
        or event.get("reasoning")
        or ""
    )


def build_proposals(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    proposals = []
    for event in events:
        if event.get("phase") == "negotiation" and event.get("action") in ("PROPOSE", "ACCEPT"):
            proposals.append(drop_empty({
                "turn": event.get("turn"),
                "speaker": event.get("speaker"),
                "action": event.get("action"),
                "contract_text": event.get("contract_text"),
                "reasoning": event.get("reasoning"),
            }))
    return proposals


def build_enforcement(
    enforcement_result: dict[str, Any],
    score_meta: dict[str, Any],
) -> dict[str, Any]:
    enforcement = {
        "success": enforcement_result.get("success"),
        "reasoning": enforcement_result.get("reasoning"),
        "modified_actions": enforcement_result.get("modified_actions"),
        "violations_detected": enforcement_result.get("violations_detected", []),
        "execution_log": [
            str(item).strip()
            for item in enforcement_result.get("execution_log", [])
            if str(item).strip()
        ],
        "payoff_adjustments": (
            enforcement_result.get("payoff_adjustments")
            or score_meta.get("payoff_adjustments")
            or {}
        ),
        "metadata": enforcement_result.get("metadata"),
    }
    cleaned = drop_empty(enforcement)
    return cleaned if cleaned else dict(NO_ENFORCEMENT_PLACEHOLDER)


def build_payoffs(score_meta: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "row_payoff",
        "col_payoff",
        "utilitarian_payoff",
        "rawlsian_payoff",
        "nash_social_welfare",
        "payoff_adjustments",
    )
    return {key: score_meta[key] for key in keys if key in score_meta}


def build_usage(input_meta: dict[str, Any], score_meta: dict[str, Any]) -> dict[str, Any]:
    usage = input_meta.get("trace_usage") or score_meta.get("trace_usage")
    if isinstance(usage, dict):
        return usage
    return {}


def build_list_item(trace: dict[str, Any]) -> dict[str, Any]:
    events = trace["events"]
    system_events = [event for event in events if event.get("phase") == "system"]
    metadata = trace["metadata"]

    enf = trace["enforcement"]
    failure_flags = {
        "contract_not_formed": metadata.get("contract_formed") is False,
        "contract_activated": metadata.get("contract_activated") is True,
        "system_feedback": bool(system_events),
        "enforcement_failed": enf.get("success") is False,
        "violations_detected": bool(enf.get("violations_detected")),
    }

    return {
        "trace_id": trace["trace_id"],
        "source": trace["source"],
        "sample_id": metadata.get("sample_id"),
        "formal_game": metadata.get("formal_game"),
        "contract_formed": metadata.get("contract_formed"),
        "contract_complied": metadata.get("contract_complied"),
        "contract_activated": metadata.get("contract_activated"),
        "turns_to_agreement": metadata.get("turns_to_agreement"),
        "event_counts": count_by_phase(events),
        "failure_flags": failure_flags,
        "label": make_label(trace, failure_flags),
    }


def make_label(trace: dict[str, Any], failure_flags: dict[str, bool]) -> str:
    metadata = trace["metadata"]
    flags = [name for name, value in failure_flags.items() if value]
    suffix = f" flags={','.join(flags)}" if flags else ""
    return (
        f"{metadata.get('sample_id')} | {metadata.get('formal_game')} | "
        f"formed={metadata.get('contract_formed')} activated={metadata.get('contract_activated')}{suffix}"
    )


def count_by_phase(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        phase = event.get("phase", "unknown")
        counts[phase] = counts.get(phase, 0) + 1
    return counts


def select_phases(
    trace: dict[str, Any],
    phases: list[str],
    pattern: re.Pattern[str] | None,
) -> dict[str, Any]:
    include_all = "all" in phases
    selected_events = [
        event
        for event in trace["events"]
        if include_all or event.get("phase") in phases
    ]
    if pattern:
        selected_events = [event for event in selected_events if object_matches(event, pattern)]

    return {
        "trace_id": trace["trace_id"],
        "source": trace["source"],
        "metadata": trace["metadata"],
        "events": selected_events,
    }


def select_fields(
    trace: dict[str, Any],
    fields: list[str],
    pattern: re.Pattern[str] | None,
) -> dict[str, Any]:
    include_all = "all" in fields
    selected = ["source", "metadata", "game_context", "contracts", "enforcement", "payoffs", "usage", "reasoning"] if include_all else fields

    result: dict[str, Any] = {"trace_id": trace["trace_id"]}
    for field in selected:
        if field == "raw" and "raw" not in trace:
            continue
        value = field_value(trace, field)
        if value is None:
            continue
        if pattern:
            value = filter_field_value(field, value, pattern)
            if is_empty_value(value):
                continue
        result[field] = value

    if pattern:
        result.setdefault("source", trace["source"])
        result.setdefault("metadata", trace["metadata"])

    return result


def field_value(trace: dict[str, Any], field: str) -> Any:
    if field in {"source", "metadata", "game_context", "contracts", "enforcement", "payoffs", "usage", "raw", "reasoning"}:
        return trace.get(field)
    if field == "proposals":
        return trace["contracts"].get("proposals", [])
    if field == "system_feedback":
        return [event for event in trace["events"] if event.get("phase") == "system"]
    return None


def filter_field_value(field: str, value: Any, pattern: re.Pattern[str]) -> Any:
    if field in {"proposals", "system_feedback"} and isinstance(value, list):
        return [item for item in value if object_matches(item, pattern)]
    return value if object_matches(value, pattern) else None


def parse_csv_arg(value: str, valid_values: set[str], label: str) -> list[str]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise TraceExtractError(f"{label} cannot be empty")
    invalid = sorted(set(parts) - valid_values)
    if invalid:
        raise TraceExtractError(f"Invalid {label} value(s): {', '.join(invalid)}")
    return parts


def compile_search(value: str | None) -> re.Pattern[str] | None:
    if not value:
        return None
    try:
        return re.compile(value, re.IGNORECASE)
    except re.error as exc:
        raise TraceExtractError(f"Invalid --search regex: {exc}") from exc


def find_matches(trace: dict[str, Any], pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    matches = []
    for path, value in iter_searchable_values(trace):
        text = str(value)
        match = pattern.search(text)
        if match:
            matches.append({
                "path": path,
                "snippet": make_snippet(text, match.start(), match.end()),
            })
    return matches


def iter_searchable_values(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "raw":
                continue
            child_path = f"{path}.{key}" if path else str(key)
            yield from iter_searchable_values(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_searchable_values(child, f"{path}[{index}]")
    elif isinstance(value, (str, int, float, bool)) and value is not None:
        yield path, value


def object_matches(value: Any, pattern: re.Pattern[str]) -> bool:
    return any(True for _path, _value in iter_searchable_values(value) if pattern.search(str(_value)))


def make_snippet(text: str, start: int, end: int, radius: int = 120) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return f"{prefix}{text[left:right]}{suffix}"


def drop_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not is_empty_value(item)}


def is_empty_value(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


if __name__ == "__main__":
    raise SystemExit(main())
