"""Utility functions for loading and parsing contracting negotiation traces.

This module provides functions for:
- Discovering experiment log directories
- Loading contracting traces from .eval files
- Parsing negotiation history from JSON strings
- Filtering traces by various criteria
"""

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from eval.analysis.contracting import get_contracting_score_metadata

_CONTRACTING_EVAL_SUBDIR_MARKERS = ('code-nl', 'code_nl', 'code-law', 'code_law')

# Sentinel segment in traces_viewer ``nav_path`` for browsing non-``eval-*`` log folders.
MISC_BROWSE_ROOT = "__logs_misc__"

_PROMPT_MODE_NAMES = frozenset(("base", "selfish", "cooperative"))


def prompt_modes_under_experiment_root(exp_root: Path) -> Dict[str, Path]:
    """Map prompt mode name -> directory for a single experiment root (any path).

    Only includes ``base`` / ``selfish`` / ``cooperative`` subdirs that exist.
    """
    exp_root = Path(exp_root)
    prompt_modes: Dict[str, Path] = {}
    if not exp_root.is_dir():
        return prompt_modes
    for prompt_dir in exp_root.iterdir():
        if not prompt_dir.is_dir():
            continue
        if prompt_dir.name in _PROMPT_MODE_NAMES:
            prompt_modes[prompt_dir.name] = prompt_dir
    return prompt_modes


def is_contracting_experiment_layout(exp_root: Path) -> bool:
    """True if ``exp_root`` looks like a GT-HarmBench contracting eval tree."""
    prompt_dirs = prompt_modes_under_experiment_root(exp_root)
    if not prompt_dirs:
        return False
    for prompt_dir in prompt_dirs.values():
        if not prompt_dir.is_dir():
            continue
        for mode_dir in prompt_dir.iterdir():
            if not mode_dir.is_dir():
                continue
            if any(name in mode_dir.name for name in _CONTRACTING_EVAL_SUBDIR_MARKERS):
                return True
    return False


def sanitize_misc_path_segments(log_base_dir: Path, misc_path_param: str) -> List[str]:
    """Split ``misc_path_param`` into safe path segments under ``log_base_dir``."""
    log_base_dir = Path(log_base_dir)
    if not misc_path_param or not log_base_dir.is_dir():
        return []
    base = log_base_dir.resolve()
    parts = [p for p in Path(misc_path_param).parts if p not in ('', '.', '..')]
    if not parts:
        return []
    candidate = (base.joinpath(*parts)).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return []
    return list(candidate.relative_to(base).parts)


def discover_misc_top_level_dirs(log_base_dir: Path) -> List[str]:
    """Top-level directory names under ``log_base_dir`` that are not ``eval-*``."""
    log_base_dir = Path(log_base_dir)
    if not log_base_dir.is_dir():
        return []
    names: List[str] = []
    for child in sorted(log_base_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("eval-"):
            continue
        names.append(child.name)
    return names


def discover_log_dirs(log_base_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Discover all experiment log directories.

    Args:
        log_base_dir: Base logs directory (e.g., Path("logs/"))

    Returns:
        Nested dict: {experiment_dir: {prompt_mode: Path}}
        Example: {"eval-20260429-153230": {"base": Path(...), "selfish": Path(...)}}
    """
    log_base_dir = Path(log_base_dir)
    result: Dict[str, Dict[str, Path]] = {}

    if not log_base_dir.exists():
        return result

    # Find all eval-* directories
    for exp_dir in sorted(log_base_dir.glob("eval-*")):
        if not exp_dir.is_dir():
            continue

        prompt_modes = prompt_modes_under_experiment_root(exp_dir)
        if prompt_modes:
            result[exp_dir.name] = prompt_modes

    return result


def resolve_experiment_prompt_dirs(
    log_base_dir: Path,
    experiment: str,
    known_log_dirs: Dict[str, Dict[str, Path]],
) -> Optional[Dict[str, Path]]:
    """Resolve ``experiment`` to prompt-mode dirs: top-level eval name or relative path."""
    log_base_dir = Path(log_base_dir)
    if experiment in known_log_dirs:
        return known_log_dirs[experiment]
    candidate = (log_base_dir / experiment).resolve()
    try:
        candidate.relative_to(log_base_dir.resolve())
    except ValueError:
        return None
    if candidate.is_dir() and is_contracting_experiment_layout(candidate):
        return prompt_modes_under_experiment_root(candidate)
    return None


def experiment_root_path(
    log_base_dir: Path,
    experiment: str,
    known_log_dirs: Dict[str, Dict[str, Path]],
) -> Optional[Path]:
    """Absolute experiment root (parent of prompt dirs), or None."""
    log_base_dir = Path(log_base_dir)
    if experiment in known_log_dirs:
        first = next(iter(known_log_dirs[experiment].values()), None)
        if first is not None:
            return first.parent
    candidate = (log_base_dir / experiment).resolve()
    try:
        candidate.relative_to(log_base_dir.resolve())
    except ValueError:
        return None
    if candidate.is_dir() and is_contracting_experiment_layout(candidate):
        return candidate
    return None


def iter_contract_eval_files(prompt_dirs_for_experiment: Dict[str, Path]):
    """Yield paths to contracting `*.eval` logs under base/selfish/cooperative trees."""
    for prompt_dir in prompt_dirs_for_experiment.values():
        yield from iter_contract_eval_files_under_prompt_dir(prompt_dir)


def iter_contract_eval_files_under_prompt_dir(prompt_dir: Path):
    """Yield contracting `*.eval` paths nested under one prompt-mode folder."""
    if not prompt_dir.is_dir():
        return
    for mode_dir in sorted(prompt_dir.iterdir()):
        if not mode_dir.is_dir():
            continue
        if not any(m in mode_dir.name for m in _CONTRACTING_EVAL_SUBDIR_MARKERS):
            continue
        yield from sorted(mode_dir.glob('*.eval'))


def count_trace_rows_in_eval_file(eval_file: Path) -> int:
    """Count scorer rows inside one `.eval` without loading negotiation bodies.

    Prefer `_journal/summaries/` JSON shards (few reads, matches journal totals); fallback
    to counting `samples/*.json` paths.
    """
    try:
        with zipfile.ZipFile(eval_file, 'r') as zf:
            names = zf.namelist()
            summary_paths = sorted(
                n for n in names
                if n.startswith('_journal/summaries/') and n.endswith('.json')
            )
            if summary_paths:
                total = 0
                for sp in summary_paths:
                    try:
                        with zf.open(sp) as fh:
                            data = json.load(fh)
                    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                        continue
                    if isinstance(data, list):
                        total += len(data)
                    elif isinstance(data, dict):
                        total += 1
                if total > 0:
                    return total
            samples = [
                n for n in names
                if n.startswith('samples/') and n.endswith('.json')
            ]
            return len(samples)
    except (zipfile.BadZipFile, OSError):
        return 0


def count_contracting_trace_rows_for_experiment(
    prompt_dirs_for_experiment: Dict[str, Path],
) -> int:
    """Total trace rows summed across contracting `.eval` files for one experiment."""
    return sum(count_trace_rows_in_eval_file(ep) for ep in iter_contract_eval_files(prompt_dirs_for_experiment))


def parse_negotiation_history(negotiation_json: str) -> List[Dict[str, Any]]:
    """Parse negotiation history from JSON string safely.

    Args:
        negotiation_json: JSON string containing negotiation history

    Returns:
        List of negotiation turn dicts, or empty list if parsing fails
    """
    if not negotiation_json:
        return []

    try:
        history = json.loads(negotiation_json)
        if isinstance(history, list):
            return history
        return []
    except (json.JSONDecodeError, TypeError):
        return []


def _format_display_name(value: Any) -> str:
    """Format raw event names for compact display labels."""
    if value is None:
        return ""
    return str(value).replace("_", " ").replace("-", " ").title()


def _extract_reasoning_from_output(output: Dict[str, Any]) -> Dict[str, Any]:
    """Extract reasoning content from model output.

    Args:
        output: Sample output dict with choices containing message content

    Returns:
        Dict with 'row_player' and 'col_player' reasoning info
        Each has: 'text' (reasoning or summary), 'summary', 'redacted',
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
                        reasoning_info["text"] = item.get("summary")

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


def _normalize_conversation_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve a raw conversation event and add display-oriented fields."""
    normalized = dict(event)
    player = event.get('player') or event.get('agent')
    action = event.get('action')
    phase = event.get('phase')
    message = (
        event.get('message')
        or event.get('raw_message')
        or event.get('reasoning')
        or event.get('contract_text')
        or ''
    )

    normalized['display_player'] = _format_display_name(player) or 'Unknown'
    normalized['display_action'] = _format_display_name(action) if action else ''
    normalized['display_phase'] = _format_display_name(phase) if phase else ''
    normalized['display_turn'] = event.get('turn')
    normalized['display_message'] = message
    return normalized


def load_eval_traces(eval_file: Path) -> List[Dict[str, Any]]:
    """Load contracting traces from a single .eval file.

    Reads from samples/ directory to get full negotiation history (not truncated).

    Args:
        eval_file: Path to .eval file

    Returns:
        List of trace dicts with negotiation and metadata
    """
    traces = []

    try:
        with zipfile.ZipFile(eval_file, 'r') as zf:
            # Read from samples/ directory for full negotiation history
            sample_files = sorted([name for name in zf.namelist()
                                   if name.startswith('samples/')])

            for sample_file in sample_files:
                with zf.open(sample_file) as f:
                    sample = json.load(f)

                    # Extract input metadata
                    input_meta = sample.get('metadata', {})

                    # Extract negotiation result and contracting metadata
                    negotiation_result = input_meta.get('negotiation_result', {})
                    if not negotiation_result:
                        continue

                    # Build negotiation history from conversations while preserving
                    # phase-specific fields such as message, agent, errors, and html.
                    conversations = negotiation_result.get('conversations', [])
                    negotiation = [
                        _normalize_conversation_event(conv)
                        for conv in conversations
                        if isinstance(conv, dict)
                    ]

                    contract_data = negotiation_result.get('contract') or {}
                    enforcement_result = input_meta.get('enforcement_result') or {}

                    # Extract model reasoning from output (if available)
                    reasoning = _extract_reasoning_from_output(sample.get('output', {}))

                    # Build trace dict
                    trace = {
                        'sample_id': input_meta.get('id', ''),
                        'input_metadata': input_meta,
                        'negotiation': negotiation,
                        'contract_formed': negotiation_result.get('agreement_reached', False),
                        'contract_complied': None,  # Defined only when formal enforcement runs
                        'contract_activated': None,  # True when formal enforcement mechanisms activate
                        'formation_failure_reason': None,
                        'compliance_failure_reason': None,
                        'is_nash': None,
                        'is_utilitarian': None,
                        'is_rawlsian': None,
                        'utilitarian_payoff': None,  # Will be computed from scores
                        'rawlsian_payoff': None,
                        'turns_to_agreement': negotiation_result.get('turns_taken', len(negotiation)),
                        'contract_text': negotiation_result.get('contract_text', ''),
                        'final_contract': contract_data,
                        'enforcement_result': enforcement_result,
                        'row_action': None,
                        'column_action': None,
                        'row_effort_level': None,
                        'col_effort_level': None,
                        'payoff_adjustments': {},  # Enforcement primitives
                        'reasoning': reasoning,  # Model reasoning (CoT)
                    }

                    # Try to get additional metadata from scorer if available
                    score_meta = get_contracting_score_metadata(sample)
                    if score_meta:
                        trace['contract_formed'] = score_meta.get('contract_formed', trace['contract_formed'])
                        trace['contract_complied'] = score_meta.get('contract_complied')
                        trace['contract_activated'] = score_meta.get(
                            'contract_activated',
                            None if trace['contract_complied'] is None else not trace['contract_complied'],
                        )
                        trace['formation_failure_reason'] = score_meta.get('formation_failure_reason')
                        trace['compliance_failure_reason'] = score_meta.get('compliance_failure_reason')
                        trace['is_nash'] = score_meta.get('is_nash')
                        trace['is_utilitarian'] = score_meta.get('is_utilitarian')
                        trace['is_rawlsian'] = score_meta.get('is_rawlsian')
                        trace['utilitarian_payoff'] = score_meta.get('utilitarian_payoff')
                        trace['rawlsian_payoff'] = score_meta.get('rawlsian_payoff')
                        trace['turns_to_agreement'] = score_meta.get('turns_to_agreement', trace['turns_to_agreement'])
                        trace['contract_text'] = score_meta.get('contract_text') or trace['contract_text']
                        trace['row_action'] = score_meta.get('row_action')
                        trace['column_action'] = score_meta.get('column_action')
                        trace['row_effort_level'] = score_meta.get('row_effort_level')
                        trace['col_effort_level'] = score_meta.get('col_effort_level')
                        # Enforcement primitives
                        trace['payoff_adjustments'] = score_meta.get('payoff_adjustments', {})

                    if not trace['contract_text'] and contract_data:
                        trace['contract_text'] = contract_data.get('content', '')

                    if not trace['payoff_adjustments'] and enforcement_result:
                        trace['payoff_adjustments'] = enforcement_result.get('payoff_adjustments', {})

                    traces.append(trace)

    except Exception as e:
        print(f"Error loading {eval_file.name}: {e}")

    return traces


def load_contracting_traces(
    log_base_dir: Path,
    experiments: Optional[Tuple[str, ...]] = None,
    prompt_modes: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    """Load contracting traces from multiple experiments.

    Args:
        log_base_dir: Base logs directory
        experiments: Optional tuple of experiment names (default: all)
        prompt_modes: Optional tuple of prompt modes (default: all)

    Returns:
        DataFrame with one row per trace
    """
    log_dirs = discover_log_dirs(log_base_dir)

    all_traces = []

    experiment_names = experiments if experiments else list(log_dirs.keys())

    for exp_name in experiment_names:
        prompt_dirs = resolve_experiment_prompt_dirs(log_base_dir, exp_name, log_dirs)
        if not prompt_dirs:
            continue

        for prompt_mode, prompt_dir in prompt_dirs.items():
            if prompt_modes and prompt_mode not in prompt_modes:
                continue

            # Find all 4x4-code-nl and 2x2-code-nl subdirectories
            for mode_dir in prompt_dir.iterdir():
                if not mode_dir.is_dir():
                    continue

                # Only look at contracting modes with negotiation
                if not any(name in mode_dir.name for name in ['code-nl', 'code_nl', 'code-law', 'code_law']):
                    continue

                # Find .eval files
                eval_files = sorted(mode_dir.glob("*.eval"))
                for eval_file in eval_files:
                    traces = load_eval_traces(eval_file)

                    # Add experiment metadata to each trace
                    for trace in traces:
                        trace['experiment'] = exp_name
                        trace['prompt_mode'] = prompt_mode
                        trace['contract_mode'] = mode_dir.name
                        trace['model'] = _extract_model_name(exp_name)

                    all_traces.extend(traces)

    if not all_traces:
        return pd.DataFrame()

    df = pd.DataFrame(all_traces)
    return df


def _extract_model_name(exp_name: str) -> str:
    """Extract model name from experiment directory name.

    Args:
        exp_name: Experiment directory name (e.g., "eval-20260429-openai-gpt-4o")

    Returns:
        Model name string
    """
    tail = Path(exp_name).name
    # Experiment names are like: eval-TIMESTAMP-MODEL
    parts = tail.split('-')
    if len(parts) >= 3:
        # Everything after the timestamp is the model
        return '-'.join(parts[2:])
    return "unknown"


def filter_traces(
    df: pd.DataFrame,
    prompt_mode: Optional[str] = None,
    contract_formed: Optional[bool] = None,
    contract_complied: Optional[bool] = None,
    game_type: Optional[str] = None,
    sample_id: Optional[str] = None,
) -> pd.DataFrame:
    """Filter traces by various criteria.

    Args:
        df: DataFrame from load_contracting_traces()
        prompt_mode: Filter by prompt mode (base/selfish/cooperative)
        contract_formed: Filter by contract formation status
        contract_complied: Filter by contract compliance status
        game_type: Filter by formal game type
        sample_id: Filter by sample ID (partial match)

    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    filtered_df = df.copy()

    if prompt_mode:
        filtered_df = filtered_df[filtered_df['prompt_mode'] == prompt_mode]

    if contract_formed is not None:
        filtered_df = filtered_df[filtered_df['contract_formed'] == contract_formed]

    if contract_complied is not None:
        filtered_df = filtered_df[filtered_df['contract_complied'] == contract_complied]

    if game_type:
        filtered_df = filtered_df[
            filtered_df['input_metadata'].apply(
                lambda x: x.get('formal_game', '') == game_type
            )
        ]

    if sample_id:
        filtered_df = filtered_df[
            filtered_df['sample_id'].astype(str).str.contains(sample_id)
        ]

    return filtered_df


def get_metadata_summary(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metadata for display.

    Args:
        trace: Single trace dict from load_eval_traces()

    Returns:
        Dict with summary metadata
    """
    input_meta = trace.get('input_metadata', {})

    return {
        'sample_id': trace.get('sample_id', ''),
        'formal_game': input_meta.get('formal_game', ''),
        'model': trace.get('model', ''),
        'prompt_mode': trace.get('prompt_mode', ''),
        'contract_mode': trace.get('contract_mode', ''),
        'contract_formed': trace.get('contract_formed', False),
        'contract_complied': trace.get('contract_complied'),
        'contract_activated': trace.get('contract_activated'),
        'formation_failure_reason': trace.get('formation_failure_reason'),
        'compliance_failure_reason': trace.get('compliance_failure_reason'),
        'utilitarian_payoff': trace.get('utilitarian_payoff'),
        'rawlsian_payoff': trace.get('rawlsian_payoff'),
        'turns_to_agreement': trace.get('turns_to_agreement', 0),
        'has_negotiation': len(trace.get('negotiation', [])) > 0,
    }


def get_game_context(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Extract game context for display.

    Args:
        trace: Single trace dict from load_eval_traces()

    Returns:
        Dict with game context
    """
    input_meta = trace.get('input_metadata', {})

    # Parse payoff matrix from JSON string if needed
    payoff_matrix_4x4 = input_meta.get('payoff_matrix_4x4')
    if payoff_matrix_4x4 and isinstance(payoff_matrix_4x4, str):
        try:
            payoff_matrix_4x4 = json.loads(payoff_matrix_4x4)
        except (json.JSONDecodeError, TypeError):
            payoff_matrix_4x4 = None

    payoff_matrix = input_meta.get('payoff_matrix')
    if payoff_matrix and isinstance(payoff_matrix, str):
        try:
            payoff_matrix = json.loads(payoff_matrix)
        except (json.JSONDecodeError, TypeError):
            payoff_matrix = None

    # For 2x2 games, build payoff matrix from separate payoff fields
    is_4x4 = input_meta.get('is_4x4', False)
    if not is_4x4 and not payoff_matrix:
        # Check for 2x2 payoff fields (1_1_payoff, 1_2_payoff, etc.)
        actions_row = input_meta.get('actions_row', [])
        actions_column = input_meta.get('actions_column', [])

        if len(actions_row) == 2 and len(actions_column) == 2:
            matrix_2x2 = []
            for i in range(1, 3):
                row_data = []
                for j in range(1, 3):
                    key = f'{i}_{j}_payoff'
                    payoff_str = input_meta.get(key, '[]')
                    try:
                        payoff = json.loads(payoff_str) if isinstance(payoff_str, str) else payoff_str
                        row_data.append(payoff if isinstance(payoff, list) else [payoff])
                    except (json.JSONDecodeError, TypeError):
                        row_data.append([0, 0])
                matrix_2x2.append(row_data)

            if all(len(row) == 2 for row in matrix_2x2):
                payoff_matrix = matrix_2x2

    return {
        'formal_game': input_meta.get('formal_game', ''),
        'story_row': input_meta.get('story_row', ''),
        'story_col': input_meta.get('story_col', ''),
        'actions_row': input_meta.get('actions_row', []),
        'actions_column': input_meta.get('actions_column', []),
        'rewards_matrix': input_meta.get('rewards_matrix', []),
        'payoff_matrix_4x4': payoff_matrix_4x4,
        'payoff_matrix': payoff_matrix,
        'is_4x4': is_4x4,
    }


def discover_analysis_dirs(exp_dir: Path) -> Dict[str, Path]:
    """Discover analysis subdirectories for an experiment.

    Args:
        exp_dir: Path to experiment directory (e.g., logs/eval-20260428-165127-model/)

    Returns:
        Dict mapping prompt mode to analysis directory path
        Example: {"base": Path(.../analysis/base), "selfish": Path(...)}
    """
    analysis_base = exp_dir / "analysis"
    result = {}

    if not analysis_base.exists():
        return result

    # Find prompt mode subdirectories
    for prompt_dir in analysis_base.iterdir():
        if not prompt_dir.is_dir():
            continue

        prompt_mode = prompt_dir.name
        if prompt_mode in ("base", "selfish", "cooperative"):
            result[prompt_mode] = prompt_dir

    return result


def load_analysis_files(analysis_dir: Path) -> Dict[str, Any]:
    """Load analysis files from a prompt mode's analysis directory.

    Args:
        analysis_dir: Path to analysis subdirectory (e.g., logs/.../analysis/base/)

    Returns:
        Dict with file contents (Paths to PNGs, DataFrames for CSVs)
        Keys: 'action_heatmap', 'combined_plots', 'summary_csv'
    """
    result = {
        'action_heatmap': None,
        'combined_plots': None,
        'summary_csv': None,
    }

    if not analysis_dir.exists():
        return result

    # Load PNG files
    heatmap_path = analysis_dir / "action_heatmaps.png"
    if heatmap_path.exists():
        result['action_heatmap'] = heatmap_path

    combined_path = analysis_dir / "combined_plots.png"
    if combined_path.exists():
        result['combined_plots'] = combined_path

    # Load summary CSV
    summary_path = analysis_dir / "summary.csv"
    if summary_path.exists():
        try:
            result['summary_csv'] = pd.read_csv(summary_path)
        except Exception as e:
            print(f"Error loading summary CSV: {e}")

    return result


def load_combined_summary_csv(exp_dir: Path) -> Optional[pd.DataFrame]:
    """Load and combine summary CSVs from all prompt modes.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        Combined DataFrame with prompt_mode column, or None if no summaries found
    """
    analysis_dirs = discover_analysis_dirs(exp_dir)

    if not analysis_dirs:
        return None

    combined_dfs = []

    for prompt_mode, analysis_dir in analysis_dirs.items():
        summary_path = analysis_dir / "summary.csv"
        if summary_path.exists():
            try:
                df = pd.read_csv(summary_path)
                df['prompt_mode'] = prompt_mode
                combined_dfs.append(df)
            except Exception as e:
                print(f"Error loading {summary_path}: {e}")

    if not combined_dfs:
        return None

    return pd.concat(combined_dfs, ignore_index=True)
