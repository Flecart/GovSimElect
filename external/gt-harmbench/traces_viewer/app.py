"""Streamlit app for browsing contracting negotiation traces.

Run with:
    conda run -n contracts uv run streamlit run traces_viewer/app.py
"""

import html as html_lib
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from traces_viewer.traces_utils import (
    _CONTRACTING_EVAL_SUBDIR_MARKERS,
    _format_display_name,
    MISC_BROWSE_ROOT,
    discover_log_dirs,
    discover_misc_top_level_dirs,
    experiment_root_path,
    is_contracting_experiment_layout,
    load_contracting_traces,
    filter_traces,
    get_metadata_summary,
    get_game_context,
    discover_analysis_dirs,
    load_analysis_files,
    load_combined_summary_csv,
    count_trace_rows_in_eval_file,
    iter_contract_eval_files_under_prompt_dir,
    iter_contract_eval_files,
    resolve_experiment_prompt_dirs,
    sanitize_misc_path_segments,
)

# Custom CSS for clean styling - dark theme
CUSTOM_CSS = """
<style>
    /* Main container styles - dark backgrounds */
    .main .block-container {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }

    /*
     * Default secondary buttons: compact outlined toolbar / breadcrumbs / Back.
     * Directory browsing cards stay secondary too (avoid primary = theme accent red).
     * Streamlit exposes st.container(..., key=...) as CSS class .st-key-<key>.
     */
    [data-testid="stBaseButton-secondary"] button {
        min-height: 2.625rem !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.9375rem !important;
        line-height: 1.35 !important;
        white-space: nowrap !important;
        text-align: center !important;
        justify-content: center !important;
        align-items: center !important;
        background-color: transparent !important;
        border: 1px solid rgba(250, 250, 250, 0.38) !important;
        border-radius: 0.5rem !important;
        color: rgb(226, 232, 240) !important;
        box-shadow: none !important;
    }

    [data-testid="stBaseButton-secondary"] button:not([disabled]):hover {
        background-color: rgba(250, 250, 250, 0.08) !important;
        border-color: rgba(250, 250, 250, 0.52) !important;
        color: rgb(247, 250, 253) !important;
    }

    [data-testid="stBaseButton-secondary"] button[disabled] {
        border-color: rgba(250, 250, 250, 0.28) !important;
        opacity: 0.88;
    }

    /* Multiline directory tiles (neutral grey — not Streamlit accent primary red) */
    .st-key-harmbench_pick_experiments [data-testid="stBaseButton-secondary"] button,
    .st-key-harmbench_pick_prompts [data-testid="stBaseButton-secondary"] button,
    .st-key-harmbench_pick_contract_modes [data-testid="stBaseButton-secondary"] button,
    .st-key-harmbench_pick_misc [data-testid="stBaseButton-secondary"] button {
        min-height: unset !important;
        height: auto !important;
        padding: 16px 20px !important;
        font-size: 15px !important;
        line-height: 1.55 !important;
        white-space: normal !important;
        text-align: left !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
        background-color: rgba(42, 42, 42, 0.95) !important;
        color: #ececec !important;
        border: 1px solid rgba(250, 250, 250, 0.32) !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    .st-key-harmbench_pick_experiments [data-testid="stBaseButton-secondary"] button:not([disabled]):hover,
    .st-key-harmbench_pick_prompts [data-testid="stBaseButton-secondary"] button:not([disabled]):hover,
    .st-key-harmbench_pick_contract_modes [data-testid="stBaseButton-secondary"] button:not([disabled]):hover,
    .st-key-harmbench_pick_misc [data-testid="stBaseButton-secondary"] button:not([disabled]):hover {
        background-color: rgba(52, 52, 52, 0.95) !important;
        border-color: rgba(250, 250, 250, 0.45) !important;
        color: #f5f5f5 !important;
    }

    [data-testid="stBaseButton-secondary"] button:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(180, 190, 210, 0.45) !important;
    }

    /* st.header — directory section titles (Experiments, folder, prompt mode) */
    section.main .block-container h2 {
        border: 1px solid #5a5a5a !important;
        border-radius: 8px !important;
        padding: 0.6rem 1rem !important;
        margin: 0.25rem 0 1rem 0 !important;
        background-color: rgba(40, 40, 40, 0.85) !important;
        display: block !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    iframe[title*="streamlit"] {
        vertical-align: middle;
    }

    /* Selection cards */
    .selection-card {
        background-color: #2d2d2d;
        border: 2px solid #4a4a4a;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .selection-card:hover {
        background-color: #3d3d3d;
        border-color: #5a9fd4;
        transform: translateY(-2px);
    }

    .selection-card-title {
        font-size: 18px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .selection-card-subtitle {
        font-size: 14px;
        color: #a0a0a0;
    }

    /* Breadcrumb navigation */
    .breadcrumb {
        background-color: #2d2d2d;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 20px;
    }

    /* Turn container */
    .turn-container {
        border: 1px solid #4a4a4a;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        background-color: #2d2d2d;
    }

    .turn-header {
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 6px;
        font-size: 14px;
    }

    .turn-propose {
        border-left: 4px solid #4a90e2;
        padding-left: 12px;
    }

    .turn-accept {
        border-left: 4px solid #52c41a;
        padding-left: 12px;
    }

    .turn-coding {
        border-left: 4px solid #a277ff;
        padding-left: 12px;
    }

    .turn-system {
        border-left: 4px solid #8a8a8a;
        padding-left: 12px;
    }

    /* Contract box */
    .contract-box {
        background-color: #1a1a1a;
        border: 1px solid #4a4a4a;
        border-radius: 4px;
        padding: 12px;
        margin: 8px 0;
        font-style: italic;
        color: #e0e0e0;
    }

    /* Reasoning box */
    .reasoning-box {
        background-color: #1e3a1e;
        border: 1px solid #2d5a2d;
        border-radius: 4px;
        padding: 12px;
        margin: 8px 0;
        white-space: pre-wrap;
        font-family: monospace;
        font-size: 13px;
        color: #d0d0d0;
    }

    .message-box {
        background-color: #1a2433;
        border: 1px solid #2c4a70;
        border-radius: 4px;
        padding: 12px;
        margin: 8px 0;
        white-space: pre-wrap;
        font-family: monospace;
        font-size: 13px;
        color: #d0d0d0;
    }

    /* Metadata items */
    .metadata-item {
        margin: 4px 0;
        color: #e0e0e0;
    }

    .status-true {
        color: #52c41a;
        font-weight: 600;
    }

    .status-false {
        color: #ff4d4f;
        font-weight: 600;
    }

    /* Summary header */
    .summary-header {
        background-color: #2d2d2d;
        border: 1px solid #4a4a4a;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 16px;
        font-weight: 600;
        color: #ffffff;
    }

    /* Table styling */
    .dataframe {
        color: #e0e0e0;
    }

    .dataframe th {
        background-color: #2d2d2d;
        color: #ffffff;
        font-weight: 600;
    }

    .dataframe td {
        border: 1px solid #4a4a4a;
    }

    /* Hide selection column in dataframes */
    .dataframe td:first-child,
    .dataframe th:first-child {
        display: none;
    }

    /* Make table rows appear clickable */
    .dataframe tbody tr {
        cursor: pointer;
    }

    .dataframe tbody tr:hover {
        background-color: #3d3d3d;
    }

    /* Metric cards */
    .metric-card {
        background-color: #2d2d2d;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #5a9fd4;
    }

    .metric-label {
        font-size: 14px;
        color: #a0a0a0;
        margin-top: 4px;
    }
</style>
"""

st.set_page_config(
    page_title="GT-HarmBench Traces Viewer",
    page_icon="💬",
    layout="wide",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

NAV_QUERY_KEYS = ("experiment", "prompt_mode", "contract_mode", "sample_id", "misc_path")


def _in_misc_browse(nav_path: list[str]) -> bool:
    return len(nav_path) >= 2 and nav_path[1] == MISC_BROWSE_ROOT


def _misc_segments(nav_path: list[str]) -> list[str]:
    """Path segments under ``logs/`` when in misc browse mode (may be empty)."""
    if not _in_misc_browse(nav_path):
        return []
    return list(nav_path[2:])


def _misc_dir_path(log_base: Path, nav_path: list[str]) -> Path | None:
    """Absolute directory for current misc browse (``logs/`` when at browse root)."""
    if not _in_misc_browse(nav_path):
        return None
    segs = _misc_segments(nav_path)
    base = Path(log_base).resolve()
    return base.joinpath(*segs).resolve() if segs else base


def _experiment_rel_for_path(log_base: Path, target: Path) -> str | None:
    """POSIX path relative to ``log_base`` for ``target``, or None if outside."""
    base = Path(log_base).resolve()
    resolved = Path(target).resolve()
    try:
        rel = resolved.relative_to(base)
    except ValueError:
        return None
    return rel.as_posix()


def _sync_eval_selections_from_nav(
    nav_path: list[str],
    log_base: Path,
    log_dirs: dict[str, dict[str, Path]],
) -> None:
    """Align session selection fields with ``nav_path`` after navigation."""
    if _in_misc_browse(nav_path) or len(nav_path) <= 1:
        st.session_state.selected_experiment = None
        st.session_state.selected_prompt_mode = None
        st.session_state.selected_contract_mode = None
        st.session_state.selected_sample_id = None
        return

    experiment = nav_path[1]
    prompt_dirs = resolve_experiment_prompt_dirs(log_base, experiment, log_dirs)
    if not prompt_dirs:
        return

    st.session_state.selected_experiment = experiment
    st.session_state.selected_prompt_mode = None
    st.session_state.selected_contract_mode = None
    st.session_state.selected_sample_id = None

    if len(nav_path) >= 3:
        pm = nav_path[2]
        if pm in prompt_dirs:
            st.session_state.selected_prompt_mode = pm
    if len(nav_path) >= 4 and st.session_state.selected_prompt_mode:
        st.session_state.selected_contract_mode = nav_path[3]
    if len(nav_path) >= 5:
        last = nav_path[4]
        if isinstance(last, str) and last.startswith("sample_"):
            st.session_state.selected_sample_id = last.removeprefix("sample_")


def _nav_breadcrumb_labels(nav_path: list[str]) -> list[str]:
    """Human-readable labels for breadcrumb buttons (parallel to ``nav_path``)."""
    out: list[str] = []
    for i, token in enumerate(nav_path):
        if i == 0:
            out.append("home")
        elif token == MISC_BROWSE_ROOT:
            out.append("Other logs")
        else:
            out.append(token)
    return out


def _query_param_value(key: str) -> str | None:
    """Return a single query param value across Streamlit API representations."""
    value = st.query_params.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    return str(value) if value else None


def _clear_loaded_trace_state() -> None:
    """Drop loaded trace data so Refresh reloads from disk while keeping navigation."""
    for key in ("traces_df", "traces_df_cache_key", "scenario_df"):
        st.session_state.pop(key, None)


def _mtime_signature_for_eval_paths(eval_paths: Iterable[Path]) -> tuple[tuple[str, int], ...]:
    """Stable cache key from eval paths plus mtimes."""
    ordered = sorted(eval_paths, key=lambda p: str(p))
    entries: list[tuple[str, int]] = []
    for ep in ordered:
        try:
            ep_stat = ep.stat()
            entries.append((str(ep.resolve()), ep_stat.st_mtime_ns))
        except OSError:
            entries.append((str(ep), -1))
    return tuple(entries)


@st.cache_data(show_spinner=False)
def _cached_contract_trace_row_total(sig: tuple[tuple[str, int], ...]) -> int:
    """Sum trace-row counts across eval files keyed by `(path, mtime_ns)` tuples."""
    return sum(count_trace_rows_in_eval_file(Path(p)) for p, _ in sig)


def initialize_navigation_state(
    log_dirs: dict[str, dict[str, Path]],
    log_base_path: Path,
) -> None:
    """Initialize navigation from URL query params, falling back to home."""
    if 'nav_path' in st.session_state:
        return

    experiment = _query_param_value("experiment")
    misc_path_param = _query_param_value("misc_path")
    prompt_mode = _query_param_value("prompt_mode")
    contract_mode = _query_param_value("contract_mode")
    sample_id = _query_param_value("sample_id")

    nav_path: list[str] = ["home"]
    selected_experiment = None
    selected_prompt_mode = None
    selected_contract_mode = None
    selected_sample_id = None

    if experiment:
        prompt_dirs = resolve_experiment_prompt_dirs(log_base_path, experiment, log_dirs)
        if prompt_dirs:
            nav_path.append(experiment)
            selected_experiment = experiment

            if prompt_mode and prompt_mode in prompt_dirs:
                nav_path.append(prompt_mode)
                selected_prompt_mode = prompt_mode

                prompt_dir = prompt_dirs[prompt_mode]
                if contract_mode and (prompt_dir / contract_mode).is_dir():
                    nav_path.append(contract_mode)
                    selected_contract_mode = contract_mode

                    if sample_id:
                        nav_path.append(f"sample_{sample_id}")
                        selected_sample_id = sample_id
    elif misc_path_param:
        misc_segs = sanitize_misc_path_segments(log_base_path, misc_path_param)
        nav_path.extend([MISC_BROWSE_ROOT, *misc_segs])

    st.session_state.nav_path = nav_path
    st.session_state.selected_experiment = selected_experiment
    st.session_state.selected_prompt_mode = selected_prompt_mode
    st.session_state.selected_contract_mode = selected_contract_mode
    st.session_state.selected_sample_id = selected_sample_id


def sync_navigation_query_params() -> None:
    """Mirror current navigation state into the URL so browser refresh is durable."""
    nav_path = st.session_state.get("nav_path") or ["home"]

    if _in_misc_browse(nav_path):
        misc_segs = _misc_segments(nav_path)
        if misc_segs:
            st.query_params["misc_path"] = "/".join(misc_segs)
        elif "misc_path" in st.query_params:
            del st.query_params["misc_path"]
        for key in ("experiment", "prompt_mode", "contract_mode", "sample_id"):
            if key in st.query_params:
                del st.query_params[key]
        return

    next_values = {
        "experiment": st.session_state.selected_experiment,
        "prompt_mode": st.session_state.selected_prompt_mode,
        "contract_mode": st.session_state.selected_contract_mode,
        "sample_id": st.session_state.selected_sample_id,
    }

    if "misc_path" in st.query_params:
        del st.query_params["misc_path"]

    for key in ("experiment", "prompt_mode", "contract_mode", "sample_id"):
        value = next_values[key]
        if value:
            st.query_params[key] = str(value)
        elif key in st.query_params:
            del st.query_params[key]


def classify_trace_event(event: dict) -> str:
    """Classify a raw conversation event into a viewer section."""
    phase = event.get('phase') or ''
    action = event.get('action') or ''
    player = event.get('player') or ''

    if player == 'system' or action == 'CODING_FAILED' or 'feedback' in phase:
        return 'system'
    if phase == 'coding_agent_translation' or player == 'coding_agent':
        return 'coding'
    if action in ('PROPOSE', 'ACCEPT'):
        return 'negotiation'
    return 'unknown'


def event_has_display_content(event: dict) -> bool:
    """Return whether an event has enough content to render."""
    content_fields = (
        'action',
        'phase',
        'message',
        'raw_message',
        'contract_text',
        'reasoning',
        'html',
        'error',
    )
    return any(event.get(field) is not None and event.get(field) != '' for field in content_fields)


def _clean_display_text(
    content: object,
    *,
    collapse_blank_lines: bool = False,
    remove_blank_lines: bool = False,
) -> str:
    """Clean trace text for visible display while preserving raw output elsewhere."""
    text = html_lib.unescape(str(content)).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = text.replace("&#x27;", '"').replace("&#x27", '"')

    if remove_blank_lines:
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines).strip()
    elif collapse_blank_lines:
        lines = [line.rstrip() for line in text.splitlines()]
        text = "\n".join(lines).strip()
        # Collapse runs of blank lines to a single newline (no empty display lines).
        text = re.sub(r"\n{2,}", "\n", text)

    return text


def _render_content_box(
    label: str,
    content: object,
    css_class: str,
    *,
    collapse_blank_lines: bool = False,
    remove_blank_lines: bool = False,
) -> None:
    """Render escaped multiline content in a styled box."""
    if content is None or content == '':
        return

    escaped_label = html_lib.escape(label)
    cleaned_content = _clean_display_text(
        content,
        collapse_blank_lines=collapse_blank_lines,
        remove_blank_lines=remove_blank_lines,
    )
    escaped_content = html_lib.escape(cleaned_content, quote=False)
    escaped_content = escaped_content.replace("```", "&#96;&#96;&#96;")
    # Newlines inside HTML confuse Streamlit markdown (<p> margins); use explicit breaks.
    escaped_content = escaped_content.replace("\n", "<br/>")
    label_markup = f"<strong>{escaped_label}:</strong><br/>" if escaped_label else ""
    st.markdown(
        f'<div class="{css_class}">{label_markup}{escaped_content}</div>',
        unsafe_allow_html=True,
    )


def _event_message(event: dict) -> str:
    """Get the most informative free-form text for an event."""
    return (
        event.get('message')
        or event.get('raw_message')
        or event.get('reasoning')
        or ''
    )


def _coding_display_message(event: dict) -> str:
    """Compact coding-agent output for code-law traces."""
    return _clean_display_text(_event_message(event), remove_blank_lines=True)


def _format_decision(original: object, final: object) -> str:
    """Format a player decision, showing enforcement changes compactly."""
    original_text = str(original) if original else ""
    final_text = str(final) if final else ""

    if original_text and final_text and original_text != final_text:
        return f"{original_text} -> {final_text}"
    return final_text or original_text or "N/A"


def _status_html(value: bool | None) -> str:
    """Render a compact Yes/No/N/A status with the existing metadata styling."""
    if value is None:
        return '<span class="status-false">N/A</span>'
    css_class = "status-true" if value else "status-false"
    label = "✓ Yes" if value else "✗ No"
    return f'<span class="{css_class}">{label}</span>'


def _event_header(event: dict, event_type: str) -> tuple[str, str, str]:
    """Return icon, CSS class, and header text for a trace event."""
    player = event.get('display_player') or _format_display_name(event.get('player') or event.get('agent')) or 'Unknown'
    action = event.get('action')
    phase = event.get('display_phase') or _format_display_name(event.get('phase'))
    turn_num = event.get('display_turn', event.get('turn'))

    if event_type == 'negotiation':
        turn_label = f"Turn {turn_num}" if turn_num is not None else "Negotiation"
        action_label = action or phase or 'MESSAGE'
        icon = "✅" if action == "ACCEPT" else "📝"
        css_class = "turn-accept" if action == "ACCEPT" else "turn-propose"
        return icon, css_class, f"{turn_label} — {player} — {action_label}"

    if event_type == 'coding':
        actor = player if player != 'Unknown' else 'Coding Agent'
        label = phase or 'Coding'
        return "🛠️", "turn-coding", f"Coding — {actor} — {label}"

    if event_type == 'system':
        label = phase or action or 'System Message'
        return "⚙️", "turn-system", f"System — {label}"

    return "ℹ️", "turn-system", "Additional Trace Event"


def render_turn(turn: dict) -> None:
    """Render a single phase-aware trace event."""
    if not event_has_display_content(turn):
        return

    event_type = classify_trace_event(turn)
    icon, turn_class, header = _event_header(turn, event_type)
    escaped_header = html_lib.escape(header)

    st.markdown(f"""
        <div class="turn-container {turn_class}">
            <div class="turn-header">
                {icon} {escaped_header}
            </div>
        """, unsafe_allow_html=True)

    if event_type == 'negotiation':
        _render_content_box("Contract", turn.get('contract_text'), "contract-box")
        _render_content_box(
            "Reasoning",
            turn.get("reasoning"),
            "reasoning-box",
            collapse_blank_lines=True,
        )
        raw_message = turn.get('raw_message')
        if raw_message and raw_message != turn.get('contract_text'):
            with st.expander("Raw model message"):
                st.text(raw_message)
    elif event_type == 'coding':
        coding_display = _coding_display_message(turn)
        _render_content_box("Coding output", coding_display, "message-box")
        _render_content_box(
            "Validation error",
            turn.get("error_message"),
            "reasoning-box",
            collapse_blank_lines=True,
        )
        raw_message = _event_message(turn)
        if raw_message and raw_message != coding_display:
            with st.expander("Raw model message"):
                st.text(raw_message)
    elif event_type == 'system':
        _render_content_box(
            "Details",
            _event_message(turn) or turn.get('reasoning'),
            "message-box",
            remove_blank_lines=True,
        )
    else:
        with st.expander("Raw event"):
            st.json(turn)

    st.markdown('</div>', unsafe_allow_html=True)


def split_trace_events(events: list[dict]) -> dict[str, list[dict]]:
    """Split trace events into display sections."""
    grouped = {
        'negotiation': [],
        'coding_system': [],
        'unknown': [],
    }

    for event in events:
        if not event_has_display_content(event):
            continue

        event_type = classify_trace_event(event)
        if event_type == 'negotiation':
            grouped['negotiation'].append(event)
        elif event_type in ('coding', 'system'):
            grouped['coding_system'].append(event)
        else:
            grouped['unknown'].append(event)

    return grouped


def render_summary(trace: dict, metadata: dict) -> None:
    """Render metadata and decisions in one compact summary section."""
    input_meta = trace.get('input_metadata', {})
    original_row = input_meta.get('original_row_action')
    original_col = input_meta.get('original_column_action')
    final_row = trace.get('row_action')
    final_col = trace.get('column_action')
    row_decision = html_lib.escape(_format_decision(original_row, final_row))
    col_decision = html_lib.escape(_format_decision(original_col, final_col))

    with st.expander("📊 Summary", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f'<div class="metadata-item">Contract formed: {_status_html(metadata["contract_formed"])}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metadata-item">Contract activated: {_status_html(metadata.get("contract_activated"))}</div>',
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f'<div class="metadata-item">Row decision: <strong>{row_decision}</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metadata-item">Column decision: <strong>{col_decision}</strong></div>',
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f'<div class="metadata-item">Turns to agreement: '
                f'<strong>{metadata["turns_to_agreement"]}</strong></div>',
                unsafe_allow_html=True,
            )
            if metadata['formation_failure_reason']:
                st.markdown(
                    f'<div class="metadata-item">Formation failed: '
                    f'<em>{metadata["formation_failure_reason"]}</em></div>',
                    unsafe_allow_html=True,
                )
            if metadata['compliance_failure_reason']:
                st.markdown(
                    f'<div class="metadata-item">Compliance failed: '
                    f'<em>{metadata["compliance_failure_reason"]}</em></div>',
                    unsafe_allow_html=True,
                )


def render_enforcement(trace: dict) -> None:
    """Render execution details and enforcement primitives together."""
    input_meta = trace.get('input_metadata', {})
    enforcement = trace.get('enforcement_result') or {}
    payoff_adjustments = trace.get('payoff_adjustments', {})
    original_row = input_meta.get('original_row_action')
    original_col = input_meta.get('original_column_action')
    final_row = trace.get('row_action')
    final_col = trace.get('column_action')
    execution_log = [
        str(item).strip()
        for item in enforcement.get('execution_log') or []
        if str(item).strip()
    ]
    modified_actions = enforcement.get('modified_actions')
    has_action_overrides = bool(
        original_row
        and original_col
        and ((original_row != final_row) or (original_col != final_col))
    )
    has_primitives = any((
        payoff_adjustments and any(payoff_adjustments.values()),
        has_action_overrides,
    ))
    has_content = any((
        enforcement.get('reasoning'),
        execution_log,
        modified_actions,
        enforcement.get('violations_detected'),
        enforcement.get('metadata'),
        enforcement.get('success') is not None,
        has_primitives,
    ))
    if not has_content:
        return

    with st.expander("⚖️ Enforcement", expanded=True):
        success = enforcement.get('success')
        if success is not None:
            status = "✓ Success" if success else "✗ Failed"
            css_class = "status-true" if success else "status-false"
            st.markdown(
                f'<div class="metadata-item">Enforcement status: '
                f'<span class="{css_class}">{status}</span></div>',
                unsafe_allow_html=True,
            )

        _render_content_box(
            "Reasoning",
            enforcement.get("reasoning"),
            "reasoning-box",
            collapse_blank_lines=True,
        )

        if has_action_overrides:
            st.markdown("**Modified Actions:**")
            st.json(modified_actions)

        violations = enforcement.get('violations_detected') or []
        if violations:
            st.markdown("**Violations Detected:**")
            for violation in violations:
                st.markdown(f"- `{violation}`")

        if execution_log:
            st.markdown("**Execution Log:**")
            st.code("\n".join(str(item) for item in execution_log), language="text")

        execution_state = (enforcement.get('metadata') or {}).get('execution_state')
        if execution_state:
            with st.expander("Execution state"):
                st.json(execution_state)

        if has_primitives:
            has_fines = False
            has_transfers = False
            st.markdown("**Outcome Adjustments:**")

            if payoff_adjustments.get('row', {}).get('fines') or payoff_adjustments.get('column', {}).get('fines'):
                has_fines = True
                st.markdown("**💰 Fines Applied:**")

                row_fines = payoff_adjustments.get('row', {}).get('fines', [])
                if row_fines:
                    total_fine = sum(row_fines)
                    st.markdown(f"  • Row player: {total_fine:.1f} ({len(row_fines)} fine(s))")

                col_fines = payoff_adjustments.get('column', {}).get('fines', [])
                if col_fines:
                    total_fine = sum(col_fines)
                    st.markdown(f"  • Column player: {total_fine:.1f} ({len(col_fines)} fine(s))")

            if any(
                payoff_adjustments.get(player, {}).get(direction)
                for player in ('row', 'column')
                for direction in ('sent', 'received')
            ):
                has_transfers = True
                st.markdown("**🔄 Reward Transfers:**")

                row_sent = payoff_adjustments.get('row', {}).get('sent', [])
                row_received = payoff_adjustments.get('row', {}).get('received', [])
                if row_sent or row_received:
                    gross_sent = abs(sum(row_sent))
                    gross_received = sum(row_received)
                    parts = []
                    if gross_sent:
                        parts.append(f"sent {gross_sent:.1f}")
                    if gross_received:
                        parts.append(f"received {gross_received:.1f}")
                    st.markdown(f"  • Row player: {', '.join(parts)}")

                col_sent = payoff_adjustments.get('column', {}).get('sent', [])
                col_received = payoff_adjustments.get('column', {}).get('received', [])
                if col_sent or col_received:
                    gross_sent = abs(sum(col_sent))
                    gross_received = sum(col_received)
                    parts = []
                    if gross_sent:
                        parts.append(f"sent {gross_sent:.1f}")
                    if gross_received:
                        parts.append(f"received {gross_received:.1f}")
                    st.markdown(f"  • Column player: {', '.join(parts)}")

            if has_action_overrides:
                st.markdown("**🔄 Action Overrides:**")
                if original_row != final_row:
                    st.markdown(f"  • Row: {original_row} → {final_row}")
                if original_col != final_col:
                    st.markdown(f"  • Column: {original_col} → {final_col}")

            if not any((has_fines, has_transfers, has_action_overrides)):
                st.caption("No payoff or action adjustments applied")


def render_model_decisions(trace: dict) -> None:
    """Render original and final model decisions for both players compactly."""
    input_meta = trace.get('input_metadata', {})
    original_row = input_meta.get('original_row_action')
    original_col = input_meta.get('original_column_action')
    final_row = trace.get('row_action')
    final_col = trace.get('column_action')

    if not any((original_row, original_col, final_row, final_col)):
        return

    st.markdown("### 🎯 Model Decisions")
    row_decision = html_lib.escape(_format_decision(original_row, final_row))
    col_decision = html_lib.escape(_format_decision(original_col, final_col))
    st.markdown(
        f"""
        <div class="summary-header">
            Row: <code>{row_decision}</code> &nbsp; | &nbsp;
            Column: <code>{col_decision}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_payoff_matrix(game_context: dict, trace_key: str) -> None:
    """Render the payoff matrix as a formatted table.

    Args:
        game_context: Game context dict from get_game_context()
        trace_key: Unique key for this trace (for UI components)
    """
    payoff_matrix = game_context.get('payoff_matrix_4x4') or game_context.get('payoff_matrix')
    actions_row = game_context.get('actions_row', [])
    actions_column = game_context.get('actions_column', [])

    # Validate payoff_matrix is a list (not string like "N/A")
    if not payoff_matrix or not isinstance(payoff_matrix, list):
        return
    if not actions_row or not actions_column:
        return

    st.markdown("#### 📊 Payoff Matrix")

    # Create a DataFrame for display
    matrix_size = len(actions_row)

    # Build display data - each cell shows (row_payoff, col_payoff)
    display_data = []
    for i in range(matrix_size):
        row_data = []
        for j in range(matrix_size):
            try:
                # Safely access the cell
                if i < len(payoff_matrix) and j < len(payoff_matrix[i]):
                    cell_payoffs = payoff_matrix[i][j]
                    if isinstance(cell_payoffs, list) and len(cell_payoffs) >= 2:
                        row_payoff = cell_payoffs[0]
                        col_payoff = cell_payoffs[1]
                        # Format as "X; Y"
                        row_data.append(f"{row_payoff:.1f}; {col_payoff:.1f}")
                    else:
                        row_data.append("N/A")
                else:
                    row_data.append("N/A")
            except (IndexError, TypeError):
                row_data.append("N/A")
        display_data.append(row_data)

    # Create DataFrame with action labels
    df = pd.DataFrame(display_data, index=actions_row, columns=actions_column)

    # Display with styled caption
    st.caption("*Each cell shows payoffs as: Row; Column*")
    st.dataframe(
        df,
        width="stretch",
        hide_index=False,
    )


def render_trace_detail(trace: dict, trace_index: int) -> None:
    """Render detailed view of a single trace.

    Args:
        trace: Trace dictionary
        trace_index: Unique index for this trace (used for Streamlit keys)
    """
    metadata = get_metadata_summary(trace)
    game_context = get_game_context(trace)

    # Header
    st.markdown(f"""
        <div class="summary-header">
            Sample: {metadata['sample_id']} | Game: {metadata['formal_game']}<br/>
            Model: {metadata['model']} | Prompt: {metadata['prompt_mode']} | Mode: {metadata['contract_mode']}
        </div>
    """, unsafe_allow_html=True)

    render_summary(trace, metadata)
    render_enforcement(trace)

    # Game context section
    with st.expander("📖 Game Context"):
        st.markdown(f"**Formal Game Type:** {game_context['formal_game']}")

        # Generate unique keys for each trace (use trace_index instead of id(trace) for stability)
        trace_key = f"{metadata['sample_id']}_{trace_index}"

        # Render payoff matrix first (if available)
        render_payoff_matrix(game_context, trace_key)

        st.markdown("---")  # Separator between payoff matrix and stories

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Row Player Story:**")
            st.text_area(
                "Row player story",
                value=game_context['story_row'],
                height=150,
                key=f"story_row_{trace_key}",
                label_visibility="collapsed",
            )
            st.markdown(f"**Row Actions:** {game_context['actions_row']}")

        with col2:
            st.markdown("**Column Player Story:**")
            st.text_area(
                "Column player story",
                value=game_context['story_col'],
                height=150,
                key=f"story_col_{trace_key}",
                label_visibility="collapsed",
            )
            st.markdown(f"**Column Actions:** {game_context['actions_column']}")

    # Final contract section
    if trace.get('contract_text'):
        with st.expander("📋 Final Contract"):
            final_contract = trace.get('final_contract') or {}
            nl_contract = (final_contract.get('metadata') or {}).get('nl_contract')
            if nl_contract:
                st.markdown("**Agreed Natural-Language Contract:**")
                _render_content_box("", nl_contract, "contract-box", collapse_blank_lines=True)
                st.markdown("**Generated Python Contract:**")
                st.code(trace['contract_text'], language="python")
            else:
                _render_content_box("", trace['contract_text'], "contract-box", collapse_blank_lines=True)

    # Negotiation and coding trace sections
    negotiation = trace.get('negotiation', [])

    if negotiation:
        grouped_events = split_trace_events(negotiation)

        if grouped_events['negotiation']:
            st.markdown("### 💬 Negotiation")
            for turn in grouped_events['negotiation']:
                render_turn(turn)

        if grouped_events['coding_system']:
            st.markdown("### 🧾 Coding / System")
            st.caption("Code-law traces include coding-agent translations and technical validation feedback.")
            for turn in grouped_events['coding_system']:
                render_turn(turn)

        if grouped_events['unknown']:
            with st.expander("Additional trace events"):
                for turn in grouped_events['unknown']:
                    render_turn(turn)

    else:
        st.info("No negotiation trace available for this sample.")

    render_model_decisions(trace)


def build_navigation_copy_path(
    log_base: Path,
    nav_path: list[str],
    experiment: str | None,
    prompt_mode: str | None,
    contract_mode: str | None,
    sample_id: str | None,
) -> str:
    """Absolute path to the selected log directory, with sample id when drilled in."""
    if _in_misc_browse(nav_path):
        cur = _misc_dir_path(log_base, nav_path)
        if cur is not None:
            return cur.as_posix()
    resolved = (Path.cwd() / log_base).resolve()
    if experiment:
        resolved = resolved / experiment
    if prompt_mode:
        resolved = resolved / prompt_mode
    if contract_mode:
        resolved = resolved / contract_mode
    text = resolved.as_posix()
    if sample_id:
        text = f"{text} sample_id={sample_id}"
    return text


def render_copy_path_button(copy_text: str) -> None:
    """Copy path via clipboard inside an iframe component.

    Matches Streamlit secondary toolbar styling beside Refresh (same border as outlined buttons).
    """
    js_str = json.dumps(copy_text)
    components.html(
        f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
  html, body {{
    margin: 0;
    padding: 0;
    height: 100%;
    box-sizing: border-box;
    background: transparent;
  }}
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    display: flex;
    align-items: stretch;
    font-family: "Source Sans Pro", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  #gt_hb_copy_path_btn {{
    flex: 1;
    width: 100%;
    align-self: stretch;
    margin: 0;
    cursor: pointer;
    min-height: 2.625rem;
    padding: 0.5rem 1rem;
    font-size: 0.9375rem;
    font-weight: 400;
    line-height: 1.35;
    white-space: nowrap;
    text-align: center;
    color: rgb(226, 232, 240);
    background-color: transparent;
    border: 1px solid rgba(250, 250, 250, 0.38);
    border-radius: 0.5rem;
    transition: background-color 120ms ease, border-color 120ms ease, color 120ms ease;
  }}
  #gt_hb_copy_path_btn:hover {{
    background-color: rgba(250, 250, 250, 0.08);
    border-color: rgba(250, 250, 250, 0.52);
    color: rgb(247, 250, 253);
  }}
  #gt_hb_copy_path_btn:focus-visible {{
    outline: none;
    box-shadow: 0 0 0 2px rgba(147, 197, 253, 0.45);
    border-color: rgba(250, 250, 250, 0.4);
  }}
</style>
</head>
<body>
<button id="gt_hb_copy_path_btn" type="button">Copy path</button>
<script>
  const b = document.getElementById("gt_hb_copy_path_btn");
  const t = {js_str};
  b.addEventListener("click", async () => {{
    try {{
      await navigator.clipboard.writeText(t);
      const prev = b.textContent;
      b.textContent = "Copied";
      setTimeout(() => {{ b.textContent = prev; }}, 1400);
    }} catch (err) {{
      b.textContent = "Copy failed";
      setTimeout(() => {{ b.textContent = "Copy path"; }}, 2000);
    }}
  }});
</script>
</body>
</html>
""",
        height=48,
        width=None,
    )


def _chunked_list(items: list, chunk_size: int):
    """Yield consecutive slices of at most chunk_size elements."""
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


# Fixed column count so the last partial row keeps the same card width as above
# (st.columns(len(chunk)) would stretch one cell across the whole row).
_SELECTION_GRID_COLUMNS = 3


def _eval_tile_button_content(exp: dict) -> str:
    """Build the multiline button label for an eval-run card."""
    game_indicator = f" [{exp['game_type']}]" if exp.get('game_type') else ""
    modes_str = ', '.join(str(m) for m in exp['prompt_modes'])  # type: ignore[arg-type]
    trace_n = int(exp['trace_rows'])  # type: ignore[arg-type]
    return f"""{exp['model']}{game_indicator}

{exp['time']} {exp['date']} | {modes_str} | {trace_n:,} traces"""


def _render_tile_grid(
    items: list,
    container_key: str,
    button_key_prefix: str,
    *,
    label_fn,
    on_click,
) -> None:
    """Render a grid of secondary buttons inside a container.

    Args:
        items: Items to render (one per tile).
        container_key: Streamlit container key (controls CSS styling).
        button_key_prefix: Prefix for unique st.button keys.
        label_fn: Callable(item, index) -> (button_label: str, unique_suffix: str).
        on_click: Callable(item) invoked when the button is pressed.
    """
    with st.container(key=container_key):
        for chunk in _chunked_list(items, _SELECTION_GRID_COLUMNS):
            cols = st.columns(_SELECTION_GRID_COLUMNS)
            for idx, item in enumerate(chunk):
                with cols[idx]:
                    label, suffix = label_fn(item, idx)
                    if st.button(label, key=f"{button_key_prefix}_{suffix}", width='stretch'):
                        on_click(item)
            for pad in range(len(chunk), _SELECTION_GRID_COLUMNS):
                with cols[pad]:
                    st.empty()


def render_breadcrumbs(
    path: list,
    *,
    log_base: Path,
    log_dirs: dict[str, dict[str, Path]],
    copy_path_text: str | None = None,
) -> None:
    """Render breadcrumb navigation with buttons; optional clipboard copy of current path."""
    if not path:
        return

    labels = _nav_breadcrumb_labels(path)

    if len(path) == 1:
        if copy_path_text:
            spacer, refresh_col, copy_col = st.columns(
                [6, 1, 1], gap='small', vertical_alignment='center'
            )
            # Explicitly occupy the spacer so Streamlit does not reserve a phantom
            # leading column that skews layouts of following horizontal rows.
            with spacer:
                st.empty()
            with refresh_col:
                if st.button(
                    "Refresh",
                    key="refresh_frontend_home",
                    width='stretch',
                ):
                    _clear_loaded_trace_state()
                    st.rerun()
            with copy_col:
                render_copy_path_button(copy_path_text)
        return

    weights = [1.0] * len(path)
    if copy_path_text:
        weights.extend([0.5, 0.42])
    cols = st.columns(weights, gap='small', vertical_alignment='center')

    for i, (item, label) in enumerate(zip(path, labels)):
        with cols[i]:
            if i == len(path) - 1:
                # Current page - display as card/button (not clickable)
                shown = label if i > 0 else "home"
                button_label = "📍 " + (shown[:25] if shown != "home" else "Home")
                st.button(
                    button_label,
                    key=f"breadcrumb_current_{i}",
                    width='stretch',
                    disabled=True,
                )
            else:
                # Clickable breadcrumb
                if i == 0:
                    button_label = "🏠"
                elif item == MISC_BROWSE_ROOT:
                    button_label = "📂 " + label[:20]
                else:
                    button_label = label[:20]
                if st.button(
                    button_label,
                    key=f"breadcrumb_{i}",
                    width='stretch',
                ):
                    st.session_state.nav_path = path[: i + 1]
                    _sync_eval_selections_from_nav(st.session_state.nav_path, log_base, log_dirs)
                    st.rerun()

    if copy_path_text:
        with cols[-2]:
            if st.button(
                "Refresh",
                key="refresh_frontend",
                width='stretch',
            ):
                _clear_loaded_trace_state()
                st.rerun()
        with cols[-1]:
            render_copy_path_button(copy_path_text)


def get_eval_file_counts(log_base_dir: Path, experiment: str, prompt_mode: str) -> dict:
    """Get count of .eval files for each contract mode."""
    log_dirs = discover_log_dirs(log_base_dir)
    prompt_dirs = resolve_experiment_prompt_dirs(log_base_dir, experiment, log_dirs)
    if not prompt_dirs:
        return {}

    prompt_dir = prompt_dirs.get(prompt_mode)
    if not prompt_dir:
        return {}

    counts = {}
    for mode_dir in prompt_dir.iterdir():
        if not mode_dir.is_dir():
            continue

        if any(name in mode_dir.name for name in _CONTRACTING_EVAL_SUBDIR_MARKERS):
            eval_files = list(mode_dir.glob("*.eval"))
            if eval_files:
                counts[mode_dir.name] = len(eval_files)

    return counts


def parse_experiment_name(exp_name: str) -> dict:
    """Parse experiment directory name to extract components.

    Args:
        exp_name: Experiment directory name (e.g., "eval-20260429-openai-gpt-4o-pd")

    Returns:
        Dictionary with date, time, model, game_type, and other metadata
    """
    parts = Path(exp_name).name.split('-')
    if len(parts) < 3:
        return {
            'date': 'N/A',
            'time': 'N/A',
            'model': exp_name,
            'game_type': None,
            'game_type_label': '',
        }

    # Format: eval-YYYYMMDD-HHMMSS-model[-gametype]
    date_part = parts[1] if len(parts) > 1 else ''
    time_part = parts[2] if len(parts) > 2 else ''

    # Known game type suffixes (lowercase)
    game_type_suffixes = {'pd', 'sh', 'co', 'coord'}

    # Extract model and potential game type suffix
    # Model parts are everything after the timestamp until a potential game type suffix
    model_parts = []
    game_type = None
    game_type_label = ''

    for part in parts[3:]:
        # Check if this is a known game type suffix
        if part in game_type_suffixes:
            # Normalize to CO for both 'co' and 'coord', otherwise uppercase
            game_type = 'CO' if part in ('co', 'coord') else part.upper()
            # Map to full labels
            game_labels = {
                'PD': "Prisoner's Dilemma",
                'SH': 'Stag Hunt',
                'CO': 'Coordination',
            }
            game_type_label = game_labels.get(game_type, game_type)
        else:
            model_parts.append(part)

    model = '-'.join(model_parts) if model_parts else exp_name

    # Format date as YYYY-MM-DD
    if len(date_part) == 8:
        formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
    else:
        formatted_date = date_part

    # Format time as HH:MM:SS
    if len(time_part) == 6:
        formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
    else:
        formatted_time = time_part

    return {
        'date': formatted_date,
        'time': formatted_time,
        'model': model,
        'game_type': game_type,
        'game_type_label': game_type_label,
    }


def build_eval_run_tile_row(
    rel_exp: str,
    log_base_path: Path,
    log_dirs: dict[str, dict[str, Path]],
) -> dict[str, object] | None:
    """One eval-run card payload (path under ``logs/`` may be nested, e.g. ``old/eval-…``)."""
    prompt_dirs = resolve_experiment_prompt_dirs(log_base_path, rel_exp, log_dirs)
    if not prompt_dirs:
        return None
    parsed = parse_experiment_name(rel_exp)
    prompt_modes = sorted(prompt_dirs.keys())
    eval_sig = _mtime_signature_for_eval_paths(iter_contract_eval_files(prompt_dirs))
    trace_rows = _cached_contract_trace_row_total(eval_sig)
    return {
        'name': rel_exp,
        'date': parsed['date'],
        'time': parsed['time'],
        'model': parsed['model'],
        'game_type': parsed.get('game_type'),
        'game_type_label': parsed.get('game_type_label') or '',
        'prompt_modes': prompt_modes,
        'trace_rows': trace_rows,
    }


def get_scenario_stats(traces_df: pd.DataFrame) -> pd.DataFrame:
    """Get statistics for each scenario (sample_id)."""
    stats = []

    for sample_id in traces_df['sample_id'].unique():
        sample_traces = traces_df[traces_df['sample_id'] == sample_id]

        first_trace = sample_traces.iloc[0]
        metadata = get_metadata_summary(first_trace.to_dict())

        stats.append({
            'Sample ID': sample_id,
            'Game Type': metadata['formal_game'],
            'Num Traces': len(sample_traces),
            'Formation Rate': f"{sample_traces['contract_formed'].mean():.1%}",
            'Activation Rate': _format_optional_rate(_contract_activation_series(sample_traces)),
            'Avg Turns': f"{sample_traces['turns_to_agreement'].mean():.1f}",
            'Nash Rate': _optional_boolean_rate(sample_traces.get('is_nash')),
            'Utilitarian Rate': _optional_boolean_rate(sample_traces.get('is_utilitarian')),
            'Rawlsian Rate': _optional_boolean_rate(sample_traces.get('is_rawlsian')),
            'High Effort Rate': _high_effort_rate(sample_traces),
        })

    return pd.DataFrame(stats)


def _optional_boolean_rate(series: pd.Series | None) -> float | None:
    """Format the fraction of True values, or None when unavailable."""
    if series is None:
        return None
    values = series.dropna()
    if values.empty:
        return None
    return float(values.astype(bool).mean())


def _high_effort_rate(traces_df: pd.DataFrame) -> float | None:
    """Compute High Effort share over available row and column effort choices."""
    effort_columns = [
        column
        for column in ('row_effort_level', 'col_effort_level')
        if column in traces_df.columns
    ]
    if not effort_columns:
        return None

    effort_values = pd.concat(
        [traces_df[column] for column in effort_columns],
        ignore_index=True,
    ).dropna()
    if effort_values.empty:
        return None

    return float((effort_values == 'High Effort').mean())


def _format_optional_rate(series: pd.Series | None) -> str:
    """Format a rate that may be undefined for non-enforced modes."""
    if series is None:
        return "N/A"
    values = series.dropna()
    if values.empty:
        return "N/A"
    return f"{values.mean():.1%}"


def _contract_activation_series(traces_df: pd.DataFrame) -> pd.Series | None:
    """Return activation values, deriving them from legacy compliance values if needed."""
    if 'contract_activated' in traces_df.columns:
        return traces_df['contract_activated']
    if 'contract_complied' not in traces_df.columns:
        return None
    complied = traces_df['contract_complied'].dropna()
    if complied.empty:
        return None
    return ~complied.astype(bool)


def render_analysis_previews(exp_dir: Path, prompt_mode: str = None) -> None:
    """Render analysis previews for an experiment.

    Displays side-by-side columns by prompt mode with:
    - Action heatmap plots
    - Combined plots
    - Summary CSV table (collated by prompt_mode, experiment)

    Args:
        exp_dir: Path to experiment directory
        prompt_mode: If specified, only show this prompt mode's summaries
    """
    analysis_dirs = discover_analysis_dirs(exp_dir)

    if not analysis_dirs:
        st.info("📊 No analysis results found for this experiment. Run the analysis script to generate plots and metrics.")
        return

    # Filter to specific prompt mode if requested
    if prompt_mode:
        analysis_dirs = {k: v for k, v in analysis_dirs.items() if k == prompt_mode}
        if not analysis_dirs:
            st.info(f"📊 No analysis results found for prompt mode '{prompt_mode}'.")
            return

    # Load combined summary CSV for joint table
    combined_summary = load_combined_summary_csv(exp_dir)

    # Display section header
    st.markdown("---")
    if prompt_mode:
        st.markdown(f"### 📊 Analysis Previews: {prompt_mode.title()}")
    else:
        st.markdown("### 📊 Analysis Previews")

    # Create columns for each prompt mode
    prompt_modes = sorted(analysis_dirs.keys())
    cols = st.columns(len(prompt_modes))

    for i, pm in enumerate(prompt_modes):
        with cols[i]:
            st.markdown(f"#### {pm.title()}")

            # Load analysis files for this prompt mode
            analysis_data = load_analysis_files(analysis_dirs[pm])

            if not any((analysis_data['action_heatmap'], analysis_data['combined_plots'])):
                st.caption("No analysis files found")
                continue

            heatmap_path = analysis_data['action_heatmap']
            combined_path = analysis_data['combined_plots']

            if prompt_mode and heatmap_path and combined_path:
                img_left, img_right = st.columns(2, gap="medium")
                with img_left:
                    st.markdown("**Action Heatmaps**")
                    st.image(str(heatmap_path), width='stretch')
                with img_right:
                    st.markdown("**Combined Plots**")
                    st.image(str(combined_path), width='stretch')
            else:
                if heatmap_path:
                    st.markdown("**Action Heatmaps**")
                    st.image(str(heatmap_path), width='stretch')
                if combined_path:
                    st.markdown("**Combined Plots**")
                    st.image(str(combined_path), width='stretch')

    # Display summary tables
    if combined_summary is not None:
        # Filter to specific prompt mode if requested
        if prompt_mode:
            combined_summary = combined_summary[combined_summary['prompt_mode'] == prompt_mode]
            st.markdown(f"#### 📋 Summary Metrics: {prompt_mode.title()}")
        else:
            st.markdown("#### 📋 Summary Metrics (All Prompt Modes)")

        # Sort by experiment first, then prompt_mode
        if 'prompt_mode' in combined_summary.columns and 'experiment' in combined_summary.columns:
            combined_summary = combined_summary.sort_values(['experiment', 'prompt_mode'])

            # Reorder columns to put experiment and prompt_mode first, then accuracy metrics
            display_cols = ['experiment', 'prompt_mode', 'nash_accuracy', 'utilitarian_accuracy',
                           'rawlsian_accuracy', 'contract_formation_rate',
                           'contract_activation_rate', 'cooperation_rate', 'high_effort_rate']
        else:
            # Fallback column order
            display_cols = ['experiment', 'nash_accuracy', 'utilitarian_accuracy',
                           'rawlsian_accuracy', 'contract_formation_rate',
                           'contract_activation_rate', 'cooperation_rate', 'high_effort_rate']

        # Only show columns that exist
        available_cols = [col for col in display_cols if col in combined_summary.columns]

        st.dataframe(
            combined_summary[available_cols],
            width='stretch',
            hide_index=True,
        )
    else:
        st.caption("No summary metrics available")


# Base path (used for breadcrumbs copy + discovery)
log_base_path = Path("logs")

# Discover logs before restoring URL navigation so stale links can fall back safely.
log_dirs = discover_log_dirs(log_base_path)

# Navigation state
initialize_navigation_state(log_dirs, log_base_path)

if 'selected_experiment' not in st.session_state:
    st.session_state.selected_experiment = None

if 'selected_prompt_mode' not in st.session_state:
    st.session_state.selected_prompt_mode = None

if 'selected_contract_mode' not in st.session_state:
    st.session_state.selected_contract_mode = None

if 'selected_sample_id' not in st.session_state:
    st.session_state.selected_sample_id = None

# Store sorted list of sample IDs for navigation (initialized when scenarios are loaded)
if 'sample_ids_list' not in st.session_state:
    st.session_state.sample_ids_list = None

# Game type filter state (for home page)
if 'selected_game_types' not in st.session_state:
    st.session_state.selected_game_types = None  # None = show all

sync_navigation_query_params()

# Get navigation path
nav_path = st.session_state.nav_path
current_level = nav_path[-1] if nav_path else 'home'

# Title
st.title("💬 GT-HarmBench Contracting Traces Viewer")

nav_copy_text = build_navigation_copy_path(
    log_base_path,
    nav_path,
    st.session_state.selected_experiment,
    st.session_state.selected_prompt_mode,
    st.session_state.selected_contract_mode,
    st.session_state.selected_sample_id,
)

# Render breadcrumbs (+ copy current path)
render_breadcrumbs(
    nav_path,
    log_base=log_base_path,
    log_dirs=log_dirs,
    copy_path_text=nav_copy_text,
)

# ==================== HOME: Select Experiment ====================
if current_level == 'home':
    misc_top = discover_misc_top_level_dirs(log_base_path)

    if log_dirs:
        # Game type filter UI
        st.markdown("#### 📁 Eval runs")

        # Collect all unique game types from experiments
        all_game_types = set()
        for exp_name in log_dirs.keys():
            parsed = parse_experiment_name(exp_name)
            if parsed.get('game_type'):
                all_game_types.add(parsed['game_type'])

        # Create filter options with "All" as default
        game_type_options = ['All'] + sorted(all_game_types)

        # Map game type codes to labels for display
        game_type_labels = {
            'All': 'All Game Types',
            'PD': "Prisoner's Dilemma",
            'SH': 'Stag Hunt',
            'CO': 'Coordination',
        }

        # Display filter as a horizontal button bar
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("Filter by game type:")
        with col2:
            selected_filter = st.selectbox(
                "Game Type Filter",
                options=game_type_options,
                format_func=lambda x: game_type_labels.get(x, x),
                label_visibility="collapsed",
                key="game_type_filter_select",
                index=0 if st.session_state.selected_game_types is None else game_type_options.index(st.session_state.selected_game_types) if st.session_state.selected_game_types in game_type_options else 0,
            )

            # Update session state when filter changes
            if selected_filter == 'All':
                st.session_state.selected_game_types = None
            else:
                st.session_state.selected_game_types = selected_filter

        # Show filter status indicator
        if st.session_state.selected_game_types is not None:
            st.info(f"🔍 Showing only **{game_type_labels.get(st.session_state.selected_game_types, st.session_state.selected_game_types)}** experiments")

        # Build experiment data
        exp_data: list[dict[str, object]] = []
        for exp_name in sorted(log_dirs.keys(), reverse=True):
            row = build_eval_run_tile_row(exp_name, log_base_path, log_dirs)
            if row:
                # Filter by game type if a filter is selected
                if st.session_state.selected_game_types is not None:
                    if row.get('game_type') != st.session_state.selected_game_types:
                        continue
                exp_data.append(row)

        # One horizontal row per chunk; always use three columns so a lone last card does
        # not expand to full row width (avoids stretched / centered mega-buttons).
        def _home_eval_label(exp, _idx):
            return _eval_tile_button_content(exp), f"exp_{exp['name']}"

        def _home_eval_click(exp):
            st.session_state.nav_path.append(str(exp['name']))
            st.session_state.selected_experiment = str(exp['name'])
            st.rerun()

        _render_tile_grid(
            exp_data, "harmbench_pick_experiments", "home_eval",
            label_fn=_home_eval_label, on_click=_home_eval_click,
        )
    else:
        st.info(f"No top-level contracting eval runs under `{log_base_path}`.")

    if misc_top:
        st.markdown("#### 📂 Other log folders")
        st.caption("Archives and other layouts; contracting runs nested here appear as eval cards.")

        eval_misc_rows: list[dict[str, object]] = []
        folder_misc: list[str] = []
        for name in sorted(misc_top):
            child = (log_base_path / name).resolve()
            rel_exp = _experiment_rel_for_path(log_base_path, child)
            if rel_exp and is_contracting_experiment_layout(child):
                row = build_eval_run_tile_row(rel_exp, log_base_path, log_dirs)
                if row:
                    # Apply game type filter to misc eval runs as well
                    if st.session_state.selected_game_types is not None:
                        if row.get('game_type') != st.session_state.selected_game_types:
                            continue
                    eval_misc_rows.append(row)
                    continue
            folder_misc.append(name)

        eval_misc_rows.sort(key=lambda r: str(r['name']), reverse=True)

        if eval_misc_rows:
            def _misc_eval_label(exp, _idx):
                return _eval_tile_button_content(exp), f"misc_eval_{str(exp['name']).replace('/', '__')}"

            def _misc_eval_click(exp):
                st.session_state.nav_path = ["home", str(exp['name'])]
                _sync_eval_selections_from_nav(
                    st.session_state.nav_path, log_base_path, log_dirs
                )
                st.rerun()

            _render_tile_grid(
                eval_misc_rows, "harmbench_pick_experiments", "misc_eval",
                label_fn=_misc_eval_label, on_click=_misc_eval_click,
            )

        if folder_misc:
            def _folder_misc_label(name, _idx):
                return name, f"misc_home_{name}"

            def _folder_misc_click(name):
                st.session_state.nav_path = ["home", MISC_BROWSE_ROOT, name]
                _sync_eval_selections_from_nav(
                    st.session_state.nav_path, log_base_path, log_dirs
                )
                st.rerun()

            _render_tile_grid(
                folder_misc, "harmbench_pick_misc", "misc_folder",
                label_fn=_folder_misc_label, on_click=_folder_misc_click,
            )

    if not log_dirs and not misc_top:
        st.warning(f"No directories to browse under `{log_base_path}`.")
        st.stop()


# ==================== MISC LOG BROWSER ====================
elif _in_misc_browse(nav_path):
    current = _misc_dir_path(log_base_path, nav_path)
    if current is None or not current.is_dir():
        st.warning("That folder is missing or is not a directory.")
        st.session_state.nav_path = ["home"]
        _sync_eval_selections_from_nav(st.session_state.nav_path, log_base_path, log_dirs)
        st.rerun()

    at_misc_top = len(nav_path) == 2
    st.markdown("#### 📂 Browse logs")
    rel = "/".join(_misc_segments(nav_path)) if _misc_segments(nav_path) else ""
    st.caption(f"`{current}`" + (f" — `{rel}`" if rel else ""))

    if at_misc_top:
        subdirs = discover_misc_top_level_dirs(log_base_path)
    else:
        subdirs = sorted(
            p.name
            for p in current.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    files = sorted(
        p for p in current.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )

    key_scope = "__".join(_misc_segments(nav_path)).replace(" ", "_") or "root"

    eval_rows: list[dict[str, object]] = []
    folder_entries: list[tuple[str, Path]] = []
    for name in subdirs:
        child = (current / name).resolve()
        try:
            child.relative_to(log_base_path.resolve())
        except ValueError:
            continue
        rel_exp = _experiment_rel_for_path(log_base_path, child)
        if rel_exp and is_contracting_experiment_layout(child):
            row = build_eval_run_tile_row(rel_exp, log_base_path, log_dirs)
            if row:
                # Apply game type filter to misc browse eval runs as well
                if st.session_state.selected_game_types is not None:
                    if row.get('game_type') != st.session_state.selected_game_types:
                        continue
                eval_rows.append(row)
                continue
        folder_entries.append((name, child))

    eval_rows.sort(key=lambda r: str(r['name']), reverse=True)

    if eval_rows:
        st.markdown("##### Eval runs")

        def _browse_eval_label(exp, _idx):
            return _eval_tile_button_content(exp), f"misc_browse_eval_{key_scope}_{str(exp['name']).replace('/', '__')}"

        def _browse_eval_click(exp):
            st.session_state.nav_path = ["home", str(exp['name'])]
            _sync_eval_selections_from_nav(
                st.session_state.nav_path, log_base_path, log_dirs
            )
            st.rerun()

        _render_tile_grid(
            eval_rows, "harmbench_pick_experiments", "browse_eval",
            label_fn=_browse_eval_label, on_click=_browse_eval_click,
        )

    if folder_entries:
        st.markdown("##### Folders")

        def _browse_folder_label(entry, _idx):
            return entry[0], f"misc_open_{key_scope}_{entry[0]}"

        def _browse_folder_click(entry):
            st.session_state.nav_path = [*nav_path, entry[0]]
            st.rerun()

        _render_tile_grid(
            folder_entries, "harmbench_pick_misc", "browse_folder",
            label_fn=_browse_folder_label, on_click=_browse_folder_click,
        )

    if files:
        with st.expander(f"Files here ({len(files)})"):
            rows = [
                {"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1)}
                for p in files
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if len(nav_path) > 2:
        if st.button("← Up one folder"):
            st.session_state.nav_path = nav_path[:-1]
            st.rerun()
    elif st.button("← Back to home"):
        st.session_state.nav_path = ["home"]
        _sync_eval_selections_from_nav(st.session_state.nav_path, log_base_path, log_dirs)
        st.rerun()


# ==================== EXPERIMENT: Select Prompt Mode ====================
elif st.session_state.selected_experiment and current_level == st.session_state.selected_experiment:
    exp_name = st.session_state.selected_experiment
    parsed = parse_experiment_name(exp_name)

    st.header(f"📁 {exp_name}")
    # Build caption with game type if present
    game_suffix = f" | {parsed['game_type_label']}" if parsed.get('game_type_label') else ""
    st.caption(f"{parsed['model']}{game_suffix} | {parsed['date']} {parsed['time']}")

    prompt_dirs = resolve_experiment_prompt_dirs(log_base_path, exp_name, log_dirs)
    if not prompt_dirs:
        st.warning("This path does not look like a contracting eval (expected base/selfish/cooperative with code-nl / code-law runs).")
        if st.button("← Back to home"):
            st.session_state.nav_path = ["home"]
            _sync_eval_selections_from_nav(st.session_state.nav_path, log_base_path, log_dirs)
            st.rerun()
        st.stop()

    prompt_modes = sorted(prompt_dirs.keys())

    # Precompute prompt-mode tile data
    _prompt_tile_data = {}
    for pm in prompt_modes:
        contract_modes = [
            md.name for md in prompt_dirs[pm].iterdir()
            if md.is_dir() and any(n in md.name for n in _CONTRACTING_EVAL_SUBDIR_MARKERS)
        ]
        sig = _mtime_signature_for_eval_paths(
            iter_contract_eval_files_under_prompt_dir(prompt_dirs[pm])
        )
        _prompt_tile_data[pm] = {
            'contract_modes': contract_modes,
            'trace_count': _cached_contract_trace_row_total(sig),
        }

    def _prompt_label(pm, _idx):
        td = _prompt_tile_data[pm]
        label = f"""{pm.title()}

{len(td['contract_modes'])} modes | {td['trace_count']:,} traces"""
        return label, f"prompt_{pm}"

    def _prompt_click(pm):
        st.session_state.nav_path.append(pm)
        st.session_state.selected_prompt_mode = pm
        st.rerun()

    _render_tile_grid(
        prompt_modes, "harmbench_pick_prompts", "prompt",
        label_fn=_prompt_label, on_click=_prompt_click,
    )

    # Back button
    if st.button("← Back to Experiments"):
        st.session_state.nav_path = st.session_state.nav_path[:-1]
        st.session_state.selected_experiment = None
        st.rerun()

    # Display analysis previews at the bottom of experiment page
    exp_root = experiment_root_path(log_base_path, exp_name, log_dirs)
    exp_dir = exp_root if exp_root is not None else (log_base_path / exp_name)
    render_analysis_previews(exp_dir)


# ==================== PROMPT MODE: Select Contract Mode ====================
elif (st.session_state.selected_experiment and
      st.session_state.selected_prompt_mode and
      current_level == st.session_state.selected_prompt_mode):

    exp_name = st.session_state.selected_experiment
    prompt_mode = st.session_state.selected_prompt_mode

    st.header(f"⚙️ {prompt_mode.title()}")
    st.caption(f"Experiment: {exp_name}")

    # Get contract modes with counts
    mode_counts = get_eval_file_counts(log_base_path, exp_name, prompt_mode)

    if not mode_counts:
        st.warning(f"No contract modes found")
        if st.button("← Back"):
            st.session_state.nav_path = st.session_state.nav_path[:-1]
            st.session_state.selected_prompt_mode = None
            st.rerun()
        st.stop()

    mode_names = sorted(mode_counts.keys())

    def _mode_label(mode_name, _idx):
        display_name = mode_name.replace('_', ' ').replace('-', ' ').title()
        label = f"**{display_name}**\n\n{mode_counts[mode_name]} eval files"
        return label, f"mode_{mode_name}"

    def _mode_click(mode_name):
        st.session_state.nav_path.append(mode_name)
        st.session_state.selected_contract_mode = mode_name
        st.rerun()

    _render_tile_grid(
        mode_names, "harmbench_pick_contract_modes", "contract_mode",
        label_fn=_mode_label, on_click=_mode_click,
    )

    # Back button
    if st.button("← Back to Prompt Modes"):
        st.session_state.nav_path = st.session_state.nav_path[:-1]
        st.session_state.selected_prompt_mode = None
        st.rerun()

    # Display analysis previews for this prompt mode
    exp_root = experiment_root_path(log_base_path, exp_name, log_dirs)
    exp_dir = exp_root if exp_root is not None else (log_base_path / exp_name)
    render_analysis_previews(exp_dir, prompt_mode=prompt_mode)


# ==================== CONTRACT MODE: Browse Scenarios (TABLE) ====================
elif (st.session_state.selected_experiment and
      st.session_state.selected_prompt_mode and
      st.session_state.selected_contract_mode and
      current_level == st.session_state.selected_contract_mode):

    # Load traces
    with st.spinner(f"Loading traces from {st.session_state.selected_contract_mode}..."):
        traces_df = load_contracting_traces(
            log_base_path,
            experiments=(st.session_state.selected_experiment,),
            prompt_modes=(st.session_state.selected_prompt_mode,)
        )

    # Filter to selected contract mode
    traces_df = traces_df[traces_df['contract_mode'] == st.session_state.selected_contract_mode]

    if traces_df.empty:
        st.warning("No traces found for the selected combination.")
        if st.button("← Back"):
            st.session_state.nav_path = st.session_state.nav_path[:-1]
            st.session_state.selected_contract_mode = None
            st.rerun()
        st.stop()

    # Get scenario statistics
    scenario_stats = get_scenario_stats(traces_df)

    # Display header
    mode_display = st.session_state.selected_contract_mode.replace('_', ' ').replace('-', ' ').title()
    st.subheader(f"📋 {mode_display}")
    st.caption(f"Loaded {len(traces_df)} traces from {len(scenario_stats)} game scenarios")

    # Display scenarios table - CLICKABLE ROWS
    st.markdown("### 🎮 Game Scenarios (click Sample ID to view traces)")

    # Store the dataframe in session state for click handling
    st.session_state.scenario_df = scenario_stats.copy()
    st.session_state.scenario_df = st.session_state.scenario_df.sort_values('Sample ID')

    # Store sorted list of sample IDs for left/right navigation
    st.session_state.sample_ids_list = st.session_state.scenario_df['Sample ID'].tolist()

    # Make Sample IDs visually distinct in the table
    def make_clickable_sample_id(sample_id: str) -> str:
        return f"🔗 {sample_id}"

    st.session_state.scenario_df['_clickable_sample'] = st.session_state.scenario_df['Sample ID'].apply(
        make_clickable_sample_id
    )

    # Reorder columns with clickable sample first
    cols_order = [
        '_clickable_sample',
        'Game Type',
        'Num Traces',
        'Formation Rate',
        'Activation Rate',
        'Avg Turns',
        'Nash Rate',
        'Utilitarian Rate',
        'Rawlsian Rate',
        'High Effort Rate',
    ]
    st.session_state.scenario_df = st.session_state.scenario_df[cols_order]

    # Use column configurator with link-like styling
    column_config = {
        "_clickable_sample": st.column_config.TextColumn(
            "Sample ID",
            width="medium",
            help="Click to view traces"
        ),
        "Game Type": st.column_config.TextColumn("Game Type", width="medium"),
        "Num Traces": st.column_config.NumberColumn("Traces", width="small"),
        "Formation Rate": st.column_config.TextColumn("Formation %", width="small"),
        "Activation Rate": st.column_config.TextColumn("Activation %", width="small"),
        "Avg Turns": st.column_config.TextColumn("Avg Turns", width="small"),
        "Nash Rate": st.column_config.NumberColumn(
            "Nash %",
            width="small",
            format="percent",
            help="Share of traces whose selected joint action matches a Nash equilibrium",
        ),
        "Utilitarian Rate": st.column_config.NumberColumn(
            "Utilitarian %",
            width="small",
            format="percent",
            help="Share of traces whose selected joint action matches a utilitarian optimum",
        ),
        "Rawlsian Rate": st.column_config.NumberColumn(
            "Rawlsian %",
            width="small",
            format="percent",
            help="Share of traces whose selected joint action matches a Rawlsian optimum",
        ),
        "High Effort Rate": st.column_config.NumberColumn(
            "High Effort %",
            width="small",
            format="percent",
            help="Share of available row and column effort choices marked High Effort",
        ),
    }

    # Display dataframe with selection enabled
    event = st.dataframe(
        st.session_state.scenario_df,
        width='stretch',
        hide_index=True,
        column_config=column_config,
        selection_mode="single-row",
        on_select="rerun",
        key="scenarios_table"
    )

    # Handle selection - works with both checkbox and row click
    if event.selection and len(event.selection['rows']) > 0:
        selected_idx = event.selection['rows'][0]
        # Get original sample ID from the dataframe (before we added _clickable_sample)
        selected_sample_id = st.session_state.scenario_df.iloc[selected_idx]['_clickable_sample'].replace('🔗 ', '')

        st.session_state.selected_sample_id = selected_sample_id
        st.session_state.nav_path.append(f"sample_{selected_sample_id}")
        st.rerun()

    # Back button
    if st.button("← Back to Contract Modes"):
        st.session_state.nav_path = st.session_state.nav_path[:-1]
        st.session_state.selected_contract_mode = None
        st.rerun()


# ==================== SAMPLE VIEW: View Traces ====================
elif st.session_state.selected_sample_id and current_level.startswith('sample_'):
    sample_id = st.session_state.selected_sample_id

    # Load traces with cache invalidation when navigation context changes (otherwise,
    # visiting e.g. 4x4-code-law then 4x4-code-nl can show stale law traces for same sample).
    traces_cache_key = (
        st.session_state.selected_experiment,
        st.session_state.selected_prompt_mode,
        st.session_state.selected_contract_mode,
    )
    needs_reload = (
        'traces_df' not in st.session_state
        or st.session_state.get('traces_df_cache_key') != traces_cache_key
    )
    if needs_reload:
        with st.spinner("Loading traces..."):
            st.session_state.traces_df = load_contracting_traces(
                log_base_path,
                experiments=(st.session_state.selected_experiment,),
                prompt_modes=(st.session_state.selected_prompt_mode,)
            )
            st.session_state.traces_df = st.session_state.traces_df[
                st.session_state.traces_df['contract_mode'] == st.session_state.selected_contract_mode
            ]
            st.session_state.traces_df_cache_key = traces_cache_key

    traces_df = st.session_state.traces_df
    sample_traces = traces_df[traces_df['sample_id'] == sample_id]

    if sample_traces.empty:
        st.warning(f"No traces found for sample {sample_id}")
        st.session_state.nav_path = st.session_state.nav_path[:-1]
        st.session_state.selected_sample_id = None
        st.rerun()

    # Display header with left/right navigation
    sample_ids = st.session_state.sample_ids_list or []
    current_idx = sample_ids.index(sample_id) if sample_id in sample_ids else 0
    total_samples = len(sample_ids)
    can_go_prev = current_idx > 0
    can_go_next = current_idx < total_samples - 1

    col_left, col_center, col_right = st.columns([2, 6, 2])

    with col_left:
        if st.button("⬅️ Previous", key="nav_prev", disabled=not can_go_prev):
            if can_go_prev:
                st.session_state.selected_sample_id = sample_ids[current_idx - 1]
                st.session_state.nav_path[-1] = f"sample_{st.session_state.selected_sample_id}"
                st.rerun()

    with col_center:
        # Position indicator and sample header
        st.markdown(f"""
            <div style="text-align: center; margin-top: 0.5rem;">
                <h3 style="margin: 0;">🎮 Sample {sample_id}</h3>
                <small style="color: #888;">{current_idx + 1} of {total_samples}</small>
            </div>
        """, unsafe_allow_html=True)

    with col_right:
        if st.button("Next ➡️", key="nav_next", disabled=not can_go_next):
            if can_go_next:
                st.session_state.selected_sample_id = sample_ids[current_idx + 1]
                st.session_state.nav_path[-1] = f"sample_{st.session_state.selected_sample_id}"
                st.rerun()

    first_trace = sample_traces.iloc[0].to_dict()
    metadata = get_metadata_summary(first_trace)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metadata['formal_game']}</div>
                <div class="metric-label">Game Type</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        formation_rate = sample_traces['contract_formed'].mean()
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{formation_rate:.1%}</div>
                <div class="metric-label">Formation Rate</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(sample_traces)}</div>
                <div class="metric-label">Num Traces</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Display traces
    for display_idx, (df_index, trace_row) in enumerate(sample_traces.iterrows()):
        trace = trace_row.to_dict()

        with st.expander(
            f"Trace {display_idx + 1}: {trace['prompt_mode']} - "
            f"{'✓ Formed' if trace['contract_formed'] else '✗ Not formed'} - "
            f"{len(trace.get('negotiation', []))} turns",
            expanded=(display_idx == 0)
        ):
            # Use DataFrame index for unique trace identifier
            render_trace_detail(trace, int(df_index))

    # Back button
    if st.button("← Back to Scenarios"):
        st.session_state.nav_path = st.session_state.nav_path[:-1]
        st.session_state.selected_sample_id = None
        st.rerun()

else:
    # Fallback - reset to home
    st.session_state.nav_path = ["home"]
    _sync_eval_selections_from_nav(st.session_state.nav_path, log_base_path, log_dirs)
    st.rerun()
