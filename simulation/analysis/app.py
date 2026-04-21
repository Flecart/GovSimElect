import importlib.util
import json
import os
import traceback
import time
from pathlib import Path
from typing import Any, Dict, Optional

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.stats.api as sms
from dash import Input, Output, State, dcc, html
from flask import jsonify, request
from flask_caching import Cache
from plotly.subplots import make_subplots

from .preprocessing import get_data, get_summary_runs


def get_available_subsets():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    if not os.path.isdir(base_path):
        return []
    return sorted(
        entry
        for entry in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, entry))
    )


def get_home_layout(message=None):
    subsets = get_available_subsets()
    children = [
        dmc.Title("GovSim Analysis", order=1),
        dmc.Text("Select a result subset to browse the recorded runs."),
    ]
    if message:
        children.append(dmc.Alert(message, color="yellow", title="Notice"))

    if subsets:
        children.append(
            dmc.Stack(
                [
                    dcc.Link(
                        subset,
                        href=f"/{subset}",
                        style={"fontSize": "1.1rem"},
                    )
                    for subset in subsets
                ],
                spacing="xs",
            )
        )
    else:
        children.append(
            dmc.Alert(
                "No subsets were found under simulation/results yet.",
                color="blue",
                title="No Data",
            )
        )

    return dmc.Container(dmc.Stack(children, spacing="md"), size="lg", pt="xl")

# Setup app
app = dash.Dash(
    __name__,
    external_stylesheets=[  # include google fonts
        "https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;900&display=swap"
    ],
    suppress_callback_exceptions=True,
)

cache = Cache()
server = app.server
cache.init_app(
    server,
    config={
        "CACHE_TYPE": "SimpleCache",
    },
)


@cache.memoize(timeout=5)
def global_store(value):
    # Used by the Dash UI; short TTL allows "live-ish" refreshes when new results land.
    return get_data(value)


# Define the app layout with Location and a content div

app.layout = dmc.MantineProvider(
    theme={
        "fontFamily": "'Inter', sans-serif",
        "primaryColor": "indigo",
        "components": {
            "Button": {"styles": {"root": {"fontWeight": 400}}},
            "Alert": {"styles": {"title": {"fontWeight": 500}}},
            "AvatarGroup": {"styles": {"truncated": {"fontWeight": 500}}},
        },
    },
    inherit=True,
    withGlobalStyles=True,
    withNormalizeCSS=True,
    children=[
        html.Div(
            [
                dcc.Location(id="url", refresh=False),
                dcc.Store(id="runs-group", storage_type="session"),
                html.Div(id="page-content"),
            ]
        )
    ],
)


# Define the callback for dynamic page loading
@app.callback(
    [Output("page-content", "children"), Output("runs-group", "data")],
    [Input("url", "pathname")],
)
def display_page(pathname):
    pathname = pathname or "/"
    clean_parts = [part for part in pathname.split("/") if part]
    if not clean_parts:
        return get_home_layout(), None

    if "details" not in pathname:
        from .group import group

        group_name = clean_parts[0]
        if group_name not in get_available_subsets():
            return get_home_layout(f"Unknown subset: {group_name}"), None

        return group, group_name
    else:
        from .details import details_layout

        group_name = clean_parts[0]
        if group_name not in get_available_subsets():
            return get_home_layout(f"Unknown subset: {group_name}"), None

        return details_layout, group_name


#
from .details import *
from .group import *


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _serialize_dataframe(df):
    if df is None:
        return []
    cleaned = df.reset_index()
    # Replace NaN / inf with None so JSON is valid
    cleaned = cleaned.replace([np.nan, np.inf, -np.inf], None)
    return cleaned.to_dict(orient="records")


def _sanitize_json_value(value):
    """Recursively convert NaN/inf values to None for strict JSON responses."""
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    return value


def _build_run_label(row):
    name = row.get("name")
    if isinstance(name, str) and name not in {"", "."}:
        return name

    group = row.get("group", "")
    if isinstance(group, str) and group:
        return group.split("/")[-1]

    run_id = row.get("run_id", "")
    if isinstance(run_id, str) and run_id:
        return run_id.split("/")[0]
    return "run"


def _build_chart_key(row):
    name = row.get("name")
    if isinstance(name, str) and name not in {"", "."}:
        return name
    return _build_run_label(row)


def _build_summary_runs(summary_df):
    summary_runs = []
    if summary_df is None or summary_df.empty:
        return summary_runs

    cleaned_summary = summary_df.reset_index().replace([np.nan, np.inf, -np.inf], None)
    for row in cleaned_summary.to_dict(orient="records"):
        row["run_key"] = row.get("run_id")
        row["run_label"] = _build_run_label(row)
        row["chart_key"] = _build_chart_key(row)
        summary_runs.append(row)
    return summary_runs


_INDEX_CACHE_TTL_SECONDS = 60.0
_GROUP_CACHE_TTL_SECONDS = 30.0
_index_cache = {}
_run_response_cache = {}
_group_response_cache = {}


def _fast_subset_index(subset_name):
    """Build lightweight run index from filesystem only (no YAML flattening)."""
    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results", subset_name)
    )
    if not os.path.isdir(base_path):
        return {"summary_groups": [], "summary_runs": []}

    summary_runs = []
    groups = {}
    for group in sorted(os.listdir(base_path)):
        group_path = os.path.join(base_path, group)
        if not os.path.isdir(group_path):
            continue

        for root, _, files in os.walk(group_path):
            if "log_env.json" not in files:
                continue
            rel_run_path = os.path.relpath(root, group_path)
            rel_parent = os.path.dirname(rel_run_path)
            full_group = (
                f"{subset_name}/{group}"
                if rel_parent in {"", "."}
                else f"{subset_name}/{group}/{rel_parent}"
            )
            run_id = f"{group}/{rel_run_path}"
            run_row = {
                "name": rel_run_path,
                "group": full_group,
                "run_id": run_id,
            }
            run_row["run_key"] = run_id
            run_row["run_label"] = _build_run_label(run_row)
            run_row["chart_key"] = _build_chart_key(run_row)
            summary_runs.append(run_row)
            groups[full_group] = {"id": full_group, "group": full_group}

    summary_runs.sort(key=lambda row: (str(row.get("group", "")), str(row.get("name", ""))))
    sorted_groups = sorted(groups.values(), key=lambda row: str(row.get("group", "")))
    return {"summary_groups": sorted_groups, "summary_runs": summary_runs}


def _cached_subset_index(subset_name):
    now = time.time()
    cached = _index_cache.get(subset_name)
    if cached and now - cached["created_at"] < _INDEX_CACHE_TTL_SECONDS:
        return cached["payload"]
    payload = _fast_subset_index(subset_name)
    _index_cache[subset_name] = {"created_at": now, "payload": payload}
    return payload


def _cached_group_response(cache_key, builder):
    now = time.time()
    cached = _group_response_cache.get(cache_key)
    if cached and now - cached["created_at"] < _GROUP_CACHE_TTL_SECONDS:
        return cached["payload"]
    payload = builder()
    _group_response_cache[cache_key] = {"created_at": now, "payload": payload}
    return payload


def _load_contracting_events(summary_df):
    if summary_df is None or summary_df.empty:
        return {}

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    contracting_data = {}
    for _, row in summary_df.iterrows():
        run_path = os.path.join(
            base_path, row["group"], row["name"], "contracting_results.jsonl"
        )
        if not os.path.exists(run_path):
            continue

        events = []
        with open(run_path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        contracting_data[row["run_id"]] = events
    return contracting_data


def _load_run_records(summary_df):
    if summary_df is None or summary_df.empty:
        return {}

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    run_records = {}
    for _, row in summary_df.iterrows():
        run_path = os.path.join(base_path, row["group"], row["name"], "log_env.json")
        if not os.path.exists(run_path):
            run_records[row["run_id"]] = []
            continue
        try:
            df = pd.read_json(run_path, orient="records")
        except (ValueError, OSError):
            # e.g. read while another process replaces the file; next poll usually succeeds
            run_records[row["run_id"]] = []
            continue
        df = df.replace([np.nan, np.inf, -np.inf], None)
        run_records[row["run_id"]] = df.to_dict(orient="records")
    return run_records


def _load_single_run_records(run_path):
    if not os.path.exists(run_path):
        return []
    try:
        df = pd.read_json(run_path, orient="records")
    except (ValueError, OSError):
        # e.g. read while another process replaces the file; next poll usually succeeds
        return []
    df = df.replace([np.nan, np.inf, -np.inf], None)
    return df.to_dict(orient="records")


def _load_single_contracting_events(contracting_path):
    if not os.path.exists(contracting_path):
        return []
    events = []
    with open(contracting_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


_SUMMARIZE_FISHING_MOD = None


def _get_summarize_fishing_module():
    """Load scripts/summarize_fishing_log.py once (same logic as CLI --json-out)."""
    global _SUMMARIZE_FISHING_MOD
    if _SUMMARIZE_FISHING_MOD is not None:
        return _SUMMARIZE_FISHING_MOD
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "summarize_fishing_log.py"
    spec = importlib.util.spec_from_file_location(
        "govsim_summarize_fishing_log", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load summarizer from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _SUMMARIZE_FISHING_MOD = mod
    return mod


def _write_fishing_run_summary_json(
    base_path: str,
    group: str,
    name: str,
    *,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return fishing run summary dict; write fishing_run_summary.json from log_env.

    If ``force`` is False, a cached ``fishing_run_summary.json`` is returned only when
    it is at least as new as ``log_env.json`` (mtime). Otherwise the summary is
    recomputed so live runs stay in sync with partial logs. If ``force`` is True,
    the summary is always recomputed.
    """
    run_dir = os.path.join(base_path, group, name)
    log_path = os.path.join(run_dir, "log_env.json")
    summary_path = os.path.join(run_dir, "fishing_run_summary.json")
    if not os.path.isfile(log_path):
        return None
    if not force and os.path.isfile(summary_path):
        try:
            log_mtime = os.path.getmtime(log_path)
            summary_mtime = os.path.getmtime(summary_path)
            if summary_mtime >= log_mtime:
                with open(summary_path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
        except (json.JSONDecodeError, OSError):
            pass
    try:
        mod = _get_summarize_fishing_module()
        records = mod.parse_log(Path(log_path))
        report = mod.summarize(
            records,
            capacity=100.0,
            expected_regen=2.0,
            collapse_threshold=0.0,
            verbose=False,
        )
        os.makedirs(run_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2))
        return report
    except Exception as exc:  # noqa: BLE001 — surface to UI
        return {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _load_fishing_summaries(summary_df):
    if summary_df is None or summary_df.empty:
        return {}
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    out = {}
    for _, row in summary_df.iterrows():
        rid = str(row["run_id"])
        summary = _write_fishing_run_summary_json(base_path, row["group"], row["name"])
        out[rid] = summary
    return out


def _load_resource_time_series(summary_df):
    if summary_df is None or summary_df.empty:
        return {}

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    grouped_frames = {}

    for _, row in summary_df.iterrows():
        run_path = os.path.join(base_path, row["group"], row["name"], "log_env.json")
        if not os.path.exists(run_path):
            continue

        try:
            df = pd.read_json(run_path, orient="records")
        except (ValueError, OSError):
            continue
        harvest = df[df["action"] == "harvesting"].copy()
        if harvest.empty:
            continue

        chart_key = _build_chart_key(row)
        points = []
        for round_num in sorted(harvest["round"].dropna().unique()):
            round_rows = harvest[harvest["round"] == round_num]
            if round_rows.empty:
                continue
            before = round_rows.iloc[0].get("resource_in_pool_before_harvesting")
            after = round_rows.iloc[0].get("resource_in_pool_after_harvesting")
            points.append({"x": float(round_num), "round": int(round_num), chart_key: before})
            points.append(
                {
                    "x": float(round_num) + 0.8,
                    "round": int(round_num),
                    chart_key: after,
                }
            )

        run_frame = pd.DataFrame(points).replace([np.nan, np.inf, -np.inf], None)
        grouped_frames.setdefault(row["group"], []).append(run_frame)

    resource_in_pool = {}
    for group, frames in grouped_frames.items():
        merged = None
        for frame in frames:
            if merged is None:
                merged = frame
            else:
                merged = pd.merge(merged, frame, on=["x", "round"], how="outer")
        if merged is not None:
            resource_in_pool[group] = merged.sort_values(["x", "round"]).reset_index(drop=True)
    return resource_in_pool


@server.route("/api/results", methods=["GET"])
def api_list_results():
    subsets = get_available_subsets()
    return _sanitize_json_value({"subsets": subsets})


@server.route("/api/results/<subset_name>", methods=["GET"])
def api_result_subset(subset_name):
    """
    Lightweight JSON API exposing the same preprocessed data used by the Dash app.
    This is designed to be consumed by a separate React frontend.
    """
    if subset_name not in get_available_subsets():
        return {"error": f"Unknown subset: {subset_name}"}, 404

    # For the React frontend, prefer fresh reads so new runs appear quickly.
    # (Dash callbacks still use `global_store` with a short TTL.)
    data = get_data(subset_name)

    summary_group_df = data.get("summary_group_df")
    summary_df = data.get("summary_df")
    ts_resource = _load_resource_time_series(summary_df)
    legacy_resource = data.get("resource_in_pool") or {}
    # Prefer the API time-series builder when it has rows; otherwise use preprocessing
    # (handles partial logs / concurrent harvest layout) so the group dashboard updates.
    resource_in_pool = {}
    for g in set(ts_resource.keys()) | set(legacy_resource.keys()):
        ts_df = ts_resource.get(g)
        leg_df = legacy_resource.get(g)
        if ts_df is not None and not ts_df.empty:
            resource_in_pool[g] = ts_df
        elif leg_df is not None and not leg_df.empty:
            resource_in_pool[g] = leg_df
    contracting_data = _load_contracting_events(summary_df)
    run_records = _load_run_records(summary_df)
    fishing_summaries = _load_fishing_summaries(summary_df)
    summary_runs = _build_summary_runs(summary_df)

    response = {
        "summary_groups": _serialize_dataframe(summary_group_df),
        "summary_runs": summary_runs,
        "run_data": run_records,
        "fishing_summaries": fishing_summaries,
        "resource_in_pool": {
            group_name: df.replace([np.nan, np.inf, -np.inf], None).to_dict(
                orient="records"
            )
            for group_name, df in resource_in_pool.items()
        },
        "contracting_data": {
            row["run_id"]: contracting_data.get(row["run_id"], [])
            for _, row in summary_df.iterrows()
        },
    }
    return _sanitize_json_value(response)


@server.route("/api/results/<subset_name>/index", methods=["GET"])
def api_result_subset_index(subset_name):
    """Lightweight subset index for sidebar/navigation."""
    if subset_name not in get_available_subsets():
        return {"error": f"Unknown subset: {subset_name}"}, 404

    payload = _cached_subset_index(subset_name)
    return _sanitize_json_value(payload)


@server.route("/api/results/<subset_name>/group/<path:group_name>", methods=["GET"])
def api_result_subset_group(subset_name, group_name):
    """Fetch group-level chart data for one experiment group only."""
    if subset_name not in get_available_subsets():
        return {"error": f"Unknown subset: {subset_name}"}, 404

    def build_response():
        summary_df, _ = get_summary_runs(subset_name)
        if summary_df is None or summary_df.empty:
            return {"resource_rows": []}

        resolved_group_name = str(group_name)
        prefix = f"{subset_name}/"
        relative_group = (
            resolved_group_name[len(prefix):]
            if resolved_group_name.startswith(prefix)
            else resolved_group_name
        )
        root_group = relative_group.split("/", 1)[0]
        nested_prefix = relative_group[len(root_group):].lstrip("/")

        root_group_name = f"{subset_name}/{root_group}"
        group_df = summary_df[summary_df["group"].astype(str) == root_group_name]
        if nested_prefix:
            group_df = group_df[
                group_df["name"].astype(str).str.startswith(f"{nested_prefix}/")
                | (group_df["name"].astype(str) == nested_prefix)
            ]
        if group_df.empty:
            return {"error": f"Unknown group: {resolved_group_name}"}, 404

        ts_resource = _load_resource_time_series(group_df)
        resource_df = ts_resource.get(root_group_name)
        if resource_df is None or resource_df.empty:
            return {"resource_rows": []}

        # Keep only run series that belong to the selected nested group.
        allowed_chart_keys = {
            _build_chart_key(row)
            for _, row in group_df.iterrows()
        }
        keep_columns = [
            col
            for col in resource_df.columns
            if col in {"x", "round"} or col in allowed_chart_keys
        ]
        # Fallback: if filtering removed all run columns, return unfiltered data.
        # This avoids empty charts when naming/path normalization mismatches occur.
        if len(keep_columns) > 2:
            resource_df = resource_df[keep_columns]

        return {
            "resource_rows": resource_df.replace([np.nan, np.inf, -np.inf], None).to_dict(
                orient="records"
            )
        }

    payload = _cached_group_response(f"{subset_name}:{group_name}", build_response)
    return _sanitize_json_value(payload)


@server.route("/api/results/<subset_name>/run/<path:run_id>", methods=["GET"])
def api_result_subset_run(subset_name, run_id):
    """Fetch heavy payload for a single run only."""
    if subset_name not in get_available_subsets():
        return {"error": f"Unknown subset: {subset_name}"}, 404

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    run_id = str(run_id)
    run_parts = run_id.split("/", 1)
    if len(run_parts) != 2 or not run_parts[0] or not run_parts[1]:
        return {"error": f"Malformed run_id: {run_id}"}, 404

    group_suffix, run_name = run_parts
    run_group = f"{subset_name}/{group_suffix}"
    run_dir = os.path.join(base_path, run_group, run_name)
    log_path = os.path.join(run_dir, "log_env.json")
    contracting_path = os.path.join(run_dir, "contracting_results.jsonl")
    summary_path = os.path.join(run_dir, "fishing_run_summary.json")

    if not os.path.exists(run_dir):
        return {"error": f"Unknown run_id: {run_id}"}, 404

    cache_key = f"{subset_name}:{run_id}"
    log_mtime = os.path.getmtime(log_path) if os.path.exists(log_path) else None
    contracting_mtime = (
        os.path.getmtime(contracting_path) if os.path.exists(contracting_path) else None
    )
    summary_mtime = os.path.getmtime(summary_path) if os.path.exists(summary_path) else None
    cache_fingerprint = (log_mtime, contracting_mtime, summary_mtime)
    cached = _run_response_cache.get(cache_key)
    if cached and cached.get("fingerprint") == cache_fingerprint:
        return _sanitize_json_value(cached["response"])

    run_row = {
        "run_id": run_id,
        "name": run_name,
        "group": run_group,
    }
    run_row["run_key"] = run_row["run_id"]
    run_row["run_label"] = _build_run_label(run_row)
    run_row["chart_key"] = _build_chart_key(run_row)

    run_records = _load_single_run_records(log_path)
    contracting_data = _load_single_contracting_events(contracting_path)
    fishing_summary = _write_fishing_run_summary_json(base_path, run_group, run_name)

    response = {
        "run": run_row,
        "run_data": run_records,
        "contracting_data": contracting_data,
        "fishing_summary": fishing_summary,
    }
    _run_response_cache[cache_key] = {
        "fingerprint": cache_fingerprint,
        "response": response,
    }
    return _sanitize_json_value(response)


def api_refresh_fishing_summary():
    """Recompute and persist fishing_run_summary.json for one run (same as CLI).

    Invoked via :func:`_intercept_fishing_summary_refresh` in ``before_request`` so POST
    is handled before Dash's catch-all static route (which only allows GET/HEAD/OPTIONS
    and would otherwise return 405).
    """
    payload = request.get_json(silent=True) or {}
    subset_name = payload.get("subset") or payload.get("subset_name")
    run_id = payload.get("run_id")
    if not subset_name:
        return {"error": "Missing subset (or subset_name) in JSON body"}, 400
    if run_id is None:
        return {"error": "Missing run_id in JSON body"}, 400
    subset_name = str(subset_name)
    run_id = str(run_id)

    if subset_name not in get_available_subsets():
        return {"error": f"Unknown subset: {subset_name}"}, 404

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    run_parts = run_id.split("/", 1)
    if len(run_parts) != 2 or not run_parts[0] or not run_parts[1]:
        return {"error": f"Malformed run_id: {run_id}"}, 404

    group_suffix, run_name = run_parts
    run_group = f"{subset_name}/{group_suffix}"
    run_dir = os.path.join(base_path, run_group, run_name)
    if not os.path.isdir(run_dir):
        return {"error": f"Unknown run_id: {run_id}"}, 404

    report = _write_fishing_run_summary_json(
        base_path, run_group, run_name, force=True
    )
    if report is None:
        return {
            "error": "log_env.json not found for this run; cannot summarize.",
        }, 400

    return _sanitize_json_value({"fishing_summary": report})


@server.before_request
def _intercept_fishing_summary_refresh():
    """Run POST /api/fishing-summary/refresh before routing (avoids Dash 405 on POST)."""
    path = request.path.rstrip("/") or "/"
    if path != "/api/fishing-summary/refresh":
        return None
    if request.method != "POST":
        return None
    rv = api_refresh_fishing_summary()
    if isinstance(rv, tuple) and len(rv) == 2:
        body, status = rv
        return jsonify(body), status
    return jsonify(rv)


if __name__ == "__main__":
    app.run(
        debug=_env_flag("SIM_ANALYSIS_DEBUG", False),
        host=os.getenv("SIM_ANALYSIS_HOST", "0.0.0.0"),
        port=int(os.getenv("SIM_ANALYSIS_PORT", "8050")),
    )
