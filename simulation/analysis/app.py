import importlib.util
import json
import os
import traceback
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
from flask_caching import Cache
from plotly.subplots import make_subplots

from .preprocessing import get_data


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
        df = pd.read_json(run_path, orient="records")
        df = df.replace([np.nan, np.inf, -np.inf], None)
        run_records[row["run_id"]] = df.to_dict(orient="records")
    return run_records


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


def _ensure_fishing_run_summary_json(
    base_path: str, group: str, name: str
) -> Optional[Dict[str, Any]]:
    """Return fishing run summary dict; create fishing_run_summary.json from log_env if missing."""
    run_dir = os.path.join(base_path, group, name)
    log_path = os.path.join(run_dir, "log_env.json")
    summary_path = os.path.join(run_dir, "fishing_run_summary.json")
    if not os.path.isfile(log_path):
        return None
    if os.path.isfile(summary_path):
        try:
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
        summary = _ensure_fishing_run_summary_json(base_path, row["group"], row["name"])
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

        df = pd.read_json(run_path, orient="records")
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
    return {"subsets": subsets}


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
    run_data = data.get("run_data", {})
    resource_in_pool = _load_resource_time_series(summary_df)
    contracting_data = _load_contracting_events(summary_df)
    run_records = _load_run_records(summary_df)
    fishing_summaries = _load_fishing_summaries(summary_df)
    summary_runs = []
    if summary_df is not None:
        cleaned_summary = summary_df.reset_index().replace([np.nan, np.inf, -np.inf], None)
        for row in cleaned_summary.to_dict(orient="records"):
            row["run_key"] = row.get("run_id")
            row["run_label"] = _build_run_label(row)
            row["chart_key"] = _build_chart_key(row)
            summary_runs.append(row)

    return {
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


if __name__ == "__main__":
    app.run(
        debug=_env_flag("SIM_ANALYSIS_DEBUG", False),
        host=os.getenv("SIM_ANALYSIS_HOST", "0.0.0.0"),
        port=int(os.getenv("SIM_ANALYSIS_PORT", "8050")),
    )
