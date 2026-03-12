import json
import os

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
    resource_in_pool = data.get("resource_in_pool", {})
    contracting_data = _load_contracting_events(summary_df)
    run_records = _load_run_records(summary_df)
    summary_runs = []
    if summary_df is not None:
        cleaned_summary = summary_df.reset_index().replace([np.nan, np.inf, -np.inf], None)
        for row in cleaned_summary.to_dict(orient="records"):
            row["run_key"] = row.get("run_id")
            row["run_label"] = _build_run_label(row)
            row["chart_key"] = row.get("name")
            summary_runs.append(row)

    return {
        "summary_groups": _serialize_dataframe(summary_group_df),
        "summary_runs": summary_runs,
        "run_data": run_records,
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
