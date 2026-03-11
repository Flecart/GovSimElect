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


@cache.memoize()
def global_store(value):
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


if __name__ == "__main__":
    app.run(
        debug=_env_flag("SIM_ANALYSIS_DEBUG", False),
        host=os.getenv("SIM_ANALYSIS_HOST", "0.0.0.0"),
        port=int(os.getenv("SIM_ANALYSIS_PORT", "8050")),
    )
