import base64

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.stats.api as sms
from dash import Input, Output, State, dcc, html

from .app import app, global_store
from .plots import get_figures_single_run
from .utils import create_table, generate_colors

details_layout = dmc.Stack(
    [
        dmc.Header(
            ml=30,
            height=100,
            children=[
                dmc.Title("Details", order=1),
                dmc.Text(id="group-name"),
            ],
        ),
        dmc.SimpleGrid(
            cols=2,
            children=[
                html.Div(
                    [
                        dcc.Graph(id="resource-over-time-graph"),
                        dcc.Graph(id="resource-per-persona-graph"),
                        dmc.Slider(
                            id="max-y-axis-slider",
                            value=100,
                            min=0,
                            max=100,
                            step=5,
                            ml=50,
                            mr=50,
                            marks=[
                                {"value": i, "label": str(i)} for i in range(0, 101, 10)
                            ],
                        ),
                    ]
                ),
                dmc.Stack(
                    [
                        dmc.SimpleGrid(
                            cols=2,
                            children=[
                                dmc.Select(id="run-dropdown", label="Run"),
                                dmc.Select(id="run-day", label="Day"),
                            ],
                        ),
                        dmc.Tabs(
                            [
                                dmc.TabsList(
                                    [
                                        dmc.Tab(
                                            "Sheep prompts", value="harvesting-prompts"
                                        ),
                                        dmc.Tab("Conversations", value="conversation"),
                                        dmc.Tab(
                                            "Conversations - anaylsis",
                                            value="conversation-anaylsis",
                                        ),
                                    ]
                                ),
                                dmc.TabsPanel(
                                    [
                                        dmc.Stack(
                                            [
                                                dmc.Select(
                                                    id="persona-dropdown",
                                                    label="Persona",
                                                    data=[
                                                        {
                                                            "label": f"persona_{i}",
                                                            "value": f"persona_{i}",
                                                        }
                                                        for i in range(5)
                                                    ],
                                                    value="persona_0",
                                                ),
                                                html.Div(
                                                    id="harvesting-prompts-render"
                                                ),
                                            ]
                                        )
                                    ],
                                    value="harvesting-prompts",
                                ),
                                dmc.TabsPanel(
                                    children=[
                                        html.Div(id="conversation-display"),
                                    ],
                                    value="conversation",
                                ),
                                dmc.TabsPanel(
                                    children=[
                                        dmc.Stack(
                                            [
                                                dmc.Select(
                                                    id="conversation-analysis-dropdown",
                                                    label="Prompt analysis",
                                                    data=[
                                                        {
                                                            "label": "conversation_resource_limit",
                                                            "value": "conversation_resource_limit",
                                                        },
                                                        {
                                                            "label": (
                                                                "conversation_summary"
                                                            ),
                                                            "value": (
                                                                "conversation_summary"
                                                            ),
                                                        },
                                                    ],
                                                    value="conversation_resource_limit",
                                                ),
                                                html.Div(
                                                    id="conversation-analysis-display"
                                                ),
                                            ]
                                        ),
                                    ],
                                    value="conversation-anaylsis",
                                ),
                            ],
                            color="blue",
                            orientation="horizontal",
                            value="conversation",
                            id="details-tabs",
                        ),
                    ]
                ),
            ],
        ),
    ]
)


@app.callback(
    [
        Output("resource-over-time-graph", "figure"),
        Output("resource-per-persona-graph", "figure"),
    ],
    [
        Input("runs-group", "data"),
        Input("url", "pathname"),
        Input("max-y-axis-slider", "value"),
    ],
)
def update_details_graph(subset_name, url, max_y_axis):
    if not subset_name or not url or "details" not in url:
        return go.Figure(), go.Figure()
    url = "/".join([u for u in url.split("/") if u != ""])
    group_name = "/".join(url.split("/")[:2])
    preprocessing_data = global_store(subset_name)

    fig_num_resource, fig_resource_by_persona, _ = get_figures_single_run(
        preprocessing_data, group_name, max_y_axis=max_y_axis
    )

    return fig_num_resource, fig_resource_by_persona


@app.callback(
    Output("group-name", "children"),
    Input("url", "pathname"),
)
def update_group_name(url):
    if not url or "details" not in url:
        return dash.no_update
    url = "/".join([u for u in url.split("/") if u != ""])
    group_name = "/".join(url.split("/")[:2])
    return group_name


@app.callback(
    [
        Output("run-dropdown", "data"),
        Output("run-dropdown", "value"),
    ],
    [
        Input("runs-group", "data"),
        Input("url", "pathname"),
        Input("resource-over-time-graph", "clickData"),
        Input("resource-per-persona-graph", "clickData"),
    ],
)
def update_conversation_run_selection(
    subset_name, url, clickData_num_resource, clickData_resource_by_persona
):
    if not subset_name:
        return [], None
    preprocessing_data = global_store(subset_name)
    summary_df = preprocessing_data["summary_df"]

    if (
        dash.callback_context.triggered[0]["prop_id"]
        == "resource-over-time-graph.clickData"
    ):
        run_name = clickData_num_resource["points"][0]["customdata"][0]

        return dash.no_update, run_name
    elif (
        dash.callback_context.triggered[0]["prop_id"]
        == "resource-per-persona-graph.clickData"
    ):
        run_name = clickData_resource_by_persona["points"][0]["customdata"][0]

        return dash.no_update, run_name
    else:
        if not url or "details" not in url:
            return [], None
        url = "/".join([u for u in url.split("/") if u != ""])
        group_name = "/".join(url.split("/")[:2])
        runs = summary_df[summary_df["group"] == group_name]["name"]
        if runs.empty:
            return [], None
        return [{"label": i, "value": i} for i in runs], runs.iloc[0]


@app.callback(
    [
        Output("run-day", "data"),
        Output("run-day", "value"),
        Output("persona-dropdown", "value"),
        Output("details-tabs", "value"),
        Output("conversation-analysis-dropdown", "value"),
    ],
    [
        Input("runs-group", "data"),
        Input("run-dropdown", "value"),
        Input("resource-over-time-graph", "clickData"),
        Input("resource-per-persona-graph", "clickData"),
    ],
)
def update_conversation_day_selection(
    subset_name, run_name, clickData_num_resource, clickData_resource_by_persona
):
    if not subset_name or not run_name:
        return [], None, dash.no_update, "conversation", dash.no_update
    preprocessing_data = global_store(subset_name)
    run_data = preprocessing_data["run_data"]
    if run_name not in run_data:
        return [], None, dash.no_update, "conversation", dash.no_update
    max_round = run_data[run_name]["round"].max()

    day = 0
    persona = dash.no_update

    tab = "conversation"
    convesation_analysis = dash.no_update
    if (
        dash.callback_context.triggered[0]["prop_id"]
        == "resource-over-time-graph.clickData"
    ):
        day = clickData_num_resource["points"][0]["customdata"][1]
        persona = dash.no_update
        tab = "conversation"
    elif (
        dash.callback_context.triggered[0]["prop_id"]
        == "resource-per-persona-graph.clickData"
    ):
        day = clickData_resource_by_persona["points"][0]["x"]
        persona = clickData_resource_by_persona["points"][0]["customdata"][1]
        tab = "harvesting-prompts"

        if persona == "resource_limit":
            persona = dash.no_update
            tab = "conversation-anaylsis"
            convesation_analysis = "conversation_resource_limit"

    return (
        [{"label": i, "value": i} for i in range(max_round + 1)],
        day,
        persona,
        tab,
        convesation_analysis,
    )


@app.callback(
    Output("conversation-display", "children"),
    [
        Input("runs-group", "data"),
        Input("url", "pathname"),
        Input("run-dropdown", "value"),
        Input("run-day", "value"),
    ],
)
def update_details_graph(subset_name, url, run, day):
    if not subset_name or not url or "details" not in url or run is None or day is None:
        return html.Div("No conversation selected.")
    preprocessing_data = global_store(subset_name)
    run_data = preprocessing_data["run_data"]
    if run not in run_data:
        return html.Div("Run not found.")

    df = run_data[run]

    acc = []

    def get_name(x):
        if x == "framework":
            return "F"
        else:
            return x.split("_")[1]

    for round, conv in df[df["action"] == "utterance"].groupby("round"):
        conv["agent_name"] = (
            conv["agent_name"] + " (" + conv["agent_id"].apply(get_name) + ")"
        )
        acc.append((f"{round+1})", conv[["agent_name", "utterance"]]))

    if day >= len(acc):
        return html.Div("Conversation not available for that round.")
    c = acc[day][1]
    return dmc.Table(
        striped=True,
        highlightOnHover=True,
        withBorder=True,
        withColumnBorders=True,
        children=create_table(c),
    )


@app.callback(
    Output("conversation-analysis-display", "children"),
    [
        Input("runs-group", "data"),
        Input("url", "pathname"),
        Input("run-dropdown", "value"),
        Input("run-day", "value"),
        Input("conversation-analysis-dropdown", "value"),
    ],
)
def update_details_graph_conv_analysis(subset_name, url, run, day, prompt):
    if not subset_name or not url or "details" not in url or run is None or day is None:
        return html.Div("No analysis selected.")
    preprocessing_data = global_store(subset_name)
    run_data = preprocessing_data["run_data"]
    if run not in run_data:
        return html.Div("Run not found.")

    df = run_data[run]

    d = df[df["action"] == prompt].groupby("round").first()
    if d.empty or day >= len(d):
        return html.Div("Analysis not available for that round.")
    h = d.iloc[day]

    return html.Iframe(
        sandbox="",
        srcDoc=("".join(h["html_interactions"]).replace("\n", "<br>")),
        width="100%",
        height="1000",
    )


@app.callback(
    Output("harvesting-prompts-render", "children"),
    [
        Input("runs-group", "data"),
        Input("url", "pathname"),
        Input("run-dropdown", "value"),
        Input("run-day", "value"),
        Input("persona-dropdown", "value"),
    ],
)
def update_harvesting_prompts(subset_name, url, run, day, persona):
    if (
        not subset_name
        or not url
        or "details" not in url
        or run is None
        or day is None
        or persona is None
    ):
        return html.Div("No harvesting prompt selected.")
    preprocessing_data = global_store(subset_name)
    run_data = preprocessing_data["run_data"]
    if run not in run_data:
        return html.Div("Run not found.")

    df = run_data[run]

    h = df[
        (df["action"] == "harvesting")
        & (df["round"] == day)
        & (df["agent_id"] == persona)
    ]
    if h.empty:
        return html.Div("Harvesting prompt not available for that selection.")
    h = h.iloc[0]

    return html.Iframe(
        sandbox="",
        srcDoc=("".join(h["html_interactions"]).replace("\n", "<br>")),
        width="100%",
        height="1000",
    )
