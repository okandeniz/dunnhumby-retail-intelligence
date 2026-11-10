"""Prepare chart labels and preserve one observation per forecast period."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BLUE = "#689EB8"
CORAL = "#FF5A60"
GRAY = "#A79B94"
XGB = "XGBoost + Log Target"
LAST = "Last 30 Days Baseline"
AVERAGE = "Last 60-Day Average Baseline"
ACTUAL = "Actual Sales"
COLORS = {XGB: CORAL, LAST: GRAY, AVERAGE: GRAY, ACTUAL: BLUE}


def style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin={"t": 55, "l": 20, "r": 20, "b": 20},
        font_color="#243640",
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "title_text": ""},
    )
    return fig


def future_chart(row: dict) -> go.Figure:
    frame = pd.DataFrame(
        {
            "Method": [XGB, LAST, AVERAGE],
            "Quantity": [
                row["xgb_prediction"],
                row["baseline_last_30d"],
                row["baseline_avg_60d"],
            ],
        }
    )
    fig = px.bar(
        frame,
        x="Quantity",
        y="Method",
        orientation="h",
        color="Method",
        color_discrete_map=COLORS,
        text_auto=",.1f",
        title=f"Demand for days {row['forecast_start_day']}–{row['forecast_end_day']}",
        labels={"Quantity": "Units across all stores"},
    )
    fig.update_layout(showlegend=False, yaxis_title=None)
    fig.update_yaxes(autorange="reversed")
    return style(fig)


def historical_frame(rows: list[dict]) -> pd.DataFrame:
    labels = {
        "actual": ACTUAL,
        "xgb_prediction": XGB,
        "baseline_prediction": LAST,
        "baseline_60d_prediction": AVERAGE,
    }
    return (
        pd.DataFrame(rows)
        .rename(columns=labels)
        .melt(
            id_vars="forecast_period",
            value_vars=list(labels.values()),
            var_name="Method",
            value_name="Units",
        )
    )


def history_chart(rows: list[dict]) -> go.Figure:
    frame = historical_frame(rows)
    fig = px.line(
        frame,
        x="forecast_period",
        y="Units",
        color="Method",
        line_dash="Method",
        markers=True,
        color_discrete_map=COLORS,
        line_dash_map={ACTUAL: "solid", XGB: "solid", LAST: "dash", AVERAGE: "dot"},
        labels={
            "forecast_period": "Test period (dataset days)",
            "Units": "Units across all stores",
        },
        title="Actual and predicted demand",
    )
    fig.update_layout(hovermode="x unified")
    return style(fig)
