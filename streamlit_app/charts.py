"""Prepare chart labels and preserve one observation per forecast period."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

GREEN = "#0D6628"
LEAF = "#6EAD50"
RED = "#FE0000"
XGB = "XGBoost + Log Target"
LAST = "Last 30 Days Baseline"
AVERAGE = "Last 60-Day Average Baseline"
ACTUAL = "Actual Sales"
COLORS = {XGB: RED, LAST: LEAF, AVERAGE: LEAF, ACTUAL: GREEN}


def style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin={"t": 55, "l": 20, "r": 20, "b": 20},
        font_color="#203025",
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "title_text": ""},
    )
    return fig


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


def sales_forecast_chart(history: list[dict], forecast: dict) -> go.Figure:
    """Connect observed 30-day totals to the next period's saved model estimate."""
    periods = [row["forecast_period"] for row in history]
    actuals = [row["actual"] for row in history]
    future = f"{forecast['forecast_start_day']}-{forecast['forecast_end_day']}"
    fig = go.Figure()
    fig.add_scatter(
        x=periods,
        y=actuals,
        name="Actual sales",
        mode="lines+markers",
        line={"color": GREEN},
        hovertemplate="%{y:,.0f} units<extra>Actual sales</extra>",
    )
    fig.add_scatter(
        x=[periods[-1], future],
        y=[actuals[-1], forecast["xgb_prediction"]],
        name="XGBoost forecast",
        mode="lines+markers",
        line={"color": RED, "dash": "dash"},
        marker={"size": [0, 10]},
        hovertemplate=[
            "%{y:,.0f} units<extra>Last actual sales</extra>",
            "%{y:,.2f} units<extra>XGBoost forecast</extra>",
        ],
    )
    fig.add_scatter(
        x=[future],
        y=[sum(actuals) / len(actuals)],
        name="Average of past 3 periods",
        mode="markers",
        marker={
            "symbol": "diamond-open",
            "size": 14,
            "color": LEAF,
            "line": {"width": 2},
        },
        hovertemplate="%{y:,.2f} units<extra>Past 3-period average</extra>",
    )
    fig.update_layout(
        title="Sales history and next 30-day forecast",
        xaxis={
            "title": "30-day period (dataset days)",
            "type": "category",
            "tickmode": "array",
            "tickvals": [*periods, future],
            "ticktext": [*periods, f"{future} (forecast)"],
        },
        yaxis_title="Units across all stores",
    )
    return style(fig)


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
