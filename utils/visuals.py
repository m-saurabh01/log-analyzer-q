# utils/visuals.py
"""
Plotly-based visualization helpers for the Streamlit log anomaly app.

Provides:
  - plot_anomaly_timeline(df)
  - plot_anomaly_counts(df)
  - plot_score_distribution(df, score_col, title)
"""

import pandas as pd
import plotly.express as px


def plot_anomaly_timeline(df: pd.DataFrame):
    """
    Build a time-series chart of anomaly counts over time.

    Expected columns in df:
        - 'timestamp' : datetime
        - 'anomaly'   : 0 or 1 (1 means anomaly)

    We aggregate anomalies at minute-level to avoid over-plotting.
    """
    if "timestamp" not in df.columns or "anomaly" not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' and 'anomaly' columns")

    df2 = df.copy().sort_values("timestamp")

    df2["ts_min"] = df2["timestamp"].dt.floor("T")

    agg = (
        df2.groupby("ts_min")["anomaly"]
        .sum()
        .reset_index()
        .rename(columns={"ts_min": "timestamp", "anomaly": "anomaly_count"})
    )

    fig = px.line(
        agg,
        x="timestamp",
        y="anomaly_count",
        markers=True,
        labels={
            "timestamp": "Time",
            "anomaly_count": "Anomaly Count",
        },
        title="Anomalies over time",
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Number of anomalies",
        hovermode="x unified",
    )

    return fig


def plot_anomaly_counts(df: pd.DataFrame):
    """
    Build a bar chart of anomaly counts per log level.

    Expected columns:
        - 'log_level'
        - 'anomaly' (0/1)
    """
    if "log_level" not in df.columns or "anomaly" not in df.columns:
        raise ValueError("DataFrame must contain 'log_level' and 'anomaly' columns")

    agg = (
        df.groupby("log_level")["anomaly"]
        .sum()
        .reset_index()
        .rename(columns={"anomaly": "anomaly_count"})
        .sort_values("anomaly_count", ascending=False)
    )

    fig = px.bar(
        agg,
        x="log_level",
        y="anomaly_count",
        labels={
            "log_level": "Log level",
            "anomaly_count": "Anomaly Count",
        },
        title="Anomaly count by log level",
    )

    fig.update_layout(
        xaxis_title="Log level",
        yaxis_title="Number of anomalies",
    )

    return fig


def plot_score_distribution(
    df: pd.DataFrame,
    score_col: str = "anomaly_score",
    title: str = "Score distribution",
):
    """
    Histogram of anomaly scores / reconstruction errors.

    Expected columns:
        - score_col (default: 'anomaly_score')
    """
    if score_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{score_col}' column")

    fig = px.histogram(
        df,
        x=score_col,
        nbins=50,
        title=title,
        labels={score_col: score_col},
    )

    fig.update_layout(
        xaxis_title=score_col,
        yaxis_title="Count",
        bargap=0.05,
    )

    return fig
