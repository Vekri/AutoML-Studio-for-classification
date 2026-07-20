"""Distribution and visualization helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def target_distribution_chart(df: pd.DataFrame, target: str) -> go.Figure:
    counts = df[target].value_counts().reset_index()
    counts.columns = ["class", "count"]
    fig = px.bar(
        counts,
        x="class",
        y="count",
        title=f"Target Distribution: {target}",
        color="class",
        text="count",
    )
    fig.update_layout(showlegend=False, height=400)
    return fig


def numeric_histogram(df: pd.DataFrame, column: str, target: str | None = None) -> go.Figure:
    if target and target in df.columns:
        fig = px.histogram(
            df,
            x=column,
            color=target,
            barmode="overlay",
            opacity=0.7,
            title=f"Distribution of {column} by {target}",
            marginal="box",
        )
    else:
        fig = px.histogram(df, x=column, title=f"Distribution of {column}", marginal="box")
    fig.update_layout(height=450)
    return fig


def correlation_heatmap(df: pd.DataFrame, target: str | None = None) -> go.Figure:
    numeric_df = df.select_dtypes(include=["number"])
    if target in numeric_df.columns:
        corr = numeric_df.corr()
    else:
        corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(height=600)
    return fig


def box_plot_by_target(df: pd.DataFrame, column: str, target: str) -> go.Figure:
    fig = px.box(df, x=target, y=column, title=f"{column} by {target}", color=target)
    fig.update_layout(height=400, showlegend=False)
    return fig


def missing_values_chart(df: pd.DataFrame) -> go.Figure:
    missing = df.isna().sum().reset_index()
    missing.columns = ["column", "missing_count"]
    missing = missing[missing["missing_count"] > 0].sort_values("missing_count", ascending=True)
    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(text="No missing values found", showarrow=False, font={"size": 16})
        fig.update_layout(title="Missing Values", height=300)
        return fig
    fig = px.bar(
        missing,
        x="missing_count",
        y="column",
        orientation="h",
        title="Missing Values by Column",
        text="missing_count",
    )
    fig.update_layout(height=max(300, len(missing) * 35))
    return fig


def categorical_bar_chart(df: pd.DataFrame, column: str, target: str | None = None) -> go.Figure:
    if target and target in df.columns:
        ct = pd.crosstab(df[column], df[target])
        fig = px.bar(
            ct.reset_index().melt(id_vars=column, var_name=target, value_name="count"),
            x=column,
            y="count",
            color=target,
            barmode="group",
            title=f"{column} vs {target}",
        )
    else:
        counts = df[column].value_counts().head(20).reset_index()
        counts.columns = [column, "count"]
        fig = px.bar(counts, x=column, y="count", title=f"Top values in {column}")
    fig.update_layout(height=450)
    return fig


def feature_target_correlation_chart(df: pd.DataFrame, target: str) -> go.Figure:
    numeric_df = df.select_dtypes(include=["number"])
    if target not in numeric_df.columns:
        return go.Figure().add_annotation(text="Target is not numeric", showarrow=False)

    corr = numeric_df.corr()[target].drop(target).sort_values(key=abs, ascending=True)
    fig = px.bar(
        x=corr.values,
        y=corr.index,
        orientation="h",
        title=f"Feature Correlation with {target}",
        labels={"x": "Correlation", "y": "Feature"},
    )
    fig.update_layout(height=max(400, len(corr) * 25))
    return fig


def get_numeric_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = exclude or []
    return [c for c in df.select_dtypes(include=["number"]).columns if c not in exclude]


def get_categorical_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = exclude or []
    return [c for c in df.select_dtypes(exclude=["number"]).columns if c not in exclude]
