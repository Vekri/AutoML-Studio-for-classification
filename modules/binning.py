"""Numeric feature binning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def create_bins(
    series: pd.Series,
    method: str = "equal_width",
    n_bins: int = 5,
    custom_edges: list[float] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Bin a numeric series and return binned labels + config."""
    clean = series.dropna()
    config: dict[str, Any] = {
        "column": series.name,
        "method": method,
        "n_bins": n_bins,
        "edges": [],
        "labels": [],
    }

    if method == "custom" and custom_edges:
        edges = sorted(custom_edges)
        labels = [f"({edges[i]}, {edges[i+1]}]" for i in range(len(edges) - 1)]
        binned = pd.cut(series, bins=edges, labels=labels, include_lowest=True)
        config["edges"] = edges
        config["labels"] = labels
        return binned, config

    if method == "equal_frequency":
        try:
            binned, edges = pd.qcut(series, q=n_bins, duplicates="drop", retbins=True)
            labels = [f"Q{i+1}" for i in range(len(edges) - 1)]
            binned = pd.cut(series, bins=edges, labels=labels[: len(edges) - 1], include_lowest=True)
            config["edges"] = edges.tolist()
            config["labels"] = labels[: len(edges) - 1]
            return binned, config
        except ValueError:
            method = "equal_width"

    edges = np.linspace(clean.min(), clean.max(), n_bins + 1)
    labels = [f"Bin_{i+1}" for i in range(n_bins)]
    binned = pd.cut(series, bins=edges, labels=labels, include_lowest=True)
    config["edges"] = edges.tolist()
    config["labels"] = labels
    return binned, config


def apply_binning_to_dataframe(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "equal_width",
    n_bins: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply binning to multiple columns."""
    result = df.copy()
    all_config: dict[str, Any] = {"bins": {}}

    for col in columns:
        if col not in result.columns or not pd.api.types.is_numeric_dtype(result[col]):
            continue
        binned, cfg = create_bins(result[col], method=method, n_bins=n_bins)
        new_col = f"{col}_binned"
        result[new_col] = binned
        all_config["bins"][col] = cfg

    return result, all_config


def woe_iv_table(df: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
    """Calculate Weight of Evidence and Information Value for a binned/categorical column."""
    ct = pd.crosstab(df[column], df[target], dropna=False)
    if ct.shape[1] != 2:
        return pd.DataFrame()

    total_good = ct.iloc[:, 0].sum()
    total_bad = ct.iloc[:, 1].sum()
    if total_good == 0 or total_bad == 0:
        return pd.DataFrame()

    rows = []
    for idx in ct.index:
        good = ct.loc[idx, ct.columns[0]]
        bad = ct.loc[idx, ct.columns[1]]
        dist_good = max(good, 0.5) / total_good
        dist_bad = max(bad, 0.5) / total_bad
        woe = np.log(dist_good / dist_bad)
        iv = (dist_good - dist_bad) * woe
        rows.append(
            {
                "bin": str(idx),
                "good": int(good),
                "bad": int(bad),
                "woe": round(woe, 4),
                "iv": round(iv, 6),
            }
        )

    result = pd.DataFrame(rows)
    result["total_iv"] = result["iv"].sum()
    return result
