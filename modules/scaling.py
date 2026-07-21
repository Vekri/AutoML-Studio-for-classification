"""Scaling recommendations and optional transforms for numeric features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def recommend_scaling(
    df: pd.DataFrame,
    target: str | None = None,
    algorithm_hint: str | None = None,
) -> dict[str, Any]:
    """
    Recommend scaler per numeric column based on skew and algorithm family.
    algorithm_hint: logistic_regression | svm | knn | tree | xgboost | None
    """
    tree_like = algorithm_hint in (
        "random_forest",
        "xgboost",
        "gradient_boosting",
        "extra_trees",
        "decision_tree",
        "ada_boost",
        "tree",
    )
    recommendations: list[dict[str, Any]] = []

    for col in df.columns:
        if target and col == target:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].dropna()
        if len(series) < 3:
            continue
        skew = float(series.skew())
        std = float(series.std()) if len(series) > 1 else 0.0
        has_outliers = False
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            has_outliers = bool(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).mean() > 0.02)

        if tree_like:
            method, reason = "none", "Tree/ensemble models are scale-invariant — scaling optional"
        elif has_outliers or abs(skew) > 1.5:
            method, reason = "robust", "Skewed / outlier-heavy — RobustScaler (IQR)"
        elif abs(skew) > 0.75:
            method, reason = "minmax", "Moderate skew — MinMaxScaler to [0,1]"
        else:
            method, reason = "standard", "Near-normal — StandardScaler (z-score)"

        recommendations.append(
            {
                "column": col,
                "skew": round(skew, 4),
                "std": round(std, 4),
                "has_outliers": has_outliers,
                "method": method,
                "reason": reason,
            }
        )

    summary = {
        "standard": sum(1 for r in recommendations if r["method"] == "standard"),
        "minmax": sum(1 for r in recommendations if r["method"] == "minmax"),
        "robust": sum(1 for r in recommendations if r["method"] == "robust"),
        "none": sum(1 for r in recommendations if r["method"] == "none"),
        "algorithm_hint": algorithm_hint,
    }
    return {"recommendations": recommendations, "summary": summary}


def apply_scaling(
    df: pd.DataFrame,
    target: str | None = None,
    recommendations: list[dict[str, Any]] | None = None,
    algorithm_hint: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit/transform scalers per recommendation; skip method='none'."""
    pack = (
        recommend_scaling(df, target, algorithm_hint)
        if recommendations is None
        else {"recommendations": recommendations}
    )
    recs = pack["recommendations"]
    result = df.copy()
    config: dict[str, Any] = {"actions": [], "algorithm_hint": algorithm_hint}

    groups: dict[str, list[str]] = {"standard": [], "minmax": [], "robust": []}
    for rec in recs:
        method = rec["method"]
        col = rec["column"]
        if method in groups and col in result.columns:
            groups[method].append(col)

    scaler_map = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    for method, cols in groups.items():
        if not cols:
            continue
        scaler = scaler_map[method]
        block = result[cols].astype(float)
        # Fill NaN with median before scaling
        block = block.fillna(block.median())
        transformed = scaler.fit_transform(block)
        result[cols] = transformed
        config["actions"].append(
            {
                "method": method,
                "columns": cols,
                "n_columns": len(cols),
            }
        )

    return result, config
