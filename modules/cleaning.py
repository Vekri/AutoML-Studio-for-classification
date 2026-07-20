"""Data cleaning analysis and transformations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def generate_cleaning_recommendations(df: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    """Generate actionable cleaning recommendations."""
    recommendations: list[dict[str, Any]] = []

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        recommendations.append(
            {
                "severity": "medium",
                "category": "duplicates",
                "issue": f"{dup_count} duplicate rows detected",
                "recommendation": "Remove duplicate rows before modeling",
                "action": "drop_duplicates",
            }
        )

    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        missing_pct = series.isna().mean() * 100

        if missing_pct > 50:
            recommendations.append(
                {
                    "severity": "high",
                    "category": "missing_values",
                    "issue": f"'{col}' has {missing_pct:.1f}% missing values",
                    "recommendation": "Consider dropping this column (>50% missing)",
                    "action": "drop_column",
                    "column": col,
                }
            )
        elif missing_pct > 0:
            strategy = "median" if pd.api.types.is_numeric_dtype(series) else "mode"
            recommendations.append(
                {
                    "severity": "low" if missing_pct < 5 else "medium",
                    "category": "missing_values",
                    "issue": f"'{col}' has {missing_pct:.1f}% missing values",
                    "recommendation": f"Impute using {strategy}",
                    "action": "impute",
                    "column": col,
                    "strategy": strategy,
                }
            )

        if pd.api.types.is_numeric_dtype(series):
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
                if outliers > 0:
                    outlier_pct = outliers / len(series) * 100
                    if outlier_pct > 1:
                        recommendations.append(
                            {
                                "severity": "low" if outlier_pct < 5 else "medium",
                                "category": "outliers",
                                "issue": f"'{col}' has {outliers} potential outliers ({outlier_pct:.1f}%)",
                                "recommendation": "Cap outliers at IQR bounds or use robust scaling",
                                "action": "cap_outliers",
                                "column": col,
                            }
                        )

        n_unique = series.nunique(dropna=True)
        if n_unique == 1:
            recommendations.append(
                {
                    "severity": "high",
                    "category": "constant",
                    "issue": f"'{col}' has only one unique value",
                    "recommendation": "Drop constant column — no predictive power",
                    "action": "drop_column",
                    "column": col,
                }
            )
        elif n_unique == len(series.dropna()) and not pd.api.types.is_numeric_dtype(series):
            recommendations.append(
                {
                    "severity": "medium",
                    "category": "high_cardinality",
                    "issue": f"'{col}' appears to be an ID column (all unique)",
                    "recommendation": "Drop ID-like columns to prevent overfitting",
                    "action": "drop_column",
                    "column": col,
                }
            )

    target_info = df[target].dropna()
    if len(target_info.unique()) != 2:
        recommendations.append(
            {
                "severity": "high",
                "category": "target",
                "issue": f"Target '{target}' is not binary ({target_info.nunique()} classes)",
                "recommendation": "Select a column with exactly 2 unique values",
                "action": "fix_target",
            }
        )

    target_missing = df[target].isna().sum()
    if target_missing > 0:
        recommendations.append(
            {
                "severity": "high",
                "category": "target",
                "issue": f"Target has {target_missing} missing values",
                "recommendation": "Drop rows with missing target before modeling",
                "action": "drop_target_missing",
            }
        )

    return recommendations


def apply_cleaning(
    df: pd.DataFrame,
    target: str,
    actions: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply selected cleaning actions and return processed data + config."""
    result = df.copy()
    config: dict[str, Any] = {"actions_applied": [], "imputation_values": {}}

    for action in actions:
        act = action.get("action")
        col = action.get("column")

        if act == "drop_duplicates":
            before = len(result)
            result = result.drop_duplicates()
            config["actions_applied"].append({"action": act, "rows_removed": before - len(result)})

        elif act == "drop_column" and col and col != target and col in result.columns:
            result = result.drop(columns=[col])
            config["actions_applied"].append({"action": act, "column": col})

        elif act == "drop_target_missing":
            before = len(result)
            result = result.dropna(subset=[target])
            config["actions_applied"].append({"action": act, "rows_removed": before - len(result)})

        elif act == "impute" and col and col in result.columns:
            strategy = action.get("strategy", "median")
            if pd.api.types.is_numeric_dtype(result[col]):
                if strategy == "median":
                    fill_val = result[col].median()
                elif strategy == "mean":
                    fill_val = result[col].mean()
                else:
                    fill_val = result[col].mode().iloc[0] if not result[col].mode().empty else 0
            else:
                fill_val = (
                    result[col].mode().iloc[0]
                    if not result[col].mode().empty
                    else "Unknown"
                )
            result[col] = result[col].fillna(fill_val)
            config["imputation_values"][col] = str(fill_val)
            config["actions_applied"].append({"action": act, "column": col, "strategy": strategy})

        elif act == "cap_outliers" and col and col in result.columns:
            q1, q3 = result[col].quantile(0.25), result[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            result[col] = result[col].clip(lower=lower, upper=upper)
            config["actions_applied"].append(
                {"action": act, "column": col, "lower": float(lower), "upper": float(upper)}
            )

    return result, config


def auto_clean(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply all recommended cleaning actions automatically."""
    recs = generate_cleaning_recommendations(df, target)
    safe_actions = {"drop_duplicates", "drop_target_missing", "impute", "cap_outliers"}
    actionable = []
    for r in recs:
        action = r.get("action")
        severity = r.get("severity")
        if action in ("drop_duplicates", "drop_target_missing"):
            actionable.append(r)
        elif action == "drop_column" and severity == "high":
            actionable.append(r)
        elif action in safe_actions and severity != "high":
            actionable.append(r)
    return apply_cleaning(df, target, actionable)
