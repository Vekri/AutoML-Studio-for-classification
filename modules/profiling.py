"""Dataset profiling: column stats, missingness, cardinality, skew."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def profile_dataframe(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    """Build a rich data-profile summary for UI and executive report."""
    rows = int(len(df))
    cols = int(df.shape[1])
    missing_cells = int(df.isna().sum().sum())
    total_cells = max(rows * cols, 1)
    dup_count = int(df.duplicated().sum())

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    datetime_cols = [
        c
        for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c])
        or _looks_like_datetime(df[c])
    ]

    columns: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        entry: dict[str, Any] = {
            "column": col,
            "dtype": str(series.dtype),
            "kind": "numeric"
            if pd.api.types.is_numeric_dtype(series)
            else ("datetime" if col in datetime_cols else "categorical"),
            "missing_count": missing,
            "missing_pct": round(missing / max(rows, 1) * 100, 2),
            "unique_count": n_unique,
            "unique_pct": round(n_unique / max(rows, 1) * 100, 2),
            "is_target": bool(target and col == target),
            "is_constant": n_unique <= 1,
            "is_id_like": n_unique == rows and rows > 1 and not pd.api.types.is_numeric_dtype(series),
        }
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean):
                entry.update(
                    {
                        "mean": round(float(clean.mean()), 4),
                        "std": round(float(clean.std()), 4) if len(clean) > 1 else 0.0,
                        "min": round(float(clean.min()), 4),
                        "max": round(float(clean.max()), 4),
                        "median": round(float(clean.median()), 4),
                        "skew": round(float(clean.skew()), 4) if len(clean) > 2 else 0.0,
                        "q1": round(float(clean.quantile(0.25)), 4),
                        "q3": round(float(clean.quantile(0.75)), 4),
                    }
                )
                q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    outlier_mask = (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)
                    entry["outlier_count_iqr"] = int(outlier_mask.sum())
                    entry["outlier_pct_iqr"] = round(float(outlier_mask.mean() * 100), 2)
                else:
                    entry["outlier_count_iqr"] = 0
                    entry["outlier_pct_iqr"] = 0.0
        else:
            top = series.astype(str).value_counts(dropna=True).head(5)
            entry["top_values"] = [
                {"value": str(k), "count": int(v)} for k, v in top.items()
            ]
        columns.append(entry)

    target_info: dict[str, Any] | None = None
    if target and target in df.columns:
        vc = df[target].value_counts(dropna=True)
        target_info = {
            "column": target,
            "n_unique": int(vc.shape[0]),
            "is_binary": int(vc.shape[0]) == 2,
            "class_counts": {str(k): int(v) for k, v in vc.items()},
            "missing": int(df[target].isna().sum()),
        }

    return {
        "rows": rows,
        "columns": cols,
        "column_names": list(df.columns.astype(str)),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_cells / total_cells * 100, 2),
        "duplicate_rows": dup_count,
        "duplicate_pct": round(dup_count / max(rows, 1) * 100, 2),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 3),
        "target": target_info,
        "column_profiles": columns,
        "warnings": _profile_warnings(columns, dup_count, rows),
    }


def _looks_like_datetime(series: pd.Series) -> bool:
    if series.dtype == object:
    sample = series.dropna().astype(str).head(20)
    if not len(sample):
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean()) > 0.8
    return False


def _profile_warnings(columns: list[dict[str, Any]], dup_count: int, rows: int) -> list[str]:
    warnings: list[str] = []
    if dup_count > 0:
        warnings.append(f"{dup_count} duplicate rows ({dup_count / max(rows, 1) * 100:.1f}%)")
    high_missing = [c["column"] for c in columns if c["missing_pct"] > 30]
    if high_missing:
        warnings.append(f"High missing (>30%): {', '.join(high_missing[:8])}")
    constants = [c["column"] for c in columns if c["is_constant"]]
    if constants:
        warnings.append(f"Constant columns: {', '.join(constants[:8])}")
    id_like = [c["column"] for c in columns if c.get("is_id_like")]
    if id_like:
        warnings.append(f"ID-like columns: {', '.join(id_like[:8])}")
    return warnings
