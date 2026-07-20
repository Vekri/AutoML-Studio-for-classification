"""Data loading and column management."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd


def load_csv(uploaded_file) -> pd.DataFrame:
    """Load CSV from Streamlit uploaded file."""
    raw = uploaded_file.read()
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8", errors="replace")


def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for all columns."""
    rows = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        rows.append(
            {
                "column": col,
                "dtype": dtype,
                "missing": n_missing,
                "missing_pct": round(n_missing / len(df) * 100, 2) if len(df) else 0,
                "unique": n_unique,
                "sample_values": ", ".join(str(v) for v in series.dropna().head(3).tolist()),
            }
        )
    return pd.DataFrame(rows)


def apply_column_selection(
    df: pd.DataFrame,
    target: str,
    keep_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Filter dataframe to selected columns, always keeping target."""
    if keep_columns:
        cols = list(dict.fromkeys([*keep_columns, target]))
        return df[[c for c in cols if c in df.columns]].copy()

    if drop_columns:
        cols = [c for c in df.columns if c not in drop_columns or c == target]
        return df[cols].copy()

    return df.copy()


def detect_binary_target(series: pd.Series) -> dict[str, Any]:
    """Analyze whether a column is suitable as binary target."""
    clean = series.dropna()
    unique_vals = clean.unique()
    n_unique = len(unique_vals)

    counts = clean.value_counts()
    # JSON-safe keys/values (numpy types break st.json)
    class_counts = {str(k): int(v) for k, v in counts.items()}

    result = {
        "is_binary": n_unique == 2,
        "n_unique": int(n_unique),
        "unique_values": [str(v) for v in unique_vals[:10]],
        "class_counts": class_counts,
        "imbalance_ratio": None,
    }

    if n_unique == 2:
        minority = int(counts.min())
        majority = int(counts.max())
        result["imbalance_ratio"] = round(minority / majority, 4) if majority else 0.0

    return result


def suggest_target_column(df: pd.DataFrame, domain_targets: list[str]) -> str | None:
    """Suggest target column based on domain common names."""
    lower_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    for name in domain_targets:
        key = name.lower()
        if key in lower_map:
            return lower_map[key]
    for col in df.columns:
        if col.lower() in ("target", "label", "class", "y", "outcome"):
            return col
    return None


def get_numeric_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = exclude or []
    return [c for c in df.select_dtypes(include=["number"]).columns if c not in exclude]


def get_categorical_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = exclude or []
    return [c for c in df.select_dtypes(exclude=["number"]).columns if c not in exclude]
