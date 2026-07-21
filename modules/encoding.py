"""Encoding recommendations and optional transforms for categoricals."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def recommend_encoding(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    """Recommend encoding strategy per categorical / object column."""
    recommendations: list[dict[str, Any]] = []
    for col in df.columns:
        if target and col == target:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            continue
        n_unique = int(series.nunique(dropna=True))
        if n_unique <= 1:
            method, reason = "drop", "Constant / near-constant categorical"
        elif n_unique == series.dropna().shape[0] and n_unique > 50:
            method, reason = "drop", "ID-like high cardinality — drop before modeling"
        elif n_unique == 2:
            method, reason = "label", "Binary categorical — label encode"
        elif n_unique <= 10:
            method, reason = "one_hot", "Low cardinality — one-hot encode"
        elif n_unique <= 30 and target and target in df.columns:
            method, reason = "target", "Medium cardinality — target (mean) encoding"
        elif n_unique <= 30:
            method, reason = "ordinal", "Medium cardinality — ordinal encode by frequency"
        else:
            method, reason = "hash_or_drop", "High cardinality — consider hashing or drop"

        recommendations.append(
            {
                "column": col,
                "n_unique": n_unique,
                "missing_pct": round(float(series.isna().mean() * 100), 2),
                "method": method,
                "reason": reason,
            }
        )

    return {
        "recommendations": recommendations,
        "summary": {
            "one_hot": sum(1 for r in recommendations if r["method"] == "one_hot"),
            "label": sum(1 for r in recommendations if r["method"] == "label"),
            "target": sum(1 for r in recommendations if r["method"] == "target"),
            "ordinal": sum(1 for r in recommendations if r["method"] == "ordinal"),
            "drop": sum(1 for r in recommendations if r["method"] in ("drop", "hash_or_drop")),
        },
    }


def apply_encoding(
    df: pd.DataFrame,
    target: str | None = None,
    recommendations: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply recommended encodings; returns transformed frame + config."""
    pack = recommend_encoding(df, target) if recommendations is None else {"recommendations": recommendations}
    recs = pack["recommendations"]
    result = df.copy()
    config: dict[str, Any] = {"actions": [], "columns_created": [], "columns_dropped": []}

    # Target means for target encoding
    target_means: dict[str, float] = {}
    global_mean = None
    if target and target in result.columns:
        y = result[target]
        if not pd.api.types.is_numeric_dtype(y):
            y_num = LabelEncoder().fit_transform(y.astype(str))
        else:
            y_num = y.to_numpy()
        global_mean = float(pd.Series(y_num).mean())

    for rec in recs:
        col = rec["column"]
        method = rec["method"]
        if col not in result.columns:
            continue

        if method in ("drop", "hash_or_drop"):
            result = result.drop(columns=[col])
            config["columns_dropped"].append(col)
            config["actions"].append({"column": col, "method": "drop"})
            continue

        if method == "one_hot":
            dummies = pd.get_dummies(result[col].astype(str).fillna("__missing__"), prefix=col)
            result = pd.concat([result.drop(columns=[col]), dummies], axis=1)
            config["columns_created"].extend(list(dummies.columns))
            config["actions"].append({"column": col, "method": "one_hot", "n_cols": int(dummies.shape[1])})
            continue

        if method == "label":
            le = LabelEncoder()
            filled = result[col].astype(str).fillna("__missing__")
            result[col] = le.fit_transform(filled)
            config["actions"].append(
                {"column": col, "method": "label", "classes": [str(x) for x in le.classes_.tolist()]}
            )
            continue

        if method == "ordinal":
            freq = result[col].astype(str).fillna("__missing__").value_counts()
            order = {k: i for i, k in enumerate(freq.index.tolist())}
            result[col] = result[col].astype(str).fillna("__missing__").map(order).astype(float)
            config["actions"].append({"column": col, "method": "ordinal", "mapping_size": len(order)})
            continue

        if method == "target" and target and target in df.columns and global_mean is not None:
            y = df[target]
            if not pd.api.types.is_numeric_dtype(y):
                y_enc = LabelEncoder().fit_transform(y.astype(str))
            else:
                y_enc = y.to_numpy()
            tmp = pd.DataFrame({"cat": df[col].astype(str).fillna("__missing__"), "y": y_enc})
            means = tmp.groupby("cat")["y"].mean().to_dict()
            result[col] = (
                result[col].astype(str).fillna("__missing__").map(means).fillna(global_mean).astype(float)
            )
            config["actions"].append({"column": col, "method": "target", "global_mean": global_mean})
            continue

    return result, config
