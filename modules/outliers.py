"""Outlier detection helpers (IQR, Z-score, Isolation Forest)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


OUTLIER_METHODS = [
    {"id": "iqr", "label": "IQR (1.5×)", "description": "Cap values outside Q1−1.5×IQR / Q3+1.5×IQR"},
    {"id": "zscore", "label": "Z-Score", "description": "Flag |z| > threshold (default 3)"},
    {"id": "isolation_forest", "label": "Isolation Forest", "description": "Multivariate anomaly detection"},
]


def list_outlier_methods() -> list[dict[str, Any]]:
    return list(OUTLIER_METHODS)


def detect_outliers(
    df: pd.DataFrame,
    target: str | None = None,
    method: str = "iqr",
    z_threshold: float = 3.0,
    contamination: float = 0.05,
) -> dict[str, Any]:
    """Detect outliers per numeric column (and row-level for Isolation Forest)."""
    cols = [
        c
        for c in df.columns
        if c != target and pd.api.types.is_numeric_dtype(df[c])
    ]
    column_results: list[dict[str, Any]] = []
    row_flags = np.zeros(len(df), dtype=bool)

    if method == "isolation_forest" and cols:
        X = df[cols].fillna(df[cols].median(numeric_only=True)).to_numpy(dtype=float)
        if len(df) >= 20:
            clf = IsolationForest(
                contamination=min(max(contamination, 0.01), 0.2),
                random_state=42,
                n_estimators=100,
            )
            preds = clf.fit_predict(X)
            row_flags = preds == -1
            column_results.append(
                {
                    "column": "__rows__",
                    "method": method,
                    "outlier_count": int(row_flags.sum()),
                    "outlier_pct": round(float(row_flags.mean() * 100), 2),
                    "note": "Row-level anomalies across numeric features",
                }
            )
        return {
            "method": method,
            "columns": column_results,
            "row_outlier_count": int(row_flags.sum()),
            "row_outlier_pct": round(float(row_flags.mean() * 100), 2),
            "row_outlier_indices": [int(i) for i in np.where(row_flags)[0][:200]],
        }

    for col in cols:
        series = df[col]
        clean = series.dropna()
        if len(clean) < 5:
            continue
        mask = pd.Series(False, index=df.index)
        meta: dict[str, Any] = {"column": col, "method": method}
        if method == "zscore":
            std = float(clean.std())
            mean = float(clean.mean())
            if std <= 0:
                continue
            z = (series - mean).abs() / std
            mask = z > z_threshold
            meta["z_threshold"] = z_threshold
            meta["mean"] = round(mean, 4)
            meta["std"] = round(std, 4)
        else:  # iqr default
            q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (series < lower) | (series > upper)
            meta["lower"] = round(lower, 4)
            meta["upper"] = round(upper, 4)

        count = int(mask.fillna(False).sum())
        if count <= 0:
            continue
        meta["outlier_count"] = count
        meta["outlier_pct"] = round(count / max(len(df), 1) * 100, 2)
        column_results.append(meta)
        row_flags = row_flags | mask.fillna(False).to_numpy()

    return {
        "method": method if method in ("iqr", "zscore", "isolation_forest") else "iqr",
        "columns": column_results,
        "row_outlier_count": int(row_flags.sum()),
        "row_outlier_pct": round(float(row_flags.mean() * 100), 2),
        "row_outlier_indices": [int(i) for i in np.where(row_flags)[0][:200]],
    }


def apply_outlier_treatment(
    df: pd.DataFrame,
    target: str | None = None,
    method: str = "iqr",
    mode: str = "cap",
    z_threshold: float = 3.0,
    contamination: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply outlier treatment.
    mode: 'cap' (winsorize numeric cols) or 'drop_rows' (remove flagged rows).
    """
    detection = detect_outliers(
        df, target=target, method=method, z_threshold=z_threshold, contamination=contamination
    )
    result = df.copy()
    config: dict[str, Any] = {
        "method": method,
        "mode": mode,
        "detection": detection,
        "actions_applied": [],
    }

    if mode == "drop_rows" and detection.get("row_outlier_indices"):
        idx = set(detection["row_outlier_indices"])
        # Recompute full mask for drop (indices list may be capped)
        full = detect_outliers(df, target=target, method=method, z_threshold=z_threshold, contamination=contamination)
        # Use column-based mask rebuild for iqr/zscore
        if method == "isolation_forest":
            cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
            if cols and len(df) >= 20:
                X = df[cols].fillna(df[cols].median(numeric_only=True)).to_numpy(dtype=float)
                preds = IsolationForest(
                    contamination=min(max(contamination, 0.01), 0.2),
                    random_state=42,
                    n_estimators=100,
                ).fit_predict(X)
                keep = preds == 1
                before = len(result)
                result = result.loc[keep].reset_index(drop=True)
                config["actions_applied"].append(
                    {"action": "drop_rows", "rows_removed": before - len(result)}
                )
        else:
            before = len(result)
            # Drop union of column outlier rows
            drop_mask = np.zeros(len(df), dtype=bool)
            for col_info in detection.get("columns") or []:
                col = col_info.get("column")
                if not col or col not in result.columns:
                    continue
                series = result[col]
                if method == "zscore":
                    std = float(series.std())
                    mean = float(series.mean())
                    if std > 0:
                        drop_mask |= ((series - mean).abs() / std > z_threshold).fillna(False).to_numpy()
                else:
                    q1, q3 = series.quantile(0.25), series.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        drop_mask |= ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).fillna(False).to_numpy()
            result = result.loc[~drop_mask].reset_index(drop=True)
            config["actions_applied"].append(
                {"action": "drop_rows", "rows_removed": before - len(result)}
            )
        return result, config

    # Cap mode
    for col_info in detection.get("columns") or []:
        col = col_info.get("column")
        if not col or col == "__rows__" or col not in result.columns:
            continue
        series = result[col]
        if method == "zscore":
            mean = float(series.mean())
            std = float(series.std())
            if std <= 0:
                continue
            lower, upper = mean - z_threshold * std, mean + z_threshold * std
        else:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        result[col] = series.clip(lower=lower, upper=upper)
        config["actions_applied"].append(
            {
                "action": "cap",
                "column": col,
                "lower": float(lower),
                "upper": float(upper),
            }
        )

    return result, config
