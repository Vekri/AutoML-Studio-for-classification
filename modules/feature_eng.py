"""Feature engineering and variable reduction helpers."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def engineer_features(
    df: pd.DataFrame,
    target: str | None = None,
    max_interactions: int = 5,
    include_datetime: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Create simple engineered features:
    - datetime parts (year/month/day/dow) when detectable
    - top numeric pairwise ratios and products (limited)
    """
    result = df.copy()
    created: list[str] = []
    config: dict[str, Any] = {"datetime_parts": [], "interactions": [], "ratios": []}

    # Datetime parts
    if include_datetime:
        for col in list(result.columns):
            if target and col == target:
                continue
            series = result[col]
            parsed = None
            if pd.api.types.is_datetime64_any_dtype(series):
                parsed = pd.to_datetime(series, errors="coerce")
            elif series.dtype == object:
                sample = series.dropna().astype(str).head(30)
                if len(sample) and float(pd.to_datetime(sample, errors="coerce").notna().mean()) > 0.8:
                    parsed = pd.to_datetime(series, errors="coerce")
            if parsed is None or parsed.notna().sum() < max(5, int(0.5 * len(result))):
                continue
            for part, extractor in (
                ("year", parsed.dt.year),
                ("month", parsed.dt.month),
                ("day", parsed.dt.day),
                ("dow", parsed.dt.dayofweek),
            ):
                name = f"{col}_{part}"
                if name not in result.columns:
                    result[name] = extractor
                    created.append(name)
                    config["datetime_parts"].append(name)

    numeric = [
        c
        for c in result.columns
        if (not target or c != target) and pd.api.types.is_numeric_dtype(result[c])
    ]
    # Prefer columns with higher variance for interactions
    variances = {c: float(result[c].var(skipna=True) or 0) for c in numeric}
    ranked = sorted(numeric, key=lambda c: variances[c], reverse=True)[:8]

    pair_count = 0
    for a, b in combinations(ranked, 2):
        if pair_count >= max_interactions:
            break
        prod = f"{a}_x_{b}"
        if prod not in result.columns:
            result[prod] = result[a].astype(float) * result[b].astype(float)
            created.append(prod)
            config["interactions"].append(prod)
            pair_count += 1
        ratio = f"{a}_div_{b}"
        if ratio not in result.columns:
            denom = result[b].astype(float).replace(0, np.nan)
            result[ratio] = (result[a].astype(float) / denom).replace([np.inf, -np.inf], np.nan)
            created.append(ratio)
            config["ratios"].append(ratio)

    config["created"] = created
    config["n_created"] = len(created)
    return result, config


def reduce_variables(
    df: pd.DataFrame,
    target: str | None = None,
    corr_threshold: float = 0.92,
    run_pca: bool = True,
    pca_variance: float = 0.95,
) -> dict[str, Any]:
    """
    Variable reduction analysis:
    - drop suggestions for highly correlated numeric pairs
    - optional PCA variance summary (does not mutate df by default)
    """
    numeric = [
        c
        for c in df.columns
        if (not target or c != target) and pd.api.types.is_numeric_dtype(df[c])
    ]
    drop_suggestions: list[dict[str, Any]] = []
    kept = set(numeric)

    if len(numeric) >= 2:
        corr = df[numeric].corr().abs()
        for i, a in enumerate(numeric):
            for b in numeric[i + 1 :]:
                val = float(corr.loc[a, b]) if a in corr.index and b in corr.columns else 0.0
                if val >= corr_threshold:
                    # Suggest dropping the one with lower variance
                    va = float(df[a].var(skipna=True) or 0)
                    vb = float(df[b].var(skipna=True) or 0)
                    drop_col = a if va <= vb else b
                    keep_col = b if drop_col == a else a
                    if drop_col in kept:
                        kept.discard(drop_col)
                        drop_suggestions.append(
                            {
                                "drop": drop_col,
                                "keep": keep_col,
                                "correlation": round(val, 4),
                                "reason": f"|corr|={val:.3f} with {keep_col}",
                            }
                        )

    pca_info: dict[str, Any] | None = None
    if run_pca and len(numeric) >= 3 and len(df) >= 20:
        X = df[numeric].fillna(df[numeric].median(numeric_only=True)).to_numpy(dtype=float)
        Xs = StandardScaler().fit_transform(X)
        n_comp = min(len(numeric), len(df) - 1, 20)
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(Xs)
        cum = np.cumsum(pca.explained_variance_ratio_)
        n_for_var = int(np.searchsorted(cum, pca_variance) + 1)
        pca_info = {
            "n_features": len(numeric),
            "n_components_fit": n_comp,
            "variance_target": pca_variance,
            "components_for_target_variance": min(n_for_var, n_comp),
            "explained_variance_ratio": [round(float(x), 4) for x in pca.explained_variance_ratio_[:10]],
            "cumulative_variance": [round(float(x), 4) for x in cum[:10]],
        }

    return {
        "corr_threshold": corr_threshold,
        "numeric_features": numeric,
        "drop_suggestions": drop_suggestions,
        "suggested_keep": sorted(kept),
        "n_drop_suggestions": len(drop_suggestions),
        "pca": pca_info,
    }


def apply_variable_reduction(
    df: pd.DataFrame,
    drop_columns: list[str],
    target: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop selected columns (never the target)."""
    safe = [c for c in drop_columns if c in df.columns and c != target]
    result = df.drop(columns=safe) if safe else df.copy()
    return result, {"dropped": safe, "n_dropped": len(safe)}
