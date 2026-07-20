"""Feature selection methods and visualizations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def _is_id_like(series: pd.Series) -> bool:
    name = str(series.name or "").lower()
    if any(tok in name for tok in ("id", "uuid", "guid", "index", "key")):
        nunique = series.nunique(dropna=True)
        if nunique >= max(10, int(0.9 * len(series))):
            return True
    if not pd.api.types.is_numeric_dtype(series):
        nunique = series.nunique(dropna=True)
        if nunique >= max(50, int(0.95 * len(series))):
            return True
    return False


def _prepare_xy(df: pd.DataFrame, target: str, features: list[str] | None = None):
    feature_cols = features or [c for c in df.columns if c != target]
    feature_cols = [c for c in feature_cols if c in df.columns and c != target]
    # Drop ID-like / all-unique columns that break models
    feature_cols = [c for c in feature_cols if not _is_id_like(df[c])]

    if not feature_cols:
        raise ValueError("No usable features found after removing ID-like columns.")

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Drop rows with missing target
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    else:
        y = y.astype(int) if set(pd.Series(y).dropna().unique()).issubset({0, 1}) else y

    encoders: dict[str, LabelEncoder] = {}
    for col in list(X.columns):
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna("__missing__"))
            encoders[col] = le
        else:
            X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)

    return X, y, encoders


def correlation_feature_selection(
    df: pd.DataFrame, target: str, threshold: float = 0.1
) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=["number"])
    if target not in numeric_df.columns:
        return pd.DataFrame()

    corr = numeric_df.corr()[target].drop(target).abs().sort_values(ascending=False)
    result = pd.DataFrame({"feature": corr.index, "correlation": corr.values})
    result["selected"] = result["correlation"] >= threshold
    return result


def _mark_top_k(result: pd.DataFrame, top_k: int) -> pd.DataFrame:
    result = result.reset_index(drop=True)
    k = min(top_k, len(result))
    result["selected"] = result.index < k
    return result


def mutual_info_selection(df: pd.DataFrame, target: str, top_k: int = 10) -> pd.DataFrame:
    X, y, _ = _prepare_xy(df, target)
    scores = mutual_info_classif(X, y, random_state=42)
    result = pd.DataFrame({"feature": X.columns, "mutual_info": scores})
    result = result.sort_values("mutual_info", ascending=False)
    return _mark_top_k(result, top_k)


def chi_square_selection(df: pd.DataFrame, target: str, top_k: int = 10) -> pd.DataFrame:
    X, y, _ = _prepare_xy(df, target)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    scores, pvalues = chi2(X_scaled, y)
    result = pd.DataFrame({"feature": list(X.columns), "chi2": scores, "p_value": pvalues})
    result = result.sort_values("chi2", ascending=False)
    return _mark_top_k(result, top_k)


def anova_selection(df: pd.DataFrame, target: str, top_k: int = 10) -> pd.DataFrame:
    X, y, _ = _prepare_xy(df, target)
    scores, pvalues = f_classif(X, y)
    result = pd.DataFrame({"feature": list(X.columns), "f_score": scores, "p_value": pvalues})
    result = result.sort_values("f_score", ascending=False)
    return _mark_top_k(result, top_k)


def rfe_selection(df: pd.DataFrame, target: str, n_features: int = 10) -> pd.DataFrame:
    X, y, _ = _prepare_xy(df, target)
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    selector = RFE(estimator, n_features_to_select=min(n_features, X.shape[1]), step=1)
    selector.fit(X, y)
    result = pd.DataFrame(
        {
            "feature": X.columns,
            "rank": selector.ranking_,
            "selected": selector.support_,
        }
    )
    return result.sort_values("rank")


def lasso_selection(df: pd.DataFrame, target: str) -> pd.DataFrame:
    X, y, _ = _prepare_xy(df, target)
    model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42)
    model.fit(X, y)
    coefs = np.abs(model.coef_[0])
    result = pd.DataFrame({"feature": X.columns, "coefficient": coefs})
    result = result.sort_values("coefficient", ascending=False)
    result["selected"] = result["coefficient"] > 0.01
    return result


def random_forest_importance(df: pd.DataFrame, target: str) -> pd.DataFrame:
    X, y, _ = _prepare_xy(df, target)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    result = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    return result.sort_values("importance", ascending=False)


def run_feature_selection(
    df: pd.DataFrame,
    target: str,
    method: str,
    top_k: int = 10,
    threshold: float = 0.1,
) -> pd.DataFrame:
    methods = {
        "correlation_threshold": lambda: correlation_feature_selection(df, target, threshold),
        "mutual_information": lambda: mutual_info_selection(df, target, top_k),
        "chi_square": lambda: chi_square_selection(df, target, top_k),
        "anova_f": lambda: anova_selection(df, target, top_k),
        "rfe": lambda: rfe_selection(df, target, top_k),
        "lasso": lambda: lasso_selection(df, target),
    }
    if method not in methods:
        raise ValueError(f"Unknown method: {method}")
    return methods[method]()


def get_selected_features(result: pd.DataFrame) -> list[str]:
    if "selected" in result.columns:
        return result.loc[result["selected"], "feature"].tolist()
    return result["feature"].head(10).tolist()
