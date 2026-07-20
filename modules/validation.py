"""Data validation and quality checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


def validate_dataset(df: pd.DataFrame, target: str, features: list[str] | None = None) -> dict[str, Any]:
    """Run comprehensive validation checks."""
    report: dict[str, Any] = {
        "checks": [],
        "passed": 0,
        "failed": 0,
        "warnings": 0,
    }

    def add_check(name: str, status: str, message: str, details: Any = None):
        report["checks"].append({"name": name, "status": status, "message": message, "details": details})
        if status == "pass":
            report["passed"] += 1
        elif status == "fail":
            report["failed"] += 1
        else:
            report["warnings"] += 1

    if df.empty:
        add_check("dataset_size", "fail", "Dataset is empty")
        return report

    add_check("dataset_size", "pass", f"Dataset has {len(df)} rows and {len(df.columns)} columns")

    target_series = df[target]
    n_unique = target_series.nunique(dropna=True)
    if n_unique == 2:
        add_check("binary_target", "pass", f"Target '{target}' is binary")
    else:
        add_check("binary_target", "fail", f"Target has {n_unique} unique values (expected 2)")

    missing_target = target_series.isna().sum()
    if missing_target == 0:
        add_check("target_missing", "pass", "No missing values in target")
    else:
        add_check("target_missing", "fail", f"{missing_target} missing values in target")

    class_counts = target_series.value_counts()
    if len(class_counts) == 2:
        ratio = class_counts.min() / class_counts.max()
        if ratio >= 0.1:
            add_check("class_balance", "pass", f"Class balance ratio: {ratio:.3f}")
        elif ratio >= 0.05:
            add_check("class_balance", "warning", f"Mild imbalance — ratio: {ratio:.3f}")
        else:
            add_check("class_balance", "warning", f"Severe imbalance — ratio: {ratio:.3f}. Consider SMOTE or class weights")

    feature_cols = features or [c for c in df.columns if c != target]
    constant_cols = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
    if not constant_cols:
        add_check("constant_features", "pass", "No constant features detected")
    else:
        add_check("constant_features", "warning", f"Constant features: {constant_cols}")

    high_missing = [
        c for c in feature_cols if df[c].isna().mean() > 0.3 and c != target
    ]
    if not high_missing:
        add_check("missing_features", "pass", "No features with >30% missing values")
    else:
        add_check("missing_features", "warning", f"High missing: {high_missing}")

    dup_pct = df.duplicated().mean() * 100
    if dup_pct < 1:
        add_check("duplicates", "pass", f"Duplicate rate: {dup_pct:.2f}%")
    else:
        add_check("duplicates", "warning", f"Duplicate rate: {dup_pct:.2f}%")

    if len(df) < 100:
        add_check("sample_size", "warning", f"Small dataset ({len(df)} rows) — results may be unreliable")
    else:
        add_check("sample_size", "pass", f"Adequate sample size: {len(df)} rows")

    return report


def run_model_validation(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Quick model validation with train/test split and cross-validation."""
    X = df[features].copy()
    y = df[target].copy()

    if not pd.api.types.is_numeric_dtype(y):
        y = LabelEncoder().fit_transform(y.astype(str))

    for col in X.select_dtypes(exclude=["number"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    results: dict[str, Any] = {"models": {}, "split": {"train": len(X_train), "test": len(X_test)}}

    for name, model in [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(model, X, y, cv=min(cv_folds, 5), scoring="roc_auc")

        results["models"][name] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "auc_roc": round(roc_auc_score(y_test, y_prob), 4),
            "cv_auc_mean": round(cv_scores.mean(), 4),
            "cv_auc_std": round(cv_scores.std(), 4),
        }

    return results
