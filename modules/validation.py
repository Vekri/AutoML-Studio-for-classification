"""Data validation and quality checks."""

from __future__ import annotations

from typing import Any

import pandas as pd

from modules.modeling import run_model_suite, run_model_validation  # noqa: F401


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

    missing_target = int(target_series.isna().sum())
    if missing_target == 0:
        add_check("target_missing", "pass", "No missing values in target")
    else:
        add_check("target_missing", "fail", f"{missing_target} missing values in target")

    class_counts = target_series.value_counts()
    if len(class_counts) == 2:
        ratio = float(class_counts.min() / class_counts.max())
        if ratio >= 0.1:
            add_check("class_balance", "pass", f"Class balance ratio: {ratio:.3f}")
        elif ratio >= 0.05:
            add_check("class_balance", "warning", f"Mild imbalance — ratio: {ratio:.3f}")
        else:
            add_check(
                "class_balance",
                "warning",
                f"Severe imbalance — ratio: {ratio:.3f}. Try SMOTE / class weights in modeling",
            )

    feature_cols = features or [c for c in df.columns if c != target]
    constant_cols = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
    if not constant_cols:
        add_check("constant_features", "pass", "No constant features detected")
    else:
        add_check("constant_features", "warning", f"Constant features: {constant_cols}")

    high_missing = [c for c in feature_cols if df[c].isna().mean() > 0.3 and c != target]
    if not high_missing:
        add_check("missing_features", "pass", "No features with >30% missing values")
    else:
        add_check("missing_features", "warning", f"High missing: {high_missing}")

    dup_pct = float(df.duplicated().mean() * 100)
    if dup_pct < 1:
        add_check("duplicates", "pass", f"Duplicate rate: {dup_pct:.2f}%")
    else:
        add_check("duplicates", "warning", f"Duplicate rate: {dup_pct:.2f}%")

    if len(df) < 100:
        add_check("sample_size", "warning", f"Small dataset ({len(df)} rows) — results may be unreliable")
    else:
        add_check("sample_size", "pass", f"Adequate sample size: {len(df)} rows")

    return report
