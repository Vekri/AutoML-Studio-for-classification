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


def compute_quality_score(
    df: pd.DataFrame,
    target: str | None = None,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute a single 0–100 data quality score from missingness, duplicates,
    constants, class balance, outliers, and target validity.
    """
    breakdown: dict[str, float] = {}
    rows = max(len(df), 1)
    feature_cols = features or [c for c in df.columns if c != target]

    missing_pct = float(df.isna().sum().sum()) / max(rows * max(df.shape[1], 1), 1)
    breakdown["completeness"] = round(max(0.0, 25.0 * (1.0 - min(missing_pct / 0.25, 1.0))), 2)

    dup_pct = float(df.duplicated().mean())
    breakdown["uniqueness"] = round(max(0.0, 15.0 * (1.0 - min(dup_pct / 0.1, 1.0))), 2)

    constants = sum(1 for c in feature_cols if df[c].nunique(dropna=True) <= 1)
    id_like = sum(
        1
        for c in feature_cols
        if df[c].nunique(dropna=True) == rows
        and rows > 1
        and not pd.api.types.is_numeric_dtype(df[c])
    )
    bad_feat_ratio = min((constants + id_like) / max(len(feature_cols), 1), 1.0)
    breakdown["feature_validity"] = round(15.0 * (1.0 - bad_feat_ratio), 2)

    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    outlier_fracs: list[float] = []
    for col in numeric:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        outlier_fracs.append(float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()))
    avg_out = float(sum(outlier_fracs) / len(outlier_fracs)) if outlier_fracs else 0.0
    breakdown["outlier_health"] = round(max(0.0, 15.0 * (1.0 - min(avg_out / 0.15, 1.0))), 2)

    target_score = 10.0
    if target and target in df.columns:
        t = df[target]
        n_unique = int(t.nunique(dropna=True))
        miss = float(t.isna().mean())
        target_score = 0.0
        if n_unique == 2:
            target_score += 12.0
        elif n_unique > 2:
            target_score += 4.0
        target_score += 8.0 * (1.0 - min(miss / 0.05, 1.0))
    breakdown["target_health"] = round(min(target_score, 20.0), 2)

    balance_score = 5.0
    if target and target in df.columns and df[target].nunique(dropna=True) == 2:
        vc = df[target].value_counts()
        ratio = float(vc.min() / vc.max()) if vc.max() else 0.0
        balance_score = 10.0 * min(ratio / 0.5, 1.0)
    breakdown["class_balance"] = round(balance_score, 2)

    score = float(max(0.0, min(100.0, round(sum(breakdown.values()), 1))))
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 55:
        grade = "C"
    else:
        grade = "D"

    issues: list[str] = []
    if breakdown["completeness"] < 18:
        issues.append("High missingness reduces completeness")
    if breakdown["uniqueness"] < 12:
        issues.append("Elevated duplicate rate")
    if breakdown["feature_validity"] < 10:
        issues.append("Constant or ID-like features present")
    if breakdown["outlier_health"] < 10:
        issues.append("Many numeric outliers detected")
    if breakdown["target_health"] < 14:
        issues.append("Target column needs attention")
    if breakdown["class_balance"] < 6:
        issues.append("Class imbalance is material")

    return {
        "score": score,
        "grade": grade,
        "breakdown": breakdown,
        "max_score": 100,
        "issues": issues,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
    }
