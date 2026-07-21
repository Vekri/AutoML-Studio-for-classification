"""Class balancing / resampling strategies for binary classification."""

from __future__ import annotations

from typing import Any

import numpy as np

BALANCE_METHODS: dict[str, dict[str, str]] = {
    "none": {
        "label": "None (original distribution)",
        "description": "No resampling — keep natural class counts",
    },
    "class_weight": {
        "label": "Class weights (balanced)",
        "description": "No resampling — models use class_weight='balanced'",
    },
    "random_oversample": {
        "label": "Random oversampling",
        "description": "Duplicate minority class samples until balanced",
    },
    "random_undersample": {
        "label": "Random undersampling",
        "description": "Remove majority class samples until balanced",
    },
    "smote": {
        "label": "SMOTE",
        "description": "Synthetic Minority Over-sampling Technique",
    },
    "borderline_smote": {
        "label": "Borderline-SMOTE",
        "description": "SMOTE focused on borderline minority samples",
    },
    "adasyn": {
        "label": "ADASYN",
        "description": "Adaptive synthetic sampling for harder minority cases",
    },
    "smote_tomek": {
        "label": "SMOTE + Tomek links",
        "description": "Oversample with SMOTE then clean with Tomek links",
    },
    "smote_enn": {
        "label": "SMOTE + ENN",
        "description": "Oversample with SMOTE then clean with Edited Nearest Neighbours",
    },
}


def list_balance_methods() -> list[dict[str, str]]:
    return [{"id": k, **v} for k, v in BALANCE_METHODS.items()]


def class_counts(y: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(y, return_counts=True)
    return {str(int(u) if str(u).isdigit() else u): int(c) for u, c in zip(unique, counts)}


def _random_oversample(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    classes, counts = np.unique(y, return_counts=True)
    max_n = int(counts.max())
    X_parts, y_parts = [], []
    rng = np.random.RandomState(random_state)
    for cls in classes:
        mask = y == cls
        X_c, y_c = X[mask], y[mask]
        if len(X_c) < max_n:
            idx = rng.choice(len(X_c), size=max_n, replace=True)
            X_c, y_c = X_c[idx], y_c[idx]
        X_parts.append(X_c)
        y_parts.append(y_c)
    X_out = np.vstack(X_parts)
    y_out = np.concatenate(y_parts)
    order = rng.permutation(len(y_out))
    return X_out[order], y_out[order]


def _random_undersample(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    classes, counts = np.unique(y, return_counts=True)
    min_n = int(counts.min())
    X_parts, y_parts = [], []
    rng = np.random.RandomState(random_state)
    for cls in classes:
        mask = y == cls
        X_c, y_c = X[mask], y[mask]
        if len(X_c) > min_n:
            idx = rng.choice(len(X_c), size=min_n, replace=False)
            X_c, y_c = X_c[idx], y_c[idx]
        X_parts.append(X_c)
        y_parts.append(y_c)
    X_out = np.vstack(X_parts)
    y_out = np.concatenate(y_parts)
    order = rng.permutation(len(y_out))
    return X_out[order], y_out[order]


def apply_balancing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "none",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply balancing to training data only. Returns X, y, meta."""
    method = method or "none"
    before = class_counts(y_train)
    meta: dict[str, Any] = {
        "method": method,
        "label": BALANCE_METHODS.get(method, {}).get("label", method),
        "before": before,
        "applied": False,
        "note": None,
    }

    if method in ("none", "class_weight"):
        meta["after"] = before
        meta["note"] = (
            "No resampling"
            if method == "none"
            else "Resampling skipped — class_weight='balanced' used in models"
        )
        return X_train, y_train, meta

    try:
        if method == "random_oversample":
            X_res, y_res = _random_oversample(X_train, y_train, random_state)
        elif method == "random_undersample":
            X_res, y_res = _random_undersample(X_train, y_train, random_state)
        else:
            from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
            from imblearn.combine import SMOTEENN, SMOTETomek

            samplers = {
                "smote": SMOTE(random_state=random_state),
                "borderline_smote": BorderlineSMOTE(random_state=random_state),
                "adasyn": ADASYN(random_state=random_state),
                "smote_tomek": SMOTETomek(random_state=random_state),
                "smote_enn": SMOTEENN(random_state=random_state),
            }
            if method not in samplers:
                raise ValueError(f"Unknown balancing method: {method}")
            X_res, y_res = samplers[method].fit_resample(X_train, y_train)

        meta["after"] = class_counts(y_res)
        meta["applied"] = True
        meta["train_rows_before"] = int(len(y_train))
        meta["train_rows_after"] = int(len(y_res))
        return X_res, y_res, meta
    except Exception as exc:
        # Fallback: keep original train set but record failure
        meta["after"] = before
        meta["note"] = f"Balancing failed ({exc}); used original train set"
        return X_train, y_train, meta
