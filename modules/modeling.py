"""Multi-model training with optional class balancing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from modules.balancing import apply_balancing, list_balance_methods

MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "supports_class_weight": True,
    },
    "random_forest": {
        "label": "Random Forest",
        "supports_class_weight": True,
    },
    "gradient_boosting": {
        "label": "Gradient Boosting",
        "supports_class_weight": False,
    },
    "extra_trees": {
        "label": "Extra Trees",
        "supports_class_weight": True,
    },
    "decision_tree": {
        "label": "Decision Tree",
        "supports_class_weight": True,
    },
    "ada_boost": {
        "label": "AdaBoost",
        "supports_class_weight": False,
    },
    "svm": {
        "label": "SVM (RBF)",
        "supports_class_weight": True,
    },
    "knn": {
        "label": "K-Nearest Neighbors",
        "supports_class_weight": False,
    },
    "naive_bayes": {
        "label": "Naive Bayes",
        "supports_class_weight": False,
    },
}


def list_models() -> list[dict[str, Any]]:
    return [{"id": k, **{kk: vv for kk, vv in v.items() if kk != "factory"}} for k, v in MODEL_CATALOG.items()]


def _build_model(model_id: str, use_class_weight: bool, random_state: int = 42):
    cw = "balanced" if use_class_weight else None
    if model_id == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=random_state, class_weight=cw)
    if model_id == "random_forest":
        return RandomForestClassifier(
            n_estimators=150, random_state=random_state, n_jobs=-1, class_weight=cw
        )
    if model_id == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    if model_id == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=150, random_state=random_state, n_jobs=-1, class_weight=cw
        )
    if model_id == "decision_tree":
        return DecisionTreeClassifier(random_state=random_state, class_weight=cw)
    if model_id == "ada_boost":
        return AdaBoostClassifier(random_state=random_state, n_estimators=100)
    if model_id == "svm":
        return SVC(probability=True, random_state=random_state, class_weight=cw)
    if model_id == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if model_id == "naive_bayes":
        return GaussianNB()
    raise ValueError(f"Unknown model: {model_id}")


def _prepare_xy(df: pd.DataFrame, target: str, features: list[str]):
    features = [c for c in features if c in df.columns and c != target]
    if not features:
        raise ValueError("No valid features available for modeling.")

    work = df[features + [target]].dropna(subset=[target]).copy()
    X = work[features].copy()
    y = work[target].copy()

    if not pd.api.types.is_numeric_dtype(y):
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        y = y.astype(int).to_numpy()

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = LabelEncoder().fit_transform(X[col].astype(str).fillna("__missing__"))
        else:
            X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)

    return X.to_numpy(dtype=float), np.asarray(y), features


def _score_model(model, X_train, y_train, X_test, y_test, X_full, y_full, n_splits: int) -> dict[str, Any]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # decision_function fallback
        scores = model.decision_function(X_test)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    try:
        cv_scores = cross_val_score(model, X_full, y_full, cv=n_splits, scoring="roc_auc")
        cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
    except Exception:
        cv_mean, cv_std = float("nan"), float("nan")

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "cv_auc_mean": round(cv_mean, 4) if cv_mean == cv_mean else None,
        "cv_auc_std": round(cv_std, 4) if cv_std == cv_std else None,
    }


def run_model_suite(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    models: list[str] | None = None,
    balance_methods: list[str] | None = None,
    test_size: float = 0.2,
    cv_folds: int = 5,
    run_all_combinations: bool = True,
) -> dict[str, Any]:
    """
    Train selected models with selected balancing strategies.
    If run_all_combinations is True, every model × balancing pair is evaluated.
    """
    X, y, used_features = _prepare_xy(df, target, features)

    models = models or ["logistic_regression", "random_forest"]
    balance_methods = balance_methods or ["none"]

    # Validate ids
    models = [m for m in models if m in MODEL_CATALOG]
    balance_methods = [b for b in balance_methods if b in {x["id"] for x in list_balance_methods()}]
    if not models:
        raise ValueError("Select at least one valid model")
    if not balance_methods:
        balance_methods = ["none"]

    unique, counts = np.unique(y, return_counts=True)
    can_stratify = len(unique) == 2 and counts.min() >= 2
    n_splits = max(2, min(cv_folds, 5, int(counts.min()) if can_stratify else 2))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=min(test_size, 0.4),
        random_state=42,
        stratify=y if can_stratify else None,
    )

    combinations: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    pairs: list[tuple[str, str]] = []
    if run_all_combinations:
        for bal in balance_methods:
            for mid in models:
                pairs.append((bal, mid))
    else:
        # Pair by index / zip-like: each balance with each model still if single lists
        for bal in balance_methods:
            for mid in models:
                pairs.append((bal, mid))

    for bal, mid in pairs:
        use_cw = bal == "class_weight" and MODEL_CATALOG[mid]["supports_class_weight"]
        X_bal, y_bal, bal_meta = apply_balancing(X_train, y_train, method=bal)

        # If class_weight requested but model doesn't support it, note it
        if bal == "class_weight" and not MODEL_CATALOG[mid]["supports_class_weight"]:
            bal_meta = {
                **bal_meta,
                "note": "Model has no class_weight — trained without balancing adjustment",
            }
            use_cw = False

        model = _build_model(mid, use_class_weight=use_cw)
        try:
            metrics = _score_model(model, X_bal, y_bal, X_test, y_test, X, y, n_splits)
            status = "ok"
            error = None
        except Exception as exc:
            metrics = {}
            status = "error"
            error = str(exc)

        row = {
            "balance_method": bal,
            "balance_label": bal_meta.get("label", bal),
            "model_id": mid,
            "model_label": MODEL_CATALOG[mid]["label"],
            "status": status,
            "error": error,
            "metrics": metrics,
            "balance_meta": bal_meta,
            "class_weight_used": bool(use_cw),
        }
        combinations.append(row)

        if status == "ok" and metrics.get("auc_roc") is not None:
            if best is None or float(metrics["auc_roc"]) > float(best["metrics"]["auc_roc"]):
                best = row

    # Leaderboard sorted by AUC
    leaderboard = sorted(
        [c for c in combinations if c["status"] == "ok"],
        key=lambda r: float(r["metrics"].get("auc_roc") or -1),
        reverse=True,
    )

    # Backward-compatible flat models map (best result per model label)
    models_map: dict[str, Any] = {}
    for row in leaderboard:
        label = row["model_label"]
        if label not in models_map:
            models_map[label] = row["metrics"]

    return {
        "split": {
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            "train_class_counts": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
            "test_class_counts": {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        },
        "features_used": used_features,
        "models_requested": models,
        "balance_methods_requested": balance_methods,
        "run_all_combinations": run_all_combinations,
        "combinations_count": len(combinations),
        "combinations": combinations,
        "leaderboard": leaderboard,
        "best": best,
        "models": models_map,
    }


# Keep old name for compatibility
def run_model_validation(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> dict[str, Any]:
    result = run_model_suite(
        df,
        target,
        features,
        models=["logistic_regression", "random_forest"],
        balance_methods=["none"],
        test_size=test_size,
        cv_folds=cv_folds,
        run_all_combinations=True,
    )
    # Shape similar to old API
    models_map = {}
    for row in result["combinations"]:
        if row["status"] == "ok":
            models_map[row["model_label"]] = row["metrics"]
    return {
        "models": models_map,
        "split": {"train": result["split"]["train"], "test": result["split"]["test"]},
        "features_used": result["features_used"],
        "combinations": result["combinations"],
        "leaderboard": result["leaderboard"],
        "best": result["best"],
    }
