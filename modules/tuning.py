"""Light hyperparameter tuning via RandomizedSearchCV."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from modules.modeling import MODEL_CATALOG, _build_model, _prepare_xy


PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {
        "C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    },
    "random_forest": {
        "n_estimators": [100, 150, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "gradient_boosting": {
        "n_estimators": [80, 120, 200],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4],
    },
    "extra_trees": {
        "n_estimators": [100, 150, 250],
        "max_depth": [None, 8, 16],
        "min_samples_split": [2, 5],
    },
    "decision_tree": {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
    },
    "svm": {
        "C": [0.1, 0.5, 1.0, 2.0],
        "gamma": ["scale", "auto"],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
    },
    "ada_boost": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.5, 1.0, 1.5],
    },
}


def _scoring_name(metric: str) -> str:
    mapping = {
        "auc_roc": "roc_auc",
        "f1": "f1",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
    }
    return mapping.get(metric, "roc_auc")


def tune_model(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model_id: str = "random_forest",
    use_class_weight: bool = False,
    selection_metric: str = "auc_roc",
    n_iter: int = 20,
    cv_folds: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run RandomizedSearchCV for one model and return best params + score."""
    if model_id not in MODEL_CATALOG:
        raise ValueError(f"Unknown model: {model_id}")
    if model_id == "naive_bayes":
        return {
            "model_id": model_id,
            "model_label": MODEL_CATALOG[model_id]["label"],
            "status": "skipped",
            "message": "Naive Bayes has no meaningful hyperparameters in this suite",
            "best_params": {},
            "best_score": None,
        }

    X, y, used = _prepare_xy(df, target, features)
    estimator = _build_model(model_id, use_class_weight=use_class_weight, random_state=random_state)
    grid = PARAM_GRIDS.get(model_id)
    if not grid:
        return {
            "model_id": model_id,
            "model_label": MODEL_CATALOG[model_id]["label"],
            "status": "skipped",
            "message": "No search grid defined",
            "best_params": {},
            "best_score": None,
        }

    n_splits = max(2, min(cv_folds, int(np.unique(y, return_counts=True)[1].min()), 5))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator,
        param_distributions=grid,
        n_iter=min(n_iter, int(np.prod([len(v) for v in grid.values()]))),
        scoring=_scoring_name(selection_metric),
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)

    return {
        "model_id": model_id,
        "model_label": MODEL_CATALOG[model_id]["label"],
        "status": "ok",
        "selection_metric": selection_metric,
        "scoring": _scoring_name(selection_metric),
        "best_params": {k: (None if v is None else (int(v) if isinstance(v, (np.integer,)) else (float(v) if isinstance(v, (np.floating,)) else v))) for k, v in search.best_params_.items()},
        "best_score": round(float(search.best_score_), 4),
        "n_iter": int(search.n_iter),
        "cv_folds": n_splits,
        "features_used": used,
        "class_weight_used": use_class_weight,
    }


def tune_selected_algorithm(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    selected_algorithm: dict[str, Any] | None,
    selection_metric: str = "auc_roc",
    n_iter: int = 20,
    cv_folds: int = 3,
) -> dict[str, Any]:
    """Tune the currently selected algorithm (or RF default)."""
    algo = selected_algorithm or {}
    model_id = algo.get("model_id") or "random_forest"
    use_cw = bool(algo.get("class_weight_used") or algo.get("balance_method") == "class_weight")
    result = tune_model(
        df,
        target,
        features,
        model_id=model_id,
        use_class_weight=use_cw,
        selection_metric=selection_metric,
        n_iter=n_iter,
        cv_folds=cv_folds,
    )
    result["balance_method"] = algo.get("balance_method")
    result["balance_label"] = algo.get("balance_label")
    return result
