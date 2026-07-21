"""Model explanation: SHAP when available, else permutation importance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from modules.modeling import MODEL_CATALOG, _build_model, _prepare_xy

try:
    import shap

    HAS_SHAP = True
except ImportError:  # pragma: no cover
    HAS_SHAP = False


def explain_model(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model_id: str = "random_forest",
    use_class_weight: bool = False,
    top_k: int = 15,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    """Return feature importance explanation for a fitted model."""
    X, y, used = _prepare_xy(df, target, features)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) == 2 else None,
    )
    model = _build_model(model_id, use_class_weight=use_class_weight, random_state=random_state)
    model.fit(X_train, y_train)

    method = "permutation"
    importances: list[dict[str, Any]] = []
    note = None

    if HAS_SHAP:
        try:
            # Limit background / explain sample for speed
            bg_n = min(100, len(X_train))
            ex_n = min(150, len(X_test))
            background = X_train[:bg_n]
            explain_x = X_test[:ex_n]
            tree_ids = {
                "random_forest",
                "gradient_boosting",
                "extra_trees",
                "decision_tree",
                "ada_boost",
            }
            if model_id in tree_ids:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(explain_x)
                if isinstance(shap_values, list):
                    vals = np.abs(shap_values[1] if len(shap_values) > 1 else shap_values[0])
                else:
                    vals = np.abs(shap_values)
                    if vals.ndim == 3:
                        vals = vals[:, :, 1]
            elif model_id == "logistic_regression":
                explainer = shap.LinearExplainer(model, background)
                shap_values = explainer.shap_values(explain_x)
                vals = np.abs(shap_values)
            else:
                explainer = shap.Explainer(model.predict_proba, background)
                sv = explainer(explain_x)
                vals = np.abs(sv.values)
                if vals.ndim == 3:
                    vals = vals[:, :, 1]
            mean_abs = vals.mean(axis=0)
            method = "shap"
            order = np.argsort(mean_abs)[::-1][:top_k]
            importances = [
                {
                    "feature": used[int(i)],
                    "importance": round(float(mean_abs[int(i)]), 6),
                }
                for i in order
            ]
        except Exception as exc:  # noqa: BLE001
            note = f"SHAP failed ({exc}); fell back to permutation importance"
            method = "permutation"

    if method == "permutation" and not importances:
        r = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=8,
            random_state=random_state,
            scoring="roc_auc",
            n_jobs=-1,
        )
        order = np.argsort(r.importances_mean)[::-1][:top_k]
        importances = [
            {
                "feature": used[int(i)],
                "importance": round(float(r.importances_mean[int(i)]), 6),
                "std": round(float(r.importances_std[int(i)]), 6),
            }
            for i in order
        ]

    return {
        "model_id": model_id,
        "model_label": MODEL_CATALOG.get(model_id, {}).get("label", model_id),
        "method": method,
        "shap_available": HAS_SHAP,
        "features_used": used,
        "importances": importances,
        "top_features": [x["feature"] for x in importances[:10]],
        "note": note,
        "n_explained_rows": int(min(150, len(X_test))),
    }


def explain_selected_algorithm(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    selected_algorithm: dict[str, Any] | None,
    top_k: int = 15,
) -> dict[str, Any]:
    algo = selected_algorithm or {}
    model_id = algo.get("model_id") or "random_forest"
    use_cw = bool(algo.get("class_weight_used") or algo.get("balance_method") == "class_weight")
    result = explain_model(
        df,
        target,
        features,
        model_id=model_id,
        use_class_weight=use_cw,
        top_k=top_k,
    )
    result["balance_method"] = algo.get("balance_method")
    result["balance_label"] = algo.get("balance_label")
    return result
