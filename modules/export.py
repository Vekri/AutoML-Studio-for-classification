"""Export artifacts for Predictions Studio."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import pandas as pd


def build_studio_manifest(
    domain: str,
    target: str,
    features: list[str],
    cleaning_config: dict[str, Any],
    binning_config: dict[str, Any],
    validation_report: dict[str, Any],
    feature_selection: dict[str, Any],
    selected_algorithm: dict[str, Any] | None = None,
    model_validation: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    quality_score: dict[str, Any] | None = None,
    encoding_config: dict[str, Any] | None = None,
    scaling_config: dict[str, Any] | None = None,
    feature_eng_config: dict[str, Any] | None = None,
    reduction_config: dict[str, Any] | None = None,
    tuning_result: dict[str, Any] | None = None,
    explanation: dict[str, Any] | None = None,
    insights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create master manifest for Predictions Studio integration."""
    return {
        "studio": "AutoML Studio for Binary Classification",
        "version": "3.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "problem_domain": domain,
        "target_column": target,
        "selected_features": features,
        "feature_count": len(features),
        "profile_summary": {
            "rows": (profile or {}).get("rows"),
            "columns": (profile or {}).get("columns"),
            "missing_pct": (profile or {}).get("missing_pct"),
            "duplicate_pct": (profile or {}).get("duplicate_pct"),
        }
        if profile
        else {},
        "quality_score": quality_score or {},
        "cleaning_config": cleaning_config,
        "encoding_config": encoding_config or {},
        "scaling_config": scaling_config or {},
        "feature_eng_config": feature_eng_config or {},
        "reduction_config": reduction_config or {},
        "binning_config": binning_config,
        "validation_report": validation_report,
        "feature_selection": feature_selection,
        "selected_algorithm": selected_algorithm or {},
        "tuning_result": tuning_result or {},
        "explanation": {
            "method": (explanation or {}).get("method"),
            "top_features": (explanation or {}).get("top_features"),
        }
        if explanation
        else {},
        "insights": insights or {},
        "model_validation_summary": {
            "combinations_count": (model_validation or {}).get("combinations_count"),
            "best": (model_validation or {}).get("best"),
            "models_requested": (model_validation or {}).get("models_requested"),
            "balance_methods_requested": (model_validation or {}).get("balance_methods_requested"),
        }
        if model_validation
        else {},
        "pipeline_steps": [
            "load_data",
            "profile_data",
            "quality_score",
            "missing_value_treatment",
            "outlier_detection",
            "duplicate_detection",
            "encoding",
            "scaling",
            "feature_engineering",
            "variable_reduction",
            "feature_selection",
            "model_selection",
            "hyperparameter_tuning",
            "model_explanation",
            "business_insights",
            "executive_report",
            "predict",
        ],
        "ready_for_predictions_studio": True,
    }


def export_zip_bundle(
    df: pd.DataFrame,
    manifest: dict[str, Any],
    feature_list: list[str],
    cleaning_config: dict[str, Any],
    binning_config: dict[str, Any],
    validation_report: dict[str, Any],
    feature_selection_results: dict[str, Any],
    domain: str,
    extra_files: dict[str, str] | None = None,
) -> bytes:
    """Package all artifacts into a downloadable ZIP."""
    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("processed_dataset.csv", df.to_csv(index=False))
        zf.writestr("feature_list.json", json.dumps(feature_list, indent=2))
        zf.writestr("cleaning_config.json", json.dumps(cleaning_config, indent=2, default=str))
        zf.writestr("binning_config.json", json.dumps(binning_config, indent=2, default=str))
        zf.writestr("validation_report.json", json.dumps(validation_report, indent=2, default=str))
        zf.writestr(
            "feature_selection.json",
            json.dumps(feature_selection_results, indent=2, default=str),
        )
        zf.writestr(
            "selected_algorithm.json",
            json.dumps(manifest.get("selected_algorithm") or {}, indent=2, default=str),
        )
        zf.writestr("domain_config.json", json.dumps({"domain": domain}, indent=2))
        zf.writestr("studio_manifest.json", json.dumps(manifest, indent=2, default=str))
        for name, content in (extra_files or {}).items():
            zf.writestr(name, content)

        algo = manifest.get("selected_algorithm") or {}
        readme = f"""AutoML Studio Export Bundle
===========================
Domain: {domain}
Target: {manifest['target_column']}
Features: {len(feature_list)}
Best algorithm: {algo.get('model_label', 'N/A')}
Balancing: {algo.get('balance_label', 'N/A')}
Quality score: {(manifest.get('quality_score') or {}).get('score', 'N/A')}
Created: {manifest['created_at']}

Files:
- processed_dataset.csv
- feature_list.json
- cleaning_config.json / encoding_config.json / scaling_config.json
- profile.json / quality_score.json
- feature_selection.json / explanation.json / insights.json
- selected_algorithm.json / tuning_result.json
- executive_report.html
- studio_manifest.json
"""
        zf.writestr("README.txt", readme)

    buffer.seek(0)
    return buffer.getvalue()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dict_to_json_bytes(data: dict) -> bytes:
    return json.dumps(data, indent=2, default=str).encode("utf-8")
