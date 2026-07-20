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
) -> dict[str, Any]:
    """Create master manifest for Predictions Studio integration."""
    return {
        "studio": "AutoML Studio for Binary Classification",
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "problem_domain": domain,
        "target_column": target,
        "selected_features": features,
        "feature_count": len(features),
        "cleaning_config": cleaning_config,
        "binning_config": binning_config,
        "validation_report": validation_report,
        "feature_selection": feature_selection,
        "pipeline_steps": [
            "load_data",
            "apply_cleaning",
            "apply_binning",
            "select_features",
            "encode_categorical",
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
        zf.writestr("domain_config.json", json.dumps({"domain": domain}, indent=2))
        zf.writestr("studio_manifest.json", json.dumps(manifest, indent=2, default=str))

        readme = f"""AutoML Studio Export Bundle
===========================
Domain: {domain}
Target: {manifest['target_column']}
Features: {len(feature_list)}
Created: {manifest['created_at']}

Files:
- processed_dataset.csv     : Cleaned and processed data
- feature_list.json         : Selected feature columns
- cleaning_config.json      : Applied cleaning transformations
- binning_config.json       : Binning rules and edges
- validation_report.json    : Data quality validation results
- feature_selection.json    : Feature selection scores
- domain_config.json        : Business domain settings
- studio_manifest.json      : Master config for Predictions Studio
"""
        zf.writestr("README.txt", readme)

    buffer.seek(0)
    return buffer.getvalue()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dict_to_json_bytes(data: dict) -> bytes:
    return json.dumps(data, indent=2, default=str).encode("utf-8")
