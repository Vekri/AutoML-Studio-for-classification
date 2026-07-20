"""FastAPI backend for AutoML Studio (Hugging Face Spaces + React)."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import FEATURE_SELECTION_METHODS, PROBLEM_DOMAINS  # noqa: E402
from modules.binning import apply_binning_to_dataframe, create_bins, woe_iv_table  # noqa: E402
from modules.cleaning import apply_cleaning, auto_clean, generate_cleaning_recommendations  # noqa: E402
from modules.data_loader import (  # noqa: E402
    apply_column_selection,
    detect_binary_target,
    get_column_summary,
    suggest_target_column,
)
from modules.export import build_studio_manifest, export_zip_bundle  # noqa: E402
from modules.feature_selection import (  # noqa: E402
    get_selected_features,
    random_forest_importance,
    run_feature_selection,
)
from modules.validation import run_model_validation, validate_dataset  # noqa: E402
from modules.visualization import get_categorical_columns, get_numeric_columns  # noqa: E402

from backend.session_store import create_session, delete_session, get_session  # noqa: E402

app = FastAPI(title="AutoML Studio API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class DomainUpdate(BaseModel):
    domain: str


class ColumnSelection(BaseModel):
    target: str
    mode: str = Field(description="keep or drop")
    columns: list[str] = Field(default_factory=list)


class CleaningApply(BaseModel):
    mode: str = Field(description="auto or all")


class BinningRequest(BaseModel):
    columns: list[str]
    method: str = "equal_width"
    n_bins: int = 5


class FeatureSelectionRequest(BaseModel):
    method: str = "mutual_information"
    top_k: int = 10
    threshold: float = 0.1


class ValidationRequest(BaseModel):
    test_size: float = 0.2
    cv_folds: int = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return obj


def _preview(df: pd.DataFrame, n: int = 12) -> list[dict[str, Any]]:
    return _json_safe(df.head(n).where(pd.notnull(df.head(n)), None).to_dict(orient="records"))


def _session_payload(session) -> dict[str, Any]:
    df = session.df
    return {
        "session_id": session.session_id,
        "filename": session.filename,
        "domain": session.domain,
        "target": session.target,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "preview": _preview(df),
        "selected_features": session.selected_features,
        "has_cleaning": bool(session.cleaning_config),
        "has_binning": bool(session.binning_config),
    }


def _require_session(session_id: str):
    try:
        return get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Meta / health
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    return {"status": "ok", "app": "AutoML Studio", "version": "2.0.0"}


@app.get("/api/domains")
def domains():
    return {
        "domains": [
            {
                "name": name,
                "description": meta["description"],
                "common_targets": meta["common_targets"],
                "key_metrics": meta["key_metrics"],
            }
            for name, meta in PROBLEM_DOMAINS.items()
        ],
        "feature_selection_methods": FEATURE_SELECTION_METHODS,
    }


# ---------------------------------------------------------------------------
# Upload / sample
# ---------------------------------------------------------------------------


@app.post("/api/upload")
async def upload_csv(
    file: UploadFile = File(...),
    domain: str = Form("General Binary Classification"),
):
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty")

    if domain not in PROBLEM_DOMAINS:
        domain = "General Binary Classification"

    session = create_session(df, filename=file.filename or "dataset.csv", domain=domain)
    suggested = suggest_target_column(df, PROBLEM_DOMAINS[domain]["common_targets"])
    summary = get_column_summary(df)
    return {
        **_session_payload(session),
        "suggested_target": suggested,
        "column_summary": _json_safe(summary.to_dict(orient="records")),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


@app.post("/api/sample")
def load_sample(domain: str = "Banking"):
    path = ROOT / "sample_data" / "banking_loan_default.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample data not found")
    df = pd.read_csv(path)
    if domain not in PROBLEM_DOMAINS:
        domain = "Banking"
    session = create_session(df, filename="banking_loan_default.csv", domain=domain)
    session.target = "default"
    session.processed_df = df.copy()
    summary = get_column_summary(df)
    return {
        **_session_payload(session),
        "suggested_target": "default",
        "column_summary": _json_safe(summary.to_dict(orient="records")),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


@app.delete("/api/session/{session_id}")
def reset_session(session_id: str):
    delete_session(session_id)
    return {"ok": True}


@app.patch("/api/session/{session_id}/domain")
def update_domain(session_id: str, body: DomainUpdate):
    session = _require_session(session_id)
    if body.domain not in PROBLEM_DOMAINS:
        raise HTTPException(status_code=400, detail="Unknown domain")
    session.domain = body.domain
    return _session_payload(session)


# ---------------------------------------------------------------------------
# Target & columns
# ---------------------------------------------------------------------------


@app.get("/api/session/{session_id}/target-info")
def target_info(session_id: str, target: str):
    session = _require_session(session_id)
    df = session.raw_df
    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target}' not found")
    info = detect_binary_target(df[target])
    counts = df[target].value_counts(dropna=False)
    return {
        "info": _json_safe(info),
        "distribution": [{"class": str(k), "count": int(v)} for k, v in counts.items()],
    }


@app.post("/api/session/{session_id}/columns")
def select_columns(session_id: str, body: ColumnSelection):
    session = _require_session(session_id)
    df = session.raw_df
    if body.target not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid target")

    info = detect_binary_target(df[body.target])
    session.target = body.target

    if body.mode == "keep":
        session.keep_cols = body.columns
        session.drop_cols = []
        session.processed_df = apply_column_selection(df, body.target, keep_columns=body.columns)
    else:
        session.drop_cols = body.columns
        session.keep_cols = []
        session.processed_df = apply_column_selection(df, body.target, drop_columns=body.columns)

    return {
        **_session_payload(session),
        "target_info": _json_safe(info),
    }


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


@app.get("/api/session/{session_id}/cleaning")
def cleaning_recommendations(session_id: str):
    session = _require_session(session_id)
    if not isinstance(session.target, str):
        raise HTTPException(status_code=400, detail="Select a target first")
    df = session.df
    if session.target not in df.columns:
        raise HTTPException(status_code=400, detail="Target missing from dataset")
    recs = generate_cleaning_recommendations(df, session.target)
    return {"recommendations": _json_safe(recs), "config": session.cleaning_config}


@app.post("/api/session/{session_id}/cleaning/apply")
def apply_cleaning_endpoint(session_id: str, body: CleaningApply):
    session = _require_session(session_id)
    if not isinstance(session.target, str):
        raise HTTPException(status_code=400, detail="Select a target first")
    df = session.df
    target = session.target
    if body.mode == "auto":
        cleaned, config = auto_clean(df, target)
    else:
        recs = generate_cleaning_recommendations(df, target)
        actionable = [r for r in recs if r.get("action") not in ("fix_target", None)]
        cleaned, config = apply_cleaning(df, target, actionable)
    session.processed_df = cleaned
    session.cleaning_config = config
    return {**_session_payload(session), "cleaning_config": _json_safe(config)}


# ---------------------------------------------------------------------------
# Visualizations (chart data for React)
# ---------------------------------------------------------------------------


@app.get("/api/session/{session_id}/visualizations")
def visualizations(session_id: str):
    session = _require_session(session_id)
    if not isinstance(session.target, str):
        raise HTTPException(status_code=400, detail="Select a target first")
    df = session.df
    target = session.target
    if target not in df.columns:
        raise HTTPException(status_code=400, detail="Target missing")

    target_dist = [
        {"class": str(k), "count": int(v)}
        for k, v in df[target].value_counts(dropna=False).items()
    ]
    missing = [
        {"column": c, "missing_count": int(v)}
        for c, v in df.isna().sum().items()
        if int(v) > 0
    ]
    numeric = get_numeric_columns(df, exclude=[target])
    categorical = get_categorical_columns(df, exclude=[target])

    corr = []
    num_df = df.select_dtypes(include=["number"])
    if target in num_df.columns and len(num_df.columns) > 1:
        series = num_df.corr()[target].drop(target).sort_values(key=abs, ascending=False)
        corr = [{"feature": str(i), "correlation": float(v)} for i, v in series.items()]

    return {
        "target_distribution": target_dist,
        "missing": missing,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "feature_correlations": corr,
    }


@app.get("/api/session/{session_id}/visualizations/numeric/{column}")
def viz_numeric(session_id: str, column: str):
    session = _require_session(session_id)
    df = session.df
    target = session.target
    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Unknown column")
    rows = []
    for _, row in df[[column] + ([target] if target and target in df.columns else [])].dropna().iterrows():
        item = {"value": float(row[column]) if pd.notna(row[column]) else None}
        if target and target in df.columns:
            item["target"] = str(row[target])
        rows.append(item)
    return {"column": column, "points": rows[:2000]}


@app.get("/api/session/{session_id}/visualizations/categorical/{column}")
def viz_categorical(session_id: str, column: str):
    session = _require_session(session_id)
    df = session.df
    target = session.target
    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Unknown column")
    if target and target in df.columns:
        ct = pd.crosstab(df[column], df[target])
        records = []
        for idx in ct.index:
            for col in ct.columns:
                records.append(
                    {
                        "category": str(idx),
                        "target": str(col),
                        "count": int(ct.loc[idx, col]),
                    }
                )
        return {"column": column, "grouped": records}
    counts = df[column].value_counts().head(20)
    return {
        "column": column,
        "grouped": [{"category": str(k), "count": int(v)} for k, v in counts.items()],
    }


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


@app.post("/api/session/{session_id}/binning")
def binning(session_id: str, body: BinningRequest):
    session = _require_session(session_id)
    if not isinstance(session.target, str):
        raise HTTPException(status_code=400, detail="Select a target first")
    df = session.df
    binned_df, config = apply_binning_to_dataframe(df, body.columns, body.method, body.n_bins)
    session.processed_df = binned_df
    session.binning_config = config

    woe_tables = {}
    for col in body.columns:
        bcol = f"{col}_binned"
        try:
            if bcol in binned_df.columns:
                woe = woe_iv_table(binned_df, bcol, session.target)
            else:
                temp, _ = create_bins(df[col], method=body.method, n_bins=body.n_bins)
                tmp = df[[session.target]].copy()
                tmp["temp_binned"] = temp
                woe = woe_iv_table(tmp, "temp_binned", session.target)
            if isinstance(woe, pd.DataFrame) and not woe.empty:
                woe_tables[col] = _json_safe(woe.to_dict(orient="records"))
        except Exception:
            continue

    return {
        **_session_payload(session),
        "binning_config": _json_safe(config),
        "woe_tables": woe_tables,
    }


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


@app.post("/api/session/{session_id}/features")
def feature_selection(session_id: str, body: FeatureSelectionRequest):
    session = _require_session(session_id)
    if not isinstance(session.target, str):
        raise HTTPException(status_code=400, detail="Select a target first")
    df = session.df
    try:
        result = run_feature_selection(df, session.target, body.method, body.top_k, body.threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result is None or result.empty:
        raise HTTPException(status_code=400, detail="No features scored")

    selected = get_selected_features(result)
    session.selected_features = selected
    session.feature_selection_result = _json_safe(result.to_dict(orient="records"))
    return {
        **_session_payload(session),
        "result": session.feature_selection_result,
        "selected_features": selected,
    }


@app.post("/api/session/{session_id}/features/rf-importance")
def rf_importance(session_id: str):
    session = _require_session(session_id)
    if not isinstance(session.target, str):
        raise HTTPException(status_code=400, detail="Select a target first")
    try:
        result = random_forest_importance(session.df, session.target)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": _json_safe(result.to_dict(orient="records"))}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@app.post("/api/session/{session_id}/validate/data")
def validate_data(session_id: str):
    session = _require_session(session_id)
    if not isinstance(session.target, str):
        raise HTTPException(status_code=400, detail="Select a target first")
    features = session.selected_features or [
        c
        for c in session.df.columns
        if c != session.target and not str(c).endswith("_binned") and "id" not in str(c).lower()
    ]
    report = validate_dataset(session.df, session.target, features)
    session.validation_report = _json_safe(report)
    return {"report": session.validation_report, "features_used": features}


@app.post("/api/session/{session_id}/validate/models")
def validate_models(session_id: str, body: ValidationRequest):
    session = _require_session(session_id)
    if not isinstance(session.target, str):
        raise HTTPException(status_code=400, detail="Select a target first")
    features = session.selected_features or [
        c
        for c in session.df.columns
        if c != session.target and not str(c).endswith("_binned") and "id" not in str(c).lower()
    ]
    try:
        result = run_model_validation(
            session.df, session.target, features, body.test_size, body.cv_folds
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.model_validation = _json_safe(result)
    return {"result": session.model_validation}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@app.get("/api/session/{session_id}/export.zip")
def export_zip(session_id: str):
    session = _require_session(session_id)
    if not isinstance(session.target, str):
        raise HTTPException(status_code=400, detail="Select a target first")
    df = session.df
    features = session.selected_features or [
        c
        for c in df.columns
        if c != session.target and not str(c).endswith("_binned") and "id" not in str(c).lower()
    ]
    manifest = build_studio_manifest(
        domain=session.domain,
        target=session.target,
        features=features,
        cleaning_config=session.cleaning_config,
        binning_config=session.binning_config,
        validation_report=session.validation_report,
        feature_selection={
            "selected_features": features,
            "result": session.feature_selection_result,
        },
    )
    data = export_zip_bundle(
        df=df,
        manifest=manifest,
        feature_list=features,
        cleaning_config=session.cleaning_config,
        binning_config=session.binning_config,
        validation_report=session.validation_report,
        feature_selection_results=manifest["feature_selection"],
        domain=session.domain,
    )
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=automl_studio_export.zip"},
    )


@app.get("/api/session/{session_id}/manifest")
def export_manifest(session_id: str):
    session = _require_session(session_id)
    if not isinstance(session.target, str):
        raise HTTPException(status_code=400, detail="Select a target first")
    features = session.selected_features or [
        c
        for c in session.df.columns
        if c != session.target and not str(c).endswith("_binned") and "id" not in str(c).lower()
    ]
    manifest = build_studio_manifest(
        domain=session.domain,
        target=session.target,
        features=features,
        cleaning_config=session.cleaning_config,
        binning_config=session.binning_config,
        validation_report=session.validation_report,
        feature_selection={
            "selected_features": features,
            "result": session.feature_selection_result,
        },
    )
    return _json_safe(manifest)


# ---------------------------------------------------------------------------
# Serve React build (Hugging Face / production)
# ---------------------------------------------------------------------------

STATIC_DIR = ROOT / "frontend" / "dist"
ASSETS_DIR = STATIC_DIR / "assets"


@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index, media_type="text/html")
    return {
        "status": "ok",
        "message": "AutoML Studio API is running. Frontend not built.",
        "docs": "/docs",
        "health": "/api/health",
    }


if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    """SPA fallback — serve index.html for client-side routes."""
    if full_path.startswith("api/") or full_path == "docs" or full_path.startswith("docs/"):
        raise HTTPException(status_code=404, detail="Not found")
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index, media_type="text/html")
    raise HTTPException(status_code=404, detail="Frontend not built. Run: cd frontend && npm run build")
