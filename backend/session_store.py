"""In-memory session store for AutoML Studio."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class StudioSession:
    session_id: str
    raw_df: pd.DataFrame
    processed_df: pd.DataFrame | None = None
    domain: str = "General Binary Classification"
    target: str | None = None
    keep_cols: list[str] = field(default_factory=list)
    drop_cols: list[str] = field(default_factory=list)
    cleaning_config: dict[str, Any] = field(default_factory=dict)
    binning_config: dict[str, Any] = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)
    feature_selection_result: list[dict[str, Any]] = field(default_factory=list)
    validation_report: dict[str, Any] = field(default_factory=dict)
    model_validation: dict[str, Any] = field(default_factory=dict)
    selected_algorithm: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    quality_score: dict[str, Any] = field(default_factory=dict)
    outlier_report: dict[str, Any] = field(default_factory=dict)
    encoding_config: dict[str, Any] = field(default_factory=dict)
    scaling_config: dict[str, Any] = field(default_factory=dict)
    feature_eng_config: dict[str, Any] = field(default_factory=dict)
    reduction_config: dict[str, Any] = field(default_factory=dict)
    tuning_result: dict[str, Any] = field(default_factory=dict)
    explanation: dict[str, Any] = field(default_factory=dict)
    insights: dict[str, Any] = field(default_factory=dict)
    report_html: str = ""
    filename: str = "dataset.csv"

    @property
    def df(self) -> pd.DataFrame:
        if isinstance(self.processed_df, pd.DataFrame):
            return self.processed_df
        return self.raw_df


_SESSIONS: dict[str, StudioSession] = {}


def create_session(df: pd.DataFrame, filename: str = "dataset.csv", domain: str = "General Binary Classification") -> StudioSession:
    sid = str(uuid.uuid4())
    session = StudioSession(session_id=sid, raw_df=df, filename=filename, domain=domain)
    _SESSIONS[sid] = session
    return session


def get_session(session_id: str) -> StudioSession:
    if session_id not in _SESSIONS:
        raise KeyError(f"Session not found: {session_id}")
    return _SESSIONS[session_id]


def delete_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)
