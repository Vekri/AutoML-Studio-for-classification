"""Smoke tests — verify all modules import and core functions run."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import PROBLEM_DOMAINS
from modules.binning import create_bins
from modules.cleaning import generate_cleaning_recommendations
from modules.data_loader import detect_binary_target
from modules.export import build_studio_manifest
from modules.feature_selection import correlation_feature_selection
from modules.validation import validate_dataset


def test_config_loads():
    assert "Banking" in PROBLEM_DOMAINS
    assert len(PROBLEM_DOMAINS) >= 5


def test_pipeline_on_sample_data():
    sample = ROOT / "sample_data" / "banking_loan_default.csv"
    assert sample.exists()

    df = pd.read_csv(sample)
    target = "default"

    assert detect_binary_target(df[target])["is_binary"]
    assert len(generate_cleaning_recommendations(df, target)) >= 0

    fs = correlation_feature_selection(df, target, threshold=0.05)
    assert not fs.empty

    report = validate_dataset(df, target, features=["age", "income", "credit_score"])
    assert report["passed"] >= 1

    manifest = build_studio_manifest(
        domain="Banking",
        target=target,
        features=["age", "income"],
        cleaning_config={},
        binning_config={},
        validation_report=report,
        feature_selection={"method": "correlation"},
    )
    assert manifest["ready_for_predictions_studio"] is True
    json.dumps(manifest)


def test_binning():
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name="test")
    binned, cfg = create_bins(series, method="equal_width", n_bins=3)
    assert len(cfg["edges"]) == 4
