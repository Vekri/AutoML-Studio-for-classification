"""Streamlit Community Cloud entry point (free public hosting)."""

import importlib.util
from pathlib import Path

_app_path = Path(__file__).parent / "app.py"
_spec = importlib.util.spec_from_file_location("automl_main", _app_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
