"""Smoke test that the Streamlit app script executes without runtime errors.

Runs headlessly via Streamlit's AppTest (auth view only, no backend needed).
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "frontend" / "app.py"


@pytest.mark.slow
def test_frontend_auth_view_runs():
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
