from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def test_dashboard_renders_without_external_data(self) -> None:
        class EmptyScoreboardResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"events": []}

        app_path = Path(__file__).parents[1] / "streamlit_app.py"
        st.cache_data.clear()

        with (
            patch("requests.get", return_value=EmptyScoreboardResponse()),
            patch("prediction_store.record_forecasts", return_value=0),
            patch("prediction_store.settle_forecasts", return_value=0),
            patch(
                "prediction_store.load_prediction_history",
                return_value=pd.DataFrame(),
            ),
        ):
            app = AppTest.from_file(str(app_path)).run(timeout=60)

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "🏆 AI Sports Analytics")

    def test_dashboard_surfaces_source_failures(self) -> None:
        app_path = Path(__file__).parents[1] / "streamlit_app.py"
        st.cache_data.clear()

        with patch(
            "requests.get",
            side_effect=requests.Timeout("offline"),
        ):
            app = AppTest.from_file(str(app_path)).run(timeout=60)

        self.assertFalse(app.exception)
        self.assertTrue(
            any("Couldn't retrieve sports data" in item.value for item in app.error)
        )


if __name__ == "__main__":
    unittest.main()
