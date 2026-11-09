from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import streamlit as st

import sports_forecasting as forecasting
from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def test_dashboard_renders_without_external_data(self) -> None:
        app_path = Path(__file__).parents[1] / "streamlit_app.py"
        st.cache_data.clear()

        with (
            patch("sports_forecasting.fetch_current_games", return_value=[]),
            patch("sports_forecasting.fetch_history", return_value=[]),
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
        rendered_tabs = [tab.label for tab in app.tabs]
        self.assertEqual(
            rendered_tabs,
            ["Overview", "Team ratings", "Game forecasts", "Accuracy & history"],
        )

    def test_dashboard_surfaces_source_failures(self) -> None:
        app_path = Path(__file__).parents[1] / "streamlit_app.py"
        st.cache_data.clear()

        with patch(
            "sports_forecasting.fetch_current_games",
            side_effect=forecasting.SportsDataError("offline"),
        ):
            app = AppTest.from_file(str(app_path)).run(timeout=60)

        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "Couldn't retrieve sports data" in item.value
                for item in app.error
            )
        )


if __name__ == "__main__":
    unittest.main()
