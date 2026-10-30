from __future__ import annotations

import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from prediction_store import (
    load_prediction_history,
    record_forecasts,
    settle_forecasts,
)


class PredictionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "forecasts.sqlite3"
        self.forecast = {
            "Event ID": "event-1",
            "Date": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "Away Team": "Away",
            "Home Team": "Home",
            "Away Win Chance (%)": 40.0,
            "Home Win Chance (%)": 60.0,
            "Away Projected Score": 2.0,
            "Home Projected Score": 3.0,
            "Model Version": "test-model-v1",
            "State": "pre",
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_forecast_is_recorded_once_and_settled(self) -> None:
        first_record = record_forecasts(
            self.database_path,
            "NHL",
            [self.forecast],
            forecasted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        changed_forecast = {**self.forecast, "Home Win Chance (%)": 95.0}
        second_record = record_forecasts(
            self.database_path, "NHL", [changed_forecast]
        )
        settled = settle_forecasts(
            self.database_path,
            "NHL",
            [{
                "Event ID": "event-1",
                "State": "post",
                "Home Score": 4,
                "Away Score": 2,
            }],
        )
        history = load_prediction_history(self.database_path, "NHL")
        row = history.iloc[0]

        self.assertEqual(first_record, 1)
        self.assertEqual(second_record, 0)
        self.assertEqual(settled, 1)
        self.assertEqual(row["Home Win Probability"], 0.6)
        self.assertEqual(row["Model Version"], "test-model-v1")
        self.assertEqual(row["Actual Home Score"], 4)
        self.assertEqual(row["Home Outcome"], 1.0)

    def test_in_progress_forecasts_are_not_persisted(self) -> None:
        forecast = {**self.forecast, "State": "in"}

        inserted = record_forecasts(
            self.database_path, "NHL", [forecast]
        )
        history = load_prediction_history(self.database_path, "NHL")

        self.assertEqual(inserted, 0)
        self.assertTrue(history.empty)

    def test_existing_database_is_migrated_without_rewriting_history(self) -> None:
        import sqlite3

        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE forecasts (
                        sport TEXT NOT NULL,
                        event_id TEXT NOT NULL,
                        forecasted_at TEXT NOT NULL,
                        event_date TEXT NOT NULL,
                        away_team TEXT NOT NULL,
                        home_team TEXT NOT NULL,
                        away_probability REAL NOT NULL,
                        home_probability REAL NOT NULL,
                        away_projected_score REAL,
                        home_projected_score REAL,
                        actual_away_score REAL,
                        actual_home_score REAL,
                        home_outcome REAL,
                        PRIMARY KEY (sport, event_id)
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO forecasts (
                        sport, event_id, forecasted_at, event_date, away_team,
                        home_team, away_probability, home_probability
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "NHL",
                        "legacy-event",
                        "2026-01-01T00:00:00+00:00",
                        "2026-01-02T00:00:00+00:00",
                        "Away",
                        "Home",
                        0.4,
                        0.6,
                    ),
                )

        history = load_prediction_history(self.database_path, "NHL")

        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]["Model Version"], "legacy-unversioned")


if __name__ == "__main__":
    unittest.main()
