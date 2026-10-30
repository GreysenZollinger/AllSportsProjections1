"""Local SQLite storage for first-seen forecasts and their eventual outcomes."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def _initialize(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS forecasts (
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
            model_version TEXT NOT NULL DEFAULT 'legacy-unversioned',
            actual_away_score REAL,
            actual_home_score REAL,
            home_outcome REAL,
            PRIMARY KEY (sport, event_id)
        )
        """
    )
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(forecasts)")
    }
    if "model_version" not in columns:
        connection.execute(
            "ALTER TABLE forecasts ADD COLUMN model_version TEXT NOT NULL "
            "DEFAULT 'legacy-unversioned'"
        )


def record_forecasts(
    database_path: str | Path,
    sport: str,
    predictions: Sequence[Mapping[str, Any]],
    forecasted_at: datetime | None = None,
) -> int:
    """Insert a pregame forecast once; later reruns never rewrite the first one."""
    timestamp = forecasted_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    forecast_timestamp = timestamp.astimezone(timezone.utc).isoformat()
    rows = []
    for prediction in predictions:
        if prediction.get("State") != "pre":
            continue
        event_id = str(prediction.get("Event ID", "")).strip()
        if not event_id:
            raise ValueError("A forecast is missing its event identifier.")
        home_probability = float(prediction["Home Win Chance (%)"]) / 100
        away_probability = float(prediction["Away Win Chance (%)"]) / 100
        if not (
            0 <= home_probability <= 1
            and 0 <= away_probability <= 1
        ):
            raise ValueError(f"Forecast {event_id} has invalid probabilities.")
        rows.append((
            sport,
            event_id,
            forecast_timestamp,
            str(prediction["Date"]),
            str(prediction["Away Team"]),
            str(prediction["Home Team"]),
            away_probability,
            home_probability,
            prediction.get("Away Projected Score"),
            prediction.get("Home Projected Score"),
            str(prediction.get("Model Version") or "unversioned"),
        ))
    if not rows:
        return 0

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            _initialize(connection)
            cursor = connection.executemany(
                """
                INSERT INTO forecasts (
                    sport, event_id, forecasted_at, event_date, away_team,
                    home_team, away_probability, home_probability,
                    away_projected_score, home_projected_score, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (sport, event_id) DO NOTHING
                """,
                rows,
            )
            return cursor.rowcount


def settle_forecasts(
    database_path: str | Path,
    sport: str,
    completed_games: Sequence[Mapping[str, Any]],
) -> int:
    """Attach the final score to stored predictions, without revising outcomes."""
    rows = []
    for game in completed_games:
        if (
            game.get("State") != "post"
            or game.get("Home Score") is None
            or game.get("Away Score") is None
        ):
            continue
        home_score = float(game["Home Score"])
        away_score = float(game["Away Score"])
        outcome = (
            1.0 if home_score > away_score
            else 0.0 if home_score < away_score
            else 0.5
        )
        rows.append((away_score, home_score, outcome, sport, str(game["Event ID"])))
    if not rows:
        return 0

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            _initialize(connection)
            cursor = connection.executemany(
                """
                UPDATE forecasts
                SET actual_away_score = ?, actual_home_score = ?,
                    home_outcome = ?
                WHERE sport = ? AND event_id = ? AND home_outcome IS NULL
                """,
                rows,
            )
            return cursor.rowcount


def load_prediction_history(
    database_path: str | Path, sport: str, limit: int = 100
) -> pd.DataFrame:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("History limit must be a positive integer.")
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            _initialize(connection)
            rows = connection.execute(
                """
                SELECT event_id, forecasted_at, event_date, away_team, home_team,
                       away_probability, home_probability, away_projected_score,
                       home_projected_score, model_version, actual_away_score,
                       actual_home_score, home_outcome
                FROM forecasts
                WHERE sport = ?
                ORDER BY forecasted_at DESC, event_date DESC
                LIMIT ?
                """,
                (sport, limit),
            ).fetchall()
    return pd.DataFrame.from_records(rows, columns=[
        "Event ID",
        "Forecasted At (UTC)",
        "Event Date",
        "Away Team",
        "Home Team",
        "Away Win Probability",
        "Home Win Probability",
        "Away Projected Score",
        "Home Projected Score",
        "Model Version",
        "Actual Away Score",
        "Actual Home Score",
        "Home Outcome",
    ])
