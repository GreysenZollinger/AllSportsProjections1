from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import sports_forecasting as forecasting


def event(
    event_id: str = "1",
    *,
    state: str = "post",
    home_score: str | None = "3",
    away_score: str | None = "2",
    game_date: str = "2026-01-01T20:00:00Z",
) -> dict:
    return {
        "id": event_id,
        "date": game_date,
        "season": {"year": 2026, "type": 2, "slug": "regular-season"},
        "competitions": [{
            "competitors": [
                {
                    "homeAway": "away",
                    "team": {"id": "away-id", "displayName": "Away Team"},
                    "score": away_score,
                },
                {
                    "homeAway": "home",
                    "team": {"id": "home-id", "displayName": "Home Team"},
                    "score": home_score,
                },
            ]
        }],
        "status": {"type": {"state": state, "shortDetail": "Final"}},
    }


def game(
    index: int,
    home_score: float = 3,
    away_score: float = 2,
) -> dict:
    return {
        "Event ID": str(index),
        "Date": datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index),
        "Away Team": f"Team {index % 4}",
        "Away Score": away_score,
        "Home Team": f"Team {(index + 1) % 4}",
        "Home Score": home_score,
        "Status": "Final",
        "State": "post",
    }


class EventParsingTests(unittest.TestCase):
    def test_parses_roles_scores_and_normalizes_date(self) -> None:
        parsed = forecasting.parse_events([
            event(home_score="7", away_score="4", game_date="2026-01-01T20:00:00-05:00")
        ])[0]

        self.assertEqual(parsed["Home Team"], "Home Team")
        self.assertEqual(parsed["Away Team"], "Away Team")
        self.assertEqual(parsed["Home Team ID"], "home-id")
        self.assertEqual(parsed["Away Team ID"], "away-id")
        self.assertEqual(parsed["Season Type"], "regular-season")
        self.assertEqual(parsed["Home Score"], 7.0)
        self.assertEqual(parsed["Away Score"], 4.0)
        self.assertEqual(parsed["Date"].utcoffset(), timedelta(0))
        self.assertEqual(parsed["Date"].hour, 1)

    def test_unavailable_scores_remain_missing(self) -> None:
        parsed = forecasting.parse_events([
            event(state="pre", home_score=None, away_score="-")
        ])[0]

        self.assertIsNone(parsed["Home Score"])
        self.assertIsNone(parsed["Away Score"])
        self.assertEqual(parsed["State"], "pre")

    def test_duplicate_event_ids_are_removed(self) -> None:
        parsed = forecasting.parse_events([event(), event()])
        self.assertEqual(len(parsed), 1)

    def test_missing_home_away_roles_are_rejected(self) -> None:
        malformed = event()
        malformed["competitions"][0]["competitors"][0]["homeAway"] = "neutral"
        with self.assertRaises(forecasting.SportsDataError):
            forecasting.parse_events([malformed])

    def test_invalid_score_is_rejected(self) -> None:
        with self.assertRaises(forecasting.SportsDataError):
            forecasting.parse_events([event(home_score="not-a-score")])


class DataRetrievalTests(unittest.TestCase):
    def test_request_failures_are_reported(self) -> None:
        with patch.object(
            forecasting.requests,
            "get",
            side_effect=forecasting.requests.Timeout("timed out"),
        ):
            with self.assertRaisesRegex(forecasting.SportsDataError, "2026-01-01"):
                forecasting._fetch_events_for_date("NFL", date(2026, 1, 1))

    def test_bad_json_is_reported(self) -> None:
        class BadResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                raise ValueError("not JSON")

        with patch.object(forecasting.requests, "get", return_value=BadResponse()):
            with self.assertRaisesRegex(forecasting.SportsDataError, "invalid JSON"):
                forecasting._fetch_events_for_date("NFL", date(2026, 1, 1))

    def test_missing_events_field_is_reported(self) -> None:
        class IncompleteResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"message": "unexpected response"}

        with patch.object(
            forecasting.requests, "get", return_value=IncompleteResponse()
        ):
            with self.assertRaisesRegex(forecasting.SportsDataError, "events field"):
                forecasting._fetch_events_for_date("NFL", date(2026, 1, 1))

    def test_history_deduplicates_and_keeps_tied_finals(self) -> None:
        tied = game(1, home_score=2, away_score=2)
        scheduled = {**game(2), "State": "pre"}
        with patch.object(
            forecasting,
            "_fetch_events_for_date",
            return_value=[tied, scheduled, tied],
        ):
            history = forecasting.fetch_history(
                "NHL", 2, end_date=date(2026, 1, 2)
            )

        self.assertEqual([item["Event ID"] for item in history], ["1"])
        self.assertEqual(history[0]["Home Score"], 2)

    def test_history_window_bounds_are_validated(self) -> None:
        for days in (0, 366, True):
            with self.subTest(days=days), self.assertRaises(ValueError):
                forecasting.fetch_history("NBA", days, end_date=date(2026, 1, 1))


class ForecastModelTests(unittest.TestCase):
    def test_all_four_sports_produce_finite_forecasts(self) -> None:
        history = [
            game(index, home_score=3 + index % 4, away_score=2 + index % 3)
            for index in range(12)
        ]
        upcoming = {**game(13), "State": "pre"}

        for sport in forecasting.SPORTS:
            with self.subTest(sport=sport):
                ratings, _ = forecasting.build_ratings(sport, history)
                scoring_model = forecasting.build_scoring_averages(history)
                prediction = forecasting.make_predictions(
                    sport, [upcoming], ratings, scoring_model
                ).iloc[0]
                self.assertTrue(
                    0 <= prediction["Home Win Chance (%)"] <= 100
                )
                self.assertAlmostEqual(
                    prediction["Home Win Chance (%)"]
                    + prediction["Away Win Chance (%)"],
                    100,
                )

    def test_ratings_sort_history_and_preserve_ties(self) -> None:
        history = [
            game(2, home_score=2, away_score=2),
            game(1, home_score=4, away_score=1),
        ]
        ratings, table = forecasting.build_ratings("NHL", history)
        home_record = table.set_index("Team").loc["Team 2", "Record"]

        self.assertEqual(home_record, "1-0-1")
        self.assertEqual(
            set(ratings),
            {"name:team 1", "name:team 2", "name:team 3"},
        )

    def test_preseason_games_do_not_influence_model_ratings(self) -> None:
        preseason_game = {**game(1), "Season Type": "preseason"}
        regular_game = game(2)

        ratings, table = forecasting.build_ratings(
            "NBA", [preseason_game, regular_game]
        )
        regular_ratings, regular_table = forecasting.build_ratings(
            "NBA", [regular_game]
        )

        self.assertEqual(ratings, regular_ratings)
        self.assertEqual(
            table.sort_values("Team").to_dict("records"),
            regular_table.sort_values("Team").to_dict("records"),
        )

    def test_small_sample_ratings_are_shrunk_toward_neutral(self) -> None:
        _, table = forecasting.build_ratings(
            "NFL",
            [game(1, home_score=35, away_score=0)],
        )
        team_row = table.set_index("Team").loc["Team 2"]

        self.assertEqual(team_row["Games"], 1)
        self.assertLess(
            abs(team_row["Rating"] - 1500),
            abs(team_row["Raw Elo"] - 1500),
        )
        self.assertLess(team_row["Rating Prior Weight (%)"], 100)

    def test_same_day_rating_updates_are_independent_of_event_order(self) -> None:
        game_one = game(1, home_score=4, away_score=1)
        game_two = {
            **game(2, home_score=2, away_score=5),
            "Date": game_one["Date"],
        }

        _, first_order = forecasting.build_ratings(
            "NBA", [game_one, game_two]
        )
        _, second_order = forecasting.build_ratings(
            "NBA", [game_two, game_one]
        )

        first_order = first_order.sort_values("Team").reset_index(drop=True)
        second_order = second_order.sort_values("Team").reset_index(drop=True)
        self.assertEqual(
            first_order.to_dict("records"),
            second_order.to_dict("records"),
        )

    def test_stable_team_ids_survive_display_name_changes(self) -> None:
        old_name_game = {
            **game(1),
            "Home Team ID": "franchise-1",
            "Home Team": "Old Franchise Name",
        }
        new_name_game = {
            **game(2),
            "Away Team ID": "franchise-1",
            "Away Team": "Current Franchise Name",
        }

        ratings, table = forecasting.build_ratings(
            "NBA", [old_name_game, new_name_game]
        )

        self.assertIn("franchise-1", ratings)
        self.assertEqual(
            table.set_index("Team ID").loc["franchise-1", "Team"],
            "Current Franchise Name",
        )

    def test_inactive_rating_decays_by_configured_half_life(self) -> None:
        ratings = {"home-id": 1600.0, "away-id": 1400.0}
        games_played = {"home-id": 10, "away-id": 10}
        last_played = {
            "home-id": date(2025, 1, 1),
            "away-id": date(2025, 1, 1),
        }
        next_game = {
            **game(2, home_score=1, away_score=1),
            "Date": datetime(2025, 6, 30, tzinfo=timezone.utc),
            "Home Team ID": "home-id",
            "Away Team ID": "away-id",
        }
        expected_home = forecasting.expected_score(1550 + 65, 1450)
        expected_change = 20 * 0.7 * (0.5 - expected_home)

        forecasting._update_ratings_for_day(
            "NBA",
            ratings,
            games_played,
            last_played,
            [next_game],
        )

        self.assertAlmostEqual(ratings["home-id"], 1550 + expected_change)
        self.assertAlmostEqual(ratings["away-id"], 1450 - expected_change)

    def test_scoring_averages_shrink_toward_observed_league_average(self) -> None:
        model = forecasting.build_scoring_averages([
            game(1, home_score=10, away_score=0),
        ])

        self.assertEqual(model.league_average, 5)
        self.assertAlmostEqual(model.teams["Team 2"]["scored"], 5 + 5 / 6)
        self.assertIsNone(model.score_standard_deviation)

    def test_predictions_have_valid_probabilities_and_ranges(self) -> None:
        history = [
            game(index, home_score=3 + index % 4, away_score=2 + index % 3)
            for index in range(10)
        ]
        scoring_model = forecasting.build_scoring_averages(history)
        ratings, _ = forecasting.build_ratings("NBA", history)
        upcoming = {
            **game(11),
            "State": "pre",
            "Status": "Scheduled",
        }
        prediction = forecasting.make_predictions(
            "NBA", [upcoming], ratings, scoring_model
        ).iloc[0]

        self.assertAlmostEqual(
            prediction["Home Win Chance (%)"]
            + prediction["Away Win Chance (%)"],
            100,
        )
        self.assertEqual(prediction["Model Version"], forecasting.MODEL_VERSION)
        self.assertGreaterEqual(prediction["Home Projected Score"], 0)
        self.assertIn(" - ", prediction["Home Score Range (approx. 80%)"])

    def test_predictions_use_stable_team_ids_for_rating_lookups(self) -> None:
        upcoming = {
            **game(11),
            "State": "pre",
            "Home Team ID": "home-id",
            "Away Team ID": "away-id",
        }
        prediction = forecasting.make_predictions(
            "NBA",
            [upcoming],
            {"home-id": 2000, "away-id": 1000},
            forecasting.build_scoring_averages([]),
        ).iloc[0]

        self.assertGreater(prediction["Home Win Chance (%)"], 50)
        self.assertEqual(prediction["Home Rating"], 2000)

    def test_no_historical_score_data_is_not_fabricated(self) -> None:
        scoring_model = forecasting.build_scoring_averages([])
        upcoming = {**game(1), "State": "pre"}
        prediction = forecasting.make_predictions(
            "NFL", [upcoming], {}, scoring_model
        ).iloc[0]

        self.assertIsNone(prediction["Home Projected Score"])
        self.assertEqual(
            prediction["Home Score Range (approx. 80%)"], "Not enough data"
        )

    def test_calibration_requires_prior_sample_and_corrects_bias(self) -> None:
        examples = [
            {"raw_home_probability": 0.7, "home_outcome": 0.0}
            for _ in range(forecasting.CALIBRATION_MIN_GAMES)
        ]
        bias = forecasting.fit_calibration_bias(examples)

        self.assertLess(bias, 0)
        self.assertEqual(
            forecasting.fit_calibration_bias(examples[:-1]),
            0.0,
        )
        self.assertLess(forecasting._apply_calibration(0.7, bias), 0.7)

    def test_walk_forward_backtest_does_not_use_future_outcomes(self) -> None:
        history = [game(index) for index in range(35)]
        original = forecasting.walk_forward_backtest(
            "MLB", history, min_train_games=3
        )
        changed_history = [
            {**item, "Home Score": 100}
            if item["Event ID"] == "30"
            else item
            for item in history
        ]
        changed = forecasting.walk_forward_backtest(
            "MLB", changed_history, min_train_games=3
        )

        original_early = [
            item for item in original["predictions"]
            if item["Date"].date() < date(2026, 1, 31)
        ]
        changed_early = [
            item for item in changed["predictions"]
            if item["Date"].date() < date(2026, 1, 31)
        ]
        self.assertEqual(original_early, changed_early)
        self.assertGreater(original["metrics"]["games"], 0)
        self.assertIsNotNone(original["metrics"]["baseline_brier_score"])
        self.assertIsNotNone(original["metrics"]["score_mae"])

    def test_walk_forward_uses_prior_history_to_warm_evaluation_window(self) -> None:
        history = [game(index) for index in range(35)]
        evaluation_start = date(2026, 1, 21)
        original = forecasting.walk_forward_backtest(
            "NBA",
            history,
            min_train_games=3,
            evaluation_start=evaluation_start,
        )
        changed_warmup = [
            {**item, "Home Score": 100}
            if item["Event ID"] == "10"
            else item
            for item in history
        ]
        changed_future = [
            {**item, "Home Score": 100}
            if item["Event ID"] == "30"
            else item
            for item in history
        ]
        warmup_changed = forecasting.walk_forward_backtest(
            "NBA",
            changed_warmup,
            min_train_games=3,
            evaluation_start=evaluation_start,
        )
        future_changed = forecasting.walk_forward_backtest(
            "NBA",
            changed_future,
            min_train_games=3,
            evaluation_start=evaluation_start,
        )

        self.assertTrue(
            all(
                item["Date"].date() >= evaluation_start
                for item in original["predictions"]
            )
        )
        self.assertNotEqual(
            original["predictions"][0]["raw_home_probability"],
            warmup_changed["predictions"][0]["raw_home_probability"],
        )
        self.assertEqual(
            original["predictions"][:10],
            future_changed["predictions"][:10],
        )

    def test_walk_forward_reports_no_metrics_for_empty_evaluation_window(self) -> None:
        result = forecasting.walk_forward_backtest(
            "NFL",
            [],
            evaluation_start=date(2026, 1, 1),
        )

        self.assertEqual(result["metrics"]["games"], 0)
        self.assertIsNone(result["metrics"]["brier_score"])
        self.assertIsNone(result["metrics"]["calibration_error"])

    def test_same_day_results_do_not_change_that_days_forecasts(self) -> None:
        first_same_day = game(1)
        second_same_day = {
            **game(2),
            "Date": first_same_day["Date"],
        }
        history = [game(0), first_same_day, second_same_day]
        changed_history = [
            {**first_same_day, "Home Score": 100},
            game(0),
            second_same_day,
        ]
        original = forecasting.walk_forward_backtest(
            "NBA", history, min_train_games=1
        )
        changed = forecasting.walk_forward_backtest(
            "NBA", changed_history, min_train_games=1
        )

        def second_game_probability(result: dict) -> float:
            return next(
                item["raw_home_probability"]
                for item in result["predictions"]
                if item["Event ID"] == "2"
            )

        self.assertEqual(
            second_game_probability(original),
            second_game_probability(changed),
        )


if __name__ == "__main__":
    unittest.main()
