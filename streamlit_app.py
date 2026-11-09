from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

import sports_forecasting as forecasting
from prediction_store import (
    load_prediction_history,
    record_forecasts,
    settle_forecasts,
)

SPORTS = tuple(forecasting.SPORTS)
DEFAULT_HISTORY_DAYS = 180
DATABASE_PATH = Path(
    os.getenv(
        "PREDICTION_DB_PATH",
        str(Path(__file__).with_name("prediction_history.sqlite3")),
    )
)

st.set_page_config(
    page_title="AI Sports Analytics",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=300, show_spinner=False)
def load_current_games(sport: str) -> list[dict]:
    return forecasting.fetch_current_games(sport)


@st.cache_data(ttl=1800, show_spinner=False)
def load_completed_history(sport: str, days: int) -> list[dict]:
    return forecasting.fetch_history(sport, days)


def format_number(value, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def build_tracking_summary(history: pd.DataFrame) -> dict:
    if history.empty:
        return {
            "saved": 0,
            "settled": 0,
            "accuracy": None,
            "brier": None,
            "score_mae": None,
        }

    settled = history.dropna(subset=["Home Outcome"]).copy()
    if settled.empty:
        return {
            "saved": len(history),
            "settled": 0,
            "accuracy": None,
            "brier": None,
            "score_mae": None,
        }

    decisive = settled[settled["Home Outcome"].isin([0.0, 1.0])]
    accuracy = None
    if not decisive.empty:
        accuracy = (
            (
                (decisive["Home Win Probability"] >= 0.5)
                == (decisive["Home Outcome"] == 1.0)
            ).mean()
            * 100
        )

    brier = (
        (settled["Home Win Probability"] - settled["Home Outcome"]) ** 2
    ).mean()

    score_errors = []
    for _, row in settled.iterrows():
        for projected_key, actual_key in (
            ("Home Projected Score", "Actual Home Score"),
            ("Away Projected Score", "Actual Away Score"),
        ):
            projected = row[projected_key]
            actual = row[actual_key]
            if pd.notna(projected) and pd.notna(actual):
                score_errors.append(abs(float(projected) - float(actual)))

    return {
        "saved": len(history),
        "settled": len(settled),
        "accuracy": accuracy,
        "brier": brier,
        "score_mae": (
            sum(score_errors) / len(score_errors) if score_errors else None
        ),
    }


def show_metric(label: str, value, suffix: str = "") -> None:
    if value is None or pd.isna(value):
        st.metric(label, "Not enough data")
    else:
        st.metric(label, f"{float(value):.1f}{suffix}")


st.title("🏆 AI Sports Analytics")
st.write(
    "Four-sport team ratings, game estimates, and historical accuracy tracking."
)

with st.sidebar:
    st.header("Dashboard controls")
    sport = st.selectbox("Sport", SPORTS, index=0)
    history_days = st.select_slider(
        "Historical data window",
        options=[60, 90, 120, 150, 180],
        value=DEFAULT_HISTORY_DAYS,
        format_func=lambda value: f"{value} days",
    )
    st.caption("Data source: ESPN's public scoreboard.")
    refresh_clicked = st.button("Refresh sports data", use_container_width=True)
    if refresh_clicked:
        load_current_games.clear()
        load_completed_history.clear()
        st.cache_data.clear()

if refresh_clicked:
    st.rerun()

try:
    with st.spinner(f"Loading {sport} data and building forecasts..."):
        current_games = load_current_games(sport)
        history = load_completed_history(sport, history_days)
        ratings, rating_table = forecasting.build_ratings(sport, history)
        scoring_model = forecasting.build_scoring_averages(history)
        predictions = forecasting.make_predictions(
            sport, current_games, ratings, scoring_model
        )

        # Save only the first pregame forecast for each event. Later reruns
        # cannot overwrite the original forecast used for accuracy tracking.
        saved_count = record_forecasts(
            DATABASE_PATH, sport, predictions.to_dict("records")
        )
        settled_count = settle_forecasts(DATABASE_PATH, sport, history)
        tracked_history = load_prediction_history(DATABASE_PATH, sport, limit=500)

    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    completed_count = len(history)
    upcoming_count = sum(game.get("State") == "pre" for game in current_games)
    live_count = sum(game.get("State") == "in" for game in current_games)

    top = st.columns(4)
    top[0].metric("Completed games loaded", completed_count)
    top[1].metric("Teams rated", len(rating_table))
    top[2].metric("Upcoming games", upcoming_count)
    top[3].metric("In progress", live_count)
    st.caption(
        f"Selected sport: {sport} · Historical window: {history_days} days · "
        f"Last dashboard load: {retrieved_at}"
    )

    overview_tab, ratings_tab, forecasts_tab, accuracy_tab = st.tabs(
        ["Overview", "Team ratings", "Game forecasts", "Accuracy & history"]
    )

    with overview_tab:
        st.subheader(f"{sport} overview")
        if current_games:
            scoreboard = pd.DataFrame(current_games)
            scoreboard["Date"] = pd.to_datetime(
                scoreboard["Date"], utc=True, errors="coerce"
            )
            st.dataframe(
                scoreboard[
                    [
                        "Date", "Away Team", "Away Score", "Home Team",
                        "Home Score", "Status", "State",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("ESPN did not return games in the current eight-day window.")

        st.caption(
            "The scoreboard refreshes from ESPN when its five-minute cache expires "
            "or when you press Refresh sports data."
        )

        st.divider()
        st.subheader("All four sports")
        st.caption(
            "Load the current eight-day scoreboard window for NFL, NBA, MLB, and NHL."
        )
        if st.button("Load all four sports scoreboards", key="load_all_sports"):
            all_sports_rows = []
            failures = []
            progress = st.progress(0, text="Loading all four sports...")
            for index, comparison_sport in enumerate(SPORTS):
                try:
                    sport_games = load_current_games(comparison_sport)
                    for game in sport_games:
                        all_sports_rows.append({
                            "Sport": comparison_sport,
                            "Date": game["Date"],
                            "Away Team": game["Away Team"],
                            "Away Score": game["Away Score"],
                            "Home Team": game["Home Team"],
                            "Home Score": game["Home Score"],
                            "Status": game["Status"],
                            "State": game["State"],
                        })
                except forecasting.SportsDataError as scoreboard_error:
                    failures.append(
                        f"{comparison_sport}: {scoreboard_error}"
                    )
                progress.progress(
                    (index + 1) / len(SPORTS),
                    text=f"Loaded {index + 1} of {len(SPORTS)} sports",
                )
            progress.empty()
            if all_sports_rows:
                all_sports_table = pd.DataFrame(all_sports_rows)
                all_sports_table["Date"] = pd.to_datetime(
                    all_sports_table["Date"], utc=True, errors="coerce"
                )
                all_sports_table = all_sports_table.sort_values(
                    ["Sport", "Date"]
                )
                st.dataframe(
                    all_sports_table,
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download all-sports scoreboard",
                    all_sports_table.to_csv(index=False),
                    file_name="all_sports_scoreboard.csv",
                    mime="text/csv",
                    key="download_all_sports_scoreboard",
                )
            else:
                st.warning("No scoreboard data was returned for any sport.")
            for failure in failures:
                st.warning(failure)

    with ratings_tab:
        st.subheader("Team performance ratings")
        if rating_table.empty:
            st.info(
                "There are no completed games in this data window yet. "
                "Try a longer window or refresh later."
            )
        else:
            st.dataframe(
                rating_table,
                use_container_width=True,
                hide_index=True,
            )
            chart_data = rating_table.set_index("Team")[["Rating"]].sort_values(
                "Rating", ascending=True
            )
            st.bar_chart(chart_data)
            st.download_button(
                "Download team ratings",
                rating_table.to_csv(index=False),
                file_name=f"{sport.lower()}_team_ratings.csv",
                mime="text/csv",
            )

    with forecasts_tab:
        st.subheader("Upcoming game estimates")
        st.caption(
            "These are pregame statistical estimates from historical team results. "
            "They are not live in-game forecasts and are not guaranteed outcomes."
        )
        pregame = (
            predictions[predictions["State"] == "pre"].copy()
            if not predictions.empty
            else pd.DataFrame()
        )
        if pregame.empty:
            st.info("No upcoming pregame matchups were returned for this sport.")
        else:
            columns = [
                "Date", "Away Team", "Away Win Chance (%)",
                "Away Projected Score", "Home Team", "Home Win Chance (%)",
                "Home Projected Score", "Projected Game Total", "Status",
            ]
            available_columns = [name for name in columns if name in pregame.columns]
            display = pregame[available_columns].copy()
            display["Date"] = pd.to_datetime(
                display["Date"], utc=True, errors="coerce"
            )
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download game estimates",
                pregame.to_csv(index=False),
                file_name=f"{sport.lower()}_game_estimates.csv",
                mime="text/csv",
            )

        st.caption(
            "Score ranges and win probabilities are model estimates. The model "
            "does not currently include confirmed injuries, starting lineups, "
            "weather, or all sport-specific advanced statistics."
        )

    with accuracy_tab:
        st.subheader("Historical walk-forward evaluation")
        st.write(
            "Each historical game is evaluated using information available before "
            "that game's date. Same-day results are added only after that day's "
            "forecasts are made."
        )
        backtest = forecasting.walk_forward_backtest(
            sport, history, min_train_games=20
        )
        metrics = backtest["metrics"]
        if metrics["games"] == 0:
            st.info(
                "There are not enough prior completed games for a reliable "
                "walk-forward evaluation yet."
            )
        else:
            metric_columns = st.columns(4)
            metric_columns[0].metric("Games evaluated", metrics["games"])
            show_metric("Winner accuracy", metrics["accuracy"] * 100 if metrics["accuracy"] is not None else None, "%")
            show_metric("Brier score", metrics["brier_score"])
            show_metric("Baseline Brier", metrics["baseline_brier_score"])

            score_columns = st.columns(4)
            score_columns[0].metric(
                "Games with paired score data", metrics.get("score_games", 0)
            )
            show_metric("Score MAE", metrics["score_mae"])
            show_metric("Baseline score MAE", metrics["baseline_score_mae"])
            show_metric("Log loss", metrics["log_loss"])

            st.caption(
                "Lower Brier score, log loss, and score MAE are better. "
                "Score MAE is the average absolute error per team score, in "
                "points, runs, or goals. Baselines are evaluated on matching "
                "observations. A backtest is not proof of future accuracy."
            )
            if backtest["predictions"]:
                detail = pd.DataFrame(backtest["predictions"])
                st.dataframe(
                    detail[
                        [
                            "Date", "Away Team", "Home Team",
                            "home_probability", "home_outcome",
                            "home_projected_score", "actual_home_score",
                            "away_projected_score", "actual_away_score",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download walk-forward evaluation",
                    detail.to_csv(index=False),
                    file_name=f"{sport.lower()}_walk_forward_evaluation.csv",
                    mime="text/csv",
                )

        st.divider()
        st.subheader("Saved forecast performance")
        tracking = build_tracking_summary(tracked_history)
        tracked_columns = st.columns(4)
        tracked_columns[0].metric("First forecasts saved", tracking["saved"])
        tracked_columns[1].metric("Final results attached", tracking["settled"])
        show_metric(
            "Saved-forecast accuracy",
            tracking["accuracy"],
            "%",
        )
        show_metric("Saved-forecast Brier", tracking["brier"])
        show_metric("Saved-forecast score MAE", tracking["score_mae"])
        st.caption(
            f"This app added {saved_count} new first forecasts and attached "
            f"{settled_count} new final results during this load. Saved forecasts "
            "are tracked separately from historical backtests."
        )
        if not tracked_history.empty:
            st.dataframe(
                tracked_history,
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download saved forecast history",
                tracked_history.to_csv(index=False),
                file_name=f"{sport.lower()}_saved_forecast_history.csv",
                mime="text/csv",
            )

        st.divider()
        st.subheader("Compare all four sports")
        st.caption(
            "This optional comparison loads the same historical window for all "
            "four leagues and evaluates each with the same walk-forward rules."
        )
        if st.button("Run four-sport accuracy comparison", key="run_four_sport_comparison"):
            comparison_rows = []
            progress = st.progress(0, text="Preparing four-sport comparison...")
            for index, comparison_sport in enumerate(SPORTS):
                try:
                    comparison_history = (
                        history
                        if comparison_sport == sport
                        else load_completed_history(comparison_sport, history_days)
                    )
                    comparison = forecasting.walk_forward_backtest(
                        comparison_sport,
                        comparison_history,
                        min_train_games=20,
                    )
                    comparison_metrics = comparison["metrics"]
                    comparison_rows.append({
                        "Sport": comparison_sport,
                        "Completed games": len(comparison_history),
                        "Games evaluated": comparison_metrics["games"],
                        "Winner accuracy (%)": (
                            round(comparison_metrics["accuracy"] * 100, 1)
                            if comparison_metrics["accuracy"] is not None else None
                        ),
                        "Brier score": comparison_metrics["brier_score"],
                        "Baseline Brier": comparison_metrics["baseline_brier_score"],
                        "Score MAE": comparison_metrics["score_mae"],
                        "Baseline score MAE": comparison_metrics["baseline_score_mae"],
                    })
                except forecasting.SportsDataError as comparison_error:
                    comparison_rows.append({
                        "Sport": comparison_sport,
                        "Completed games": None,
                        "Games evaluated": 0,
                        "Winner accuracy (%)": None,
                        "Brier score": None,
                        "Baseline Brier": None,
                        "Score MAE": None,
                        "Baseline score MAE": None,
                    })
                    st.warning(
                        f"{comparison_sport} data was unavailable: {comparison_error}"
                    )
                progress.progress(
                    (index + 1) / len(SPORTS),
                    text=f"Compared {index + 1} of {len(SPORTS)} sports",
                )
            progress.empty()
            comparison_table = pd.DataFrame(comparison_rows)
            st.dataframe(
                comparison_table,
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download four-sport accuracy comparison",
                comparison_table.to_csv(index=False),
                file_name="four_sport_accuracy_comparison.csv",
                mime="text/csv",
                key="download_four_sport_comparison",
            )

    st.divider()
    st.caption(
        "Prototype sports analytics. ESPN is an unofficial public data source "
        "with no service guarantee. Forecast quality depends on available "
        "historical data; player availability, lineups, weather, and other "
        "advanced inputs are not yet included. No result is guaranteed."
    )

except forecasting.SportsDataError as error:
    st.error("Couldn't retrieve sports data or validate the ESPN response.")
    st.warning(str(error))
    st.info("Try Refresh sports data again later.")
except Exception as error:
    st.error("The dashboard couldn't finish loading. Try Refresh sports data.")
    st.caption(f"Technical detail: {type(error).__name__}: {error}")
