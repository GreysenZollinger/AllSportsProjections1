
import math
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="AI Sports Analytics",
    page_icon="🏆",
    layout="wide",
)

SPORTS = {
    "NFL": {"category": "football", "league": "nfl", "home_adv": 55, "k": 24},
    "NBA": {"category": "basketball", "league": "nba", "home_adv": 65, "k": 20},
    "MLB": {"category": "baseball", "league": "mlb", "home_adv": 45, "k": 18},
    "NHL": {"category": "hockey", "league": "nhl", "home_adv": 50, "k": 20},
}

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"


def scoreboard_url(sport):
    info = SPORTS[sport]
    return f"{BASE_URL}/{info['category']}/{info['league']}/scoreboard"


def parse_events(events):
    rows = []

    for event in events:
        competitions = event.get("competitions", [])
        if not competitions:
            continue

        competition = competitions[0]
        competitors = competition.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = next(
            (c for c in competitors if c.get("homeAway") == "home"),
            competitors[0],
        )
        away = next(
            (c for c in competitors if c.get("homeAway") == "away"),
            competitors[1],
        )

        status = event.get("status", {}).get("type", {})
        home_team = home.get("team", {})
        away_team = away.get("team", {})

        try:
            home_score = float(home.get("score", 0) or 0)
            away_score = float(away.get("score", 0) or 0)
        except (ValueError, TypeError):
            home_score = 0
            away_score = 0

        rows.append({
            "Event ID": event.get("id", ""),
            "Date": event.get("date", ""),
            "Away Team": away_team.get("displayName", "Away"),
            "Away Abbreviation": away_team.get("abbreviation", ""),
            "Away Score": away_score,
            "Home Team": home_team.get("displayName", "Home"),
            "Home Abbreviation": home_team.get("abbreviation", ""),
            "Home Score": home_score,
            "Status": status.get(
                "shortDetail", status.get("description", "Scheduled")
            ),
            "State": status.get("state", "pre"),
        })

    return rows


@st.cache_data(ttl=300, show_spinner=False)
def get_current_games(sport):
    response = requests.get(scoreboard_url(sport), timeout=20)
    response.raise_for_status()
    return parse_events(response.json().get("events", []))


@st.cache_data(ttl=1800, show_spinner=False)
def get_history(sport, days):
    """Fetch history one calendar day at a time."""
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    rows_by_event = {}
    current_date = start_date

    while current_date <= end_date:
        response = requests.get(
            scoreboard_url(sport),
            params={
                "dates": current_date.strftime("%Y%m%d"),
                "limit": 100,
            },
            timeout=30,
        )
        response.raise_for_status()

        for row in parse_events(response.json().get("events", [])):
            event_id = row.get("Event ID")
            key = str(event_id) if event_id else (
                f"{row['Date']}|{row['Away Team']}|{row['Home Team']}"
            )
            rows_by_event[key] = row

        current_date += timedelta(days=1)

    # Only completed games with a decisive score can be used for ratings.
    return [
        row for row in rows_by_event.values()
        if row["State"] == "post"
        and row["Home Team"] != row["Away Team"]
        and row["Home Score"] != row["Away Score"]
    ]

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def build_ratings(sport, history):
    info = SPORTS[sport]
    ratings = {}
    records = {}
    recent_results = {}
    games_used = 0

    def ensure_team(team):
        if team not in ratings:
            ratings[team] = 1500.0
            records[team] = {"W": 0, "L": 0, "T": 0, "Games": 0}
            recent_results[team] = []

    history = sorted(history, key=lambda x: x["Date"])

    for game in history:
        home = game["Home Team"]
        away = game["Away Team"]
        hs = game["Home Score"]
        aws = game["Away Score"]

        ensure_team(home)
        ensure_team(away)

        home_rating = ratings[home]
        away_rating = ratings[away]

        home_expected = expected_score(
            home_rating + info["home_adv"], away_rating
        )

        if hs > aws:
            home_result = 1.0
            records[home]["W"] += 1
            records[away]["L"] += 1
            recent_results[home].append(1)
            recent_results[away].append(0)
        else:
            home_result = 0.0
            records[home]["L"] += 1
            records[away]["W"] += 1
            recent_results[home].append(0)
            recent_results[away].append(1)

        records[home]["Games"] += 1
        records[away]["Games"] += 1

        # Small margin-of-victory adjustment, capped to limit blowout effects.
        margin = abs(hs - aws)
        margin_factor = min(1.5, math.log1p(margin) / 2.5 + 0.7)
        change = info["k"] * margin_factor * (home_result - home_expected)

        ratings[home] += change
        ratings[away] -= change
        games_used += 1

    rows = []
    for team, rating in ratings.items():
        record = records[team]
        recent = recent_results[team][-5:]
        rows.append({
            "Team": team,
            "Rating": round(rating),
            "Record (window)": (
                f"{record['W']}-{record['L']}-{record['T']}"
            ),
            "Games": record["Games"],
            "Last 5 wins": sum(recent),
            "Last 5 games": len(recent),
            "Win rate": (
                round(record["W"] / record["Games"] * 100, 1)
                if record["Games"] else 0
            ),
        })

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("Rating", ascending=False).reset_index(drop=True)

    return ratings, table, games_used



def backtest_model(sport, history):
    """Walk-forward evaluation with an initial warm-up period and simple baseline."""
    info = SPORTS[sport]
    ratings = {}
    predictions = []
    ordered_history = sorted(history, key=lambda game: game["Date"])

    if len(ordered_history) < 2:
        return pd.DataFrame(), {
            "training_games": len(ordered_history),
            "games": 0,
            "accuracy": None,
            "baseline_accuracy": None,
            "brier": None,
        }

    # Do not grade the cold-start games: use them to establish initial team ratings.
    warmup_games = max(1, int(len(ordered_history) * 0.30))

    for index, game in enumerate(ordered_history):
        home = game["Home Team"]
        away = game["Away Team"]
        home_score = game["Home Score"]
        away_score = game["Away Score"]

        ratings.setdefault(home, 1500.0)
        ratings.setdefault(away, 1500.0)

        # Make and record predictions only after warm-up, before updating on this game.
        home_probability = expected_score(
            ratings[home] + info["home_adv"], ratings[away]
        )
        actual_home_win = 1 if home_score > away_score else 0
        if index >= warmup_games:
            predicted_home_win = 1 if home_probability >= 0.5 else 0
            predictions.append({
                "Date": game["Date"],
                "Away Team": away,
                "Home Team": home,
                "Home Win Probability (%)": round(home_probability * 100, 1),
                "Actual Home Win": actual_home_win,
                "Correct Pick": predicted_home_win == actual_home_win,
                "Home-Only Baseline Correct": actual_home_win == 1,
                "Brier Contribution": (home_probability - actual_home_win) ** 2,
            })

        # Update ratings after this game's prediction opportunity.
        margin = abs(home_score - away_score)
        margin_factor = min(1.5, math.log1p(margin) / 2.5 + 0.7)
        change = info["k"] * margin_factor * (actual_home_win - home_probability)
        ratings[home] += change
        ratings[away] -= change

    results = pd.DataFrame(predictions)
    if results.empty:
        return results, {
            "training_games": warmup_games,
            "games": 0,
            "accuracy": None,
            "baseline_accuracy": None,
            "brier": None,
        }

    metrics = {
        "training_games": warmup_games,
        "games": len(results),
        "accuracy": round(results["Correct Pick"].mean() * 100, 1),
        "baseline_accuracy": round(results["Home-Only Baseline Correct"].mean() * 100, 1),
        "brier": round(results["Brier Contribution"].mean(), 4),
    }
    return results, metrics



def _project_scores(home, away, team_scored, team_allowed, league_home_scores, league_away_scores):
    """Estimate team scores using only games available before the forecast."""
    all_scores = league_home_scores + league_away_scores
    league_average = sum(all_scores) / len(all_scores) if all_scores else 1.0
    league_home_average = (
        sum(league_home_scores) / len(league_home_scores)
        if league_home_scores else league_average
    )
    league_away_average = (
        sum(league_away_scores) / len(league_away_scores)
        if league_away_scores else league_average
    )
    home_edge = (league_home_average - league_away_average) / 2

    def blended_average(values):
        # Shrink small team samples toward the league average.
        prior_weight = 5
        return (sum(values) + league_average * prior_weight) / (
            len(values) + prior_weight
        )

    home_offense = blended_average(team_scored.get(home, []))
    away_offense = blended_average(team_scored.get(away, []))
    home_defense = blended_average(team_allowed.get(home, []))
    away_defense = blended_average(team_allowed.get(away, []))

    projected_home = max(0.0, (home_offense + away_defense) / 2 + home_edge)
    projected_away = max(0.0, (away_offense + home_defense) / 2 - home_edge)
    return projected_home, projected_away


def backtest_score_model(history):
    """Walk-forward score projection test; each game is graded before its score is added."""
    ordered_history = sorted(history, key=lambda game: game["Date"])
    warmup_games = max(1, int(len(ordered_history) * 0.30)) if ordered_history else 0
    team_scored = {}
    team_allowed = {}
    league_home_scores = []
    league_away_scores = []
    predictions = []

    for index, game in enumerate(ordered_history):
        home = game["Home Team"]
        away = game["Away Team"]
        actual_home = float(game["Home Score"])
        actual_away = float(game["Away Score"])

        if index >= warmup_games:
            projected_home, projected_away = _project_scores(
                home, away, team_scored, team_allowed,
                league_home_scores, league_away_scores,
            )
            predictions.append({
                "Date": game["Date"],
                "Away Team": away,
                "Home Team": home,
                "Projected Away Score": round(projected_away, 2),
                "Actual Away Score": actual_away,
                "Projected Home Score": round(projected_home, 2),
                "Actual Home Score": actual_home,
                "Away Absolute Error": abs(projected_away - actual_away),
                "Home Absolute Error": abs(projected_home - actual_home),
            })

        team_scored.setdefault(home, []).append(actual_home)
        team_scored.setdefault(away, []).append(actual_away)
        team_allowed.setdefault(home, []).append(actual_away)
        team_allowed.setdefault(away, []).append(actual_home)
        team_scored[home] = team_scored[home][-10:]
        team_scored[away] = team_scored[away][-10:]
        team_allowed[home] = team_allowed[home][-10:]
        team_allowed[away] = team_allowed[away][-10:]
        league_home_scores.append(actual_home)
        league_away_scores.append(actual_away)
        league_home_scores = league_home_scores[-200:]
        league_away_scores = league_away_scores[-200:]

    results = pd.DataFrame(predictions)
    if results.empty:
        return results, {
            "training_games": warmup_games,
            "games": 0,
            "mae": None,
            "rmse": None,
            "home_mae": None,
            "away_mae": None,
        }

    errors = pd.concat(
        [results["Home Absolute Error"], results["Away Absolute Error"]],
        ignore_index=True,
    )
    metrics = {
        "training_games": warmup_games,
        "games": len(results),
        "mae": round(errors.mean(), 3),
        "rmse": round(math.sqrt((errors ** 2).mean()), 3),
        "home_mae": round(results["Home Absolute Error"].mean(), 3),
        "away_mae": round(results["Away Absolute Error"].mean(), 3),
    }
    return results, metrics


def make_score_predictions(games, history):
    """Create pre-game score estimates from prior completed games only."""
    predictions = []
    ordered_history = sorted(history, key=lambda game: game["Date"])

    for game in games:
        if game["State"] not in ("pre", "in"):
            continue
        # Strictly earlier timestamps prevent same-day future games leaking in.
        prior = [row for row in ordered_history if row["Date"] < game["Date"]]
        team_scored = {}
        team_allowed = {}
        home_scores = []
        away_scores = []

        for row in prior:
            home = row["Home Team"]
            away = row["Away Team"]
            hs = float(row["Home Score"])
            aws = float(row["Away Score"])
            team_scored.setdefault(home, []).append(hs)
            team_scored.setdefault(away, []).append(aws)
            team_allowed.setdefault(home, []).append(aws)
            team_allowed.setdefault(away, []).append(hs)
            team_scored[home] = team_scored[home][-10:]
            team_scored[away] = team_scored[away][-10:]
            team_allowed[home] = team_allowed[home][-10:]
            team_allowed[away] = team_allowed[away][-10:]
            home_scores.append(hs)
            away_scores.append(aws)

        projected_home, projected_away = _project_scores(
            game["Home Team"], game["Away Team"],
            team_scored, team_allowed, home_scores[-200:], away_scores[-200:],
        )
        predictions.append({
            "Away Team": game["Away Team"],
            "Projected Away Score": round(projected_away, 1),
            "Home Team": game["Home Team"],
            "Projected Home Score": round(projected_home, 1),
        })

    return pd.DataFrame(predictions)


def make_predictions(sport, games, ratings):
    info = SPORTS[sport]
    predictions = []

    for game in games:
        home = game["Home Team"]
        away = game["Away Team"]

        if game["State"] not in ("pre", "in"):
            continue

        home_rating = ratings.get(home, 1500.0)
        away_rating = ratings.get(away, 1500.0)

        home_probability = expected_score(
            home_rating + info["home_adv"], away_rating
        )

        if game["State"] == "in":
            note = "In progress — estimate may change"
        else:
            note = "Pre-game estimate"

        predictions.append({
            "Date": game["Date"],
            "Away Team": away,
            "Home Team": home,
            "Away Rating": round(away_rating),
            "Home Rating": round(home_rating),
            "Away Win Chance (%)": round((1 - home_probability) * 100, 1),
            "Home Win Chance (%)": round(home_probability * 100, 1),
            "Model Lean": home if home_probability >= 0.5 else away,
            "Status": game["Status"],
            "Note": note,
        })

    return pd.DataFrame(predictions)


st.title("🏆 AI Sports Analytics")
st.write(
    "Multi-sport team ratings and game estimates based on recent completed results."
)

with st.sidebar:
    st.header("Dashboard settings")
    sport = st.selectbox("Choose a sport", list(SPORTS.keys()))
    days = st.slider("Historical window (days)", 14, 90, 60, step=7)
    st.caption("Data source: ESPN public scoreboard.")
    refresh = st.button("🔄 Refresh data")

if refresh:
    get_current_games.clear()
    get_history.clear()

try:
    with st.spinner(f"Loading {sport} data and calculating ratings..."):
        current_games = get_current_games(sport)
        history = get_history(sport, days)
        ratings, rating_table, games_used = build_ratings(sport, history)
        predictions = make_predictions(sport, current_games, ratings)
        score_predictions = make_score_predictions(current_games, history)
        if not predictions.empty and not score_predictions.empty:
            predictions = predictions.merge(
                score_predictions, on=["Away Team", "Home Team"], how="left"
            )

    c1, c2, c3 = st.columns(3)
    c1.metric("Games in current scoreboard", len(current_games))
    c2.metric("Completed games used", games_used)
    c3.metric("Teams rated", len(rating_table))

    st.caption(
        f"Ratings use completed games from approximately the last {days} days. "
        "Teams start at 1500 for this window, so early-season ratings are less reliable."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Team Ratings",
        "🎯 Game Predictions",
        "🏈 Scoreboard",
        "📈 Historical Accuracy",
    ])

    with tab1:
        st.subheader(f"{sport} Team Performance Ratings")
        if rating_table.empty:
            st.info(
                "No completed games were found in this historical window. "
                "Try a longer window or refresh later."
            )
        else:
            st.dataframe(
                rating_table,
                use_container_width=True,
                hide_index=True,
            )
            st.bar_chart(rating_table.set_index("Team")["Rating"])
            st.download_button(
                "Download team ratings CSV",
                data=rating_table.to_csv(index=False),
                file_name=f"{sport.lower()}_team_ratings.csv",
                mime="text/csv",
            )

    with tab2:
        st.subheader("Game Winner Estimates")
        st.caption(
            "Probabilities are model estimates, not guarantees. "
            "They are based on Elo-style ratings and home advantage."
        )

        if predictions.empty:
            st.info(
                "No upcoming or in-progress games were returned by the current "
                "scoreboard. Check back closer to game time."
            )
        else:
            st.dataframe(
                predictions[
                    [
                        "Away Team", "Away Win Chance (%)",
                        "Home Team", "Home Win Chance (%)",
                        "Model Lean", "Projected Away Score",
                        "Projected Home Score", "Status", "Note",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download game estimates CSV",
                data=predictions.to_csv(index=False),
                file_name=f"{sport.lower()}_game_estimates.csv",
                mime="text/csv",
            )

    with tab3:
        st.subheader(f"{sport} Scoreboard")
        if not current_games:
            st.info("No games were returned for the current scoreboard.")
        else:
            scoreboard = pd.DataFrame(current_games)
            st.dataframe(
                scoreboard[
                    [
                        "Away Team", "Away Score",
                        "Home Team", "Home Score", "Status",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download scoreboard CSV",
                data=scoreboard.to_csv(index=False),
                file_name=f"{sport.lower()}_scoreboard.csv",
                mime="text/csv",
            )


    with tab4:
        st.subheader("Historical Prediction Accuracy")
        st.caption(
            "This replays completed games in date order. Each prediction uses "
            "only earlier results; the current game's score is applied afterward. "
            "This is a historical backtest, not a guarantee of future performance."
        )
        backtest_results, backtest_metrics = backtest_model(sport, history)
        if backtest_results.empty:
            st.info("Not enough completed games are available to evaluate this sport.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Warm-up games", backtest_metrics["training_games"])
            m2.metric("Games evaluated", backtest_metrics["games"])
            m3.metric("Model picks correct", f'{backtest_metrics["accuracy"]}%')
            m4.metric("Always-pick-home baseline", f'{backtest_metrics["baseline_accuracy"]}%')
            st.caption(
                "Only games after the warm-up period are scored. The baseline always "
                "picks the home team. Brier score is calculated on evaluated games; "
                "lower is better. This is a limited historical test, not a future guarantee."
            )
            st.metric("Brier score (lower is better)", backtest_metrics["brier"])
            st.dataframe(
                backtest_results[
                    ["Date", "Away Team", "Home Team", "Home Win Probability (%)",
                     "Correct Pick", "Home-Only Baseline Correct"]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download historical backtest CSV",
                data=backtest_results.to_csv(index=False),
                file_name=f"{sport.lower()}_historical_backtest.csv",
                mime="text/csv",
            )

        st.divider()
        st.subheader("Projected Score Accuracy")
        score_results, score_metrics = backtest_score_model(history)
        if score_results.empty:
            st.info("Not enough completed games to evaluate score projections.")
        else:
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Games evaluated", score_metrics["games"])
            sm2.metric("Mean absolute error (MAE)", score_metrics["mae"])
            sm3.metric("Root mean squared error (RMSE)", score_metrics["rmse"])
            st.caption(
                "Score errors are measured in points/runs/goals. Lower is better. "
                "Each estimate uses only earlier completed games; the current game's "
                "score is added after its forecast is recorded."
            )
            st.dataframe(
                score_results[
                    ["Date", "Away Team", "Projected Away Score", "Actual Away Score",
                     "Home Team", "Projected Home Score", "Actual Home Score"]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download score projection backtest CSV",
                data=score_results.to_csv(index=False),
                file_name=f"{sport.lower()}_score_projection_backtest.csv",
                mime="text/csv",
            )

        st.divider()
        st.subheader("Accuracy by Sport")
        st.caption(
            "Compare the same walk-forward method across NFL, NBA, MLB, and NHL. "
            "Each sport is evaluated independently; results may be unavailable "
            "when ESPN has too little historical data."
        )
        if st.button("Run all-sport accuracy comparison", key="run_all_sport_accuracy"):
            comparison_rows = []
            progress = st.progress(0, text="Preparing sport-by-sport comparison...")
            for index, comparison_sport in enumerate(SPORTS):
                try:
                    comparison_history = (
                        history if comparison_sport == sport
                        else get_history(comparison_sport, days)
                    )
                    _, comparison_metrics = backtest_model(
                        comparison_sport, comparison_history
                    )
                    comparison_rows.append({
                        "Sport": comparison_sport,
                        "Games Available": len(comparison_history),
                        "Warm-up Games": comparison_metrics["training_games"],
                        "Games Evaluated": comparison_metrics["games"],
                        "Model Accuracy (%)": comparison_metrics["accuracy"],
                        "Home Baseline (%)": comparison_metrics["baseline_accuracy"],
                        "Brier Score": comparison_metrics["brier"],
                    })
                except Exception as comparison_error:
                    comparison_rows.append({
                        "Sport": comparison_sport,
                        "Games Available": None,
                        "Warm-up Games": None,
                        "Games Evaluated": 0,
                        "Model Accuracy (%)": None,
                        "Home Baseline (%)": None,
                        "Brier Score": None,
                    })
                    st.warning(
                        f"{comparison_sport} comparison unavailable: "
                        f"{comparison_error}"
                    )
                progress.progress(
                    (index + 1) / len(SPORTS),
                    text=f"Compared {index + 1} of {len(SPORTS)} sports",
                )
            progress.empty()
            comparison_table = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_table, use_container_width=True, hide_index=True)
            st.download_button(
                "Download all-sport accuracy CSV",
                data=comparison_table.to_csv(index=False),
                file_name="all_sports_accuracy_comparison.csv",
                mime="text/csv",
                key="download_all_sport_accuracy",
            )


    st.divider()
    st.caption(
        "Prototype analytics only. The model is not trained on player injuries, "
        "lineups, weather, or other advanced factors. ESPN data availability "
        "and historical coverage may vary."
    )

except requests.RequestException:
    st.error(
        "Couldn't retrieve sports data. Check your internet connection and "
        "try Refresh data."
    )
except Exception as error:
    st.error("The dashboard encountered an error.")
    st.caption(f"Technical detail: {error}")
