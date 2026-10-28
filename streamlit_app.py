
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
    """Fetch history in short date ranges to avoid ESPN rejecting long queries."""
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    rows_by_event = {}
    chunk_start = start_date

    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=2), end_date)
        date_range = f"{chunk_start:%Y%m%d}-{chunk_end:%Y%m%d}"
        response = requests.get(
            scoreboard_url(sport),
            params={"dates": date_range, "limit": 100},
            timeout=30,
        )
        response.raise_for_status()

        for row in parse_events(response.json().get("events", [])):
            event_id = row.get("Event ID")
            key = str(event_id) if event_id else (
                f"{row['Date']}|{row['Away Team']}|{row['Home Team']}"
            )
            rows_by_event[key] = row

        chunk_start = chunk_end + timedelta(days=1)

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
    """Evaluate each historical game using ratings available before that game."""
    info = SPORTS[sport]
    ratings = {}
    predictions = []
    ordered_history = sorted(history, key=lambda game: game["Date"])

    for game in ordered_history:
        home = game["Home Team"]
        away = game["Away Team"]
        home_score = game["Home Score"]
        away_score = game["Away Score"]

        ratings.setdefault(home, 1500.0)
        ratings.setdefault(away, 1500.0)

        # Predict before using this game's final score to update ratings.
        home_probability = expected_score(
            ratings[home] + info["home_adv"], ratings[away]
        )
        actual_home_win = 1 if home_score > away_score else 0
        predicted_home_win = 1 if home_probability >= 0.5 else 0

        predictions.append({
            "Date": game["Date"],
            "Away Team": away,
            "Home Team": home,
            "Home Win Probability": home_probability,
            "Actual Home Win": actual_home_win,
            "Correct Pick": predicted_home_win == actual_home_win,
            "Brier Contribution": (home_probability - actual_home_win) ** 2,
        })

        margin = abs(home_score - away_score)
        margin_factor = min(1.5, math.log1p(margin) / 2.5 + 0.7)
        change = info["k"] * margin_factor * (actual_home_win - home_probability)
        ratings[home] += change
        ratings[away] -= change

    results = pd.DataFrame(predictions)
    if results.empty:
        return results, {"games": 0, "accuracy": None, "brier": None}

    metrics = {
        "games": len(results),
        "accuracy": round(results["Correct Pick"].mean() * 100, 1),
        "brier": round(results["Brier Contribution"].mean(), 4),
    }
    return results, metrics


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
                        "Model Lean", "Status", "Note",
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
            m1, m2, m3 = st.columns(3)
            m1.metric("Games evaluated", backtest_metrics["games"])
            m2.metric("Winner picks correct", f'{backtest_metrics["accuracy"]}%')
            m3.metric("Brier score", backtest_metrics["brier"])
            st.caption(
                "For Brier score, lower is better; 0 is perfect. Accuracy and "
                "Brier score describe this selected historical window only."
            )
            st.dataframe(
                backtest_results[
                    ["Date", "Away Team", "Home Team", "Correct Pick"]
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
