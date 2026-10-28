
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

# Authentication is handled by Streamlit's OpenID Connect (OIDC) support.
# Credentials stay in Streamlit secrets; no passwords or tokens are hard-coded.
auth_config = st.secrets.get("auth", {})
app_auth_config = st.secrets.get("app_auth", {})
required_auth_settings = (
    "redirect_uri",
    "cookie_secret",
    "client_id",
    "client_secret",
    "server_metadata_url",
)
missing_auth_settings = [
    key for key in required_auth_settings if not auth_config.get(key)
]
allowed_emails = [
    str(email).strip().lower()
    for email in app_auth_config.get("allowed_emails", [])
    if str(email).strip()
]

if missing_auth_settings or not allowed_emails:
    st.title("🏆 AI Sports Analytics")
    st.error("Sign-in is not configured yet, so the dashboard is locked.")
    st.markdown(
        "Follow the **Authentication setup** section in the repository README. "
        "Add the OIDC values and an email allowlist to your local "
        "`.streamlit/secrets.toml` or your host's Streamlit secrets settings. "
        "Never commit real credentials to GitHub."
    )
    st.stop()

if not st.user.is_logged_in:
    st.title("🏆 AI Sports Analytics")
    st.write("Sign in with your authorized Google account to access the dashboard.")
    st.button("Sign in with Google", on_click=st.login)
    st.stop()

user_email = str(st.user.get("email", "")).strip().lower()
email_verified = st.user.get("email_verified", False) is True

if not email_verified or user_email not in allowed_emails:
    st.title("🏆 AI Sports Analytics")
    st.error("This Google account is not authorized to use this dashboard.")
    st.caption("If you believe this is a mistake, ask the app administrator to add your verified email to the allowlist.")
    st.button("Sign out", on_click=st.logout)
    st.stop()

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
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    date_range = f"{start_date:%Y%m%d}-{end_date:%Y%m%d}"

    response = requests.get(
        scoreboard_url(sport),
        params={"dates": date_range, "limit": 500},
        timeout=30,
    )
    response.raise_for_status()
    rows = parse_events(response.json().get("events", []))

    if not rows:
        return []

    # Only completed games can be used to calculate ratings.
    return [
        row for row in rows
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
    st.caption(f"Signed in as {user_email}")
    st.button("Log out", on_click=st.logout)
    st.divider()
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

    tab1, tab2, tab3 = st.tabs([
        "📊 Team Ratings",
        "🎯 Game Predictions",
        "🏈 Scoreboard",
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
