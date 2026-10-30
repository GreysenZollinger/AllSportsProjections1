"""Data ingestion and leakage-aware forecasting helpers for four sports."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

import pandas as pd
import requests

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"
CALIBRATION_MIN_GAMES = 20
SCORE_PRIOR_GAMES = 5
SCORE_UNCERTAINTY_MIN_SCORES = 20
RATING_PRIOR_GAMES = 8
RATING_HALF_LIFE_DAYS = 180
RATING_HISTORY_DAYS = 180
MODEL_VERSION = "four-sport-forecast-v2"


@dataclass(frozen=True)
class SportConfig:
    category: str
    league: str
    home_advantage: float
    k_factor: float
    margin_scale: float


SPORTS: dict[str, SportConfig] = {
    "NFL": SportConfig("football", "nfl", 55, 24, 2.5),
    "NBA": SportConfig("basketball", "nba", 65, 20, 2.8),
    "MLB": SportConfig("baseball", "mlb", 45, 18, 1.8),
    "NHL": SportConfig("hockey", "nhl", 50, 20, 1.8),
}


class SportsDataError(RuntimeError):
    """Raised when a scoreboard response cannot be safely used."""


@dataclass(frozen=True)
class ScoringModel:
    teams: dict[str, dict[str, float | int]]
    league_average: float | None
    home_average: float | None
    away_average: float | None
    score_standard_deviation: float | None


def scoreboard_url(sport: str) -> str:
    try:
        info = SPORTS[sport]
    except KeyError as error:
        raise ValueError(f"Unsupported sport: {sport}") from error
    return f"{BASE_URL}/{info.category}/{info.league}/scoreboard"


def _event_datetime(value: Any, event_id: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise SportsDataError(f"Event {event_id} has no valid start date.")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise SportsDataError(
            f"Event {event_id} has an invalid start date: {value!r}."
        ) from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _score(value: Any, event_id: str) -> float | None:
    if value is None or value == "" or value == "-":
        return None
    if isinstance(value, bool):
        raise SportsDataError(f"Event {event_id} has an invalid score: {value!r}.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise SportsDataError(
            f"Event {event_id} has an invalid score: {value!r}."
        ) from error
    if not math.isfinite(parsed) or parsed < 0:
        raise SportsDataError(
            f"Event {event_id} has an invalid score: {value!r}."
        )
    return parsed


def parse_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Parse ESPN events without inventing scores or team identities."""
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise SportsDataError("Scoreboard events must be a list.")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for event in events:
        if not isinstance(event, Mapping):
            raise SportsDataError("Scoreboard contains a non-object event.")
        event_id = str(event.get("id", "")).strip()
        if not event_id:
            raise SportsDataError("Scoreboard event is missing its identifier.")
        if event_id in seen_ids:
            continue
        seen_ids.add(event_id)

        competitions = event.get("competitions")
        if not isinstance(competitions, list) or not competitions:
            raise SportsDataError(f"Event {event_id} has no competition.")
        competition = competitions[0]
        if not isinstance(competition, Mapping):
            raise SportsDataError(f"Event {event_id} has invalid competition data.")
        competitors = competition.get("competitors")
        if not isinstance(competitors, list) or len(competitors) < 2:
            raise SportsDataError(f"Event {event_id} has fewer than two teams.")
        if any(not isinstance(item, Mapping) for item in competitors):
            raise SportsDataError(f"Event {event_id} has invalid team data.")

        home = next(
            (item for item in competitors if item.get("homeAway") == "home"),
            None,
        )
        away = next(
            (item for item in competitors if item.get("homeAway") == "away"),
            None,
        )
        if not isinstance(home, Mapping) or not isinstance(away, Mapping):
            raise SportsDataError(
                f"Event {event_id} does not identify home and away teams."
            )

        def team_name(competitor: Mapping[str, Any]) -> str:
            team = competitor.get("team")
            if not isinstance(team, Mapping):
                raise SportsDataError(
                    f"Event {event_id} contains a team without identity data."
                )
            name = team.get("displayName") or team.get("name")
            if not isinstance(name, str) or not name.strip():
                raise SportsDataError(f"Event {event_id} contains an unnamed team.")
            return name.strip()

        def team_id(competitor: Mapping[str, Any]) -> str:
            team = competitor["team"]
            return str(team.get("id") or competitor.get("id") or "").strip()

        status_data = event.get("status")
        status_data = status_data if isinstance(status_data, Mapping) else {}
        status_type = status_data.get("type")
        status_type = status_type if isinstance(status_type, Mapping) else {}
        state = status_type.get("state")
        if state not in {"pre", "in", "post"}:
            state = "unknown"
        season = event.get("season")
        season = season if isinstance(season, Mapping) else {}
        season_slug = season.get("slug")
        season_type = (
            str(season_slug).strip().casefold()
            if isinstance(season_slug, str) and season_slug.strip()
            else {
                "1": "preseason",
                "2": "regular-season",
                "3": "postseason",
            }.get(str(season.get("type", "")), "unknown")
        )

        rows.append({
            "Event ID": event_id,
            "Date": _event_datetime(event.get("date"), event_id),
            "Season Type": season_type,
            "Away Team ID": team_id(away),
            "Away Team": team_name(away),
            "Away Score": _score(away.get("score"), event_id),
            "Home Team ID": team_id(home),
            "Home Team": team_name(home),
            "Home Score": _score(home.get("score"), event_id),
            "Status": (
                status_type.get("shortDetail")
                or status_type.get("description")
                or "Status unavailable"
            ),
            "State": state,
        })
    return rows


def _fetch_events_for_date(sport: str, game_date: date) -> list[dict[str, Any]]:
    """Fetch one scoreboard date, retrying temporary network failures."""
    last_error = None

    for attempt in range(3):
        try:
            response = requests.get(
                scoreboard_url(sport),
                params={"dates": game_date.strftime("%Y%m%d")},
                timeout=25,
            )
            response.raise_for_status()
            break
        except requests.RequestException as error:
            last_error = error
            if attempt < 2:
                import time
                time.sleep(attempt + 1)
    else:
        raise SportsDataError(
            f"Could not retrieve {sport} scoreboard for {game_date} "
            f"after 3 attempts: {last_error}"
        ) from last_error

    try:
        payload = response.json()
    except ValueError as error:
        raise SportsDataError(
            f"{sport} scoreboard for {game_date} returned invalid JSON."
        ) from error

    if not isinstance(payload, Mapping):
        raise SportsDataError(
            f"{sport} scoreboard for {game_date} returned an invalid response."
        )
    if "events" not in payload or not isinstance(payload["events"], list):
        raise SportsDataError(
            f"{sport} scoreboard for {game_date} has a missing or invalid "
            "events field."
        )

    return parse_events(payload["events"])

def _deduplicate_games(games: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    state_priority = {"unknown": 0, "pre": 1, "in": 2, "post": 3}
    for game in games:
        event_id = str(game["Event ID"])
        existing = by_id.get(event_id)
        if existing is None or state_priority[game["State"]] > state_priority[
            existing["State"]
        ]:
            by_id[event_id] = dict(game)
    return sorted(by_id.values(), key=lambda item: (item["Date"], item["Event ID"]))


def fetch_current_games(
    sport: str, today: date | None = None
) -> list[dict[str, Any]]:
    current_date = today or datetime.now(timezone.utc).date()
    games: list[dict[str, Any]] = []
    for offset in range(8):
        games.extend(_fetch_events_for_date(sport, current_date + timedelta(days=offset)))
    return _deduplicate_games(games)


def fetch_history(
    sport: str, days: int, end_date: date | None = None
) -> list[dict[str, Any]]:
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 365:
        raise ValueError("Historical window must be between 1 and 365 days.")
    last_date = end_date or datetime.now(timezone.utc).date()
    first_date = last_date - timedelta(days=days - 1)
    games: list[dict[str, Any]] = []
    for offset in range(days):
        game_date = first_date + timedelta(days=offset)
        games.extend(_fetch_events_for_date(sport, game_date))

    completed = [
        game for game in games
        if game["State"] == "post"
        and game["Home Score"] is not None
        and game["Away Score"] is not None
    ]
    return _deduplicate_games(completed)


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError(f"Invalid game date: {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _completed_games(
    history: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    games = [
        dict(game) for game in history
        if game.get("State", "post") == "post"
        and game.get("Home Score") is not None
        and game.get("Away Score") is not None
        and not _is_preseason(game)
    ]
    for game in games:
        game["Date"] = _as_datetime(game["Date"])
    return _deduplicate_games(games)


def expected_score(rating_a: float, rating_b: float) -> float:
    exponent = max(-8.0, min(8.0, (rating_b - rating_a) / 400))
    return 1 / (1 + 10**exponent)


def _home_outcome(home_score: float, away_score: float) -> float:
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5


def _team_key(game: Mapping[str, Any], side: str) -> str:
    team_id = str(game.get(f"{side} Team ID") or "").strip()
    if team_id:
        return team_id
    team_name = str(game[f"{side} Team"]).strip()
    return f"name:{team_name.casefold()}"


def _is_preseason(game: Mapping[str, Any]) -> bool:
    season_type = str(game.get("Season Type") or "").casefold()
    normalized = season_type.replace("_", "-").replace(" ", "-")
    return any(
        marker in normalized
        for marker in ("preseason", "pre-season", "spring-training", "exhibition")
    )


def _stabilize_rating(rating: float, games_played: int) -> float:
    evidence_weight = games_played / (games_played + RATING_PRIOR_GAMES)
    return 1500.0 + (rating - 1500.0) * evidence_weight


def _update_ratings_for_day(
    sport: str,
    ratings: dict[str, float],
    games_played: dict[str, int],
    last_played: dict[str, date],
    games: Sequence[Mapping[str, Any]],
) -> None:
    if not games:
        return
    info = SPORTS[sport]
    game_day = games[0]["Date"].date()
    participating_teams = {
        _team_key(game, side)
        for game in games
        for side in ("Home", "Away")
    }

    for team in participating_teams:
        ratings.setdefault(team, 1500.0)
        games_played.setdefault(team, 0)
        previous_date = last_played.get(team)
        if previous_date is not None:
            days_inactive = max(0, (game_day - previous_date).days)
            decay = 0.5 ** (days_inactive / RATING_HALF_LIFE_DAYS)
            ratings[team] = 1500.0 + (ratings[team] - 1500.0) * decay

    changes: dict[str, float] = defaultdict(float)
    for game in games:
        home = _team_key(game, "Home")
        away = _team_key(game, "Away")
        expected_home = expected_score(
            ratings[home] + info.home_advantage,
            ratings[away],
        )
        outcome = _home_outcome(game["Home Score"], game["Away Score"])
        margin = abs(game["Home Score"] - game["Away Score"])
        margin_factor = min(
            1.5, math.log1p(margin) / info.margin_scale + 0.7
        )
        change = info.k_factor * margin_factor * (outcome - expected_home)
        changes[home] += change
        changes[away] -= change

    for team, change in changes.items():
        ratings[team] += change
    for game in games:
        home = _team_key(game, "Home")
        away = _team_key(game, "Away")
        games_played[home] += 1
        games_played[away] += 1
        last_played[home] = game_day
        last_played[away] = game_day


def build_ratings(
    sport: str, history: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, float], pd.DataFrame]:
    if sport not in SPORTS:
        raise ValueError(f"Unsupported sport: {sport}")
    games = _completed_games(history)
    ratings: dict[str, float] = {}
    games_played: dict[str, int] = {}
    last_played: dict[str, date] = {}
    records: dict[str, dict[str, int]] = {}
    recent_results: dict[str, list[float]] = defaultdict(list)
    team_names: dict[str, str] = {}

    games_by_day: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        games_by_day[game["Date"].date()].append(game)
        home = _team_key(game, "Home")
        away = _team_key(game, "Away")
        team_names[home] = str(game["Home Team"])
        team_names[away] = str(game["Away Team"])
        outcome = _home_outcome(game["Home Score"], game["Away Score"])
        for team in (home, away):
            records.setdefault(team, {"W": 0, "L": 0, "T": 0, "Games": 0})
        records[home]["W" if outcome == 1 else "L" if outcome == 0 else "T"] += 1
        records[away]["L" if outcome == 1 else "W" if outcome == 0 else "T"] += 1
        records[home]["Games"] += 1
        records[away]["Games"] += 1
        recent_results[home].append(outcome)
        recent_results[away].append(1 - outcome)
    for game_day in sorted(games_by_day):
        _update_ratings_for_day(
            sport,
            ratings,
            games_played,
            last_played,
            games_by_day[game_day],
        )

    rows = []
    for team, rating in ratings.items():
        record = records[team]
        recent = recent_results[team][-5:]
        total = record["Games"]
        recent_record = (
            f'{sum(result == 1 for result in recent)}-'
            f'{sum(result == 0 for result in recent)}-'
            f'{sum(result == 0.5 for result in recent)}'
        )
        rows.append({
            "Team ID": team,
            "Team": team_names[team],
            "Rating": round(_stabilize_rating(rating, total)),
            "Raw Elo": round(rating),
            "Rating Prior Weight (%)": round(
                total / (total + RATING_PRIOR_GAMES) * 100, 1
            ),
            "Record": f'{record["W"]}-{record["L"]}-{record["T"]}',
            "Games": total,
            "Last 5": recent_record,
            "Last 5 games": len(recent),
            "Last Game (UTC)": last_played[team].isoformat(),
            "Win rate (%)": round(
                (record["W"] + record["T"] / 2) / total * 100, 1
            ) if total else 0,
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["Rating", "Team"], ascending=[False, True]
        ).reset_index(drop=True)
    forecast_ratings = {
        team: _stabilize_rating(rating, games_played[team])
        for team, rating in ratings.items()
    }
    return forecast_ratings, table


def build_scoring_averages(
    history: Sequence[Mapping[str, Any]],
) -> ScoringModel:
    team_scores: dict[str, dict[str, list[float]]] = {}
    home_scores: list[float] = []
    away_scores: list[float] = []
    for game in _completed_games(history):
        _increment_score_state(game, team_scores, home_scores, away_scores)
    return _scoring_snapshot(team_scores, home_scores, away_scores)


def fit_calibration_bias(
    calibration_examples: Sequence[Mapping[str, Any]],
) -> float:
    """Fit an intercept-only logistic correction from prior OOF predictions."""
    if len(calibration_examples) < CALIBRATION_MIN_GAMES:
        return 0.0
    examples = []
    for example in calibration_examples:
        probability = float(example["raw_home_probability"])
        outcome = float(example["home_outcome"])
        if not math.isfinite(probability) or not math.isfinite(outcome):
            raise ValueError("Calibration values must be finite.")
        probability = min(1 - 1e-6, max(1e-6, probability))
        if not 0 <= outcome <= 1:
            raise ValueError("Calibration outcomes must be between zero and one.")
        examples.append((math.log(probability / (1 - probability)), outcome))

    bias = 0.0
    for _ in range(30):
        probabilities = [
            1 / (1 + math.exp(-max(-20, min(20, logit + bias))))
            for logit, _ in examples
        ]
        gradient = sum(
            probability - outcome
            for probability, (_, outcome) in zip(probabilities, examples)
        )
        hessian = sum(
            probability * (1 - probability) for probability in probabilities
        )
        if hessian < 1e-9:
            break
        step = gradient / hessian
        bias = max(-2.5, min(2.5, bias - step))
        if abs(step) < 1e-7:
            break
    return bias


def _apply_calibration(probability: float, bias: float) -> float:
    clipped = min(1 - 1e-9, max(1e-9, probability))
    logit = math.log(clipped / (1 - clipped)) + bias
    return 1 / (1 + math.exp(-max(-20, min(20, logit))))


def _score_projection(
    home: str, away: str, scoring_model: ScoringModel
) -> tuple[float | None, float | None, float | None]:
    if scoring_model.home_average is None or scoring_model.away_average is None:
        return None, None, None
    home_stats = scoring_model.teams.get(
        home,
        {
            "home_scored": scoring_model.home_average,
            "home_allowed": scoring_model.away_average,
        },
    )
    away_stats = scoring_model.teams.get(
        away,
        {
            "away_scored": scoring_model.away_average,
            "away_allowed": scoring_model.home_average,
        },
    )
    projected_home = (
        float(home_stats["home_scored"]) + float(away_stats["away_allowed"])
    ) / 2
    projected_away = (
        float(away_stats["away_scored"]) + float(home_stats["home_allowed"])
    ) / 2
    return (
        projected_home,
        projected_away,
        scoring_model.score_standard_deviation,
    )


def fair_moneyline(probability: float) -> int:
    """Convert a model win probability to fair American odds."""
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Win probability must be between 0 and 1.")

    # Avoid undefined odds at exactly 0% or 100%.
    probability = min(0.999, max(0.001, probability))

    if probability >= 0.5:
        return round(-100 * probability / (1 - probability))
    return round(100 * (1 - probability) / probability)


def make_predictions(
    sport: str,
    games: Sequence[Mapping[str, Any]],
    ratings: Mapping[str, float],
    scoring_model: ScoringModel,
    calibration_examples: Sequence[Mapping[str, Any]] = (),
) -> pd.DataFrame:
    if sport not in SPORTS:
        raise ValueError(f"Unsupported sport: {sport}")
    info = SPORTS[sport]
    calibration_bias = (
        fit_calibration_bias(calibration_examples)
        if sport in {"NFL", "MLB"} else 0.0
    )
    rows = []
    for game in games:
        if game.get("State") not in {"pre", "in"}:
            continue
        home = str(game["Home Team"])
        away = str(game["Away Team"])
        home_key = _team_key(game, "Home")
        away_key = _team_key(game, "Away")
        raw_home = expected_score(
            float(ratings.get(home_key, 1500.0)) + info.home_advantage,
            float(ratings.get(away_key, 1500.0)),
        )
        home_chance = _apply_calibration(raw_home, calibration_bias)
        projected_home, projected_away, score_std = _score_projection(
            home, away, scoring_model
        )
        projected_total = (
            round(projected_home + projected_away, 1)
            if projected_home is not None and projected_away is not None
            else None
        )
        uncertainty = 1.2816 * score_std if score_std is not None else None

        def score_range(score: float | None) -> str:
            if score is None or uncertainty is None:
                return "Not enough data"
            low = max(0.0, score - uncertainty)
            high = score + uncertainty
            return f"{low:.1f} - {high:.1f}"

        rows.append({
            "Event ID": str(game["Event ID"]),
            "Model Version": MODEL_VERSION,
            "Date": game["Date"],
            "Away Team": away,
            "Away Rating": round(float(ratings.get(away_key, 1500.0))),
            "Away Projected Score": (
                round(projected_away, 1) if projected_away is not None else None
            ),
            "Away Score Range (approx. 80%)": score_range(projected_away),
            "Away Win Chance (%)": round((1 - home_chance) * 100, 1),
            "Away Fair Moneyline": fair_moneyline(1 - home_chance),
            "Home Team": home,
            "Home Rating": round(float(ratings.get(home_key, 1500.0))),
            "Home Projected Score": (
                round(projected_home, 1) if projected_home is not None else None
            ),
            "Projected Game Total": projected_total,
            "Home Score Range (approx. 80%)": score_range(projected_home),
            "Home Win Chance (%)": round(home_chance * 100, 1),
            "Home Fair Moneyline": fair_moneyline(home_chance),
            "Raw Home Win Chance (%)": round(raw_home * 100, 1),
            "Model Lean": home if home_chance >= 0.5 else away,
            "Status": game.get("Status", "Status unavailable"),
            "State": game["State"],
        })
    return pd.DataFrame(rows)


def _increment_score_state(
    game: Mapping[str, Any],
    team_scores: dict[str, dict[str, list[float]]],
    home_scores: list[float],
    away_scores: list[float],
) -> None:
    home_score = float(game["Home Score"])
    away_score = float(game["Away Score"])
    home = str(game["Home Team"])
    away = str(game["Away Team"])
    home_stats = team_scores.setdefault(home, {
        "scored": [],
        "allowed": [],
        "home_scored": [],
        "home_allowed": [],
        "away_scored": [],
        "away_allowed": [],
    })
    away_stats = team_scores.setdefault(away, {
        "scored": [],
        "allowed": [],
        "home_scored": [],
        "home_allowed": [],
        "away_scored": [],
        "away_allowed": [],
    })
    home_stats["scored"].append(home_score)
    home_stats["allowed"].append(away_score)
    home_stats["home_scored"].append(home_score)
    home_stats["home_allowed"].append(away_score)
    away_stats["scored"].append(away_score)
    away_stats["allowed"].append(home_score)
    away_stats["away_scored"].append(away_score)
    away_stats["away_allowed"].append(home_score)
    home_scores.append(home_score)
    away_scores.append(away_score)


def _recency_weighted_mean(
    values: Sequence[float],
    half_life_games: float = 10.0,
) -> float:
    """Weighted average that favors the most recent observations."""
    if not values:
        raise ValueError("Cannot average an empty sequence.")
    if half_life_games <= 0:
        raise ValueError("half_life_games must be positive.")

    newest = len(values) - 1
    weights = [
        0.5 ** ((newest - index) / half_life_games)
        for index in range(len(values))
    ]
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def _scoring_snapshot(
    team_scores: Mapping[str, Mapping[str, Sequence[float]]],
    home_scores: Sequence[float],
    away_scores: Sequence[float],
) -> ScoringModel:
    all_scores = [*home_scores, *away_scores]
    if not all_scores:
        return ScoringModel({}, None, None, None, None)
    league_average = statistics.fmean(all_scores)
    home_average = _recency_weighted_mean(home_scores)
    away_average = _recency_weighted_mean(away_scores)
    score_std = (
        statistics.pstdev(all_scores)
        if len(all_scores) >= SCORE_UNCERTAINTY_MIN_SCORES
        else None
    )
    teams = {}
    for team, stats in team_scores.items():
        count = len(stats["scored"])
        overall_weight = count / (count + SCORE_PRIOR_GAMES)
        home_count = len(stats["home_scored"])
        home_weight = home_count / (home_count + SCORE_PRIOR_GAMES)
        away_count = len(stats["away_scored"])
        away_weight = away_count / (away_count + SCORE_PRIOR_GAMES)
        teams[team] = {
            "scored": overall_weight * _recency_weighted_mean(stats["scored"])
            + (1 - overall_weight) * league_average,
            "allowed": overall_weight * _recency_weighted_mean(stats["allowed"])
            + (1 - overall_weight) * league_average,
            "home_scored": (
                home_weight * _recency_weighted_mean(stats["home_scored"])
                + (1 - home_weight) * home_average
            ) if home_count else home_average,
            "home_allowed": (
                home_weight * _recency_weighted_mean(stats["home_allowed"])
                + (1 - home_weight) * away_average
            ) if home_count else away_average,
            "away_scored": (
                away_weight * _recency_weighted_mean(stats["away_scored"])
                + (1 - away_weight) * away_average
            ) if away_count else away_average,
            "away_allowed": (
                away_weight * _recency_weighted_mean(stats["away_allowed"])
                + (1 - away_weight) * home_average
            ) if away_count else home_average,
            "games": count,
        }
    return ScoringModel(
        teams,
        league_average,
        home_average,
        away_average,
        score_std,
    )


def _metrics(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {
            "games": 0,
            "decisive_games": 0,
            "accuracy": None,
            "brier_score": None,
            "raw_brier_score": None,
            "baseline_brier_score": None,
            "log_loss": None,
            "baseline_log_loss": None,
            "score_mae": None,
            "baseline_score_mae": None,
            "calibration_error": None,
        }
    if not predictions:
        return {
            "games": 0,
            "decisive_games": 0,
            "accuracy": None,
            "brier_score": None,
            "raw_brier_score": None,
            "baseline_brier_score": None,
            "log_loss": None,
            "baseline_log_loss": None,
            "score_mae": None,
            "baseline_score_mae": None,
            "calibration_error": None,
        }

    outcomes = [float(item["home_outcome"]) for item in predictions]
    probabilities = [float(item["home_probability"]) for item in predictions]
    raw_probabilities = [
        float(item["raw_home_probability"]) for item in predictions
    ]
    raw_brier = statistics.fmean(
        (probability - outcome) ** 2
        for probability, outcome in zip(raw_probabilities, outcomes)
    )
    brier = statistics.fmean(
        (probability - outcome) ** 2
        for probability, outcome in zip(probabilities, outcomes)
    )
    baseline_brier = statistics.fmean((0.5 - outcome) ** 2 for outcome in outcomes)

    def log_loss(probability: float, outcome: float) -> float:
        probability = min(1 - 1e-9, max(1e-9, probability))
        return -(
            outcome * math.log(probability)
            + (1 - outcome) * math.log(1 - probability)
        )

    decisive = [
        item for item in predictions if item["home_outcome"] in {0.0, 1.0}
    ]
    accuracy = (
        sum(
            (item["home_probability"] >= 0.5)
            == (item["home_outcome"] == 1.0)
            for item in decisive
        ) / len(decisive)
        if decisive else None
    )
    model_score_errors = [
        abs(item["home_projected_score"] - item["actual_home_score"])
        for item in predictions
        if item["home_projected_score"] is not None
    ] + [
        abs(item["away_projected_score"] - item["actual_away_score"])
        for item in predictions
        if item["away_projected_score"] is not None
    ]
    baseline_score_errors = [
        abs(item["baseline_home_score"] - item["actual_home_score"])
        for item in predictions
        if item["baseline_home_score"] is not None
    ] + [
        abs(item["baseline_away_score"] - item["actual_away_score"])
        for item in predictions
        if item["baseline_away_score"] is not None
    ]
    bin_errors = []
    for bin_index in range(5):
        bin_items = [
            (probability, outcome)
            for probability, outcome in zip(probabilities, outcomes)
            if min(int(probability * 5), 4) == bin_index
        ]
        if bin_items:
            bin_errors.append(
                len(bin_items) / len(predictions)
                * abs(
                    statistics.fmean(p for p, _ in bin_items)
                    - statistics.fmean(y for _, y in bin_items)
                )
            )
    return {
        "games": len(predictions),
        "decisive_games": len(decisive),
        "accuracy": accuracy,
        "brier_score": brier,
        "raw_brier_score": raw_brier,
        "baseline_brier_score": baseline_brier,
        "log_loss": statistics.fmean(
            log_loss(probability, outcome)
            for probability, outcome in zip(probabilities, outcomes)
        ),
        "baseline_log_loss": statistics.fmean(
            log_loss(0.5, outcome) for outcome in outcomes
        ),
        "score_mae": (
            statistics.fmean(model_score_errors) if model_score_errors else None
        ),
        "baseline_score_mae": (
            statistics.fmean(baseline_score_errors)
            if baseline_score_errors else None
        ),
        "calibration_error": sum(bin_errors),
    }


def walk_forward_backtest(
    sport: str,
    history: Sequence[Mapping[str, Any]],
    min_train_games: int = 20,
    evaluation_start: date | None = None,
) -> dict[str, Any]:
    """Evaluate forecasts in an optional period after chronological warm-up."""
    if sport not in SPORTS:
        raise ValueError(f"Unsupported sport: {sport}")
    if (
        isinstance(min_train_games, bool)
        or not isinstance(min_train_games, int)
        or min_train_games < 1
    ):
        raise ValueError("Minimum training games must be a positive integer.")
    if evaluation_start is not None and (
        not isinstance(evaluation_start, date)
        or isinstance(evaluation_start, datetime)
    ):
        raise ValueError("Evaluation start must be a calendar date.")
    games = _completed_games(history)
    ratings: dict[str, float] = {}
    games_played: dict[str, int] = {}
    last_played: dict[str, date] = {}
    team_scores: dict[str, dict[str, list[float]]] = {}
    home_scores: list[float] = []
    away_scores: list[float] = []
    calibration_examples: list[dict[str, float]] = []
    predictions: list[dict[str, Any]] = []

    by_day: dict[date, list[dict[str, Any]]] = {}
    for game in games:
        by_day.setdefault(game["Date"].date(), []).append(game)

    prior_games = 0
    info = SPORTS[sport]
    for game_day in sorted(by_day):
        day_predictions: list[dict[str, Any]] = []
        scoring_model = _scoring_snapshot(
            team_scores, home_scores, away_scores
        )
        if prior_games >= min_train_games:
            bias = (
                fit_calibration_bias(calibration_examples)
                if sport in {"NFL", "MLB"} else 0.0
            )
            for game in by_day[game_day]:
                home = str(game["Home Team"])
                away = str(game["Away Team"])
                home_key = _team_key(game, "Home")
                away_key = _team_key(game, "Away")
                raw_probability = expected_score(
                    _stabilize_rating(
                        ratings.get(home_key, 1500.0),
                        games_played.get(home_key, 0),
                    ) + info.home_advantage,
                    _stabilize_rating(
                        ratings.get(away_key, 1500.0),
                        games_played.get(away_key, 0),
                    ),
                )
                probability = _apply_calibration(raw_probability, bias)
                home_score, away_score, _ = _score_projection(
                    home, away, scoring_model
                )
                outcome = _home_outcome(game["Home Score"], game["Away Score"])
                item = {
                    "Event ID": game["Event ID"],
                    "Date": game["Date"],
                    "Home Team": home,
                    "Away Team": away,
                    "raw_home_probability": raw_probability,
                    "home_probability": probability,
                    "home_outcome": outcome,
                    "home_projected_score": home_score,
                    "away_projected_score": away_score,
                    "baseline_home_score": scoring_model.home_average,
                    "baseline_away_score": scoring_model.away_average,
                    "actual_home_score": float(game["Home Score"]),
                    "actual_away_score": float(game["Away Score"]),
                }
                day_predictions.append(item)

        _update_ratings_for_day(
            sport,
            ratings,
            games_played,
            last_played,
            by_day[game_day],
        )
        for game in by_day[game_day]:
            _increment_score_state(
                game, team_scores, home_scores, away_scores
            )
            prior_games += 1

        if evaluation_start is None or game_day >= evaluation_start:
            predictions.extend(day_predictions)
        calibration_examples.extend(
            {
                "raw_home_probability": item["raw_home_probability"],
                "home_outcome": item["home_outcome"],
            }
            for item in day_predictions
        )

    return {
        "metrics": _metrics(predictions),
        "predictions": predictions,
        "minimum_training_games": min_train_games,
    }
