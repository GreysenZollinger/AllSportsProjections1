# Four-Sport Forecasting Reliability Plan

## Current build

The active development branch is `build/complete-dashboard`, based on
`accuracy/score-projection-backtest`. Work is kept separate from the existing
working branch until the new build passes its checks and is reviewed.

The application uses Streamlit, ESPN's free public scoreboard endpoints, the
four-sport forecasting library in `sports_forecasting.py`, and local SQLite
forecast history in `prediction_store.py`. No paid API subscription is
required.

## Implemented

1. **Validated data ingestion**
   - Validate ESPN response shapes, event identifiers, dates, team roles,
     game states, and numeric scores.
   - Keep unavailable scores missing instead of converting them to zero.
   - De-duplicate event IDs and exclude identified preseason, spring-training,
     and exhibition results from model fitting.
   - Fetch history in one-day requests to avoid unsupported long date ranges;
     retry temporary request failures and show a clear data error when a
     response cannot be trusted.

2. **Sport-aware team and score estimates**
   - Separate Elo configuration for NFL, NBA, MLB, and NHL.
   - Stable team IDs when ESPN provides them, same-day batch rating updates,
     inactivity regression, and small-sample shrinkage.
   - Venue-aware score estimates from historical home/away scoring and
     opponent scoring allowed.
   - Approximate score ranges are withheld until there is enough historical
     score data; the ranges are not yet empirically coverage-calibrated.

3. **Chronological backtesting**
   - Expanding-window walk-forward evaluation forms each day's forecasts before
     incorporating that day's final results.
   - Probability metrics include decisive-game accuracy, Brier score, log loss,
     and calibration error.
   - Score MAE is measured per team score and compared with a baseline only on
     matching model/baseline score observations.
   - The dashboard labels sample sizes and explains that historical results do
     not guarantee future performance.

4. **First-forecast tracking**
   - Store the first pregame forecast per event and model version in SQLite.
   - Attach final scores later without overwriting the original forecast.
   - Migrate older databases additively when the model-version column is
     missing.
   - Saved forecast tracking is separate from historical walk-forward testing.

5. **Dashboard**
   - Overview, team ratings, game estimates, and accuracy/history sections.
   - One-sport detail selection plus an optional all-four-sports scoreboard.
   - Optional four-sport walk-forward accuracy comparison.
   - CSV exports for ratings, forecasts, scoreboards, backtests, and saved
     forecast history.
   - Cache controls, retrieval timestamp, source notes, and readable error
     messages.

6. **Automated verification**
   - Unit tests cover parsing, data failures, duplicates, ratings, score
     estimates, calibration, chronological leakage, paired score MAE, and
     SQLite persistence.
   - Streamlit smoke tests use mocked data and do not depend on live ESPN.
   - GitHub Actions uses Python 3.14 and the locked `uv` environment.

## Known limitations / not yet complete

- ESPN is an unofficial public source with no service guarantee. Historical
  coverage can vary by league and season.
- Historical ingestion currently requests dates individually. A 180-day
  window can take time, especially when ESPN is slow.
- The current forecasts use team-level history. Confirmed injuries, player
  availability, starting lineups, player-level stats, weather, travel,
  rest, and richer sport-specific statistics are not included.
- In-game scoreboards are displayed, but the app does not yet generate
  validated live in-game forecasts.
- Probability calibration is applied only to NFL and MLB and is based on
  prior out-of-sample forecasts. Calibration quality still needs larger,
  recent samples.
- Score uncertainty ranges are approximate and have not yet been validated
  for empirical coverage.
- Saved forecasts live in the local SQLite file. They persist where that file
  persists; a fresh hosted container may not retain local files.
- Model quality is not yet consistently better than simple baselines across
  all sports. Do not claim a proven accuracy edge.

## Baseline evaluation snapshot

A historical evaluation recorded on 2026-10-28 used a 180-day warm-up and
60-day evaluation window. Reported Brier scores were NFL .2494 vs .2468
baseline, NBA .2673 vs .2500, MLB .2441 vs .2483, and NHL .2510 vs .2500.
Reported score MAE was NFL 7.202 vs 7.070, NBA 10.847 vs 10.562, MLB 2.402
vs 2.433, and NHL 1.432 vs 1.434. Only MLB beat both reported baseline
families in that snapshot. Results are noisy, historical, and not evidence
of durable future accuracy.

## Next priorities

1. Get both GitHub Actions workflows green on the current branch.
2. Run the Streamlit app against live ESPN data and check all four sports.
3. Verify saved forecasts settle correctly after games finish and survive an
   app restart in the user's Codespace.
4. Expand player/team context only when reliable, free sources are available.
5. Tune each sport only through out-of-sample backtests, retaining a change
   only when it improves the appropriate metrics against baselines.
