# AI Sports Analytics

A Streamlit dashboard for NFL, NBA, MLB, and NHL team ratings, pregame estimates, score projections, and accuracy tracking. It prioritizes free public data sources and does not require paid API subscriptions.

## Run in GitHub Codespaces

Open the repository in Codespaces, then run:

```bash
uv sync --locked
uv run streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Open port **8501** from the Codespaces **Ports** tab.

## Dashboard sections

- **Overview:** Current ESPN scoreboard and game status.
- **Team ratings:** Sport-specific Elo-style ratings, records, recent form, and rating support.
- **Game forecasts:** Pregame win-probability estimates and projected scores based on historical team results.
- **Accuracy & history:** Chronological walk-forward backtests, baseline comparisons, first-seen saved forecasts, and completed outcomes.

## Data and refresh behavior

- Data comes from ESPN's public, unofficial scoreboard endpoints.
- Current scoreboard results are cached for 5 minutes.
- Completed-game history is cached for 30 minutes.
- The sidebar's **Refresh sports data** button clears the caches.
- The historical window can be changed from 60 to 180 days.
- Forecast history is stored in a local SQLite database named `prediction_history.sqlite3`. Set `PREDICTION_DB_PATH` to choose another database location.

## Accuracy methodology

Historical backtests use a chronological walk-forward process: a game's forecast is formed before that day's results are added to model state. Score MAE and its baseline are compared only on the same available team-score observations. The dashboard also records the first pregame forecast for each event and attaches the final result later, so repeated refreshes do not overwrite the original forecast.

Accuracy is not guaranteed. Results depend on sample size, source coverage, and model assumptions. A historical backtest is not proof of future performance.

## Important limitations

This is a prototype sports analytics project, not a validated prediction system. The current models use team-level historical results and do not yet include confirmed injuries, starting lineups, player-level statistics, weather, or all sport-specific advanced factors. ESPN's public endpoints are unofficial, can change without notice, and may have incomplete historical coverage. Score ranges, when shown by the forecasting library, are approximate and are not yet validated for coverage.

## Tests

Run the automated test suite with:

```bash
uv run python -m unittest discover -s tests -v
```
