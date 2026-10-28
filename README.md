# AI Sports Analytics

A Streamlit dashboard for team ratings, game estimates, and scoreboards across NFL, NBA, MLB, and NHL. It uses ESPN's public scoreboard endpoints and does not require paid API subscriptions.

## Run in GitHub Codespaces

From the repository terminal:

```bash
uv sync
uv run streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Open the forwarded port 8501 from the Codespaces **Ports** tab. The dashboard does not require Google OAuth credentials or a `.streamlit/secrets.toml` file.

## What the dashboard does

- Retrieves the current ESPN scoreboard for the selected sport.
- Retrieves completed games in short date-range chunks to reduce ESPN HTTP 400 errors from long date-range requests.
- Builds Elo-style team ratings from the selected historical window.
- Produces basic pre-game winner estimates using team ratings and a home advantage adjustment.
- Allows CSV downloads for team ratings, game estimates, and scoreboard results.
- Caches current scoreboard data for 5 minutes and historical data for 30 minutes; use **Refresh data** to clear the caches.

## Important limitations

These are prototype estimates, not validated betting or outcome predictions. The model currently does not include player-level statistics, injuries, confirmed lineups, weather, or advanced team metrics. ESPN's public endpoints can change or return incomplete historical coverage. The app does not use paid APIs.
