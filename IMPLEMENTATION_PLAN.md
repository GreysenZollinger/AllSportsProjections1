# Four-Sport Forecasting Reliability Plan

## Repository audit

- The application is currently a single Streamlit script, `streamlit_app.py`.
- It reads ESPN's free, unofficial scoreboard API; the endpoint has no documented
  service-level guarantee in this project.
- Historical request exceptions are silently ignored, and missing scores are
  converted to zero. Historical games are not consistently validated or
  de-duplicated.
- Two unreachable duplicate implementations follow `return` statements.
- All four sports use one Elo-style win model. Ties are treated as losses, score
  forecasts are unshrunk means, and the app has no uncertainty, evaluation, or
  persistent forecast history.
- There are no automated tests or project-specific user/data-source docs.
- `streamlit_app.py` was already modified and three backup snapshots were
  already untracked when work began. They are preserved and are not to be
  overwritten. Work is isolated on branch
  `forecasting/reliable-four-sport-app`.

## Prioritized implementation and status

1. **P0 — Reliable ingestion and parsing — complete**
   - Validate API response shapes, event identifiers, UTC dates, team roles,
     states, and numeric scores.
   - Keep unavailable scores as missing, include ties, de-duplicate by event ID,
     and surface request/schema failures instead of silently returning partial
     history. Parse ESPN season phases and exclude identified preseason,
     spring-training, and exhibition results from model fitting.
   - Remove unreachable duplicate code and make the app importable/testable
     without running the UI.
2. **P1 — Sport-aware forecasts — complete**
   - Use stable team IDs where available, same-day batch Elo updates, at least
     180 days of dashboard rating history, inactivity regression, and
     small-sample shrinkage alongside per-sport Elo settings.
   - Produce venue-aware scoring estimates with low-sample shrinkage and
     explicitly approximate score ranges; handle teams with no history safely.
3. **P1 — No-leak chronological evaluation — complete**
   - Add expanding-window walk-forward backtests. Every forecast is formed
     before its game is added to model state.
   - Report sample size, accuracy, Brier score, log loss, score MAE, and
     appropriate simple baselines; do not make unsupported accuracy claims.
4. **P2 — Calibration and forecast tracking — complete**
   - Calibrate probabilities only from prior out-of-sample forecasts, falling
     back to raw probabilities when the calibration sample is small.
   - Persist the first pregame forecast and its model version for each event,
     then attach actual results when available; migrate legacy history
     additively without rewriting forecasts.
5. **P2 — Dashboard and documentation — complete**
   - Show data retrieval timestamps, cache windows, rating support, error,
     missing-data, uncertainty, source, and limitation messaging.
   - Replace template README instructions with setup, test, run, model, and
     data-source documentation.
6. **Verification — complete**
   - Add deterministic tests for parsing, failure paths, duplicates, ratings,
     stable IDs, season filtering, same-day order, decay/shrinkage, forecasts,
     calibration, leakage, metrics, and database migration.
   - The 31 unit and mocked dashboard tests pass; Python compilation,
     whitespace checks, and Pylance syntax/diagnostic checks pass.
   - A live ESPN 180-day warm-up / 60-day evaluation run on 2026-10-28 was
     compared with 50% probability and prior-score baselines. It did not show
     consistent gains: calibrated Brier was NFL .2494 vs .2468 baseline,
     NBA .2673 vs .2500, MLB .2441 vs .2483, and NHL .2510 vs .2500.
     Score MAE was NFL 7.202 vs 7.070, NBA 10.847 vs 10.562, MLB 2.402 vs
     2.433, and NHL 1.432 vs 1.434. Only MLB improved on both reported
     baseline families in this window; results are noisy and not a claim of
     durable accuracy.
