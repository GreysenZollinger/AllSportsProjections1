# AI Sports Analytics

A Streamlit dashboard for team ratings and game estimates across NFL, NBA, MLB, and NHL using ESPN's public scoreboard endpoints.

## Run locally

Prerequisites: Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run streamlit run streamlit_app.py
```

The app is intentionally locked until authentication is configured. This prevents an accidental public deployment with no access control.

## Authentication setup

The app uses Streamlit's built-in OpenID Connect (OIDC) login with Google. It does not store user passwords. Google authenticates the user; the app then checks that the user's email is verified and is listed in the app's allowlist.

### 1. Create a Google OAuth client

In Google Cloud Console, create or select a project and configure an OAuth consent screen. Create an OAuth client ID of type **Web application**. Add the correct redirect URI for where the app will run:

- Local development: `http://localhost:8501/oauth2callback`
- Hosted deployment: use the exact HTTPS URL shown/configured by your hosting setup, followed by `/oauth2callback`.

Keep the client secret private. Do not paste it into chat or commit it to GitHub.

### 2. Configure Streamlit secrets

Create `.streamlit/secrets.toml` locally (the folder/file is excluded from Git by `.gitignore`) or enter the same values in your hosting provider's Streamlit secrets settings. Start from [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example), then replace all placeholders.

- `client_id` and `client_secret`: from the OAuth client.
- `redirect_uri`: the exact callback URI registered with Google.
- `cookie_secret`: a long, random, private value used by Streamlit to secure the authentication cookie.
- `allowed_emails`: only the verified Google email addresses permitted to use this app.

Example shape:

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "REPLACE_WITH_A_LONG_RANDOM_SECRET"
client_id = "YOUR_GOOGLE_CLIENT_ID"
client_secret = "YOUR_GOOGLE_CLIENT_SECRET"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

[app_auth]
allowed_emails = ["you@example.com"]
```

Generate a random cookie secret locally; for example, run `python -c "import secrets; print(secrets.token_urlsafe(48))"` and copy the output directly into your local secrets file. Never commit the real `secrets.toml`.

### 3. Run and verify

```bash
uv sync
uv run streamlit run streamlit_app.py
```

Test all of the following:
1. The dashboard is locked when secrets are missing or incomplete.
2. A signed-out visitor sees the Google sign-in button, not sports data.
3. A verified Google email on the allowlist can access the dashboard.
4. A different or unverified email is denied access.
5. Logging out returns the user to the signed-out screen.

### Authentication request flow

1. A visitor opens the Streamlit app.
2. The app checks that OIDC settings and a non-empty email allowlist exist. Otherwise, it stops before showing dashboard data.
3. If the visitor is signed out, `st.login` redirects to Google.
4. Google authenticates the user and redirects to the configured `/oauth2callback` URI.
5. Streamlit validates the OIDC response and establishes its signed authentication session.
6. The app checks the verified email against `[app_auth].allowed_emails`. Only an allowlisted, verified email reaches the dashboard.
7. `st.logout` clears the app's Streamlit authentication session.

### Credentials and tokens

- The app never handles or stores users' Google passwords.
- The OAuth client secret and cookie secret belong in Streamlit secrets, not Python source code or Git.
- OIDC tokens are handled by Streamlit's authentication implementation; this app does not print or persist them.
- The email allowlist is an access policy, not a password database.
- HTTPS should be used for any deployed app. The hosted redirect URI must match the provider configuration exactly.
- GitHub repository permissions are separate from app login. This code protects the Streamlit dashboard; it does not make the GitHub repository private.
- Keep the Google OAuth client secret and cookie secret private. If exposed, rotate them in the relevant provider/hosting settings.

## Model limitations

The estimates are based on completed scoreboard results and a simple Elo-style rating calculation. They are not guarantees and do not currently account for player availability, lineups, weather, or every advanced statistic.
