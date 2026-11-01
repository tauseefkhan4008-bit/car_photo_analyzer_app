# Car Identifier

Upload a photo of a car and get its make, model, approximate year, body type, an estimated used-market price range, key specifications (engine, fuel type, mileage / fuel economy, transmission), pros and cons, and tips for buying a used one.

Built with **Streamlit** (UI) and **Google Gemini Flash** (vision + structured JSON output) through the `google-genai` Python SDK.

## Features

- Upload a JPG or PNG (up to 10 MB). The photo is validated, auto-rotated and downscaled before it is sent.
- Results are shown in a clear layout: headline metrics, price range, and tabs for specifications, pros & cons, used-buying tips, and how the car was identified.
- Choose a market region and currency in the sidebar so prices and specs fit your market.
- Friendly handling of: no image, unreadable or tiny files, photos with no car or an unclear car, a missing or invalid API key, rate limits, timeouts, network problems, blocked or empty responses, and malformed model output.
- The API key is never hardcoded or committed. It comes from Streamlit secrets or an environment variable.

## Project structure

```
car-identifier/
├── app.py             # the whole app
├── requirements.txt   # dependencies
├── README.md
├── .gitignore         # keeps secrets out of Git (see below)
└── .streamlit/
    └── secrets.toml   # LOCAL ONLY, never commit this
```

## Requirements

- Python 3.10 or newer
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Run locally

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key (pick ONE of the options below)

# Option A: Streamlit secrets file (recommended)
mkdir -p .streamlit
echo 'GEMINI_API_KEY = "your-real-key"' > .streamlit/secrets.toml

# Option B: environment variable
export GEMINI_API_KEY="your-real-key"       # Windows PowerShell: $env:GEMINI_API_KEY="your-real-key"

# 4. Start the app
streamlit run app.py
```

The app opens at http://localhost:8501.

## Configuration

| Setting          | Where                    | Required | Purpose                                                                                 |
| ---------------- | ------------------------ | -------- | --------------------------------------------------------------------------------------- |
| `GEMINI_API_KEY` | secrets or env variable  | Yes      | Your Gemini API key. `GOOGLE_API_KEY` is also accepted as an environment variable.     |
| `GEMINI_MODEL`   | secrets or env variable  | No       | Model ID to use. Defaults to `gemini-flash-latest` (Google's alias for the newest stable Flash model). Set it to pin a specific model. |

Secrets are checked first, then environment variables.

## Push to GitHub

```bash
# Make sure secrets never get committed
printf '.streamlit/secrets.toml\n.env\nvenv/\n__pycache__/\n*.pyc\n' > .gitignore

git init
git add .
git status                         # confirm secrets.toml is NOT listed
git commit -m "Initial commit: car identifier app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Create the empty repository on GitHub first (no README or .gitignore, since you already have them).

## Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** and choose to deploy from an existing repo.
3. Select your repository, the `main` branch, and `app.py` as the main file.
4. Open **Advanced settings**, and paste this into **Secrets**:
   ```toml
   GEMINI_API_KEY = "your-real-key"
   ```
   Optionally choose a Python version (3.10 or newer).
5. Click **Deploy**.

To change the key later: open your app, then **Manage app → Settings → Secrets**, edit, and save. The app restarts automatically.

## How errors are handled

| Situation                                | What the user sees                                               |
| ---------------------------------------- | ---------------------------------------------------------------- |
| No image uploaded                        | Prompt to upload one; the Identify button stays disabled         |
| Corrupt, empty, oversized or tiny file   | Specific message; button stays disabled                          |
| No car in photo / photo too unclear      | Explanation from the model plus tips for a better photo          |
| Partly clear photo or low confidence     | Result shown with a "best guess" warning                         |
| Missing or placeholder API key           | Setup instructions; button disabled                              |
| Invalid key / no permission (401, 403)   | "Gemini rejected the API key"                                    |
| Rate limit or quota (429)                | Asks the user to wait and retry                                  |
| Server error (5xx)                       | One automatic retry, then a "try again" message                  |
| Timeout or no connection                 | Clear message; request times out after 60 s                      |
| Blocked / empty / malformed model output | Friendly message with a "Technical details" expander             |

## Assumptions and limitations

- **"Mileage" means fuel economy** (km/l, L/100 km or mpg), not the odometer reading. Odometer advice is part of the buying tips.
- **All figures are AI estimates**, not live market data: identification, year range, specs and especially prices can be wrong. Pricing is only as good as the model's knowledge of your market. Always check listings and a vehicle history report.
- Year is a **range** (generation or facelift), not an exact year, and trim level is not guessed.
- Specs are typical for the identified model and generation; your specific car may differ.
- Photos are sent to Google's Gemini API for analysis. The app does not store them. Do not upload photos with people or licence plates you don't want processed.
- Free-tier API keys have rate limits and Google may use free-tier data to improve its products. Check Google's current terms.

## Tested with

Python 3.12, Streamlit 1.66, google-genai 2.34, Pillow 13, Pydantic 2.14.
