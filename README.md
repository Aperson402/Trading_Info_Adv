# Trading Intel Agent — Phase 1

Async Python system that monitors primary sources for oil and gold trading
intelligence and delivers alerts via Telegram.

## Architecture

```
trading-intel/
├── main.py          # Entry point, scheduler setup
├── monitor.py       # Runs all sources concurrently
├── sources.py       # One async function per data source
├── telegram_bot.py  # Message formatting and Telegram delivery
├── database.py      # SQLite deduplication via aiosqlite
├── config.py        # Env-var config
├── requirements.txt
├── .env.example
└── README.md
```

## Sources monitored

| Source | Method |
|---|---|
| EIA Petroleum Supply Weekly | HTML scrape |
| OPEC Press Releases | HTML scrape |
| State Dept Middle East Briefings | RSS feed |
| IAEA Press Releases | RSS feed |
| Reuters Business News | RSS feed |
| AP News | RSS feed |
| Baker Hughes Rig Count | HTML scrape |

## Quick start (local)

### 1. Prerequisites

- Python 3.11+
- A Telegram bot token — create one via [@BotFather](https://t.me/BotFather)
- Your Telegram chat ID — send a message to your bot then call
  `https://api.telegram.org/bot<TOKEN>/getUpdates`

### 2. Install dependencies

```bash
cd trading-intel
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID at minimum
```

### 4. Run

```bash
python main.py
```

On startup the system will:
1. Initialise the SQLite database
2. Run an immediate full-source monitor cycle
3. Schedule monitor cycles every 10 minutes
4. Schedule the morning brief at 06:45 UTC daily

Logs are written to stdout. You should see output like:

```
2025-01-15 06:00:00  INFO      __main__ — === Trading Intel Agent — Phase 1 starting ===
2025-01-15 06:00:00  INFO      database — Database initialised at trading_intel.db
2025-01-15 06:00:00  INFO      __main__ — ▶  Monitor cycle started
2025-01-15 06:00:02  INFO      monitor — [Reuters] returned 3 new item(s)
2025-01-15 06:00:02  INFO      monitor — [IAEA] returned 1 new item(s)
...
2025-01-15 06:00:05  INFO      __main__ — Monitor cycle complete — 4 new item(s)
```

---

## Deploying to Railway

1. Push your repo to GitHub (ensure `.env` is in `.gitignore`).
2. Create a new Railway project → **Deploy from GitHub repo**.
3. In **Variables**, add `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`.
4. Railway auto-detects Python. Set the **Start Command**:
   ```
   python main.py
   ```
5. Deploy. Railway will restart the service automatically on crashes.

### Persistent SQLite on Railway

Railway's filesystem is ephemeral. To persist the seen-items database between
deploys, use a Railway Volume:

1. Railway project → **Add Volume** → mount path `/data`
2. Set env var `DATABASE_PATH=/data/trading_intel.db`

---

## Deploying to Render

1. Push repo to GitHub.
2. Create a new **Background Worker** service (not a Web Service).
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `python main.py`
5. Add environment variables under **Environment**.
6. For persistence, add a **Render Disk** and set `DATABASE_PATH` to a path
   on that disk (e.g. `/data/trading_intel.db`).

---

## Adding a new source

1. Open `sources.py`.
2. Write an async function `fetch_<name>(session: aiohttp.ClientSession) -> list[dict]`.
   - Use `_process_rss_feed()` for RSS/Atom feeds.
   - Use `_fetch_text()` + BeautifulSoup for HTML scraping.
   - Call `await is_seen(url)` before adding an item.
   - Call `await mark_seen(...)` after adding.
3. Append the function to `ALL_SOURCES` at the bottom of the file.

No changes needed anywhere else.

---

## Phase 2 preview

Phase 2 will add Claude-powered relevance classification: each new item will be
scored for relevance to oil/gold trading, and only high-signal items will trigger
Telegram alerts, with a synthesised summary.
