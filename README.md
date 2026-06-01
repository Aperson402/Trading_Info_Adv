# Trading Intel Agent

Async Python system that monitors 8 news sources every 3 minutes for oil and gold trading intelligence, classifies each item with Claude, and delivers real-time alerts plus a daily morning brief via Telegram.

## What it does

- **Monitors** 8 sources concurrently every 3 minutes — RSS feeds, HTML scrapes, government releases
- **Classifies** every new item using Claude Haiku: instrument (oil/gold/both/neither), direction (bullish/bearish/neutral), urgency, confidence 1–10, one-line reasoning
- **Weights** classifier confidence by source reliability (historical signal rate builds over time)
- **Tracks** 24-hour sentiment ratios per instrument (e.g. 8/10 bearish oil = different environment than 5/5)
- **Delivers** enriched Telegram alerts with live market context (price, BB, Stoch, RSI, regime)
- **Generates** a daily Claude Sonnet morning brief synthesising news + technicals + COT + DXY + sentiment + economic calendar
- **Alerts** before and after high-impact economic events (forecast → actual with beat/miss)
- **Monitors** price levels every minute and fires alerts when crossed
- **Tracks** trades with auto-close on SL/TP and P&L history

## Architecture

```
├── main.py           — Entry point, scheduler, Telegram command polling
├── monitor.py        — Runs all sources concurrently
├── sources.py        — One async function per data source
├── classifier.py     — Claude Haiku relevance + direction classifier
├── market_context.py — Live price, BB, Stoch, RSI, regime, DXY via yfinance
├── morning_brief.py  — Claude Sonnet daily synthesis
├── advice.py         — Claude Sonnet on-demand real-time advice
├── telegram_bot.py   — Message formatting and Telegram delivery
├── database.py       — SQLite via aiosqlite (items, trades, sentiment)
├── cot.py            — CFTC COT positioning data
├── econ_calendar.py  — ForexFactory economic calendar
├── config.py         — Env-var configuration
└── requirements.txt
```

## Sources monitored

| Source | Method | Frequency |
|---|---|---|
| CNBC Energy | RSS | Every 3 min |
| OilPrice.com | RSS | Every 3 min |
| Financial Times | RSS | Every 3 min |
| AP News Energy | HTML scrape | Every 3 min |
| World Oil | RSS | Every 3 min |
| State Dept Press Releases | HTML scrape | Every 3 min |
| IAEA Press Releases | HTML scrape | Every 3 min |
| EIA Weekly Petroleum Report | HTML scrape | Weekly (on publish) |

## Telegram commands

| Command | Description |
|---|---|
| `/brief` | Generate morning brief on demand |
| `/advice` | Real-time market analysis with specific entry levels |
| `/calendar` | High-impact USD economic events for the next 7 days |
| `/setoil 91` | Alert when WTI crosses $91 |
| `/setgold 2500` | Alert when gold crosses $2500 |
| `/trade long oil 89.50 sl:88.00 tp:91.50` | Log a trade |
| `/trades` | Open positions with live P&L + recent closed trades |
| `/close oil` | Manually close all open oil trades at current price |

## Quick start

### Prerequisites

- Python 3.11+
- Telegram bot token — create via [@BotFather](https://t.me/BotFather)
- Telegram chat ID — send a message to your bot then check `https://api.telegram.org/bot<TOKEN>/getUpdates`
- Anthropic API key — [console.anthropic.com](https://console.anthropic.com)

### Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
ANTHROPIC_API_KEY=your_anthropic_key

# Optional — defaults shown
DATABASE_PATH=trading_intel.db
MONITOR_INTERVAL_MINUTES=3
MORNING_BRIEF_HOUR=6
MORNING_BRIEF_MINUTE=45
MORNING_BRIEF_LOOKBACK_HOURS=12
REQUEST_TIMEOUT_SECONDS=30
LOG_LEVEL=INFO
```

### Run

```bash
python main.py
```

On startup the agent will initialise the database, run an immediate monitor cycle, and start the scheduler.

---

## Deployment

### Railway

1. Push repo to GitHub (confirm `.env` is in `.gitignore`)
2. New Railway project → **Deploy from GitHub repo**
3. Add environment variables: `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, `ANTHROPIC_API_KEY`
4. Start command: `python main.py`
5. Add a **Volume** mounted at `/data` and set `DATABASE_PATH=/data/trading_intel.db` to persist the database across deploys

### Render

1. New **Background Worker** service
2. Build command: `pip install -r requirements.txt`
3. Start command: `python main.py`
4. Add environment variables under **Environment**
5. Add a **Render Disk** and set `DATABASE_PATH` to a path on that disk

---

## Adding a source

1. Open `sources.py`
2. Write `async def fetch_<name>(session: aiohttp.ClientSession) -> list[dict]`
   - Use `_process_rss_feed()` for RSS feeds
   - Use `_fetch_text()` + BeautifulSoup for HTML scrapes
   - Call `await is_seen(url)` before adding, `await mark_seen(...)` after
3. Append to `ALL_SOURCES` at the bottom

No other changes needed.
