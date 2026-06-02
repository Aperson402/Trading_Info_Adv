# Trading Intel Agent

Async Python system that monitors 14 news sources every 10 minutes for oil and gold trading intelligence, classifies each item with Claude, and delivers real-time alerts, a daily morning brief, on-demand advice, and deep drill analysis via Telegram.

## What it does

- **Monitors** 14 sources concurrently — RSS feeds, HTML scrapes, government releases
- **Classifies** every new item using Claude Haiku: instrument, direction, urgency, confidence 1–10, one-line reasoning
- **Weights** classifier confidence by source reliability (historical signal rate builds over time)
- **Batches** alerts — confidence ≥8 fires immediately; 6–7 goes into a grouped digest
- **Tracks** 24-hour sentiment ratios per instrument
- **Delivers** enriched Telegram alerts with live market context (price, BB, Stoch, RSI, regime, signal grade)
- **Generates** a daily Claude Sonnet morning brief synthesising news + technicals + COT + FRED + DXY + sentiment + economic calendar
- **Advises** on demand with specific entry levels via `/advice`
- **Drills** deep into a single instrument with multi-tool Claude analysis via `/drill`
- **Watches** for specific conditions set by drill and sends periodic status updates
- **Alerts** before and after high-impact economic events with beat/miss reaction analysis
- **Monitors** price levels every minute and fires alerts when crossed
- **Tracks** trades with auto-close on SL/TP and P&L history
- **Records** signal outcomes (2h and 4h resolution) and reports accuracy via `/signals`
- **Pulls** FRED macro data (real yields, breakeven inflation, GLD ETF flows) daily
- **Analyses** WTI futures curve structure (contango vs backwardation)
- **Filters** signals against the weekly 20-SMA trend, downgrading counter-trend setups

## Architecture

```
├── main.py            — Entry point, scheduler, Telegram command polling
├── monitor.py         — Runs all sources concurrently
├── sources.py         — 14 async source functions
├── classifier.py      — Claude Haiku relevance + direction classifier
├── event_classifier.py — Post-event reaction classification
├── market_context.py  — Live price, BB, Stoch, RSI, regime, DXY, weekly trend via yfinance
├── morning_brief.py   — Claude Sonnet daily synthesis
├── advice.py          — Claude Sonnet on-demand real-time advice
├── drill.py           — Two-phase tool-use deep-dive analysis
├── ib_orderbook.py    — Interactive Brokers Level 2 order book snapshots
├── fred.py            — FRED macro data (real yields, ETF flows)
├── cot.py             — CFTC COT positioning data
├── econ_calendar.py   — ForexFactory economic calendar
├── telegram_bot.py    — Message formatting and Telegram delivery
├── database.py        — SQLite via aiosqlite (items, trades, signals, watches)
├── config.py          — Env-var configuration
└── requirements.txt
```

## Sources monitored

| Source | Method | Focus |
|---|---|---|
| OilPrice.com | RSS | Oil, OPEC, Iran, Saudi |
| Financial Times | RSS | Macro, markets |
| CNBC Energy | RSS | Breaking commodity news |
| World Oil | RSS | Upstream, drilling, OPEC |
| Mining.com | RSS | Gold price drivers, central bank buying |
| Arab News | RSS | Gulf energy, Saudi position |
| TASS English | RSS | Russia energy exports, OPEC+ Russia |
| BBC World | RSS | Geopolitical, Middle East |
| MarketWatch | RSS | Fed/rates reactions, dollar moves |
| EIA Weekly Report | HTML scrape | Petroleum supply/demand data |
| AP News Energy | HTML scrape | Energy hub articles |
| State Dept | HTML scrape | Sanctions, Iran, geopolitical |
| US Treasury | HTML scrape | OFAC sanctions, dollar policy |
| Federal Reserve | RSS | FOMC statements, rate decisions |

## Telegram commands

| Command | Description |
|---|---|
| `/brief` | Generate morning brief on demand |
| `/advice` | Real-time market analysis with specific entry/SL/TP levels |
| `/drill oil` | Deep-dive analysis on oil with multi-tool Claude |
| `/drill gold overnight` | Drill with timeframe focus (see below) |
| `/signals` | Signal accuracy report for the last 30 days |
| `/watches` | List active watch conditions |
| `/cancelwatch <id>` | Cancel a watch by ID |
| `/calendar` | High-impact USD events for the next 7 days |
| `/setoil 91` | Alert when WTI crosses $91 |
| `/setgold 2500` | Alert when gold crosses $2500 |
| `/trade long oil 89.50 sl:88.00 tp:91.50` | Log a trade |
| `/trades` | Open positions with live P&L + recent closed |
| `/close oil` | Manually close all open oil trades at current price |
| `/monitor` | Trigger an immediate monitor scan |
| `/log` | Tail the last 50 lines of the log file |

## Drill command

`/drill` runs a multi-phase Claude Sonnet analysis on a single instrument. Claude decides what additional data it needs, fetches it, then produces a structured condition check with probability.

### Timeframe focus

Append a timeframe to direct Claude's attention to a specific period:

```
/drill oil              # default 1H structure
/drill oil 5m           # 5-minute scalping
/drill oil 15m          # 15-minute intraday
/drill oil 30m          # 30-minute swing
/drill oil 2h           # last 2 hours of action
/drill oil 4h           # 4-hour swing structure
/drill oil 1d           # daily chart macro
/drill oil 1w           # weekly macro context
/drill oil overnight    # overnight session (yesterday's close → now)
/drill gold overnight   # same for gold
```

### Tools available to Claude during drill

| Tool | What it fetches |
|---|---|
| `fetch_market_data` | OHLCV for any Yahoo Finance ticker — correlates, ETFs, FX, bonds |
| `compute_spread_ratio` | Ratio or spread between two tickers with z-score (Brent/WTI, GSR, GLD/GDX) |
| `fetch_implied_volatility` | ^OVX / ^GVZ / ^VIX with 30-day percentile and implied daily move |
| `fetch_order_book` | Live IB Level 2 depth — bid/ask walls, spread, order imbalance |
| `set_watch` | Register a condition to monitor with periodic updates |

### Watch system

If Claude identifies a specific unresolved condition (e.g. "price breaks above $73.20 with RSI > 52"), it calls `set_watch`. The system checks every 15–60 minutes using a fast Claude Haiku evaluation and sends:
- **Status updates** — current values vs. the condition, time remaining
- **Triggered alert** — fires when the condition is met, with a prompt to re-drill

## FRED macro data

Pulled daily from the St. Louis Fed (no API key required):
- 10-year real yield (DFII10) — rising real yields are bearish gold
- 5-year breakeven inflation (T5YIE) — rising breakeven = inflation bid = gold positive
- GLD ETF market cap → estimated tonnes held, day-over-day flow delta

## Interactive Brokers order book

`/drill` can request a live Level 2 snapshot from IB Gateway / TWS. Returns:
- Best bid/ask and spread
- Order imbalance (bid vs ask lots, flags BID-HEAVY / ASK-HEAVY at ≥60%)
- Walls — price levels with >2× mean resting size, labelled as support/resistance
- Full depth table (up to 5 levels per side)

Requires IB Gateway or TWS running locally with the API enabled and a CME/COMEX real-time market data subscription.

## Weekly trend filter

Market context fetches a 20-week SMA alongside 1H and 4H data. When a LONG/SHORT signal opposes the weekly trend, the signal grade is downgraded (`STRONG→MODERATE`, `MODERATE→WEAK`) and tagged `↓WEEKLY`. The weekly trend and % distance from 20W SMA appear in all prompts.

## Signal outcome tracking

Every LONG/SHORT signal from `/advice` or the morning brief is recorded with entry price. Two resolution windows (2h and 4h) auto-check whether price moved in the signal direction. `/signals` shows:
- Hit rate by instrument and direction for the last 30 days
- Last 8 signals with entry price, 2h and 4h outcomes (✅/❌/⏳)

## Quick start

### Prerequisites

- Python 3.11+
- Telegram bot token — create via [@BotFather](https://t.me/BotFather)
- Telegram chat ID — send a message to your bot then check `https://api.telegram.org/bot<TOKEN>/getUpdates`
- Anthropic API key — [console.anthropic.com](https://console.anthropic.com)
- IB Gateway or TWS (optional, for order book in drill)

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
MONITOR_INTERVAL_MINUTES=10
MORNING_BRIEF_HOUR=6
MORNING_BRIEF_MINUTE=45
MORNING_BRIEF_LOOKBACK_HOURS=12
REQUEST_TIMEOUT_SECONDS=30
LOG_LEVEL=INFO

# Interactive Brokers (optional — required for order book in /drill)
IB_HOST=127.0.0.1
IB_PORT=7497          # 7497=TWS paper, 7496=TWS live, 4002=Gateway paper, 4001=Gateway live
IB_CLIENT_ID=99
```

### Run

```bash
python main.py
```

On startup the agent initialises the database, runs an immediate monitor cycle, and starts the scheduler.

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
