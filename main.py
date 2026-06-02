"""
main.py — entry point.

Phase 2: classifier inserted between monitor and Telegram delivery.
"""

import asyncio
import json
import logging
import logging.handlers
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from telegram import Bot

from advice import generate_advice, parse_advice_signals
from drill import generate_drill
from fred import get_fred_context
from classifier import classify_item
from event_classifier import classify_event_reaction
from morning_brief import generate_morning_brief, parse_brief_signals
from cot import get_cot_data
from econ_calendar import get_week_calendar
from market_context import get_both_contexts, get_correlated_context, get_current_prices
from config import (
    LOG_LEVEL,
    MONITOR_INTERVAL_MINUTES,
    MORNING_BRIEF_HOUR,
    MORNING_BRIEF_LOOKBACK_HOURS,
    MORNING_BRIEF_MINUTE,
    TELEGRAM_TOKEN,
)
from database import (
    advance_watch, cancel_watch, close_trade, create_watch,
    get_active_watches, get_due_watches, get_items_since,
    get_open_trades, get_pending_signals, get_recent_trades,
    get_sentiment_window, get_signal_stats,
    fire_watch, init_db, open_trade, record_signal, resolve_signal, update_classification,
)
from monitor import run_all_sources
from telegram_bot import (
    send_advice, send_batch_digest, send_calendar, send_drill,
    send_event_alert, send_event_reaction, send_log, send_morning_brief,
    send_new_items, send_price_alert, send_signal_stats,
    send_trade_closed, send_trade_opened, send_trades_list,
    send_watch_triggered, send_watch_update, send_watches_list,
)

class _ColorFormatter(logging.Formatter):
    _COLORS = {
        logging.DEBUG:    "\033[37m",      # grey
        logging.INFO:     "\033[32m",      # green
        logging.WARNING:  "\033[33m",      # yellow
        logging.ERROR:    "\033[31m",      # red
        logging.CRITICAL: "\033[1;31m",    # bold red
    }
    _RESET = "\033[0m"
    _DIM   = "\033[2m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, "")
        ts    = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level = f"{color}{record.levelname:<8}{self._RESET}"
        name  = f"{self._DIM}{record.name}{self._RESET}"
        msg   = f"{color}{record.getMessage()}{self._RESET}"
        return f"{ts}  {level}  {name} — {msg}"

_LOG_FILE = Path(__file__).parent / "trading_intel.log"

_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(_ColorFormatter())

_file = logging.handlers.RotatingFileHandler(
    _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

logging.root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logging.root.addHandler(_console)
logging.root.addHandler(_file)
logger = logging.getLogger(__name__)

_TF_ALIASES: dict[str, str] = {
    # 5-minute
    "5m": "5m", "5min": "5m", "5minute": "5m",
    # 15-minute
    "15m": "15m", "15min": "15m", "15minute": "15m",
    # 30-minute
    "30m": "30m", "30min": "30m", "30minute": "30m",
    # 2-hour
    "2h": "2h", "2hr": "2h", "2hour": "2h",
    # 4-hour
    "4h": "4h", "4hr": "4h", "4hour": "4h",
    # Daily
    "1d": "1d", "daily": "1d", "day": "1d",
    # Weekly
    "1w": "1w", "weekly": "1w", "week": "1w",
    # Overnight
    "overnight": "overnight", "on": "overnight", "night": "overnight",
}

def _normalise_timeframe(raw: str) -> str:
    """Map user input like '5M', '2H', 'overnight' to a canonical key used by drill.py."""
    return _TF_ALIASES.get(raw.lower().strip(), "")


# Track which events have already been alerted to avoid duplicates
_alerted_upcoming: set[str] = set()
_alerted_result:   set[str] = set()

# Pre-event price snapshots keyed by event key — used for reaction classification
_event_snapshots: dict[str, dict] = {}

# Price alerts — persisted to disk so restarts don't clear them
_ALERTS_FILE = Path(__file__).parent / "price_alerts.json"

def _load_alerts() -> dict:
    try:
        return json.loads(_ALERTS_FILE.read_text())
    except Exception:
        return {}

def _save_alerts(alerts: dict) -> None:
    _ALERTS_FILE.write_text(json.dumps(alerts, indent=2))

_price_alerts: dict = _load_alerts()  # {"oil": {"target": 91.0, "direction": "above"}, ...}


async def poll_commands(bot: Bot) -> None:
    # Drain any updates that accumulated before startup
    try:
        pending = await bot.get_updates(timeout=0)
        offset = pending[-1].update_id + 1 if pending else 0
    except Exception:
        offset = 0

    while True:
        try:
            updates = await bot.get_updates(offset=offset, timeout=10, allowed_updates=["message"])
            for upd in updates:
                offset = upd.update_id + 1
                msg = upd.message
                if msg and msg.text and msg.text.startswith("/monitor"):
                    await msg.reply_text("⏳ Running monitor scan across all sources…")
                    await job_monitor()
                    await msg.reply_text("✅ Monitor scan complete.")
                elif msg and msg.text and msg.text.startswith("/log"):
                    await send_log(str(_LOG_FILE))
                elif msg and msg.text and msg.text.startswith("/brief"):
                    await msg.reply_text("⏳ Generating brief…")
                    await job_morning_brief()
                elif msg and msg.text and msg.text.startswith("/calendar"):
                    await msg.reply_text("⏳ Fetching calendar…")
                    events = await get_week_calendar()
                    await send_calendar(events)
                elif msg and msg.text and (msg.text.startswith("/setoil") or msg.text.startswith("/setgold")):
                    instrument = "oil" if msg.text.startswith("/setoil") else "gold"
                    raw = msg.text.replace("/setoil", "").replace("/setgold", "").strip("( )")
                    try:
                        target = float(raw)
                    except ValueError:
                        await msg.reply_text(f"⚠️ Couldn't parse price. Use /setoil 91 or /setgold 2500")
                        continue
                    prices = await get_current_prices()
                    current = prices.get(instrument)
                    direction = "above" if (current is None or current < target) else "below"
                    _price_alerts[instrument] = {"target": target, "direction": direction}
                    _save_alerts(_price_alerts)
                    symbol = "🛢" if instrument == "oil" else "🥇"
                    current_str = f"${current:,.2f}" if current else "unknown"
                    await msg.reply_text(
                        f"{symbol} Alert set: {instrument.upper()} @ ${target:,.2f}\n"
                        f"Current price: {current_str} — will fire when price goes {direction} ${target:,.2f}"
                    )
                elif msg and msg.text and msg.text.startswith("/advice"):
                    await msg.reply_text("⏳ Analysing market…")
                    since = datetime.now(timezone.utc) - timedelta(hours=4)
                    items, mkt, sentiment, cot_data, calendar_events, fred_ctx = await asyncio.gather(
                        get_items_since(since),
                        get_both_contexts(),
                        get_sentiment_window(hours=24),
                        get_cot_data(),
                        get_week_calendar(),
                        get_fred_context(),
                    )
                    oil_ctx   = mkt.get("oil")       or {}
                    gold_ctx  = mkt.get("gold")      or {}
                    dxy_ctx   = mkt.get("dxy")       or {}
                    oil_curve = mkt.get("oil_curve") or {}
                    gld_flow  = mkt.get("gld_flow")  or {}
                    advice = await generate_advice(
                        items, oil_ctx, gold_ctx, sentiment, cot_data, calendar_events,
                        dxy_ctx, fred_ctx, oil_curve, gld_flow,
                    )
                    await send_advice(advice)
                    # Record any LONG/SHORT signals for outcome tracking
                    prices = await get_current_prices()
                    for inst, direction in parse_advice_signals(advice).items():
                        if direction and prices.get(inst):
                            sid = await record_signal(inst, direction, "advice", prices[inst])
                            if sid:
                                logger.info("Signal recorded #%d: %s %s @ %.2f", sid, direction.upper(), inst, prices[inst])
                elif msg and msg.text and msg.text.startswith("/drill"):
                    parts = msg.text.split()
                    instrument = parts[1].lower() if len(parts) > 1 else ""
                    if instrument not in ("oil", "gold"):
                        await msg.reply_text(
                            "⚠️ Use: /drill oil [timeframe]  or  /drill gold [timeframe]\n"
                            "Timeframes: 5m  15m  30m  2h  4h  1d  1w  overnight"
                        )
                        continue
                    raw_tf   = parts[2].lower() if len(parts) > 2 else ""
                    timeframe = _normalise_timeframe(raw_tf)
                    tf_label  = f" [{timeframe}]" if timeframe else ""
                    await msg.reply_text(f"🔬 Drilling into {instrument.upper()}{tf_label} — fetching data…")
                    since = datetime.now(timezone.utc) - timedelta(hours=4)
                    items, mkt, corr, calendar_events, fred_ctx = await asyncio.gather(
                        get_items_since(since),
                        get_both_contexts(),
                        get_correlated_context(instrument),
                        get_week_calendar(),
                        get_fred_context(),
                    )
                    ctx       = mkt.get(instrument)   or {}
                    dxy_ctx   = mkt.get("dxy")        or {}
                    oil_curve = mkt.get("oil_curve")  or {}
                    gld_flow  = mkt.get("gld_flow")   or {}
                    drill_text = await generate_drill(
                        instrument, ctx, dxy_ctx, corr, items, calendar_events,
                        fred_ctx=fred_ctx, oil_curve=oil_curve, gld_flow=gld_flow,
                        timeframe=timeframe,
                    )
                    await send_drill(instrument, drill_text, timeframe=timeframe)
                elif msg and msg.text and msg.text.startswith("/signals"):
                    stats = await get_signal_stats(days=30)
                    await send_signal_stats(stats)
                elif msg and msg.text and msg.text.startswith("/watches"):
                    watches = await get_active_watches()
                    await send_watches_list(watches)
                elif msg and msg.text and msg.text.startswith("/cancelwatch"):
                    parts = msg.text.split()
                    try:
                        watch_id = int(parts[1])
                    except (IndexError, ValueError):
                        await msg.reply_text("⚠️ Use: /cancelwatch <id>  (see /watches for IDs)")
                        continue
                    ok = await cancel_watch(watch_id)
                    if ok:
                        await msg.reply_text(f"✅ Watch #{watch_id} cancelled.")
                    else:
                        await msg.reply_text(f"⚠️ Watch #{watch_id} not found or already done.")
                elif msg and msg.text and msg.text.startswith("/trade "):
                    parts = msg.text.split()
                    try:
                        direction  = parts[1].lower()
                        instrument = parts[2].lower()
                        entry      = float(parts[3])
                        sl = next((float(p.split(":")[1]) for p in parts if p.startswith("sl:")), None)
                        tp = next((float(p.split(":")[1]) for p in parts if p.startswith("tp:")), None)
                        if direction not in ("long", "short") or instrument not in ("oil", "gold"):
                            raise ValueError
                    except (IndexError, ValueError):
                        await msg.reply_text("⚠️ Use: /trade long oil 89.50 sl:88.00 tp:91.50")
                        continue
                    trade = await open_trade(instrument, direction, entry, sl, tp)
                    await send_trade_opened(trade)
                elif msg and msg.text and msg.text.startswith("/trades"):
                    open_t, recent_t, prices = await asyncio.gather(
                        get_open_trades(),
                        get_recent_trades(limit=5),
                        get_current_prices(),
                    )
                    await send_trades_list(open_t, recent_t, prices)
                elif msg and msg.text and msg.text.startswith("/close"):
                    parts = msg.text.split()
                    instrument = parts[1].lower() if len(parts) > 1 else None
                    if instrument not in ("oil", "gold"):
                        await msg.reply_text("⚠️ Use: /close oil  or  /close gold")
                        continue
                    open_t = await get_open_trades()
                    to_close = [t for t in open_t if t["instrument"] == instrument]
                    if not to_close:
                        await msg.reply_text(f"No open {instrument} trades.")
                        continue
                    prices = await get_current_prices()
                    price  = prices.get(instrument)
                    for trade in to_close:
                        closed = await close_trade(trade["id"], price, "manual")
                        await send_trade_closed(closed)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("Command polling error: %s", exc)
            await asyncio.sleep(5)


async def job_monitor() -> None:
    logger.info("▶  Monitor cycle started")
    try:
        new_items = await run_all_sources()
        logger.info("Fetched %d new item(s) — classifying and sending as they come…", len(new_items))

        if not new_items:
            logger.info("Monitor cycle complete — nothing new")
            return

        mkt_contexts = await get_both_contexts()

        sent = 0
        medium_conf: list[dict] = []  # confidence 6-7: batched into digest

        for item in new_items:
            classified = await classify_item(item)
            if classified is None:
                continue

            # Attach market context based on instrument
            instrument = classified.get("instrument", "neither")
            if instrument == "oil":
                classified["market_context"] = mkt_contexts.get("oil")
            elif instrument == "gold":
                classified["market_context"] = mkt_contexts.get("gold")
            elif instrument == "both":
                oil_ctx  = mkt_contexts.get("oil")
                gold_ctx = mkt_contexts.get("gold")
                if gold_ctx and gold_ctx.get("signal") != "NONE":
                    classified["market_context"] = gold_ctx
                else:
                    classified["market_context"] = oil_ctx

            await update_classification(classified["url"], classified)

            confidence = classified.get("confidence") or 0
            if confidence >= 8:
                await send_new_items([classified])
            else:
                medium_conf.append(classified)
            sent += 1

        if medium_conf:
            await send_batch_digest(medium_conf)

        logger.info(
            "Monitor cycle complete — %d/%d passed classifier (%d high, %d batched)",
            sent, len(new_items), sent - len(medium_conf), len(medium_conf),
        )

    except Exception as exc:
        logger.exception("Unhandled error in monitor job: %s", exc)


async def job_cot_update() -> None:
    """Friday 16:00 UTC — fetch new COT data and send a positioning summary."""
    logger.info("▶  COT update job started")
    try:
        from telegram_bot import send_cot_update
        cot_data = await get_cot_data()
        await send_cot_update(cot_data)
    except Exception as exc:
        logger.exception("Unhandled error in COT update job: %s", exc)


async def job_price_check() -> None:
    open_trades = await get_open_trades()
    if not _price_alerts and not open_trades:
        return
    try:
        prices = await get_current_prices()
    except Exception as exc:
        logger.error("Price check failed: %s", exc)
        return

    # Price alerts
    for instrument, alert in list(_price_alerts.items()):
        price = prices.get(instrument)
        if price is None:
            continue
        target    = alert["target"]
        direction = alert["direction"]
        if (direction == "above" and price >= target) or \
           (direction == "below" and price <= target):
            await send_price_alert(instrument, price, target, direction)
            del _price_alerts[instrument]
            _save_alerts(_price_alerts)
            logger.info("Price alert fired and cleared: %s @ %.2f", instrument, price)

    # Trade auto-close on SL/TP
    for trade in open_trades:
        price = prices.get(trade["instrument"])
        if price is None:
            continue
        sl = trade.get("sl_price")
        tp = trade.get("tp_price")
        is_long = trade["direction"] == "long"
        outcome = None
        if tp and ((is_long and price >= tp) or (not is_long and price <= tp)):
            outcome = "tp"
        elif sl and ((is_long and price <= sl) or (not is_long and price >= sl)):
            outcome = "sl"
        if outcome:
            closed = await close_trade(trade["id"], price, outcome)
            await send_trade_closed(closed)
            logger.info("Trade auto-closed [%s] %s @ %.2f", outcome, trade["instrument"], price)


async def job_event_alerts() -> None:
    now = datetime.now(timezone.utc)
    try:
        events = await get_week_calendar()
    except Exception as exc:
        logger.error("Event alert calendar fetch failed: %s", exc)
        return

    for event in events:
        dt = event.get("datetime")
        if not dt:
            continue
        key = f"{event['title']}|{dt.isoformat()}"
        minutes_away = (dt - now).total_seconds() / 60

        # Pre-event: fire once when 10–20 minutes out; snapshot prices for reaction analysis
        if 10 <= minutes_away <= 20 and key not in _alerted_upcoming:
            await send_event_alert(event, kind="upcoming")
            _alerted_upcoming.add(key)
            try:
                snap = await get_current_prices()
                _event_snapshots[key] = snap
            except Exception:
                pass
            logger.info("Upcoming alert sent: %s", event["title"])

        # Post-event: fire once when 0–15 minutes past and actual value is available
        if -15 <= minutes_away < 0 and event.get("actual") and key not in _alerted_result:
            await send_event_alert(event, kind="result")
            _alerted_result.add(key)
            logger.info("Result alert sent: %s", event["title"])

            # Reaction classification — compare pre-event snapshot to current price
            snap = _event_snapshots.get(key, {})
            try:
                post = await get_current_prices()
                analysis = await classify_event_reaction(
                    event,
                    oil_pre=snap.get("oil") or post.get("oil", 0),
                    gold_pre=snap.get("gold") or post.get("gold", 0),
                    oil_post=post.get("oil", 0),
                    gold_post=post.get("gold", 0),
                )
                if analysis:
                    await send_event_reaction(
                        event,
                        oil_pre=snap.get("oil") or post.get("oil", 0),
                        gold_pre=snap.get("gold") or post.get("gold", 0),
                        oil_post=post.get("oil", 0),
                        gold_post=post.get("gold", 0),
                        analysis=analysis,
                    )
            except Exception as exc:
                logger.error("Event reaction failed: %s", exc)


async def job_signal_resolution() -> None:
    """Resolve pending signal outcomes by checking current prices against recorded entry prices."""
    pending = await get_pending_signals()
    if not pending:
        return
    try:
        prices = await get_current_prices()
        now    = datetime.now(timezone.utc)
        for sig in pending:
            price = prices.get(sig["instrument"])
            if not price:
                continue
            is_long = sig["direction"] == "long"
            updates: dict = {}
            at_2h = datetime.fromisoformat(sig["resolve_at_2h"]).replace(tzinfo=timezone.utc)
            if now >= at_2h and sig["correct_2h"] is None:
                updates["price_2h"]   = price
                updates["correct_2h"] = 1 if (price >= sig["entry_price"] if is_long else price <= sig["entry_price"]) else 0
            at_4h = datetime.fromisoformat(sig["resolve_at_4h"]).replace(tzinfo=timezone.utc)
            if now >= at_4h and sig["correct_4h"] is None:
                updates["price_4h"]   = price
                updates["correct_4h"] = 1 if (price >= sig["entry_price"] if is_long else price <= sig["entry_price"]) else 0
            if updates:
                await resolve_signal(sig["id"], **updates)
                logger.debug(
                    "Signal #%d resolved — %s %s entry=%.2f now=%.2f %s",
                    sig["id"], sig["direction"], sig["instrument"],
                    sig["entry_price"], price, updates,
                )
    except Exception as exc:
        logger.error("Signal resolution error: %s", exc)


async def job_watch_checker() -> None:
    """Every 5 min: check due watches, evaluate condition via Claude Haiku, send update or trigger."""
    due = await get_due_watches()
    if not due:
        return

    import anthropic as _anthropic

    async def _evaluate(condition: str, instrument: str, ctx: dict) -> tuple[bool, str]:
        def g(k):
            v = (ctx or {}).get(k)
            return v if v is not None else "n/a"
        prompt = (
            f"Market state for {instrument.upper()} right now:\n"
            f"Price: ${g('price')}  |  Regime: {g('regime')}  |  Bias: {g('bias')}\n"
            f"RSI: {g('rsi')}  |  Stoch K={g('k')} D={g('d')} ({g('stoch_zone')})\n"
            f"BB pos: {g('bb_pos')}  |  Signal: {g('signal')}  |  ATR: {g('atr')}\n\n"
            f"Watch condition: {condition}\n\n"
            "Is this condition currently met? Reply in exactly this format:\n"
            "MET: yes|no\n"
            "STATUS: [one sentence — current values vs what the condition requires]"
        )
        try:
            client = _anthropic.AsyncAnthropic()
            resp = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            met = "met: yes" in text.lower()
            status = next(
                (l.replace("STATUS:", "").strip() for l in text.splitlines() if l.startswith("STATUS:")),
                text,
            )
            return met, status
        except Exception as exc:
            logger.error("Watch evaluation error: %s", exc)
            return False, f"evaluation error: {exc}"

    mkt = await get_both_contexts()

    for watch in due:
        instrument = watch["instrument"]
        ctx        = mkt.get(instrument) or {}
        try:
            met, status = await _evaluate(watch["condition"], instrument, ctx)
            if met:
                await send_watch_triggered(watch, status)
                await fire_watch(watch["id"])
                logger.info("Watch #%d triggered [%s]: %s", watch["id"], instrument, status)
            else:
                await send_watch_update(watch, status)
                await advance_watch(watch["id"], watch["check_interval"])
                logger.info("Watch #%d updated [%s]: %s", watch["id"], instrument, status)
        except Exception as exc:
            logger.error("Watch #%d check failed: %s", watch["id"], exc)


async def job_morning_brief() -> None:
    logger.info("▶  Morning brief job started")
    try:
        since = datetime.now(timezone.utc) - timedelta(hours=MORNING_BRIEF_LOOKBACK_HOURS)

        # Fetch everything concurrently
        items, mkt, cot_data, calendar_events, sentiment, fred_ctx = await asyncio.gather(
            get_items_since(since),
            get_both_contexts(),
            get_cot_data(),
            get_week_calendar(),
            get_sentiment_window(hours=24),
            get_fred_context(),
        )

        oil_ctx   = mkt.get("oil")       or {}
        gold_ctx  = mkt.get("gold")      or {}
        dxy_ctx   = mkt.get("dxy")       or {}
        oil_curve = mkt.get("oil_curve") or {}
        gld_flow  = mkt.get("gld_flow")  or {}

        logger.info(
            "Morning brief: %d item(s), COT oil=%s gold=%s, calendar=%d events",
            len(items),
            cot_data.get("oil",  {}).get("label", "n/a") if cot_data.get("oil")  else "n/a",
            cot_data.get("gold", {}).get("label", "n/a") if cot_data.get("gold") else "n/a",
            len(calendar_events),
        )

        if items:
            logger.info("Generating Sonnet synthesis…")
            brief_text = await generate_morning_brief(
                items, oil_ctx, gold_ctx, cot_data, calendar_events, sentiment,
                dxy_ctx, fred_ctx, oil_curve, gld_flow,
            )
        else:
            brief_text = ""

        await send_morning_brief(items, brief_text, oil_ctx, gold_ctx)

        # Record BULLISH/BEARISH calls from the brief as signals
        if brief_text:
            prices = await get_current_prices()
            for inst, direction in parse_brief_signals(brief_text).items():
                if direction and prices.get(inst):
                    sid = await record_signal(inst, direction, "brief", prices[inst])
                    if sid:
                        logger.info("Signal recorded #%d: %s %s @ %.2f (brief)", sid, direction.upper(), inst, prices[inst])

    except Exception as exc:
        logger.exception("Unhandled error in morning brief job: %s", exc)


async def main() -> None:
    logger.info("=== Trading Intel Agent — Phase 2 starting ===")
    await init_db()

    bot = Bot(token=TELEGRAM_TOKEN)
    polling_task = asyncio.create_task(poll_commands(bot))

    scheduler = AsyncIOScheduler(timezone="UTC")

    scheduler.add_job(
        job_monitor,
        trigger=IntervalTrigger(minutes=MONITOR_INTERVAL_MINUTES),
        id="monitor",
        name="Source Monitor",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.add_job(
        job_morning_brief,
        trigger=CronTrigger(
            hour=MORNING_BRIEF_HOUR,
            minute=MORNING_BRIEF_MINUTE,
            timezone="UTC",
        ),
        id="morning_brief",
        name="Morning Brief",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.add_job(
        job_price_check,
        trigger=IntervalTrigger(seconds=60),
        id="price_check",
        name="Price Alerts",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.add_job(
        job_event_alerts,
        trigger=IntervalTrigger(minutes=5),
        id="event_alerts",
        name="Event Alerts",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.add_job(
        job_signal_resolution,
        trigger=IntervalTrigger(minutes=15),
        id="signal_resolution",
        name="Signal Resolution",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.add_job(
        job_watch_checker,
        trigger=IntervalTrigger(minutes=5),
        id="watch_checker",
        name="Watch Checker",
        replace_existing=True,
        max_instances=1,
    )

    # COT update: every Friday at 16:00 UTC (30 min after CFTC releases at 15:30 ET)
    scheduler.add_job(
        job_cot_update,
        trigger=CronTrigger(day_of_week="fri", hour=16, minute=0, timezone="UTC"),
        id="cot_update",
        name="COT Update",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.start()
    logger.info(
        "Scheduler started — monitor every %d min, morning brief at %02d:%02d UTC",
        MONITOR_INTERVAL_MINUTES,
        MORNING_BRIEF_HOUR,
        MORNING_BRIEF_MINUTE,
    )

    # Trigger an immediate monitor run through the scheduler so it doesn't block startup
    scheduler.add_job(job_monitor, id="monitor_startup", name="Startup Monitor",
                      max_instances=1, replace_existing=True)

    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down…")
        polling_task.cancel()
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())