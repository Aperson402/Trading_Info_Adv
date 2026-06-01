"""
main.py — entry point.

Phase 2: classifier inserted between monitor and Telegram delivery.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from telegram import Bot

from advice import generate_advice
from classifier import classify_item
from morning_brief import generate_morning_brief
from cot import get_cot_data
from econ_calendar import get_week_calendar
from market_context import get_both_contexts, get_current_prices
from config import (
    LOG_LEVEL,
    MONITOR_INTERVAL_MINUTES,
    MORNING_BRIEF_HOUR,
    MORNING_BRIEF_LOOKBACK_HOURS,
    MORNING_BRIEF_MINUTE,
    TELEGRAM_TOKEN,
)
from database import (
    close_trade, get_items_since, get_open_trades, get_recent_trades,
    get_sentiment_window, init_db, open_trade, update_classification,
)
from monitor import run_all_sources
from telegram_bot import (
    send_advice, send_calendar, send_event_alert, send_morning_brief,
    send_new_items, send_price_alert, send_trade_closed, send_trade_opened, send_trades_list,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Track which events have already been alerted to avoid duplicates
_alerted_upcoming: set[str] = set()
_alerted_result:   set[str] = set()

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
                if msg and msg.text and msg.text.startswith("/brief"):
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
                    items, mkt, sentiment, cot_data, calendar_events = await asyncio.gather(
                        get_items_since(since),
                        get_both_contexts(),
                        get_sentiment_window(hours=24),
                        get_cot_data(),
                        get_week_calendar(),
                    )
                    oil_ctx  = mkt.get("oil")  or {}
                    gold_ctx = mkt.get("gold") or {}
                    dxy_ctx  = mkt.get("dxy")  or {}
                    advice = await generate_advice(items, oil_ctx, gold_ctx, sentiment, cot_data, calendar_events, dxy_ctx)
                    await send_advice(advice)
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
            await send_new_items([classified])
            sent += 1

        logger.info(
            "Monitor cycle complete — %d/%d passed classifier",
            sent, len(new_items),
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

        # Pre-event: fire once when 10–20 minutes out
        if 10 <= minutes_away <= 20 and key not in _alerted_upcoming:
            await send_event_alert(event, kind="upcoming")
            _alerted_upcoming.add(key)
            logger.info("Upcoming alert sent: %s", event["title"])

        # Post-event: fire once when 0–15 minutes past and actual value is available
        if -15 <= minutes_away < 0 and event.get("actual") and key not in _alerted_result:
            await send_event_alert(event, kind="result")
            _alerted_result.add(key)
            logger.info("Result alert sent: %s", event["title"])


async def job_morning_brief() -> None:
    logger.info("▶  Morning brief job started")
    try:
        since = datetime.now(timezone.utc) - timedelta(hours=MORNING_BRIEF_LOOKBACK_HOURS)

        # Fetch everything concurrently
        items, mkt, cot_data, calendar_events, sentiment = await asyncio.gather(
            get_items_since(since),
            get_both_contexts(),
            get_cot_data(),
            get_week_calendar(),
            get_sentiment_window(hours=24),
        )

        oil_ctx  = mkt.get("oil")  or {}
        gold_ctx = mkt.get("gold") or {}
        dxy_ctx  = mkt.get("dxy")  or {}

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
                items, oil_ctx, gold_ctx, cot_data, calendar_events, sentiment, dxy_ctx
            )
        else:
            brief_text = ""

        await send_morning_brief(items, brief_text, oil_ctx, gold_ctx)

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

    await job_monitor()

    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down…")
        polling_task.cancel()
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())