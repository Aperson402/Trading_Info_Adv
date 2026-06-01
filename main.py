"""
main.py — entry point.

Phase 2: classifier inserted between monitor and Telegram delivery.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from telegram import Bot

from classifier import classify_item
from morning_brief import generate_morning_brief
from cot import get_cot_data
from econ_calendar import get_week_calendar
from market_context import get_both_contexts
from config import (
    LOG_LEVEL,
    MONITOR_INTERVAL_MINUTES,
    MORNING_BRIEF_HOUR,
    MORNING_BRIEF_LOOKBACK_HOURS,
    MORNING_BRIEF_MINUTE,
    TELEGRAM_TOKEN,
)
from database import get_items_since, init_db, update_classification
from monitor import run_all_sources
from telegram_bot import send_morning_brief, send_new_items

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


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


async def job_morning_brief() -> None:
    logger.info("▶  Morning brief job started")
    try:
        since = datetime.now(timezone.utc) - timedelta(hours=MORNING_BRIEF_LOOKBACK_HOURS)

        # Fetch everything concurrently
        items, mkt, cot_data, calendar_events = await asyncio.gather(
            get_items_since(since),
            get_both_contexts(),
            get_cot_data(),
            get_week_calendar(),
        )

        oil_ctx  = mkt.get("oil")  or {}
        gold_ctx = mkt.get("gold") or {}

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
                items, oil_ctx, gold_ctx, cot_data, calendar_events
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