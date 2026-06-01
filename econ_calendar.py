"""
calendar.py — economic calendar for the week ahead.

Uses jblanked.com free API (ForexFactory data, clean JSON, no auth).
Filters for high-impact USD events and commodity-relevant releases.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

_cache: list[dict] = []
_cache_expires: datetime | None = None
_CACHE_TTL = timedelta(minutes=30)

FF_THISWEEK = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FF_NEXTWEEK = "https://nfs.faireconomy.media/ff_calendar_nextweek.json"

# High-impact events relevant to oil and gold trading
RELEVANT_EVENTS = {
    # Fed / rates / dollar → gold
    "fomc", "fed", "federal reserve", "interest rate", "powell",
    "cpi", "inflation", "pce", "core pce",
    "nfp", "non-farm", "unemployment", "jobs",
    "gdp", "pmi",
    # Oil-specific
    "crude oil", "eia", "petroleum", "opec", "natural gas storage",
    "api crude", "inventory",
    # General macro
    "retail sales", "durable goods", "ism",
    "treasury", "auctions",
}


def _is_relevant(event_title: str) -> bool:
    t = event_title.lower()
    return any(kw in t for kw in RELEVANT_EVENTS)


def _parse_event(raw: dict) -> Optional[dict]:
    """Parse a raw calendar event into a clean dict."""
    try:
        title    = raw.get("event") or raw.get("title") or raw.get("name") or ""
        currency = raw.get("currency") or raw.get("country") or ""
        impact   = (raw.get("impact") or raw.get("importance") or "").lower()
        date_str = raw.get("date") or raw.get("datetime") or ""
        time_str = raw.get("time") or ""
        forecast = raw.get("forecast") or raw.get("consensus") or ""
        previous = raw.get("previous") or ""

        # Parse datetime
        dt = None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(date_str[:19], fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue

        return {
            "title":    title,
            "currency": currency.upper(),
            "impact":   impact,
            "datetime": dt,
            "time_str": time_str or (dt.strftime("%H:%M UTC") if dt else "TBD"),
            "forecast": forecast,
            "previous": previous,
            "actual":   raw.get("actual") or "",
        }
    except Exception:
        return None


async def get_week_calendar() -> list[dict]:
    """
    Fetch high-impact economic events for the next 7 days from ForexFactory.
    Results are cached for 30 minutes to avoid rate-limiting.
    """
    global _cache, _cache_expires
    now = datetime.now(timezone.utc)

    if _cache_expires and now < _cache_expires:
        return _cache

    cutoff = now + timedelta(days=7)
    timeout = aiohttp.ClientTimeout(total=15)
    events: list[dict] = []

    async def _fetch(url: str) -> list:
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, ssl=True) as resp:
                    if resp.status == 404:
                        # nextweek endpoint returns 404 when not yet published — expected
                        logger.debug("Calendar %s not yet published (404)", url)
                        return []
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as exc:
            logger.warning("Calendar fetch failed (%s): %s", url, exc)
            return []

    raw_all = await asyncio.gather(_fetch(FF_THISWEEK), _fetch(FF_NEXTWEEK))

    for raw_events in raw_all:
        for raw in raw_events:
            if (raw.get("country") != "USD") or (raw.get("impact") != "High"):
                continue
            event = _parse_event(raw)
            if not event or not event["datetime"]:
                continue
            if event["datetime"] < now or event["datetime"] > cutoff:
                continue
            if _is_relevant(event["title"]):
                events.append(event)

    events.sort(key=lambda e: e["datetime"])
    # Deduplicate by title+datetime
    seen: set = set()
    unique = []
    for e in events:
        key = (e["title"], e["datetime"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    _cache = unique
    _cache_expires = now + _CACHE_TTL
    logger.info("Calendar: %d relevant high-impact USD events in next 7 days (cached 30m)", len(unique))
    return unique


def fmt_calendar_for_brief(events: list[dict]) -> str:
    """Format calendar events as a compact string for the morning brief prompt."""
    if not events:
        return "No high-impact events in the next 7 days."

    lines = []
    for e in events[:8]:  # cap at 8 events
        dt = e.get("datetime")
        when = dt.strftime("%a %d %b %H:%M UTC") if dt else e.get("time_str", "TBD")
        forecast = f" (f: {e['forecast']})" if e.get("forecast") else ""
        previous = f" prev: {e['previous']}" if e.get("previous") else ""
        lines.append(f"• {when}  {e['title']}{forecast}{previous}")

    return "\n".join(lines)
