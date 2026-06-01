"""
calendar.py — economic calendar for the week ahead.

Uses jblanked.com free API (ForexFactory data, clean JSON, no auth).
Filters for high-impact USD events and commodity-relevant releases.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

CALENDAR_API = "https://www.jblanked.com/news/api/forex-factory/calendar/range/"

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
        }
    except Exception:
        return None


async def get_week_calendar() -> list[dict]:
    """
    Fetch high-impact economic events for the next 7 days.
    Returns list of relevant events sorted by datetime.
    """
    now   = datetime.now(timezone.utc)
    end   = now + timedelta(days=7)
    from_str = now.strftime("%Y-%m-%d")
    to_str   = end.strftime("%Y-%m-%d")

    params = {
        "from":     from_str,
        "to":       to_str,
        "currency": "USD",
        "impact":   "High",
    }

    timeout = aiohttp.ClientTimeout(total=15)
    events: list[dict] = []

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(CALENDAR_API, params=params, ssl=True) as resp:
                resp.raise_for_status()
                raw_events = await resp.json()

        for raw in raw_events:
            event = _parse_event(raw)
            if event and _is_relevant(event["title"]):
                events.append(event)

        # Sort by datetime
        events.sort(key=lambda e: e["datetime"] or datetime.max.replace(tzinfo=timezone.utc))
        logger.info("Calendar: %d relevant high-impact events in next 7 days", len(events))

    except Exception as exc:
        logger.warning("Calendar fetch failed: %s", exc)

    return events


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
