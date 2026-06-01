"""
fred.py — FRED economic data: real yields and breakeven inflation.

Uses the public CSV download endpoint — no API key required.
Data is daily with ~1 business day lag.

Key series:
  DFII10  — 10-year TIPS real yield (primary gold suppressor when positive)
  T5YIE   — 5-year breakeven inflation (inflation hedge demand signal)
  T10YIE  — 10-year breakeven inflation
"""

import asyncio
import csv
import io
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

SERIES = {
    "real_yield_10y": "DFII10",
    "breakeven_5y":   "T5YIE",
    "breakeven_10y":  "T10YIE",
}

# Cache — FRED data is daily, no point re-fetching within the same session
_cache: dict = {}
_cache_expires: Optional[datetime] = None
_CACHE_TTL = timedelta(hours=4)


async def _fetch_series(
    session: aiohttp.ClientSession, series_id: str
) -> Optional[tuple[float, float]]:
    """Returns (latest, prev) values, or None on failure."""
    url = _FRED_CSV.format(series_id)
    try:
        async with session.get(url) as resp:
            resp.raise_for_status()
            text = await resp.text()
        reader = csv.reader(io.StringIO(text))
        next(reader)  # skip header row
        rows = [(r[0], r[1]) for r in reader if len(r) >= 2 and r[1] and r[1] != "."]
        if not rows:
            return None
        latest = float(rows[-1][1])
        prev   = float(rows[-2][1]) if len(rows) >= 2 else latest
        return latest, prev
    except Exception as exc:
        logger.warning("FRED %s failed: %s", series_id, exc)
        return None


async def get_fred_context() -> dict:
    """
    Fetch 10yr real yield, 5yr and 10yr breakeven inflation from FRED.
    Returns current values, day-over-day changes, and gold-relevant interpretations.
    Cached for 4 hours — FRED only updates once per business day.
    """
    global _cache, _cache_expires
    now = datetime.now(timezone.utc)
    if _cache_expires and now < _cache_expires:
        return _cache

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *[_fetch_series(session, sid) for sid in SERIES.values()],
            return_exceptions=True,
        )

    out: dict = {}
    for key, result in zip(SERIES.keys(), results):
        if isinstance(result, Exception) or result is None:
            out[key]          = None
            out[f"{key}_chg"] = None
        else:
            out[key]          = round(result[0], 3)
            out[f"{key}_chg"] = round(result[0] - result[1], 3)

    real_yield = out.get("real_yield_10y")
    be5y       = out.get("breakeven_5y")

    if real_yield is not None:
        if real_yield > 2.5:
            out["gold_yield_signal"] = "strong headwind — real yields very elevated"
        elif real_yield > 1.5:
            out["gold_yield_signal"] = "headwind — positive real yields suppress gold"
        elif real_yield > 0.5:
            out["gold_yield_signal"] = "mild headwind"
        elif real_yield >= 0:
            out["gold_yield_signal"] = "neutral"
        else:
            out["gold_yield_signal"] = "tailwind — negative real yields support gold"

    if be5y is not None:
        if be5y > 2.8:
            out["inflation_signal"] = "elevated — inflation hedge demand active"
        elif be5y > 2.2:
            out["inflation_signal"] = "anchored — neutral for gold"
        else:
            out["inflation_signal"] = "low — weak inflation bid for gold"

    logger.info(
        "FRED: real yield %.3f%% (%s), breakeven 5y %.3f%%",
        real_yield or 0, out.get("gold_yield_signal", "n/a"), be5y or 0,
    )

    _cache         = out
    _cache_expires = now + _CACHE_TTL
    return out
