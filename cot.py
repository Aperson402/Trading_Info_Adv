"""
cot.py — CFTC Commitments of Traders positioning data.

Uses the CFTC Public Reporting Environment API (no token required).
Fetches legacy futures-only report for WTI crude oil and gold.
Computes net speculative positioning and 3-year historical percentile.

CFTC codes:
  WTI Crude Oil (NYMEX): 067651
  Gold (COMEX):          088691
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# CFTC Public Reporting Environment - Legacy Futures Only
# Dataset ID: 6dca-aqww
CFTC_API = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

# CFTC market codes
CODES = {
    "oil":  "067651",  # CRUDE OIL, LIGHT SWEET - NYMEX
    "gold": "088691",  # GOLD - COMMODITY EXCHANGE INC.
}

HISTORY_WEEKS = 156  # 3 years of weekly data for percentile calculation


async def _fetch_cot(session: aiohttp.ClientSession, cftc_code: str, limit: int = 160) -> list[dict]:
    """Fetch COT records for a given CFTC code, most recent first."""
    params = {
        "$where": f"cftc_contract_market_code='{cftc_code}'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": str(limit),
    }
    try:
        async with session.get(CFTC_API, params=params, ssl=True) as resp:
            resp.raise_for_status()
            return await resp.json()
    except Exception as exc:
        logger.warning("CFTC API fetch failed for code %s: %s", cftc_code, exc)
        return []


def _compute_positioning(records: list[dict]) -> Optional[dict]:
    """
    Given a list of COT records (newest first), compute:
    - Latest net speculative position (non-commercial longs - shorts)
    - Net position as % of open interest
    - 3-year percentile rank
    - Week-over-week change
    - Positioning label: "extreme long" / "crowded long" / "neutral" /
                         "crowded short" / "extreme short"
    """
    if not records:
        return None

    def _net(rec: dict) -> Optional[float]:
        try:
            longs  = float(rec.get("noncomm_positions_long_all",  0))
            shorts = float(rec.get("noncomm_positions_short_all", 0))
            return longs - shorts
        except (TypeError, ValueError):
            return None

    def _oi(rec: dict) -> Optional[float]:
        try:
            return float(rec.get("open_interest_all", 1))
        except (TypeError, ValueError):
            return None

    nets = []
    for rec in records:
        n = _net(rec)
        if n is not None:
            nets.append(n)

    if not nets:
        return None

    current_net = nets[0]
    prev_net    = nets[1] if len(nets) > 1 else current_net
    wow_change  = current_net - prev_net

    # Percentile rank over available history (up to 3 years)
    history = nets[1:]  # exclude current week
    if history:
        pct_rank = sum(1 for n in history if n < current_net) / len(history) * 100
    else:
        pct_rank = 50.0

    # Net as % of open interest
    oi = _oi(records[0])
    net_pct_oi = (current_net / oi * 100) if oi else None

    # Label
    if pct_rank >= 90:
        label = "extreme long"
    elif pct_rank >= 70:
        label = "crowded long"
    elif pct_rank <= 10:
        label = "extreme short"
    elif pct_rank <= 30:
        label = "crowded short"
    else:
        label = "neutral"

    # Report date
    report_date = records[0].get("report_date_as_yyyy_mm_dd", "")[:10]

    return {
        "net_contracts":   int(current_net),
        "net_pct_oi":      round(net_pct_oi, 1) if net_pct_oi is not None else None,
        "wow_change":      int(wow_change),
        "pct_rank":        round(pct_rank, 0),
        "label":           label,
        "report_date":     report_date,
        "history_weeks":   len(history),
    }


async def get_cot_data() -> dict:
    """
    Fetch and compute COT positioning for oil and gold.
    Returns dict with keys 'oil' and 'gold', each containing positioning dict or None.
    """
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        import asyncio
        oil_records, gold_records = await asyncio.gather(
            _fetch_cot(session, CODES["oil"],  HISTORY_WEEKS + 4),
            _fetch_cot(session, CODES["gold"], HISTORY_WEEKS + 4),
        )

    oil_pos  = _compute_positioning(oil_records)
    gold_pos = _compute_positioning(gold_records)

    if oil_pos:
        logger.info(
            "COT Oil:  net=%+d  pct_rank=%.0f%%  label=%s  date=%s",
            oil_pos["net_contracts"], oil_pos["pct_rank"],
            oil_pos["label"], oil_pos["report_date"],
        )
    if gold_pos:
        logger.info(
            "COT Gold: net=%+d  pct_rank=%.0f%%  label=%s  date=%s",
            gold_pos["net_contracts"], gold_pos["pct_rank"],
            gold_pos["label"], gold_pos["report_date"],
        )

    return {"oil": oil_pos, "gold": gold_pos}
