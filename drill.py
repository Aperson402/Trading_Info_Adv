"""
drill.py — AI-directed deep-dive analysis via Anthropic tool use.

Two-phase flow:
  1. Claude sees the current context and calls fetch_market_data to request
     whatever additional tickers/timeframes/indicators it thinks it needs
  2. All requests are fetched in parallel via yfinance
  3. Claude receives the results and produces a focused WAIT condition assessment

Claude can request any Yahoo Finance ticker at any interval — it decides what's
relevant based on the instrument and current setup.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta, timezone
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)
SONNET_MODEL = "claude-sonnet-4-6"

# ── Tool definition ───────────────────────────────────────────────────────────

FETCH_TOOL = {
    "name": "fetch_market_data",
    "description": (
        "Fetch OHLCV price data for any asset available on Yahoo Finance. "
        "Use this to pull correlated assets, alternative timeframes, spread instruments, "
        "ETFs, or volatility measures that would help assess whether a WAIT condition "
        "is about to resolve. Results include recent bars, % change over the period, "
        "trend direction, RSI (where enough history exists), and Bollinger Band position "
        "for hourly+ timeframes. "
        "Useful tickers: ^TNX (10yr yield), ^VIX (volatility), TLT (bond ETF), "
        "GLD/GDX (gold ETFs), SI=F (silver), XLE/USO/BNO (energy ETFs/Brent), "
        "NG=F (nat gas), DX-Y.NYB (DXY), EURUSD=X, USDCNH=X (dollar/yuan)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "requests": {
                "type": "array",
                "description": "Data requests. Maximum 6.",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Yahoo Finance ticker symbol.",
                        },
                        "interval": {
                            "type": "string",
                            "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                            "description": (
                                "Bar interval. "
                                "5m/15m = intraday momentum; "
                                "1h/4h = structure and regime; "
                                "1d = trend context and COT-level positioning."
                            ),
                        },
                        "period": {
                            "type": "string",
                            "enum": ["1d", "5d", "1mo", "3mo"],
                            "description": (
                                "How far back to fetch. "
                                "1d for intraday bars, 5d for recent structure, "
                                "1mo+ for trend and percentile context."
                            ),
                        },
                        "label": {
                            "type": "string",
                            "description": "Short human-readable name shown in results.",
                        },
                    },
                    "required": ["ticker", "interval", "period", "label"],
                },
            }
        },
        "required": ["requests"],
    },
}

# ── Prompts ───────────────────────────────────────────────────────────────────

PHASE1_PROMPT = """\
You are assessing a WAIT signal on {instrument_upper}. This is a two-stage process.

STAGE 1 (now): Call fetch_market_data with the specific data you need.
Be intentional. Request the exact tickers and timeframes that would tell you whether
the blocking conditions are about to resolve — not generic data.

STAGE 2 (after data): Produce a tight CONDITION CHECK with probability.

── CURRENT STATE ────────────────────────────────────────────────────────────────

TIME: {now} UTC

{instrument_upper} — 1H:
  Price: ${price}  |  BB: {upper} / {basis} / {lower}
  Regime: {regime}  |  Bias: {bias}  |  BB pos: {bb_pos}  |  Weekly: {weekly}
  RSI: {rsi}  |  Stoch K={k} D={d} ({stoch_zone})  |  ATR: {atr}

DXY: {dxy_price} ({dxy_change:+.2f}% today, {dxy_trend} vs SMA20)

ALREADY AVAILABLE — do not re-request:
{already_fetched}

REAL RATES & ETF FLOWS (FRED, prev business day):
  10yr real yield: {real_yield}% ({real_yield_chg:+.3f}%) — {gold_yield_signal}
  5yr breakeven inflation: {breakeven_5y}% — {inflation_signal}
  GLD ETF: {gld_tonnes}t ({gld_delta:+.1f}t today) — {gld_signal}

OIL FUTURES CURVE:
  Structure: {curve_structure}  |  Front: ${oil_front}  |  6m fwd: ${oil_6m}  |  Spread: {curve_spread:+.2f}  |  {curve_signal}
{silver_section}

RECENT NEWS ({n_items} items, last 4h):
{items_text}

EVENTS NEXT 4H: {upcoming_events}

────────────────────────────────────────────────────────────────────────────────
Call fetch_market_data now with what you need.\
"""

PHASE2_INSTRUCTION = """\
You now have all the data you requested. Give the final analysis.
Under 220 words. No hedging. Exact format:

CONDITION CHECK:
• [first blocking condition]: [CONFIRMING / DENYING / UNCLEAR] — [one sentence why, cite the data]
• [second blocking condition]: [CONFIRMING / DENYING / UNCLEAR] — [one sentence why, cite the data]
• [third blocking condition]: [CONFIRMING / DENYING / UNCLEAR] — [one sentence why, cite the data]

PROBABILITY WAIT RESOLVES IN 2H: [LOW / MEDIUM / HIGH]
[One sentence on the decisive factor]

NARRATIVE:
[What fundamental catalyst is actually driving the price against the expected narrative?
If geopolitics should be bullish but price is falling — explain what is overriding it and why.
Cite specific news items, data points, or macro flows. Two sentences max.]

WATCH NOW:
• [Most important specific level or signal to monitor, with exact number]
• [Second most important thing]\
"""


# ── Data fetcher ──────────────────────────────────────────────────────────────

def _fetch_one_sync(ticker: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


def _wilder_rsi(c: pd.Series, n: int = 14) -> float:
    """Wilder's RMA RSI — matches TradingView. Returns last bar value."""
    if len(c) < n * 2:
        return float("nan")
    delta = c.diff().to_numpy()
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    # Find first non-nan
    start = int(np.argmax(~np.isnan(delta[1:]))) + 1
    if len(gain) - start < n:
        return float("nan")
    alpha = 1.0 / n
    avg_g = float(np.nanmean(gain[start: start + n]))
    avg_l = float(np.nanmean(loss[start: start + n]))
    for i in range(start + n, len(gain)):
        avg_g = avg_g * (1 - alpha) + gain[i] * alpha
        avg_l = avg_l * (1 - alpha) + loss[i] * alpha
    return 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)


def _summarise_df(df: pd.DataFrame, ticker: str, label: str, interval: str) -> str:
    c = df["Close"]
    n_show = min(16, len(c))
    recent = c.iloc[-n_show:]
    last   = float(c.iloc[-1])
    first  = float(recent.iloc[0])
    chg    = (last - first) / first * 100 if first else 0.0

    # Direction from slope across the window
    vals = recent.to_numpy(dtype=float)
    if len(vals) >= 4:
        slope = (vals[-1] - vals[0]) / (len(vals) - 1)
        thresh = abs(last) * 0.0002
        direction = "rising" if slope > thresh else "falling" if slope < -thresh else "flat"
    else:
        direction = "flat"

    bars_str = " → ".join(f"{float(x):.3f}" for x in recent.tolist())

    # RSI
    rsi_val = _wilder_rsi(c)
    rsi_str = f"  RSI: {rsi_val:.1f}" if not np.isnan(rsi_val) else ""

    # BB for hourly+ only
    bb_str = ""
    if interval in ("1h", "4h", "1d") and len(c) >= 20:
        basis_val = float(c.rolling(20).mean().iloc[-1])
        std_val   = float(c.rolling(20).std(ddof=0).iloc[-1])
        upper_val = basis_val + 2 * std_val
        lower_val = basis_val - 2 * std_val
        pos       = "above mid" if last > basis_val else "below mid"
        bb_str    = f"  BB: {lower_val:.3f}/{basis_val:.3f}/{upper_val:.3f} ({pos})"

    return (
        f"[{label}] {ticker} @ {interval}  →  {last:.3f}  "
        f"({chg:+.2f}% over period)  {direction}"
        f"{rsi_str}{bb_str}\n"
        f"  bars: {bars_str}"
    )


async def _fetch_all(requests: list[dict]) -> str:
    """Fetch up to 6 requested tickers concurrently; return formatted string for tool result."""
    loop     = asyncio.get_event_loop()
    capped   = requests[:6]

    dfs = await asyncio.gather(
        *[
            loop.run_in_executor(None, _fetch_one_sync, r["ticker"], r["interval"], r["period"])
            for r in capped
        ],
        return_exceptions=True,
    )

    sections = []
    for req, df in zip(capped, dfs):
        label    = req.get("label", req["ticker"])
        ticker   = req["ticker"]
        interval = req["interval"]
        if isinstance(df, Exception) or df is None:
            sections.append(f"[{label}] {ticker}: fetch failed (invalid ticker or market closed)")
        else:
            sections.append(_summarise_df(df, ticker, label, interval))

    return "\n\n".join(sections)


# ── Prompt helpers ────────────────────────────────────────────────────────────

def _fmt_silver_section(corr: dict) -> str:
    """
    Compute Gold/Silver ratio and 15m divergence for gold drills.
    Silver outperforming gold = bullish divergence (silver leads spot).
    Returns empty string for non-gold instruments.
    """
    inst = (corr or {}).get("instrument_15m") or {}
    silver = (corr or {}).get("silver") or {}
    if not inst or not silver:
        return ""

    gold_px   = inst.get("price")
    silver_px = silver.get("price")
    if not gold_px or not silver_px:
        return ""

    gsr        = gold_px / silver_px
    gold_chg   = inst.get("change_1h_pct", 0.0)
    silver_chg = silver.get("change_1h_pct", 0.0)
    divergence = gold_chg - silver_chg  # positive = gold leading; negative = silver leading

    if divergence < -0.4:
        div_note = f"BULLISH DIVERGENCE — silver outperforming by {abs(divergence):.2f}% (silver leads gold up)"
    elif divergence > 0.4:
        div_note = f"BEARISH DIVERGENCE — gold outperforming silver by {divergence:.2f}% (unusual, may mean-revert)"
    else:
        div_note = f"no divergence — moving together ({divergence:+.2f}%)"

    # Historical GSR context
    if gsr > 90:
        gsr_note = "historically elevated (gold expensive vs silver — mean-reversion favours gold drop or silver catch-up)"
    elif gsr > 80:
        gsr_note = "elevated"
    elif gsr < 65:
        gsr_note = "low (silver leading — bullish for both metals)"
    else:
        gsr_note = "normal range"

    return (
        f"\nGOLD/SILVER DIVERGENCE (15m):\n"
        f"  GSR: {gsr:.1f}x ({gsr_note})\n"
        f"  Gold 1H: {gold_chg:+.2f}%  |  Silver 1H: {silver_chg:+.2f}%\n"
        f"  → {div_note}"
    )


def _weekly_str(ctx: dict) -> str:
    wt  = (ctx or {}).get("weekly_trend", "")
    pct = (ctx or {}).get("weekly_pct_from_sma")
    if not wt:
        return "n/a"
    pct_str = f" ({pct:+.1f}% vs 20W SMA)" if pct is not None else ""
    return f"{wt.upper()}{pct_str}"


def _fmt_already_fetched(instrument: str, corr: dict) -> str:
    lines = [
        f"• {instrument.upper()} — 15m bars  (direction, 1H change)",
        "• DXY — 15m bars  (direction, 1H change)",
    ]
    if instrument == "gold":
        if corr.get("10yr_yield"):
            lines.append("• 10yr yield (^TNX) — 15m")
        if corr.get("silver"):
            lines.append("• Silver (SI=F) — 15m")
    elif instrument == "oil":
        if corr.get("nat_gas"):
            lines.append("• Natural Gas (NG=F) — 15m")
    return "\n".join(lines)


def _fmt_items(items: list) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"• {i.get('title', '')}" for i in items[-6:])


def _fmt_upcoming(calendar_events: list) -> str:
    now    = datetime.now(timezone.utc)
    window = [
        e for e in (calendar_events or [])
        if e.get("datetime") and
        timedelta(0) <= (e["datetime"] - now) <= timedelta(hours=4)
    ]
    if not window:
        return "None"
    return " | ".join(
        f"{e['datetime'].strftime('%H:%M')} {e['title']}" for e in window
    )


def _build_phase1(instrument, ctx, dxy_ctx, corr, items, calendar_events,
                  fred_ctx=None, oil_curve=None, gld_flow=None) -> str:
    def g(d, k, default="n/a"):
        v = (d or {}).get(k, default)
        return v if v is not None else default

    fred      = fred_ctx  or {}
    curve     = oil_curve or {}
    flow      = gld_flow  or {}

    return PHASE1_PROMPT.format(
        instrument_upper=instrument.upper(),
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        price=g(ctx, "price"),
        upper=g(ctx, "upper"),
        basis=g(ctx, "basis"),
        lower=g(ctx, "lower"),
        regime=g(ctx, "regime"),
        bias=g(ctx, "bias"),
        rsi=g(ctx, "rsi"),
        k=g(ctx, "k"),
        d=g(ctx, "d"),
        stoch_zone=g(ctx, "stoch_zone"),
        bb_pos=g(ctx, "bb_pos"),
        weekly=_weekly_str(ctx),
        atr=g(ctx, "atr"),
        dxy_price=g(dxy_ctx, "price", "n/a"),
        dxy_change=(dxy_ctx or {}).get("change_pct", 0.0),
        dxy_trend=g(dxy_ctx, "trend", "n/a"),
        already_fetched=_fmt_already_fetched(instrument, corr),
        silver_section=_fmt_silver_section(corr) if instrument == "gold" else "",
        # FRED
        real_yield=fred.get("real_yield_10y", "n/a"),
        real_yield_chg=fred.get("real_yield_10y_chg") or 0.0,
        gold_yield_signal=fred.get("gold_yield_signal", "n/a"),
        breakeven_5y=fred.get("breakeven_5y", "n/a"),
        inflation_signal=fred.get("inflation_signal", "n/a"),
        # GLD flow
        gld_tonnes=flow.get("tonnes", "n/a"),
        gld_delta=flow.get("delta_1d") or 0.0,
        gld_signal=flow.get("signal", "n/a"),
        # Oil curve
        curve_structure=curve.get("structure", "n/a"),
        oil_front=curve.get("front", "n/a"),
        oil_6m=curve.get("p6m", "n/a"),
        curve_spread=curve.get("spread_6m") or 0.0,
        curve_signal=curve.get("signal", "n/a"),
        n_items=len(items),
        items_text=_fmt_items(items),
        upcoming_events=_fmt_upcoming(calendar_events),
    )


# ── Main entry point ──────────────────────────────────────────────────────────

async def generate_drill(
    instrument: str,
    ctx: dict,
    dxy_ctx: dict,
    corr: dict,
    items: list,
    calendar_events: list = None,
    fred_ctx: dict = None,
    oil_curve: dict = None,
    gld_flow: dict = None,
) -> str:
    client        = anthropic.AsyncAnthropic()
    phase1_prompt = _build_phase1(
        instrument, ctx, dxy_ctx, corr, items, calendar_events or [],
        fred_ctx=fred_ctx, oil_curve=oil_curve, gld_flow=gld_flow,
    )
    messages      = [{"role": "user", "content": phase1_prompt}]

    # Phase 1 — Claude declares what it needs
    try:
        resp1 = await client.messages.create(
            model=SONNET_MODEL,
            max_tokens=600,
            tools=[FETCH_TOOL],
            tool_choice={"type": "any"},  # force at least one tool call
            messages=messages,
        )
    except Exception as exc:
        logger.error("Drill phase-1 failed: %s", exc)
        return ""

    tool_blocks = [b for b in resp1.content if b.type == "tool_use"]
    if not tool_blocks:
        logger.warning("Drill: Claude returned no tool call")
        return ""

    # Phase 2 — fetch all requested data in parallel
    all_requests = []
    for tb in tool_blocks:
        all_requests.extend(tb.input.get("requests", []))

    logger.info(
        "Drill [%s]: fetching %d dataset(s): %s",
        instrument,
        len(all_requests),
        ", ".join(f"{r['label']}@{r['interval']}" for r in all_requests[:6]),
    )
    fetched_str = await _fetch_all(all_requests)

    # Phase 3 — give Claude the results, get the analysis
    tool_results = [
        {"type": "tool_result", "tool_use_id": tb.id, "content": fetched_str}
        for tb in tool_blocks
    ]
    messages = messages + [
        {"role": "assistant", "content": resp1.content},
        {"role": "user",      "content": tool_results + [{"type": "text", "text": PHASE2_INSTRUCTION}]},
    ]

    try:
        resp2 = await client.messages.create(
            model=SONNET_MODEL,
            max_tokens=450,
            messages=messages,
        )
        text = resp2.content[0].text.strip()
        logger.info(
            "Drill [%s] complete — %d dataset(s) fetched, %d chars output",
            instrument, len(all_requests), len(text),
        )
        return text
    except Exception as exc:
        logger.error("Drill phase-3 failed: %s", exc)
        return ""
