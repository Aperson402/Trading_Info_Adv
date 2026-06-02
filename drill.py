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

# ── Tool definitions ──────────────────────────────────────────────────────────

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

SPREAD_TOOL = {
    "name": "compute_spread_ratio",
    "description": (
        "Fetch two Yahoo Finance tickers and compute their ratio (A÷B) or spread (A−B), "
        "returning the current value, 20-period mean, standard deviation, and z-score. "
        "Use for: Brent/WTI spread (BZ=F minus CL=F), gold/silver ratio (GC=F ÷ SI=F), "
        "GLD/GDX ratio (ETF premium vs miners), oil/natgas ratio (CL=F ÷ NG=F), "
        "WTI/Brent arb, or any pair where the relationship matters more than absolute price."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker_a": {
                "type": "string",
                "description": "Numerator ticker (ratio mode) or minuend (difference mode).",
            },
            "ticker_b": {
                "type": "string",
                "description": "Denominator ticker (ratio mode) or subtrahend (difference mode).",
            },
            "mode": {
                "type": "string",
                "enum": ["ratio", "difference"],
                "description": "ratio = A÷B  |  difference = A−B",
            },
            "interval": {
                "type": "string",
                "enum": ["1h", "4h", "1d"],
                "description": "Bar interval.",
            },
            "period": {
                "type": "string",
                "enum": ["5d", "1mo", "3mo"],
                "description": "Lookback period.",
            },
            "label": {
                "type": "string",
                "description": "Human-readable name shown in results (e.g. 'Brent-WTI spread').",
            },
        },
        "required": ["ticker_a", "ticker_b", "mode", "interval", "period", "label"],
    },
}

VOL_TOOL = {
    "name": "fetch_implied_volatility",
    "description": (
        "Fetch commodity implied-volatility indices from Yahoo Finance. "
        "^OVX = crude oil volatility index (CBOE); "
        "^GVZ = gold volatility index (CBOE); "
        "^VIX = equity volatility / risk-off proxy. "
        "Returns current level, 30-day percentile rank, 5-day change, "
        "and implied daily price move (price × vol ÷ √252) when a reference price is supplied. "
        "Use to gauge whether options markets are pricing in a breakout or expecting calm — "
        "high OVX/GVZ percentile = market expects a large move."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "indices": {
                "type": "array",
                "description": "Volatility indices to fetch. ^OVX for oil, ^GVZ for gold, ^VIX for risk-off.",
                "items": {
                    "type": "string",
                    "enum": ["^OVX", "^GVZ", "^VIX"],
                },
                "minItems": 1,
            },
            "reference_price": {
                "type": "number",
                "description": (
                    "Current instrument price used to compute implied daily move. "
                    "Pass the live WTI or gold price so the output is actionable. Optional."
                ),
            },
        },
        "required": ["indices"],
    },
}

ORDER_BOOK_TOOL = {
    "name": "fetch_order_book",
    "description": (
        "Fetch a live Level 2 order book snapshot from Interactive Brokers "
        "for the front-month WTI (CL/NYMEX) or Gold (GC/COMEX) futures contract. "
        "Returns: best bid/ask and spread, total depth imbalance (bid vs ask lots), "
        "identified walls (price levels with unusually large resting orders), "
        "and the full depth table. "
        "Use to identify where large resting orders are acting as support or resistance, "
        "and whether order flow is skewed to buyers or sellers. "
        "Requires IB Gateway or TWS to be running locally with API enabled."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "instrument": {
                "type": "string",
                "enum": ["oil", "gold"],
                "description": "oil = CL (WTI) on NYMEX  |  gold = GC on COMEX",
            },
            "num_rows": {
                "type": "integer",
                "description": "Depth levels per side (1–5; capped by IBKR subscription tier).",
                "minimum": 1,
                "maximum": 5,
                "default": 5,
            },
        },
        "required": ["instrument"],
    },
}

SET_WATCH_TOOL = {
    "name": "set_watch",
    "description": (
        "Register a specific condition to monitor with periodic automated updates. "
        "Call this when your analysis identifies a concrete, observable condition that "
        "hasn't resolved yet — a price level to break, an indicator crossover to happen, "
        "a spread to reach a threshold. The system will check the condition every N minutes, "
        "send a status update each time, and fire an alert when the condition is met. "
        "Only call this if there is a real, specific thing to watch — not for vague conditions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "condition": {
                "type": "string",
                "description": (
                    "Precise, observable condition. Include exact numbers so the automated "
                    "checker can evaluate it unambiguously. "
                    "Examples: 'Price breaks above $73.20 with RSI > 52' — "
                    "'Stochastic K crosses above D from below 25' — "
                    "'Brent-WTI spread narrows below $1.50'"
                ),
            },
            "check_interval_minutes": {
                "type": "integer",
                "enum": [15, 30, 60],
                "description": "How often to check and send a status update.",
            },
            "expires_minutes": {
                "type": "integer",
                "enum": [60, 120, 240, 480],
                "description": "Stop watching after this many minutes if the condition hasn't fired.",
            },
        },
        "required": ["condition", "check_interval_minutes", "expires_minutes"],
    },
}

ALL_TOOLS = [FETCH_TOOL, SPREAD_TOOL, VOL_TOOL, ORDER_BOOK_TOOL]

# ── Prompts ───────────────────────────────────────────────────────────────────

# ── Timeframe focus hints ─────────────────────────────────────────────────────

_TF_HINTS: dict[str, str] = {
    "5m":  (
        "TIMEFRAME FOCUS: 5-minute scalping. "
        "Fetch 5m bars for today (period='1d'). Focus on micro-structure, momentum, "
        "and immediate entry/exit levels within the next 15-30 minutes."
    ),
    "15m": (
        "TIMEFRAME FOCUS: 15-minute intraday. "
        "Fetch 15m bars for the last 1-2 days. Identify intraday trend direction, "
        "key intraday levels, and whether the current candle structure confirms or reverses the bias."
    ),
    "30m": (
        "TIMEFRAME FOCUS: 30-minute swing. "
        "Fetch 30m bars for the last 2-5 days. Identify the intraday swing structure, "
        "momentum continuation vs. reversal, and optimal entry timing."
    ),
    "2h":  (
        "TIMEFRAME FOCUS: Last 2 hours of price action. "
        "Fetch 15m bars for today. Analyse what specifically happened in the last 2 hours — "
        "the developing micro-trend, recent high/low, and whether momentum is building or fading."
    ),
    "4h":  (
        "TIMEFRAME FOCUS: 4-hour swing structure. "
        "Fetch 4H bars for the last 30 days. Identify the current swing trend, key 4H levels, "
        "and whether the setup is a trend continuation or potential reversal."
    ),
    "1d":  (
        "TIMEFRAME FOCUS: Daily chart macro perspective. "
        "Fetch 1d bars for the last 3 months. Identify the macro trend, daily key levels, "
        "and where this instrument sits in the broader picture."
    ),
    "1w":  (
        "TIMEFRAME FOCUS: Weekly macro context. "
        "Fetch weekly bars for 1-2 years. Identify long-term trend, major support/resistance, "
        "and whether price is at a historically significant level."
    ),
    "overnight": (
        "TIMEFRAME FOCUS: Overnight session (yesterday's close → now). "
        "Fetch 15m bars for the last 2 days. Determine: what happened while markets were thin, "
        "whether there are unfilled gaps, significant overnight moves or news spikes, "
        "and what bias is established for the London open."
    ),
}

def _timeframe_focus(timeframe: str) -> str:
    """Return the prompt hint for the requested timeframe, or empty string for default 1H."""
    hint = _TF_HINTS.get(timeframe.lower() if timeframe else "")
    return f"\n{hint}\n" if hint else ""


PHASE1_PROMPT = """\
You are assessing a WAIT signal on {instrument_upper}. This is a two-stage process.

STAGE 1 (now): Call fetch_market_data with the specific data you need.
Be intentional. Request the exact tickers and timeframes that would tell you whether
the blocking conditions are about to resolve — not generic data.

STAGE 2 (after data): Produce a tight CONDITION CHECK with probability.
{timeframe_focus}
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
AVAILABLE TOOLS (call any combination in parallel):
• fetch_market_data        — OHLCV for any Yahoo Finance ticker (correlated assets, ETFs, FX, bonds)
• compute_spread_ratio     — ratio or spread between two tickers with z-score context
                             (e.g. Brent−WTI spread, Gold/Silver ratio, GLD÷GDX, Oil÷Natgas)
• fetch_implied_volatility — ^OVX / ^GVZ / ^VIX with 30-day percentile rank and implied daily move
• fetch_order_book         — live IB Level 2 depth: bid/ask walls, spread, order imbalance
                             (oil = CL NYMEX, gold = GC COMEX; requires IB Gateway running)

Call the tools you need now. Be intentional — choose what actually resolves your uncertainty.\
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
• [Second most important thing]

After writing the analysis, if there is a specific, concrete condition worth monitoring
(a price level, crossover, or threshold not yet reached), call set_watch with the exact
condition, how often to check, and when to stop. Only call it if there is a real trigger
to watch — skip if the situation is already clear or has no actionable wait condition.\
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


# ── Spread and vol fetchers ───────────────────────────────────────────────────

async def _run_spread(inp: dict) -> str:
    """Fetch two tickers concurrently and compute spread or ratio with z-score."""
    loop = asyncio.get_event_loop()
    ticker_a = inp["ticker_a"]
    ticker_b = inp["ticker_b"]
    mode     = inp["mode"]
    interval = inp["interval"]
    period   = inp["period"]
    label    = inp["label"]

    df_a, df_b = await asyncio.gather(
        loop.run_in_executor(None, _fetch_one_sync, ticker_a, interval, period),
        loop.run_in_executor(None, _fetch_one_sync, ticker_b, interval, period),
    )

    if df_a is None or df_b is None:
        missing = ticker_a if df_a is None else ticker_b
        return f"[{label}]: fetch failed for {missing}"

    ca = df_a["Close"].squeeze()
    cb = df_b["Close"].squeeze()
    aligned = pd.DataFrame({"a": ca, "b": cb}).dropna()
    if len(aligned) < 5:
        return f"[{label}]: insufficient overlapping data ({len(aligned)} bars)"

    series = aligned["a"] / aligned["b"] if mode == "ratio" else aligned["a"] - aligned["b"]

    last   = float(series.iloc[-1])
    n      = min(20, len(series))
    mean_n = float(series.rolling(n).mean().iloc[-1])
    std_n  = float(series.rolling(n).std().iloc[-1])
    zscore = (last - mean_n) / std_n if std_n > 0 else 0.0

    prev   = float(series.iloc[-2]) if len(series) >= 2 else last
    trend  = "widening" if last > prev else "narrowing"

    z_note = (
        "EXTREME HIGH" if zscore >  2.0 else
        "HIGH"         if zscore >  1.0 else
        "EXTREME LOW"  if zscore < -2.0 else
        "LOW"          if zscore < -1.0 else
        "normal range"
    )

    n_show   = min(12, len(series))
    bars_str = " → ".join(f"{float(x):.4f}" for x in series.iloc[-n_show:].tolist())

    return (
        f"[{label}] {ticker_a} {mode} {ticker_b} @ {interval}  →  {last:.4f}\n"
        f"  {n}-bar mean: {mean_n:.4f}  |  z-score: {zscore:+.2f} ({z_note})  |  {trend}\n"
        f"  bars: {bars_str}"
    )


def _fetch_vol_one_sync(ticker: str, ref_price: float = None) -> str:
    """Sync fetch for a single vol index."""
    label_map = {"^OVX": "OVX (Crude Oil Vol)", "^GVZ": "GVZ (Gold Vol)", "^VIX": "VIX (Equity Vol)"}
    label = label_map.get(ticker, ticker)
    try:
        df = yf.download(ticker, period="1mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return f"[{label}]: no data"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        c    = df["Close"].squeeze()
        last = float(c.iloc[-1])

        prev5 = float(c.iloc[-6]) if len(c) >= 6 else float(c.iloc[0])
        chg5d = (last - prev5) / prev5 * 100 if prev5 else 0.0

        pct_rank = float((c <= last).mean() * 100)

        implied = ""
        if ref_price:
            daily_move = ref_price * (last / 100) / (252 ** 0.5)
            implied = f"  |  implied daily move: ±${daily_move:.2f}"

        trend = "rising" if chg5d > 1 else "falling" if chg5d < -1 else "flat"
        return (
            f"[{label}] {last:.2f}  ({chg5d:+.1f}% last 5d, {trend})  "
            f"30d percentile: {pct_rank:.0f}th{implied}"
        )
    except Exception as exc:
        return f"[{label}]: error — {exc}"


async def _run_vol(inp: dict, instrument_price: float = None) -> str:
    """Fetch multiple vol indices concurrently."""
    loop      = asyncio.get_event_loop()
    indices   = inp.get("indices", [])
    ref_price = inp.get("reference_price") or instrument_price
    results   = await asyncio.gather(
        *[loop.run_in_executor(None, _fetch_vol_one_sync, ticker, ref_price) for ticker in indices]
    )
    return "\n".join(results)


async def _dispatch_tool_calls(tool_blocks: list, instrument_price: float = None,
                               instrument: str = "") -> list[str]:
    """Run all tool calls from phase-1 concurrently; return result strings in same order."""
    from ib_orderbook import fetch_order_book

    async def _one(tb) -> str:
        name = tb.name
        inp  = tb.input
        if name == "fetch_market_data":
            return await _fetch_all(inp.get("requests", []))
        if name == "compute_spread_ratio":
            return await _run_spread(inp)
        if name == "fetch_implied_volatility":
            return await _run_vol(inp, instrument_price)
        if name == "fetch_order_book":
            inst     = inp.get("instrument") or instrument
            num_rows = inp.get("num_rows", 5)
            return await fetch_order_book(inst, num_rows)
        return f"Unknown tool: {name}"

    return list(await asyncio.gather(*[_one(tb) for tb in tool_blocks]))


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
                  fred_ctx=None, oil_curve=None, gld_flow=None, timeframe: str = "") -> str:
    def g(d, k, default="n/a"):
        v = (d or {}).get(k, default)
        return v if v is not None else default

    fred      = fred_ctx  or {}
    curve     = oil_curve or {}
    flow      = gld_flow  or {}

    return PHASE1_PROMPT.format(
        instrument_upper=instrument.upper(),
        timeframe_focus=_timeframe_focus(timeframe),
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
    timeframe: str = "",
) -> str:
    client        = anthropic.AsyncAnthropic()
    phase1_prompt = _build_phase1(
        instrument, ctx, dxy_ctx, corr, items, calendar_events or [],
        fred_ctx=fred_ctx, oil_curve=oil_curve, gld_flow=gld_flow,
        timeframe=timeframe,
    )
    messages      = [{"role": "user", "content": phase1_prompt}]

    # Phase 1 — Claude declares what it needs (may call multiple tools)
    try:
        resp1 = await client.messages.create(
            model=SONNET_MODEL,
            max_tokens=700,
            tools=ALL_TOOLS,
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

    tool_names = [tb.name for tb in tool_blocks]
    logger.info("Drill [%s]: %d tool call(s): %s", instrument, len(tool_blocks), tool_names)

    # Phase 2 — execute all tool calls concurrently
    instrument_price = ctx.get("price")
    results = await _dispatch_tool_calls(tool_blocks, instrument_price, instrument)

    # Phase 3 — give Claude the results, get the analysis
    tool_results = [
        {"type": "tool_result", "tool_use_id": tb.id, "content": result}
        for tb, result in zip(tool_blocks, results)
    ]
    messages = messages + [
        {"role": "assistant", "content": resp1.content},
        {"role": "user",      "content": tool_results + [{"type": "text", "text": PHASE2_INSTRUCTION}]},
    ]

    try:
        resp2 = await client.messages.create(
            model=SONNET_MODEL,
            max_tokens=600,
            tools=[SET_WATCH_TOOL],
            tool_choice={"type": "auto"},
            messages=messages,
        )

        # Extract text analysis and any set_watch calls
        text = ""
        watch_inp = None
        for block in resp2.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use" and block.name == "set_watch":
                watch_inp = block.input
        text = text.strip()

        # Create the watch if Claude requested one
        if watch_inp:
            from database import create_watch
            watch_id = await create_watch(
                instrument=instrument,
                condition=watch_inp["condition"],
                check_interval=watch_inp["check_interval_minutes"],
                expires_minutes=watch_inp["expires_minutes"],
            )
            interval = watch_inp["check_interval_minutes"]
            expires  = watch_inp["expires_minutes"]
            text += (
                f"\n\n🔭 Watch #{watch_id} set — checking every {interval}min "
                f"for up to {expires}min:\n{watch_inp['condition']}"
            )
            logger.info(
                "Watch #%d created [%s] every %dmin/%dmin: %s",
                watch_id, instrument, interval, expires, watch_inp["condition"][:80],
            )

        logger.info(
            "Drill [%s] complete — %d tool call(s), %d chars output",
            instrument, len(tool_blocks), len(text),
        )
        return text
    except Exception as exc:
        logger.error("Drill phase-3 failed: %s", exc)
        return ""
