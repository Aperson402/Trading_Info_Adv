"""
market_context.py — fetches live 1H OHLCV and computes BB + Stochastic context.

Exact translation of the Pine Script indicator:
  BB:    SMA(20), 2.0 std devs
  Stoch: K=14, D=SMA(K,3), OB=80, OS=20
  Regime: squeeze / trending / breakout / ranging
  Signal: longSignal  = nearLower AND oversold  AND kCrossedAbove
           shortSignal = nearUpper AND overbought AND kCrossedBelow
  Grade:  STRONG / MODERATE / WEAK based on volume percentile + HTF stoch alignment
  SL/TP:  ATR-based, multiplier varies by grade
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Ticker map ────────────────────────────────────────────────────────────────
TICKERS = {
    "oil":  "CL=F",   # WTI Crude front-month futures
    "gold": "GC=F",   # Gold front-month futures
}

# ── Pine Script parameters (exact match) ─────────────────────────────────────
BB_LENGTH    = 20
BB_MULT      = 2.0
STOCH_LENGTH = 14
STOCH_SMOOTH = 3
OB_LEVEL     = 80
OS_LEVEL     = 20
ATR_LENGTH   = 14
VOL_AVG_LEN  = 20
BW_SMA_LEN   = 20    # for squeeze detection
HTF_TF       = "4h"  # Pine Script default was 60min HTF → use 4H as meaningful higher TF


# ── Indicator calculations ────────────────────────────────────────────────────

def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def _stdev(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).std(ddof=0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    hl  = high - low
    hpc = (high - close.shift(1)).abs()
    lpc = (low  - close.shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    lowest  = low.rolling(n).min()
    highest = high.rolling(n).max()
    return (close - lowest) / (highest - lowest) * 100


def _vol_percentile(volume: pd.Series, n: int = 100) -> pd.Series:
    """Rolling percentile rank of volume — equivalent to ta.percentrank."""
    def prank(x):
        return (x[:-1] < x[-1]).sum() / max(len(x) - 1, 1) * 100
    return volume.rolling(n + 1, min_periods=2).apply(prank, raw=True)


def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Given a DataFrame with columns [Open, High, Low, Close, Volume],
    compute all Pine Script indicator values for the most recent bar.
    Returns a dict of scalar values.
    """
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    # Bollinger Bands
    basis = _sma(c, BB_LENGTH)
    dev   = _stdev(c, BB_LENGTH)
    upper = basis + BB_MULT * dev
    lower = basis - BB_MULT * dev
    bw    = upper - lower
    bw_sma = _sma(bw, BW_SMA_LEN)

    # ATR
    atr = _atr(h, l, c, ATR_LENGTH)

    # Stochastic
    k_raw = _stoch_k(h, l, c, STOCH_LENGTH)
    k     = k_raw
    d     = _sma(k, STOCH_SMOOTH)

    # Volume
    vol_avg  = _sma(v, VOL_AVG_LEN)
    vol_ratio = v / vol_avg
    vol_pct  = _vol_percentile(v, 100)

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - 100 / (1 + rs)

    # Pull last 4 bars for directional checks (Pine uses [3] lookback)
    def last(s, n=0):
        return s.iloc[-(1 + n)] if len(s) > n else np.nan

    # Current values
    close_now  = last(c)
    upper_now  = last(upper)
    lower_now  = last(lower)
    basis_now  = last(basis)
    bw_now     = last(bw)
    bw_sma_now = last(bw_sma)
    atr_now    = last(atr)
    k_now      = last(k)
    d_now      = last(d)
    k_prev     = last(k, 1)
    d_prev     = last(d, 1)
    rsi_now    = last(rsi)
    vol_pct_now  = last(vol_pct)
    vol_ratio_now = last(vol_ratio)

    # Directional checks (3 bars ago)
    lower_3    = last(lower, 3)
    upper_3    = last(upper, 3)
    bw_3       = last(bw, 3)

    lb_rising  = lower_now > lower_3
    ub_falling = upper_now < upper_3
    expanding  = bw_now > bw_3
    squeeze    = bw_now < bw_sma_now * 0.75

    # Regime (exact Pine Script logic)
    if squeeze:
        regime = "Squeeze"
    elif lb_rising and not ub_falling:
        regime = "Trending ↑"
    elif ub_falling and not lb_rising:
        regime = "Trending ↓"
    elif expanding and lb_rising:
        regime = "Breakout ↑"
    elif expanding and ub_falling:
        regime = "Breakout ↓"
    elif not lb_rising and not ub_falling:
        regime = "Ranging"
    else:
        regime = "Transition"

    # Signal conditions
    near_lower     = close_now <= lower_now * 1.002
    near_upper     = close_now >= upper_now * 0.998
    oversold       = k_now < OS_LEVEL
    overbought     = k_now > OB_LEVEL
    k_crossed_above = k_prev < d_prev and k_now >= d_now   # crossover
    k_crossed_below = k_prev > d_prev and k_now <= d_now   # crossunder

    long_signal  = near_lower and oversold  and k_crossed_above
    short_signal = near_upper and overbought and k_crossed_below

    # Signal quality grade
    vol_strong   = vol_pct_now > 70
    vol_moderate = vol_ratio_now > 1.2

    # Bias
    if k_now > 50 and close_now > basis_now:
        bias = "BULLISH"
    elif k_now < 50 and close_now < basis_now:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    # BB position label
    if close_now > upper_now:
        bb_pos = "Above Upper"
    elif close_now < lower_now:
        bb_pos = "Below Lower"
    elif close_now > basis_now:
        bb_pos = "Above Mid"
    else:
        bb_pos = "Below Mid"

    # Stoch zone
    if k_now > OB_LEVEL:
        stoch_zone = "Overbought"
    elif k_now < OS_LEVEL:
        stoch_zone = "Oversold"
    elif k_now > 50:
        stoch_zone = "Mid-High"
    else:
        stoch_zone = "Mid-Low"

    # Trend string
    if lb_rising and not ub_falling:
        trend_str = "Up"
    elif ub_falling and not lb_rising:
        trend_str = "Down"
    else:
        trend_str = "Sideways"

    # Grade scoring
    def _grade(is_long: bool) -> str:
        htf_aligned = htf_k_val < 40 if is_long else htf_k_val > 60
        score = (2 if vol_strong else 1 if vol_moderate else 0) + (1 if htf_aligned else 0)
        return "STRONG" if score >= 3 else "MODERATE" if score >= 1 else "WEAK"

    # SL/TP multipliers
    def _sl_mult(quality: str) -> float:
        return 1.2 if quality == "STRONG" else 1.5 if quality == "MODERATE" else 2.0

    return {
        # Price
        "price":        round(close_now, 3),
        "upper":        round(upper_now, 3),
        "lower":        round(lower_now, 3),
        "basis":        round(basis_now, 3),
        "atr":          round(atr_now, 3),
        # Stochastic
        "k":            round(k_now, 1),
        "d":            round(d_now, 1),
        "stoch_zone":   stoch_zone,
        # Regime / bias
        "regime":       regime,
        "bias":         bias,
        "bb_pos":       bb_pos,
        "trend":        trend_str,
        "squeeze":      squeeze,
        # Signals
        "long_signal":  long_signal,
        "short_signal": short_signal,
        # Context
        "rsi":          round(rsi_now, 1),
        "vol_pct":      round(vol_pct_now, 0),
        "vol_ratio":    round(vol_ratio_now, 2),
        # HTF placeholder — filled after second fetch
        "htf_k":        None,
        # Grade (set after htf_k is known)
        "_vol_strong":  vol_strong,
        "_vol_moderate": vol_moderate,
        "_sl_mult_fn":  _sl_mult,
        "_grade_fn":    _grade,
    }


def finalize_with_htf(ctx: dict, htf_k: float) -> dict:
    """Add HTF stoch and compute grade + SL/TP."""
    ctx = {**ctx, "htf_k": round(htf_k, 1)}

    grade_fn  = ctx.pop("_grade_fn")
    sl_mult_fn = ctx.pop("_sl_mult_fn")
    ctx.pop("_vol_strong", None)
    ctx.pop("_vol_moderate", None)

    if ctx["long_signal"]:
        quality = grade_fn(True)
        mult    = sl_mult_fn(quality)
        ctx["signal"]       = "LONG"
        ctx["signal_grade"] = quality
        ctx["suggested_sl"] = round(ctx["price"] - ctx["atr"] * mult, 3)
        ctx["suggested_tp"] = round(ctx["price"] + ctx["atr"] * mult * 2, 3)
    elif ctx["short_signal"]:
        quality = grade_fn(False)
        mult    = sl_mult_fn(quality)
        ctx["signal"]       = "SHORT"
        ctx["signal_grade"] = quality
        ctx["suggested_sl"] = round(ctx["price"] + ctx["atr"] * mult, 3)
        ctx["suggested_tp"] = round(ctx["price"] - ctx["atr"] * mult * 2, 3)
    else:
        # Suggested levels from bias
        mult = 1.5
        ctx["signal"]       = "NONE"
        ctx["signal_grade"] = "—"
        if ctx["bias"] == "BULLISH":
            ctx["suggested_sl"] = round(ctx["price"] - ctx["atr"] * mult, 3)
            ctx["suggested_tp"] = round(ctx["price"] + ctx["atr"] * mult * 2, 3)
        elif ctx["bias"] == "BEARISH":
            ctx["suggested_sl"] = round(ctx["price"] + ctx["atr"] * mult, 3)
            ctx["suggested_tp"] = round(ctx["price"] - ctx["atr"] * mult * 2, 3)
        else:
            ctx["suggested_sl"] = None
            ctx["suggested_tp"] = None

    return ctx


def _fetch_ohlcv(ticker: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    """Download OHLCV — runs in a thread (yfinance is synchronous)."""
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as exc:
        logger.warning("yfinance download failed for %s: %s", ticker, exc)
        return None


async def get_market_context(instrument: str) -> Optional[dict]:
    """
    Fetch 1H OHLCV + 4H HTF for the given instrument ('oil' or 'gold').
    Returns a fully computed context dict, or None on failure.
    """
    ticker = TICKERS.get(instrument)
    if not ticker:
        logger.warning("Unknown instrument: %s", instrument)
        return None

    loop = asyncio.get_event_loop()

    # Fetch 1H data (60 bars = ~3 trading days, enough for all indicators)
    df_1h = await loop.run_in_executor(
        None, _fetch_ohlcv, ticker, "1h", "10d"
    )
    if df_1h is None or len(df_1h) < BB_LENGTH + 5:
        logger.warning("Insufficient 1H data for %s", ticker)
        return None

    # Fetch 4H for HTF stoch
    df_4h = await loop.run_in_executor(
        None, _fetch_ohlcv, ticker, "4h", "30d"
    )

    # Compute 1H indicators
    ctx = compute_indicators(df_1h)

    # Compute HTF K
    htf_k = 50.0  # default neutral if unavailable
    if df_4h is not None and len(df_4h) >= STOCH_LENGTH:
        htf_k_series = _stoch_k(df_4h["High"], df_4h["Low"], df_4h["Close"], STOCH_LENGTH)
        if not htf_k_series.empty and not pd.isna(htf_k_series.iloc[-1]):
            htf_k = float(htf_k_series.iloc[-1])

    ctx = finalize_with_htf(ctx, htf_k)
    ctx["instrument"] = instrument
    ctx["ticker"]     = ticker
    ctx["fetched_at"] = datetime.now(timezone.utc).strftime("%H:%M UTC")

    logger.info(
        "[%s] price=%.3f regime=%s bias=%s K=%.1f signal=%s",
        ticker, ctx["price"], ctx["regime"], ctx["bias"],
        ctx["k"], ctx["signal"],
    )
    return ctx


async def get_both_contexts() -> dict:
    """Fetch oil and gold contexts concurrently."""
    oil_ctx, gold_ctx = await asyncio.gather(
        get_market_context("oil"),
        get_market_context("gold"),
        return_exceptions=True,
    )
    return {
        "oil":  oil_ctx  if not isinstance(oil_ctx,  Exception) else None,
        "gold": gold_ctx if not isinstance(gold_ctx, Exception) else None,
    }
