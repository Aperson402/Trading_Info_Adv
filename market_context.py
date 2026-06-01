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
    "dxy":  "DX-Y.NYB",   # US Dollar Index (ICE)
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

    # RSI — Wilder's RMA to match TradingView (seed with SMA, then alpha=1/14)
    RSI_LEN = 14
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta.clip(upper=0))

    def _wilder_rma(s: pd.Series, n: int) -> pd.Series:
        result = np.full(len(s), np.nan)
        if len(s) < n:
            return pd.Series(result, index=s.index)
        result[n - 1] = s.iloc[:n].mean()          # SMA seed (TradingView behaviour)
        alpha = 1.0 / n
        for i in range(n, len(s)):
            result[i] = result[i - 1] * (1 - alpha) + s.iloc[i] * alpha
        return pd.Series(result, index=s.index)

    avg_gain = _wilder_rma(gain, RSI_LEN)
    avg_loss = _wilder_rma(loss, RSI_LEN)
    rs    = avg_gain / avg_loss.replace(0, np.nan)
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
    """Add HTF stoch and compute grade + SL/TP. Downgrades grade when signal opposes weekly trend."""
    ctx = {**ctx, "htf_k": round(htf_k, 1)}

    grade_fn   = ctx.pop("_grade_fn")
    sl_mult_fn = ctx.pop("_sl_mult_fn")
    ctx.pop("_vol_strong", None)
    ctx.pop("_vol_moderate", None)

    wt = ctx.get("weekly_trend", "")  # "bullish" | "bearish" | ""

    def _apply_weekly_downgrade(quality: str, is_long: bool) -> tuple[str, str]:
        against = (is_long and wt == "bearish") or (not is_long and wt == "bullish")
        if against and quality != "WEAK":
            new_quality = "MODERATE" if quality == "STRONG" else "WEAK"
            return new_quality, "↓WEEKLY"
        return quality, ""

    if ctx["long_signal"]:
        quality = grade_fn(True)
        quality, wk_note = _apply_weekly_downgrade(quality, True)
        mult    = sl_mult_fn(quality)
        ctx["signal"]          = "LONG"
        ctx["signal_grade"]    = f"{quality}{wk_note}" if wk_note else quality
        ctx["suggested_sl"]    = round(ctx["price"] - ctx["atr"] * mult, 3)
        ctx["suggested_tp"]    = round(ctx["price"] + ctx["atr"] * mult * 2, 3)
    elif ctx["short_signal"]:
        quality = grade_fn(False)
        quality, wk_note = _apply_weekly_downgrade(quality, False)
        mult    = sl_mult_fn(quality)
        ctx["signal"]          = "SHORT"
        ctx["signal_grade"]    = f"{quality}{wk_note}" if wk_note else quality
        ctx["suggested_sl"]    = round(ctx["price"] + ctx["atr"] * mult, 3)
        ctx["suggested_tp"]    = round(ctx["price"] - ctx["atr"] * mult * 2, 3)
    else:
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

    # Fetch all timeframes concurrently
    df_1h, df_4h, df_wk = await asyncio.gather(
        loop.run_in_executor(None, _fetch_ohlcv, ticker, "1h",  "10d"),
        loop.run_in_executor(None, _fetch_ohlcv, ticker, "4h",  "30d"),
        loop.run_in_executor(None, _fetch_ohlcv, ticker, "1wk", "2y"),
    )
    if df_1h is None or len(df_1h) < BB_LENGTH + 5:
        logger.warning("Insufficient 1H data for %s", ticker)
        return None

    # Compute 1H indicators
    ctx = compute_indicators(df_1h)

    # Weekly trend vs 20-week SMA — used to downgrade grades on counter-trend signals
    if df_wk is not None and len(df_wk) >= 20:
        wk_c = df_wk["Close"]
        if isinstance(wk_c, pd.DataFrame):
            wk_c = wk_c.iloc[:, 0]
        wk_sma20 = float(wk_c.rolling(20).mean().iloc[-1])
        wk_price = float(wk_c.iloc[-1])
        if not (np.isnan(wk_sma20) or np.isnan(wk_price)):
            ctx["weekly_trend"]        = "bullish" if wk_price > wk_sma20 else "bearish"
            ctx["weekly_sma20"]        = round(wk_sma20, 2)
            ctx["weekly_pct_from_sma"] = round((wk_price - wk_sma20) / wk_sma20 * 100, 1)

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

    # Override price with real-time tick so /advice and /brief show current price
    def _live_price(t: str) -> Optional[float]:
        try:
            p = yf.Ticker(t).fast_info.last_price
            return float(p) if p else None
        except Exception:
            return None

    live = await loop.run_in_executor(None, _live_price, ticker)
    if live:
        ctx["price"] = round(live, 3)

    logger.info(
        "[%s] price=%.3f regime=%s bias=%s K=%.1f signal=%s",
        ticker, ctx["price"], ctx["regime"], ctx["bias"],
        ctx["k"], ctx["signal"],
    )
    return ctx


async def get_dxy_context() -> dict:
    """Fetch DXY price, daily change, and trend vs 20-day SMA."""
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(None, _fetch_ohlcv, TICKERS["dxy"], "1d", "30d")
    if df is None or len(df) < 2:
        return {}
    price      = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    change_pct = (price - prev_close) / prev_close * 100
    sma20      = float(df["Close"].rolling(20).mean().iloc[-1])
    trend      = "bullish" if price > sma20 else "bearish"
    return {
        "price":      round(price, 2),
        "change_pct": round(change_pct, 2),
        "trend":      trend,
        "sma20":      round(sma20, 2),
        "gold_implication": "headwind" if trend == "bullish" else "tailwind",
    }


async def get_current_prices() -> dict:
    """Lightweight price-only fetch — used for 1-minute price alert checks."""
    loop = asyncio.get_event_loop()

    def _price(ticker: str) -> Optional[float]:
        try:
            p = yf.Ticker(ticker).fast_info.last_price
            return float(p) if p else None
        except Exception:
            return None

    oil, gold = await asyncio.gather(
        loop.run_in_executor(None, _price, TICKERS["oil"]),
        loop.run_in_executor(None, _price, TICKERS["gold"]),
    )
    return {"oil": oil, "gold": gold}


_WTI_MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}


def _wti_forward_ticker(months_ahead: int) -> str:
    now   = datetime.now(timezone.utc)
    m     = now.month + months_ahead
    year  = now.year + (m - 1) // 12
    month = ((m - 1) % 12) + 1
    return f"CL{_WTI_MONTH_CODES[month]}{year % 100:02d}.NYM"


async def get_oil_curve() -> dict:
    """
    Fetch WTI front, 3-month, and 6-month futures prices.
    Classifies curve structure: backwardation (tight physical = bullish)
    vs contango (oversupply = bearish).
    """
    loop  = asyncio.get_event_loop()
    t3m   = _wti_forward_ticker(3)
    t6m   = _wti_forward_ticker(6)

    def _px(ticker: str) -> Optional[float]:
        try:
            p = yf.Ticker(ticker).fast_info.last_price
            return float(p) if p else None
        except Exception:
            return None

    front, p3m, p6m = await asyncio.gather(
        loop.run_in_executor(None, _px, "CL=F"),
        loop.run_in_executor(None, _px, t3m),
        loop.run_in_executor(None, _px, t6m),
    )

    if front is None:
        return {}

    out: dict = {"front": round(front, 2), "ticker_3m": t3m, "ticker_6m": t6m}

    if p3m:
        out["p3m"] = round(p3m, 2)
    if p6m:
        out["p6m"] = round(p6m, 2)
        spread = round(front - p6m, 2)
        out["spread_6m"] = spread
        if spread > 1.0:
            out["structure"] = "backwardation"
            out["signal"]    = "bullish — physical market tight"
        elif spread > 0.25:
            out["structure"] = "mild backwardation"
            out["signal"]    = "slightly bullish"
        elif spread < -1.0:
            out["structure"] = "contango"
            out["signal"]    = "bearish — oversupply"
        elif spread < -0.25:
            out["structure"] = "mild contango"
            out["signal"]    = "slightly bearish"
        else:
            out["structure"] = "flat"
            out["signal"]    = "neutral"

    logger.info("Oil curve: front=%.2f 6m=%.2f spread=%s structure=%s",
                front, p6m or 0, out.get("spread_6m", "n/a"), out.get("structure", "n/a"))
    return out


async def get_gld_flow() -> dict:
    """
    Estimate GLD ETF holdings in tonnes from yfinance market cap + gold price.
    Stores daily snapshots; computes day-over-day and 5-day flow deltas.
    ETF outflows during geopolitical events = institutions selling, not buying.
    """
    from database import store_etf_snapshot, get_etf_snapshots
    loop = asyncio.get_event_loop()

    def _get_gld_info() -> tuple[Optional[float], Optional[float]]:
        import logging as _logging
        # Suppress yfinance's own error logger — 401 crumb errors are transient
        # and handled gracefully; no need to pollute the terminal with them.
        _yf_log = _logging.getLogger("yfinance")
        _prev   = _yf_log.level
        _yf_log.setLevel(_logging.CRITICAL)
        try:
            fi = yf.Ticker("GLD").fast_info
            mc = getattr(fi, "market_cap", None) or getattr(fi, "marketCap", None)
            return float(mc) if mc else None, None
        except Exception:
            return None, None
        finally:
            _yf_log.setLevel(_prev)

    market_cap, _ = await loop.run_in_executor(None, _get_gld_info)
    if not market_cap:
        return {}

    gold_price = await loop.run_in_executor(None, lambda: (
        float(yf.Ticker("GC=F").fast_info.last_price or 0)
    ))
    if not gold_price:
        return {}

    TROY_OZ_PER_TONNE = 32_150.7
    tonnes = market_cap / gold_price / TROY_OZ_PER_TONNE

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await store_etf_snapshot("GLD", today, market_cap, round(tonnes, 1))
    snapshots = await get_etf_snapshots("GLD", days=6)

    out: dict = {
        "tonnes":          round(tonnes, 1),
        "aum_bn":          round(market_cap / 1e9, 1),
    }

    if len(snapshots) >= 2:
        prev_tonnes        = snapshots[-2]["tonnes"]
        delta_1d           = round(tonnes - prev_tonnes, 1)
        out["delta_1d"]    = delta_1d
        if delta_1d < -5:
            out["signal"]  = "bearish — large institutional outflow"
        elif delta_1d < -1:
            out["signal"]  = "mildly bearish — small outflow"
        elif delta_1d > 5:
            out["signal"]  = "bullish — large institutional inflow"
        elif delta_1d > 1:
            out["signal"]  = "mildly bullish — small inflow"
        else:
            out["signal"]  = "neutral — flat holdings"

    if len(snapshots) >= 5:
        out["delta_5d"] = round(tonnes - snapshots[0]["tonnes"], 1)

    logger.info("GLD ETF: %.1ft (Δ1d=%+.1ft) — %s",
                tonnes, out.get("delta_1d", 0), out.get("signal", "n/a"))
    return out


async def get_correlated_context(instrument: str) -> dict:
    """
    Fetch short-timeframe correlated data for /drill:
      - 15-min bars for the instrument and DXY (last 2 hours of micro-movement)
      - US 10-year yield (^TNX)  — primary gold driver, not otherwise in system
      - Silver (SI=F)            — gold confirmation signal
      - Natural gas (NG=F)       — oil/energy correlation
    """
    loop = asyncio.get_event_loop()

    CORR_TICKERS: dict[str, str] = {
        "instrument_15m": TICKERS.get(instrument, ""),
        "dxy_15m":        TICKERS["dxy"],
        "10yr_yield":     "^TNX",
    }
    if instrument == "gold":
        CORR_TICKERS["silver"] = "SI=F"
    elif instrument == "oil":
        CORR_TICKERS["nat_gas"] = "NG=F"

    def _summarise_15m(ticker: str) -> Optional[dict]:
        try:
            df = yf.download(ticker, period="2d", interval="15m",
                             progress=False, auto_adjust=True)
            if df.empty or len(df) < 4:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            c = df["Close"]
            recent = c.iloc[-8:] if len(c) >= 8 else c
            bars   = [round(float(x), 3) for x in recent.tolist()]
            last   = bars[-1]
            chg_1h = round((last - bars[-4]) / bars[-4] * 100, 2) if len(bars) >= 4 else 0.0
            # simple linear direction over last 8 bars
            if len(bars) >= 4:
                first_half = sum(bars[:len(bars)//2]) / (len(bars)//2)
                second_half = sum(bars[len(bars)//2:]) / (len(bars) - len(bars)//2)
                direction = "rising" if second_half > first_half * 1.0005 else \
                            "falling" if second_half < first_half * 0.9995 else "flat"
            else:
                direction = "flat"
            return {"price": last, "change_1h_pct": chg_1h,
                    "direction": direction, "bars": bars}
        except Exception:
            return None

    results = await asyncio.gather(
        *[loop.run_in_executor(None, _summarise_15m, t) for t in CORR_TICKERS.values()],
        return_exceptions=True,
    )
    out = {}
    for key, res in zip(CORR_TICKERS.keys(), results):
        out[key] = None if isinstance(res, Exception) else res
    return out


async def get_both_contexts() -> dict:
    """Fetch oil, gold, DXY, oil futures curve, and GLD ETF flow concurrently."""
    oil_ctx, gold_ctx, dxy_ctx, oil_curve, gld_flow = await asyncio.gather(
        get_market_context("oil"),
        get_market_context("gold"),
        get_dxy_context(),
        get_oil_curve(),
        get_gld_flow(),
        return_exceptions=True,
    )
    return {
        "oil":       oil_ctx   if not isinstance(oil_ctx,   Exception) else None,
        "gold":      gold_ctx  if not isinstance(gold_ctx,  Exception) else None,
        "dxy":       dxy_ctx   if not isinstance(dxy_ctx,   Exception) else {},
        "oil_curve": oil_curve if not isinstance(oil_curve, Exception) else {},
        "gld_flow":  gld_flow  if not isinstance(gld_flow,  Exception) else {},
    }
