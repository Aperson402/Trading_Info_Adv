"""
advice.py — on-demand real-time trading advice via /advice command.

Answers: "where is the market right now and what do I do?"
Uses a 4-hour news lookback, current technicals, sentiment, and COT.
"""

import logging
import re
from datetime import datetime, timezone

import anthropic

logger = logging.getLogger(__name__)

SONNET_MODEL = "claude-sonnet-4-6"

ADVICE_PROMPT = """\
You are a commodity trading advisor. A trader just asked for real-time advice \
on oil (WTI/Brent CFDs) and gold (XAU/USD CFDs).

CURRENT TIME: {now} UTC

RECENT NEWS (last 4 hours, {n_items} items):
{items_text}

OIL (WTI) — ${oil_price}  |  BB: {oil_upper} / {oil_basis} / {oil_lower}  |  Trend: {oil_trend}
Regime: {oil_regime} | Bias: {oil_bias} | Squeeze: {oil_squeeze}
BB Pos: {oil_bb} | K={oil_k} D={oil_d} ({oil_stoch_zone}) | HTF K={oil_htf_k} | RSI: {oil_rsi}
ATR: {oil_atr} | Vol: {oil_vol_pct}% ({oil_vol_ratio}x avg)
Signal: {oil_signal} {oil_grade} | Weekly: {oil_weekly}
{oil_sl_tp}

GOLD (XAU/USD) — ${gold_price}  |  BB: {gold_upper} / {gold_basis} / {gold_lower}  |  Trend: {gold_trend}
Regime: {gold_regime} | Bias: {gold_bias} | Squeeze: {gold_squeeze}
BB Pos: {gold_bb} | K={gold_k} D={gold_d} ({gold_stoch_zone}) | HTF K={gold_htf_k} | RSI: {gold_rsi}
ATR: {gold_atr} | Vol: {gold_vol_pct}% ({gold_vol_ratio}x avg)
Signal: {gold_signal} {gold_grade} | Weekly: {gold_weekly}
{gold_sl_tp}

24H SENTIMENT:
{sentiment}

DXY — {dxy_price} ({dxy_change:+.2f}% today, {dxy_trend} vs SMA20) — Gold: {dxy_implication}

REAL RATES & FLOWS (FRED — prev biz day):
10yr real yield: {real_yield}% ({real_yield_chg:+.3f}%) — {gold_yield_signal}
5yr breakeven: {breakeven_5y}% — {inflation_signal}
GLD ETF: {gld_tonnes}t ({gld_delta:+.1f}t today) — {gld_signal}
Oil curve: {curve_structure} — spread {curve_spread:+.2f} — {curve_signal}

COT POSITIONING:
{cot_oil}
{cot_gold}

EVENTS TODAY:
{today_events}

Respond in exactly this format. Be direct. No hedging. Under 200 words total.

OIL — [LONG / SHORT / WAIT]
[1-2 sentences on what's driving price right now]
[If setup exists: Entry: $X  SL: $X  TP: $X]
[If no setup: "Wait for: [specific condition]"]

GOLD — [LONG / SHORT / WAIT]
[1-2 sentences on what's driving price right now]
[If setup exists: Entry: $X  SL: $X  TP: $X]
[If no setup: "Wait for: [specific condition]"]

WATCH:
[2 bullet points — things that could change the picture in the next few hours]\
"""


def _fmt_items(items: list[dict]) -> str:
    if not items:
        return "(no new items in last 4 hours)"
    lines = []
    for item in items[-10:]:  # cap at 10 most recent
        inst = item.get("instrument", "")
        dirn = item.get("direction", "")
        tag = f"[{inst.upper()} {dirn.upper()}] " if inst and inst != "neither" else ""
        lines.append(f"• {tag}{item.get('title', '')}")
    return "\n".join(lines)


def _fmt_sentiment(sentiment: dict) -> str:
    lines = []
    for inst in ("oil", "gold"):
        s = sentiment.get(inst, {})
        total = s.get("total", 0)
        if total == 0:
            lines.append(f"{inst.upper()}: no data")
            continue
        bull, bear, neut = s.get("bullish", 0), s.get("bearish", 0), s.get("neutral", 0)
        lines.append(f"{inst.upper()}: {bull}↑ {bear}↓ {neut}→ of {total}")
    return "  ".join(lines)


def _fmt_cot(cot, instrument: str) -> str:
    if not cot:
        return f"{instrument.upper()} COT: unavailable"
    net = cot["net_contracts"]
    direction = "NET LONG" if net > 0 else "NET SHORT"
    return (
        f"{instrument.upper()} COT: {direction} {abs(net):,} "
        f"({cot['pct_rank']:.0f}th pct, {cot['label'].upper()})"
    )


def _weekly_str(ctx: dict) -> str:
    wt  = (ctx or {}).get("weekly_trend", "")
    pct = (ctx or {}).get("weekly_pct_from_sma")
    if not wt:
        return "n/a"
    pct_str = f" ({pct:+.1f}% vs 20W SMA)" if pct is not None else ""
    return f"{wt.upper()}{pct_str}"


def parse_advice_signals(advice_text: str) -> dict:
    """Parse LONG/SHORT per instrument from advice output. Returns {'oil': 'long', 'gold': None, ...}"""
    signals: dict = {}
    for m in re.finditer(r'^(OIL|GOLD) — (LONG|SHORT|WAIT)', advice_text, re.MULTILINE):
        inst      = m.group(1).lower()
        direction = m.group(2).lower()
        signals[inst] = direction if direction in ("long", "short") else None
    return signals


def _fmt_sl_tp(ctx: dict) -> str:
    if ctx.get("signal") and ctx["signal"] != "NONE":
        return f"SL: {ctx.get('suggested_sl', 'n/a')} | TP: {ctx.get('suggested_tp', 'n/a')}"
    return ""


def _fmt_today_events(calendar_events: list[dict]) -> str:
    now = datetime.now(timezone.utc)
    today = [
        e for e in calendar_events
        if e.get("datetime") and e["datetime"].date() == now.date()
    ]
    if not today:
        return "No high-impact events today."
    lines = []
    for e in today:
        when = e["datetime"].strftime("%H:%M UTC")
        forecast = f" (f: {e['forecast']})" if e.get("forecast") else ""
        lines.append(f"• {when} {e['title']}{forecast}")
    return "\n".join(lines)


def _build_prompt(items, oil_ctx, gold_ctx, sentiment, cot_data, calendar_events,
                  dxy_ctx=None, fred_ctx=None, oil_curve=None, gld_flow=None) -> str:
    def g(ctx, key, default="n/a"):
        v = (ctx or {}).get(key, default)
        return v if v is not None else default

    cot_data  = cot_data  or {}
    dxy_ctx   = dxy_ctx   or {}
    fred_ctx  = fred_ctx  or {}
    oil_curve = oil_curve or {}
    gld_flow  = gld_flow  or {}

    return ADVICE_PROMPT.format(
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        n_items=len(items),
        items_text=_fmt_items(items),
        oil_price=g(oil_ctx, "price"),
        oil_upper=g(oil_ctx, "upper"),
        oil_basis=g(oil_ctx, "basis"),
        oil_lower=g(oil_ctx, "lower"),
        oil_trend=g(oil_ctx, "trend"),
        oil_regime=g(oil_ctx, "regime"),
        oil_bias=g(oil_ctx, "bias"),
        oil_squeeze="YES" if oil_ctx.get("squeeze") else "NO",
        oil_bb=g(oil_ctx, "bb_pos"),
        oil_k=g(oil_ctx, "k"),
        oil_d=g(oil_ctx, "d"),
        oil_stoch_zone=g(oil_ctx, "stoch_zone"),
        oil_htf_k=g(oil_ctx, "htf_k"),
        oil_rsi=g(oil_ctx, "rsi"),
        oil_atr=g(oil_ctx, "atr"),
        oil_vol_pct=g(oil_ctx, "vol_pct"),
        oil_vol_ratio=g(oil_ctx, "vol_ratio"),
        oil_signal=g(oil_ctx, "signal", "NONE"),
        oil_grade=g(oil_ctx, "signal_grade", ""),
        oil_weekly=_weekly_str(oil_ctx),
        oil_sl_tp=_fmt_sl_tp(oil_ctx),
        gold_price=g(gold_ctx, "price"),
        gold_upper=g(gold_ctx, "upper"),
        gold_basis=g(gold_ctx, "basis"),
        gold_lower=g(gold_ctx, "lower"),
        gold_trend=g(gold_ctx, "trend"),
        gold_regime=g(gold_ctx, "regime"),
        gold_bias=g(gold_ctx, "bias"),
        gold_squeeze="YES" if gold_ctx.get("squeeze") else "NO",
        gold_bb=g(gold_ctx, "bb_pos"),
        gold_k=g(gold_ctx, "k"),
        gold_d=g(gold_ctx, "d"),
        gold_stoch_zone=g(gold_ctx, "stoch_zone"),
        gold_htf_k=g(gold_ctx, "htf_k"),
        gold_rsi=g(gold_ctx, "rsi"),
        gold_atr=g(gold_ctx, "atr"),
        gold_vol_pct=g(gold_ctx, "vol_pct"),
        gold_vol_ratio=g(gold_ctx, "vol_ratio"),
        gold_signal=g(gold_ctx, "signal", "NONE"),
        gold_grade=g(gold_ctx, "signal_grade", ""),
        gold_weekly=_weekly_str(gold_ctx),
        gold_sl_tp=_fmt_sl_tp(gold_ctx),
        sentiment=_fmt_sentiment(sentiment or {}),
        dxy_price=dxy_ctx.get("price", "n/a"),
        dxy_change=dxy_ctx.get("change_pct", 0.0),
        dxy_trend=dxy_ctx.get("trend", "n/a"),
        dxy_implication=dxy_ctx.get("gold_implication", "n/a"),
        # FRED
        real_yield=fred_ctx.get("real_yield_10y", "n/a"),
        real_yield_chg=fred_ctx.get("real_yield_10y_chg") or 0.0,
        gold_yield_signal=fred_ctx.get("gold_yield_signal", "n/a"),
        breakeven_5y=fred_ctx.get("breakeven_5y", "n/a"),
        inflation_signal=fred_ctx.get("inflation_signal", "n/a"),
        # GLD flow
        gld_tonnes=gld_flow.get("tonnes", "n/a"),
        gld_delta=gld_flow.get("delta_1d") or 0.0,
        gld_signal=gld_flow.get("signal", "n/a"),
        # Oil curve
        curve_structure=oil_curve.get("structure", "n/a"),
        curve_spread=oil_curve.get("spread_6m") or 0.0,
        curve_signal=oil_curve.get("signal", "n/a"),
        cot_oil=_fmt_cot(cot_data.get("oil"), "oil"),
        cot_gold=_fmt_cot(cot_data.get("gold"), "gold"),
        today_events=_fmt_today_events(calendar_events or []),
    )


async def generate_advice(
    items: list[dict],
    oil_ctx: dict,
    gold_ctx: dict,
    sentiment: dict,
    cot_data: dict = None,
    calendar_events: list = None,
    dxy_ctx: dict = None,
    fred_ctx: dict = None,
    oil_curve: dict = None,
    gld_flow: dict = None,
) -> str:
    client = anthropic.AsyncAnthropic()
    prompt = _build_prompt(items, oil_ctx, gold_ctx, sentiment, cot_data, calendar_events,
                           dxy_ctx, fred_ctx, oil_curve, gld_flow)

    try:
        response = await client.messages.create(
            model=SONNET_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        logger.info("Advice generated (%d chars)", len(text))
        return text
    except Exception as exc:
        logger.error("Advice generation failed: %s", exc)
        return ""
