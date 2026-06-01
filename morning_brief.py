"""
morning_brief.py — Claude Sonnet synthesis for the 06:45 UTC morning brief.

Incorporates:
  - Overnight classified items
  - Live market context (price, BB, stoch, regime)
  - COT positioning (CFTC percentile rank)
  - Economic calendar (high-impact events this week)
"""

import logging
import re

import anthropic
from econ_calendar import fmt_calendar_for_brief

logger = logging.getLogger(__name__)

SONNET_MODEL = "claude-sonnet-4-6"

MORNING_BRIEF_PROMPT = """\
You are preparing a pre-market briefing for a commodity trader who trades \
oil (WTI/Brent CFDs) and gold (XAU/USD CFDs) during London open (07:00-10:00 UTC).

OVERNIGHT ITEMS ({n_items} items, last 12 hours):
{items_text}

CURRENT MARKET STATE:

OIL (WTI) - {oil_price}  |  BB: {oil_upper} / {oil_basis} / {oil_lower}  |  Trend: {oil_trend}
Regime: {oil_regime} | Bias: {oil_bias} | Squeeze: {oil_squeeze}
BB Position: {oil_bb} | Stoch K={oil_k} D={oil_d} ({oil_stoch_zone}) | HTF K={oil_htf_k}
RSI: {oil_rsi} | ATR: {oil_atr} | Vol: {oil_vol_pct}% ({oil_vol_ratio}x avg)
Signal: {oil_signal} {oil_grade} | Weekly: {oil_weekly}
{oil_sl_tp}

GOLD (XAU/USD) - {gold_price}  |  BB: {gold_upper} / {gold_basis} / {gold_lower}  |  Trend: {gold_trend}
Regime: {gold_regime} | Bias: {gold_bias} | Squeeze: {gold_squeeze}
BB Position: {gold_bb} | Stoch K={gold_k} D={gold_d} ({gold_stoch_zone}) | HTF K={gold_htf_k}
RSI: {gold_rsi} | ATR: {gold_atr} | Vol: {gold_vol_pct}% ({gold_vol_ratio}x avg)
Signal: {gold_signal} {gold_grade} | Weekly: {gold_weekly}
{gold_sl_tp}

US DOLLAR INDEX (DXY) — {dxy_price} ({dxy_change:+.2f}% today)
Trend: {dxy_trend} vs SMA20 ({dxy_sma20}) — Gold implication: {dxy_implication}

REAL RATES & ETF FLOWS (FRED — prev business day):
10yr real yield: {real_yield}% ({real_yield_chg:+.3f}%) — {gold_yield_signal}
5yr breakeven inflation: {breakeven_5y}% — {inflation_signal}
GLD ETF: {gld_tonnes}t ({gld_delta:+.1f}t today) — {gld_signal}

OIL FUTURES CURVE: {curve_structure} — front ${oil_front} vs 6m ${oil_6m} (spread {curve_spread:+.2f}) — {curve_signal}

24-HOUR SENTIMENT WINDOW:
{sentiment}

COT POSITIONING (latest CFTC report):
{cot_oil}
{cot_gold}

HIGH-IMPACT EVENTS THIS WEEK:
{calendar}

Write a morning brief covering exactly these four sections. Be direct and specific. \
No fluff, no disclaimers. Under 300 words total.

Format your response exactly like this:

OIL - [BULLISH/BEARISH/NEUTRAL] $PRICE
[2-3 sentences on dominant overnight narrative and direction]
[1 sentence on COT - is the trade crowded?]
[1-2 sentences on whether technical setup aligns with fundamentals]

GOLD - [BULLISH/BEARISH/NEUTRAL] $PRICE
[2-3 sentences on dominant overnight narrative and direction]
[1 sentence on COT - is the trade crowded?]
[1-2 sentences on whether technical setup aligns with fundamentals]

KEY RISK TODAY:
[1-2 sentences on the most important scheduled event or developing story]

AT LONDON OPEN:
[2-3 sentences on what to watch for each instrument and what entry requires]\
"""


def _weekly_str(ctx: dict) -> str:
    wt  = (ctx or {}).get("weekly_trend", "")
    pct = (ctx or {}).get("weekly_pct_from_sma")
    if not wt:
        return "n/a"
    pct_str = f" ({pct:+.1f}% vs 20W SMA)" if pct is not None else ""
    return f"{wt.upper()}{pct_str}"


def parse_brief_signals(brief_text: str) -> dict:
    """Parse BULLISH/BEARISH per instrument from morning brief output. Returns {'oil': 'long', ...}"""
    signals: dict = {}
    for m in re.finditer(r'^(OIL|GOLD) - (BULLISH|BEARISH|NEUTRAL)', brief_text, re.MULTILINE):
        inst      = m.group(1).lower()
        direction = m.group(2).lower()
        signals[inst] = "long" if direction == "bullish" else "short" if direction == "bearish" else None
    return signals


def _fmt_items(items: list[dict]) -> str:
    if not items:
        return "(no relevant items)"
    lines = []
    for item in items:
        inst = item.get("instrument", "")
        dirn = item.get("direction", "")
        conf = item.get("confidence", "")
        reasoning = item.get("reasoning", "")
        tag = f"[{inst.upper()} {dirn.upper()} {conf}/10] " if inst and inst != "neither" else ""
        lines.append(f"* {tag}{item.get('title', '')} - {reasoning}")
    return "\n".join(lines)


def _fmt_sl_tp(ctx: dict) -> str:
    if ctx.get("signal") and ctx["signal"] != "NONE":
        return f"SL: {ctx.get('suggested_sl', 'n/a')} | TP: {ctx.get('suggested_tp', 'n/a')}"
    return ""


def _fmt_cot(cot, instrument: str) -> str:
    if not cot:
        return f"{instrument.upper()} COT: unavailable"
    net = cot["net_contracts"]
    direction = "NET LONG" if net > 0 else "NET SHORT"
    wow = cot["wow_change"]
    wow_str = f"+{wow:,}" if wow >= 0 else f"{wow:,}"
    return (
        f"{instrument.upper()} COT ({cot['report_date']}): "
        f"{direction} {abs(net):,} contracts "
        f"({cot['pct_rank']:.0f}th percentile, {cot['label'].upper()}) "
        f"WoW: {wow_str}"
    )


def _fmt_sentiment(sentiment: dict) -> str:
    lines = []
    for inst in ("oil", "gold"):
        s = sentiment.get(inst, {})
        total = s.get("total", 0)
        if total == 0:
            lines.append(f"{inst.upper()}: no classified items in last 24h")
            continue
        bull = s.get("bullish", 0)
        bear = s.get("bearish", 0)
        neut = s.get("neutral", 0)
        dominant = max(("bullish", bull), ("bearish", bear), ("neutral", neut), key=lambda x: x[1])[0]
        lines.append(
            f"{inst.upper()}: {bull} bullish / {bear} bearish / {neut} neutral "
            f"({total} total) — dominant: {dominant.upper()}"
        )
    return "\n".join(lines)


def _build_prompt(items, oil_ctx, gold_ctx, cot_data=None, calendar_events=None,
                  sentiment=None, dxy_ctx=None, fred_ctx=None, oil_curve=None, gld_flow=None) -> str:
    def g(ctx, key, default="n/a"):
        v = (ctx or {}).get(key, default)
        return v if v is not None else default

    cot_data  = cot_data  or {}
    dxy_ctx   = dxy_ctx   or {}
    fred_ctx  = fred_ctx  or {}
    oil_curve = oil_curve or {}
    gld_flow  = gld_flow  or {}
    cal_str   = fmt_calendar_for_brief(calendar_events or [])
    sent_str  = _fmt_sentiment(sentiment or {})

    return MORNING_BRIEF_PROMPT.format(
        n_items=len(items),
        items_text=_fmt_items(items),
        sentiment=sent_str,
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
        dxy_price=dxy_ctx.get("price", "n/a"),
        dxy_change=dxy_ctx.get("change_pct", 0.0),
        dxy_trend=dxy_ctx.get("trend", "n/a"),
        dxy_sma20=dxy_ctx.get("sma20", "n/a"),
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
        oil_front=oil_curve.get("front", "n/a"),
        oil_6m=oil_curve.get("p6m", "n/a"),
        curve_spread=oil_curve.get("spread_6m") or 0.0,
        curve_signal=oil_curve.get("signal", "n/a"),
        cot_oil=_fmt_cot(cot_data.get("oil"),  "oil"),
        cot_gold=_fmt_cot(cot_data.get("gold"), "gold"),
        calendar=cal_str,
    )


async def generate_morning_brief(
    items: list[dict],
    oil_ctx: dict,
    gold_ctx: dict,
    cot_data: dict = None,
    calendar_events: list = None,
    sentiment: dict = None,
    dxy_ctx: dict = None,
    fred_ctx: dict = None,
    oil_curve: dict = None,
    gld_flow: dict = None,
) -> str:
    client = anthropic.AsyncAnthropic()
    prompt = _build_prompt(items, oil_ctx, gold_ctx, cot_data, calendar_events,
                           sentiment, dxy_ctx, fred_ctx, oil_curve, gld_flow)

    try:
        response = await client.messages.create(
            model=SONNET_MODEL,
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        brief = response.content[0].text.strip()
        logger.info("Morning brief generated (%d chars)", len(brief))
        return brief
    except Exception as exc:
        logger.error("Morning brief generation failed: %s", exc)
        return ""