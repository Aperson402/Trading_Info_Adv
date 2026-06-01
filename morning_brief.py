"""
morning_brief.py — Claude Sonnet synthesis for the 06:45 UTC morning brief.

Incorporates:
  - Overnight classified items
  - Live market context (price, BB, stoch, regime)
  - COT positioning (CFTC percentile rank)
  - Economic calendar (high-impact events this week)
"""

import logging

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

OIL (WTI) - {oil_price}
Regime: {oil_regime} | Bias: {oil_bias}
BB Position: {oil_bb} | Stoch K={oil_k} ({oil_stoch_zone}) | HTF K={oil_htf_k}
RSI: {oil_rsi} | ATR: {oil_atr}
Signal: {oil_signal} {oil_grade}
{oil_sl_tp}

GOLD (XAU/USD) - {gold_price}
Regime: {gold_regime} | Bias: {gold_bias}
BB Position: {gold_bb} | Stoch K={gold_k} ({gold_stoch_zone}) | HTF K={gold_htf_k}
RSI: {gold_rsi} | ATR: {gold_atr}
Signal: {gold_signal} {gold_grade}
{gold_sl_tp}

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


def _build_prompt(items, oil_ctx, gold_ctx, cot_data=None, calendar_events=None) -> str:
    def g(ctx, key, default="n/a"):
        v = ctx.get(key, default)
        return v if v is not None else default

    cot_data = cot_data or {}
    cal_str  = fmt_calendar_for_brief(calendar_events or [])

    return MORNING_BRIEF_PROMPT.format(
        n_items=len(items),
        items_text=_fmt_items(items),
        oil_price=g(oil_ctx, "price"),
        oil_regime=g(oil_ctx, "regime"),
        oil_bias=g(oil_ctx, "bias"),
        oil_bb=g(oil_ctx, "bb_pos"),
        oil_k=g(oil_ctx, "k"),
        oil_stoch_zone=g(oil_ctx, "stoch_zone"),
        oil_htf_k=g(oil_ctx, "htf_k"),
        oil_rsi=g(oil_ctx, "rsi"),
        oil_atr=g(oil_ctx, "atr"),
        oil_signal=g(oil_ctx, "signal", "NONE"),
        oil_grade=g(oil_ctx, "signal_grade", ""),
        oil_sl_tp=_fmt_sl_tp(oil_ctx),
        gold_price=g(gold_ctx, "price"),
        gold_regime=g(gold_ctx, "regime"),
        gold_bias=g(gold_ctx, "bias"),
        gold_bb=g(gold_ctx, "bb_pos"),
        gold_k=g(gold_ctx, "k"),
        gold_stoch_zone=g(gold_ctx, "stoch_zone"),
        gold_htf_k=g(gold_ctx, "htf_k"),
        gold_rsi=g(gold_ctx, "rsi"),
        gold_atr=g(gold_ctx, "atr"),
        gold_signal=g(gold_ctx, "signal", "NONE"),
        gold_grade=g(gold_ctx, "signal_grade", ""),
        gold_sl_tp=_fmt_sl_tp(gold_ctx),
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
) -> str:
    client = anthropic.AsyncAnthropic()
    prompt = _build_prompt(items, oil_ctx, gold_ctx, cot_data, calendar_events)

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