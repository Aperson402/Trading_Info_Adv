"""
event_classifier.py — post-event reaction classifier.

After a high-impact event releases, compares pre-event vs post-event prices
and uses Claude to classify whether the reaction was expected, complete, or
has follow-through potential.
"""

import logging

import anthropic

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"

REACTION_PROMPT = """\
A high-impact economic event just released. Classify the market reaction.

EVENT: {title}
FORECAST: {forecast}  |  ACTUAL: {actual}  |  {beat_miss}

PRICE MOVE (snapshot before event → ~15 min after):
OIL (WTI):   ${oil_pre:.2f} → ${oil_post:.2f}  ({oil_move:+.2f}%)
GOLD (XAU):  ${gold_pre:.2f} → ${gold_post:.2f}  ({gold_move:+.2f}%)

Respond in exactly 4 lines. No labels, no preamble:
Line 1: Was the reaction expected given the beat/miss, and which instrument led?
Line 2: Is the move likely complete (already priced) or is there follow-through potential?
Line 3: One risk to the current direction in the next hour.
Line 4: One-line trade implication — be specific (e.g. "fade the spike on oil above $X" or "gold long valid above $Y").\
"""


async def classify_event_reaction(
    event: dict,
    oil_pre: float,
    gold_pre: float,
    oil_post: float,
    gold_post: float,
) -> str:
    forecast = event.get("forecast") or "n/a"
    actual   = event.get("actual")   or "n/a"

    try:
        f_val = float(str(forecast).replace("%", "").replace("K", ""))
        a_val = float(str(actual).replace("%", "").replace("K", ""))
        beat_miss = "BEAT" if a_val > f_val else "MISS" if a_val < f_val else "IN LINE"
    except Exception:
        beat_miss = "vs forecast"

    oil_move  = (oil_post  - oil_pre)  / oil_pre  * 100 if oil_pre  else 0.0
    gold_move = (gold_post - gold_pre) / gold_pre * 100 if gold_pre else 0.0

    prompt = REACTION_PROMPT.format(
        title=event.get("title", ""),
        forecast=forecast,
        actual=actual,
        beat_miss=beat_miss,
        oil_pre=oil_pre,
        oil_post=oil_post,
        oil_move=oil_move,
        gold_pre=gold_pre,
        gold_post=gold_post,
        gold_move=gold_move,
    )

    client = anthropic.AsyncAnthropic()
    try:
        response = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.error("Event reaction classification failed: %s", exc)
        return ""
