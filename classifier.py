"""
classifier.py — Claude Haiku relevance classifier for Phase 2.

Takes a raw item from Phase 1 and returns an enriched version with:
  - instrument: oil | gold | both | neither
  - direction:  bullish | bearish | neutral | unclear
  - urgency:    breaking | developing | routine
  - confidence: 1-10
  - reasoning:  one sentence
  - ignore:     bool

Items with ignore=True or confidence < MIN_CONFIDENCE are dropped.
"""

import json
import logging
from typing import Optional

import anthropic

from database import get_source_reliability

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 6  # items below this threshold are dropped even if ignore=False
HAIKU_MODEL = "claude-haiku-4-5-20251001"

CLASSIFIER_PROMPT = """\
You are a commodity trading analyst specialising in oil and gold.

Classify this news item for a trader who trades oil (WTI/Brent) and gold (XAU/USD).

SOURCE: {source}
TITLE: {title}
SUMMARY: {summary}

Return JSON only — no other text, no markdown fences:
{{
  "instrument": "oil" | "gold" | "both" | "neither",
  "direction": "bullish" | "bearish" | "neutral" | "unclear",
  "urgency": "breaking" | "developing" | "routine",
  "confidence": <integer 1-10>,
  "reasoning": "<one sentence explanation of market implication>",
  "ignore": <true | false>
}}

Classification rules:

OIL is affected by: OPEC decisions and output changes, Middle East geopolitical \
events, Russia/Ukraine supply disruption, US/Iran sanctions, inventory data (EIA/API), \
Strait of Hormuz developments, Saudi/UAE/Iraq production, tanker movements, \
rig count trends, refinery capacity.

GOLD is affected by: Fed policy and interest rate decisions, US dollar strength/weakness, \
inflation data (CPI/PCE), geopolitical risk broadly, central bank buying/selling, \
real yields, recession fears, safe-haven demand.

Mark ignore=true if:
- Domestic politics with no commodity implication (elections, scandals, court cases)
- Sports, entertainment, human interest
- Corporate earnings unrelated to oil/gas/mining
- Travel, weather, or lifestyle content
- Pure technology news with no energy angle
- Military/geopolitical events with no direct energy supply chain implication
  (e.g. conventional troop movements, diplomatic summits without energy agenda)
- Renewable energy news (solar, wind, EV) unless directly displacing oil demand

Mark confidence low (1-5) if the connection to commodities is indirect or speculative.
Mark confidence high (7-10) if the item directly affects supply, demand, or sentiment.

Important examples:
- IAEA transporting nuclear fuel → instrument=neither, ignore=true (no commodity price impact)
- Strait of Hormuz disruption → instrument=oil, bullish, confidence=9
- Fed rate decision → instrument=gold, relevant, confidence=8
- Military offensive near oil infrastructure → instrument=oil, bullish, confidence=8
- Belarus troop movements with no energy angle → ignore=true
- Country selecting nuclear plant sites → instrument=neither, developing story, confidence=3\
"""


def _build_prompt(item: dict) -> str:
    return CLASSIFIER_PROMPT.format(
        source=item.get("source_name", "Unknown"),
        title=item.get("title", ""),
        summary=item.get("summary") or "(no summary available)",
    )


def _parse_response(text: str) -> Optional[dict]:
    """Parse Claude's JSON response, handling minor formatting issues."""
    text = text.strip()
    # Strip markdown fences if Claude added them despite instructions
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object from surrounding text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


async def classify_item(item: dict) -> Optional[dict]:
    """
    Classify a single item using Claude Haiku.

    Returns the item dict enriched with classification fields,
    or None if the item should be dropped (ignore=True or low confidence).
    """
    client = anthropic.AsyncAnthropic()

    try:
        response = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": _build_prompt(item)}],
        )
    except Exception as exc:
        logger.error("Classifier API call failed: %s", exc)
        # On API failure, pass item through unclassified rather than drop it
        return {**item, "classification_error": str(exc)}

    raw = response.content[0].text if response.content else ""
    classification = _parse_response(raw)

    if not classification:
        logger.warning("Could not parse classifier response for: %s | raw: %s",
                       item.get("title", "")[:60], raw[:200])
        return {**item, "classification_error": "parse_failed"}

    confidence = int(classification.get("confidence", 0))
    ignore = bool(classification.get("ignore", False))

    if ignore:
        logger.info("FILTERED [ignore=True] %s", item.get("title", "")[:70])
        return None

    reliability = await get_source_reliability(item.get("source_name", ""))
    weighted_confidence = min(10, round(confidence * reliability))
    if weighted_confidence != confidence:
        logger.info(
            "SOURCE WEIGHT [%s] %.2fx → conf %d→%d",
            item.get("source_name", ""), reliability, confidence, weighted_confidence,
        )
    confidence = weighted_confidence

    if confidence < MIN_CONFIDENCE:
        logger.info("FILTERED [conf=%d after weighting] %s", confidence, item.get("title", "")[:70])
        return None

    logger.info(
        "CLASSIFIED [%s/%s/%s conf=%d] %s",
        classification.get("instrument", "?"),
        classification.get("direction", "?"),
        classification.get("urgency", "?"),
        confidence,
        item.get("title", "")[:60],
    )

    return {
        **item,
        "instrument":  classification.get("instrument", "unclear"),
        "direction":   classification.get("direction", "unclear"),
        "urgency":     classification.get("urgency", "routine"),
        "confidence":  confidence,
        "reasoning":   classification.get("reasoning", ""),
    }


async def classify_items(items: list[dict]) -> list[dict]:
    """
    Classify items sequentially with a small delay to respect rate limits.
    50 req/min limit = 1.2s between requests minimum; we use 1.5s to be safe.
    Returns only items that pass the filter.
    """
    import asyncio

    classified = []
    for i, item in enumerate(items):
        if i > 0:
            await asyncio.sleep(1.5)  # 40 req/min sustained — well under 50/min limit
        try:
            result = await classify_item(item)
            if result is not None:
                classified.append(result)
        except Exception as exc:
            logger.error("classify_item raised for %s: %s",
                         item.get("title", "")[:50], exc)

    logger.info(
        "Classifier: %d in → %d passed (%.0f%% filtered)",
        len(items),
        len(classified),
        100 * (1 - len(classified) / max(len(items), 1)),
    )
    return classified