"""
sources.py — one async function per data source.

Feed status as of 2026-05-31 (verified from Mac):
  EIA              — RSS times out; scrape HTML weekly report page
  OPEC             — HTML scrape (no RSS)
  State Dept       — state.gov/feed/ RSS ✅
  IAEA             — HTML scrape (feed URLs unstable)
  FT               — ft.com/rss/home RSS ✅  (replaces dead Reuters)
  AP               — hub/business?format=rss loads HTML; scrape it
  Baker Hughes     — rigcount.bakerhughes.com times out; scrape bakerhughes.com/rig-count
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp
import feedparser
from bs4 import BeautifulSoup

from database import is_seen, mark_seen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Minimal headers for government/CDN sites that block browser fingerprinting
HEADERS_MINIMAL = {
    "User-Agent": "curl/8.4.0",
    "Accept": "*/*",
}

# Keywords for relevance pre-filter (Phase 1 stopgap until Phase 2 Claude scoring)
# Items from FT, AP, IAEA are only kept if title contains at least one of these.
RELEVANCE_KEYWORDS = {
    # Commodities
    "oil", "crude", "petroleum", "opec", "opec+", "brent", "wti",
    "lng", "natural gas", "pipeline", "refin",
    "gold", "xau", "bullion", "silver", "commodity", "commodities",
    # Geopolitical regions that move energy markets (paired to avoid false positives)
    "iran", "iraq", "saudi", "aramco", "uae", "kuwait", "qatar",
    "hormuz", "strait of",
    "russia sanction", "ukraine energy", "ukraine gas", "ukraine oil",
    "israel iran", "israel oil", "israel gas", "lebanon oil",
    "syria oil", "yemen oil", "libya oil",
    # Macro / markets — only when paired with finance/commodity context
    "federal reserve", "fed rate", "interest rate",
    "oil price", "gold price", "commodity price",
    "bond yield", "dollar index",
    # Broader market context relevant to commodities
    "wall street", "stock market", "rally", "selloff",
    # Weapons/conflict with supply implications
    "weapons", "missile", "attack on", "strike on", "drone attack",
    "offensive in ukraine", "offensive in russia",
    # Energy — specific phrases to avoid matching solar/wind/renewable noise
    "oil and gas", "energy market", "energy price", "energy crisis",
    "energy supply", "energy demand", "energy export", "energy import",
    "barrel", "rig count", "refinery", "refining",
    "geopolit", "supply chain", "sanctions",
    "lng", "liquefied natural gas",
    # Nuclear — specific to energy/weapons context
    "nuclear weapon", "nuclear deal", "enriched uranium", "enrichment",
    "nuclear power plant", "nuclear fuel", "nuclear program",
    "uranium grip", "uranium supply",
}

def _is_relevant(title: str) -> bool:
    """Return True if title contains any relevance keyword (case-insensitive)."""
    t = title.lower()
    return any(kw in t for kw in RELEVANCE_KEYWORDS)

MAX_ITEM_AGE_HOURS = 48


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_struct_time(st) -> datetime:
    if st is None:
        return _now_utc()
    try:
        return datetime(*st[:6], tzinfo=timezone.utc)
    except Exception:
        return _now_utc()


def _is_too_old(ts: datetime) -> bool:
    cutoff = _now_utc() - timedelta(hours=MAX_ITEM_AGE_HOURS)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts < cutoff


async def _fetch_text(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """
    Fetch URL text with progressive fallback:
      1. Browser UA + SSL on
      2. Browser UA + SSL off  
      3. Minimal curl UA + SSL on  (for WAF-protected gov/org sites)
    """
    attempts = [
        (HEADERS,         True),
        (HEADERS,         False),
        (HEADERS_MINIMAL, True),
    ]
    for hdrs, use_ssl in attempts:
        try:
            async with session.get(url, headers=hdrs, ssl=use_ssl) as resp:
                resp.raise_for_status()
                return await resp.text()
        except Exception as exc:
            last_exc = exc
            continue
    logger.warning("Fetch failed for %s: %s", url, last_exc)
    return None


async def _process_rss_feed(
    session: aiohttp.ClientSession,
    feed_url: str,
    source_name: str,
    max_new: int = 15,
    filter_fn=None,
) -> list[dict]:
    """
    Generic RSS handler. If filter_fn is provided, items that don't pass it
    are skipped BEFORE mark_seen — keeps the DB clean.
    """
    text = await _fetch_text(session, feed_url)
    if not text:
        return []

    feed = feedparser.parse(text)
    new_items: list[dict] = []

    for entry in feed.entries:
        if len(new_items) >= max_new:
            break

        url = entry.get("link") or entry.get("id") or ""
        if not url:
            continue

        ts = _parse_struct_time(
            entry.get("published_parsed") or entry.get("updated_parsed")
        )

        if _is_too_old(ts):
            continue

        title = entry.get("title", "(no title)").strip()

        # Apply filter before DB write if provided
        if filter_fn and not filter_fn(title):
            continue

        if await is_seen(url):
            continue

        summary = entry.get("summary") or entry.get("description") or None
        if summary:
            summary = BeautifulSoup(summary, "html.parser").get_text(
                separator=" ", strip=True
            )[:500]

        item = {
            "source_name": source_name,
            "title": title,
            "url": url,
            "timestamp": ts,
            "summary": summary,
        }
        await mark_seen(url, source_name, title, summary, ts)
        new_items.append(item)
        logger.info("[%s] new item: %s", source_name, title[:80])

    return new_items


# ---------------------------------------------------------------------------
# Source 1 — EIA Weekly Petroleum Status Report (HTML scrape, targeted)
# ---------------------------------------------------------------------------

async def fetch_eia(session: aiohttp.ClientSession) -> list[dict]:
    """
    Scrapes the EIA weekly petroleum page but only picks up dated report links
    (e.g. 'May 28, 2026') — ignores navigation, table headers, and section anchors.
    """
    source_name = "EIA"
    page_url = "https://www.eia.gov/petroleum/supply/weekly/"

    text = await _fetch_text(session, page_url)
    if not text:
        return []

    soup = BeautifulSoup(text, "html.parser")
    new_items: list[dict] = []

    # Match only links whose visible text looks like a date: "May 28, 2026"
    date_pattern = re.compile(
        r"^(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},\s+\d{4}$"
    )

    base = "https://www.eia.gov"
    # Collect all dated report links first, then only take the most recent unseen one.
    # This prevents alerting on weeks of backlog on first run.
    candidates: list[tuple[str, str]] = []  # (href, title)
    for a in soup.find_all("a", href=True):
        title = a.get_text(strip=True)
        if not date_pattern.match(title):
            continue
        href = a["href"]
        if href.startswith("/"):
            href = base + href
        if not href.startswith("http"):
            continue
        candidates.append((href, title))

    # Page lists newest first — take only the first unseen report
    for href, title in candidates[:1]:
        if await is_seen(href):
            break
        ts = _now_utc()
        item = {
            "source_name": source_name,
            "title": f"EIA Weekly Petroleum Status Report — {title}",
            "url": href,
            "timestamp": ts,
            "summary": None,
        }
        await mark_seen(href, source_name, item["title"], None, ts)
        new_items.append(item)
        logger.info("[%s] new item: %s", source_name, item["title"])

    return new_items


# ---------------------------------------------------------------------------
# Source 2 — OPEC Press Releases (HTML scrape)
# ---------------------------------------------------------------------------

async def fetch_oilprice(session: aiohttp.ClientSession) -> list[dict]:
    """
    OilPrice.com — dedicated oil, gold, OPEC, Iran, Saudi coverage.
    Verified active RSS feed. Replaces dead OPEC scrape.
    """
    return await _process_rss_feed(
        session,
        "https://oilprice.com/rss/main",
        "OilPrice",
        max_new=10,
        filter_fn=_is_relevant,
    )


# ---------------------------------------------------------------------------
# Source 3 — State Department RSS ✅ (verified working)
# ---------------------------------------------------------------------------

async def fetch_state_dept(session: aiohttp.ClientSession) -> list[dict]:
    """
    state.gov/feed/ returns 200 to curl but hangs for aiohttp with browser UA.
    Use minimal curl-like headers and scrape the press-releases listing instead.
    """
    source_name = "State Dept"
    page_url = "https://www.state.gov/press-releases/"

    text = await _fetch_text(session, page_url)

    soup = BeautifulSoup(text, "html.parser")
    new_items: list[dict] = []
    base = "https://www.state.gov"

    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if "/press-releases/" not in href:
            continue
        if href.rstrip("/") in ("/press-releases", ""):
            continue
        if href.startswith("/"):
            href = base + href
        if not href.startswith("http"):
            continue

        title = a.get_text(strip=True)
        if len(title) < 10:
            continue

        if await is_seen(href):
            continue

        ts = _now_utc()
        item = {
            "source_name": source_name,
            "title": title,
            "url": href,
            "timestamp": ts,
            "summary": None,
        }
        await mark_seen(href, source_name, title, None, ts)
        new_items.append(item)
        logger.info("[%s] new item: %s", source_name, title[:80])

    return new_items


# ---------------------------------------------------------------------------
# Source 4 — IAEA Press Releases (HTML scrape)
# ---------------------------------------------------------------------------

async def fetch_iaea(session: aiohttp.ClientSession) -> list[dict]:
    source_name = "IAEA"
    page_url = "https://www.iaea.org/newscenter/pressreleases"

    text = await _fetch_text(session, page_url)
    if not text:
        return []

    soup = BeautifulSoup(text, "html.parser")
    new_items: list[dict] = []
    base = "https://www.iaea.org"

    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if "/newscenter/pressreleases/" not in href:
            continue
        if href.rstrip("/") == "/newscenter/pressreleases":
            continue
        if href.startswith("/"):
            href = base + href
        if not href.startswith("http"):
            continue

        title = a.get_text(strip=True)
        if len(title) < 10:
            continue

        if await is_seen(href):
            continue

        ts = _now_utc()
        item = {
            "source_name": source_name,
            "title": title,
            "url": href,
            "timestamp": ts,
            "summary": None,
        }
        await mark_seen(href, source_name, title, None, ts)
        new_items.append(item)
        logger.info("[%s] new item: %s", source_name, title[:80])

    IAEA_KEEP = {
        "iran", "iraq", "ukraine", "russia", "weapons", "enrichment",
        "nuclear fuel", "nuclear program", "nuclear deal", "nonprolifer",
        "safeguards", "inspections", "uranium", "plutonium",
        "venezuela", "north korea", "pakistan", "middle east",
    }
    def _iaea_relevant(title: str) -> bool:
        t = title.lower()
        return any(kw in t for kw in IAEA_KEEP)

    return [i for i in new_items if _iaea_relevant(i["title"])]


# ---------------------------------------------------------------------------
# Source 5 — Financial Times RSS ✅ (replaces dead Reuters feed)
# ---------------------------------------------------------------------------

async def fetch_ft(session: aiohttp.ClientSession) -> list[dict]:
    return await _process_rss_feed(session, "https://www.ft.com/rss/home", "FT", filter_fn=_is_relevant)


# ---------------------------------------------------------------------------
# Source 6 — AP News Energy (HTML scrape of hub page)
# ---------------------------------------------------------------------------

async def fetch_ap(session: aiohttp.ClientSession) -> list[dict]:
    """
    AP's RSS paths are dead. Their hub pages load as HTML.
    Scrape the energy hub for article links.
    """
    source_name = "AP News"
    page_url = "https://apnews.com/hub/energy"  # energy hub — relevance filter still applied

    text = await _fetch_text(session, page_url)
    if not text:
        return []

    soup = BeautifulSoup(text, "html.parser")
    new_items: list[dict] = []
    base = "https://apnews.com"
    seen_hrefs: set[str] = set()

    for a in soup.find_all("a", href=True):
        href: str = a["href"]

        # AP article URLs look like /article/<slug> or start with /article/
        if "/article/" not in href:
            continue
        if href.startswith("/"):
            href = base + href
        if href in seen_hrefs:
            continue
        seen_hrefs.add(href)

        title = a.get_text(strip=True)
        if len(title) < 15:
            continue

        # Apply relevance filter BEFORE writing to DB — keeps seen-items store clean
        if not _is_relevant(title):
            continue

        if await is_seen(href):
            continue

        ts = _now_utc()
        item = {
            "source_name": source_name,
            "title": title,
            "url": href,
            "timestamp": ts,
            "summary": None,
        }
        await mark_seen(href, source_name, title, None, ts)
        new_items.append(item)
        logger.info("[%s] new item: %s", source_name, title[:80])

        if len(new_items) >= 10:
            break

    return new_items


# ---------------------------------------------------------------------------
# Source 7 — Baker Hughes Rig Count (scrape main rig-count page)
# ---------------------------------------------------------------------------

async def fetch_worldoil(session: aiohttp.ClientSession) -> list[dict]:
    """
    World Oil — industry publication since 1916.
    Covers drilling, upstream, OPEC, supply/demand.
    Replaces Baker Hughes (hard 403) and IEA (403).
    """
    return await _process_rss_feed(
        session,
        "https://worldoil.com/rss?feed=news",
        "World Oil",
        max_new=10,
        filter_fn=_is_relevant,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_SOURCES = [
    fetch_eia,
    fetch_oilprice,
    fetch_state_dept,
    fetch_iaea,
    fetch_ft,
    fetch_ap,
    fetch_worldoil,
]