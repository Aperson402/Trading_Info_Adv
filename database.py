import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiosqlite

from config import DATABASE_PATH

logger = logging.getLogger(__name__)


def make_url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


async def init_db() -> None:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS seen_items (
                url_hash        TEXT PRIMARY KEY,
                url             TEXT NOT NULL,
                source_name     TEXT NOT NULL,
                title           TEXT NOT NULL,
                summary         TEXT,
                discovered_at   TEXT NOT NULL,
                -- Phase 2 classification fields (nullable for pre-classifier items)
                instrument      TEXT,
                direction       TEXT,
                urgency         TEXT,
                confidence      INTEGER,
                reasoning       TEXT,
                ignored         INTEGER DEFAULT 0
            )
        """)
        # Add classification columns if upgrading from Phase 1 DB
        for col, coltype in [
            ("instrument", "TEXT"),
            ("direction",  "TEXT"),
            ("urgency",    "TEXT"),
            ("confidence", "INTEGER"),
            ("reasoning",  "TEXT"),
            ("ignored",    "INTEGER DEFAULT 0"),
        ]:
            try:
                await db.execute(f"ALTER TABLE seen_items ADD COLUMN {col} {coltype}")
            except Exception:
                pass  # column already exists
        await db.commit()
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument  TEXT NOT NULL,
                direction   TEXT NOT NULL,
                entry_price REAL NOT NULL,
                sl_price    REAL,
                tp_price    REAL,
                opened_at   TEXT NOT NULL,
                closed_at   TEXT,
                close_price REAL,
                outcome     TEXT,
                pnl_pct     REAL
            )
        """)
        await db.commit()
    logger.info("Database initialised at %s", DATABASE_PATH)


async def is_seen(url: str) -> bool:
    url_hash = make_url_hash(url)
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT 1 FROM seen_items WHERE url_hash = ?", (url_hash,)
        )
        row = await cursor.fetchone()
    return row is not None


async def mark_seen(
    url: str,
    source_name: str,
    title: str,
    summary: Optional[str] = None,
    discovered_at: Optional[datetime] = None,
) -> None:
    url_hash = make_url_hash(url)
    ts = (discovered_at or datetime.now(timezone.utc)).isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            """
            INSERT OR IGNORE INTO seen_items
                (url_hash, url, source_name, title, summary, discovered_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (url_hash, url, source_name, title, summary, ts),
        )
        await db.commit()


async def update_classification(url: str, classification: dict) -> None:
    """Store Phase 2 classification results against an existing seen item."""
    url_hash = make_url_hash(url)
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            """
            UPDATE seen_items SET
                instrument  = ?,
                direction   = ?,
                urgency     = ?,
                confidence  = ?,
                reasoning   = ?,
                ignored     = ?
            WHERE url_hash = ?
            """,
            (
                classification.get("instrument"),
                classification.get("direction"),
                classification.get("urgency"),
                classification.get("confidence"),
                classification.get("reasoning"),
                1 if classification.get("ignored") else 0,
                url_hash,
            ),
        )
        await db.commit()


async def get_sentiment_window(hours: int = 24) -> dict:
    """
    Return bullish/bearish/neutral counts for oil and gold over the last N hours.
    Only counts classified, non-ignored items.
    """
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    result = {
        "oil":  {"bullish": 0, "bearish": 0, "neutral": 0, "total": 0},
        "gold": {"bullish": 0, "bearish": 0, "neutral": 0, "total": 0},
    }
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            """
            SELECT instrument, direction, COUNT(*) as cnt
            FROM seen_items
            WHERE discovered_at >= ?
              AND ignored = 0
              AND confidence IS NOT NULL
              AND direction IN ('bullish', 'bearish', 'neutral')
              AND instrument IN ('oil', 'gold', 'both')
            GROUP BY instrument, direction
            """,
            (since,),
        )
        rows = await cursor.fetchall()

    for instrument, direction, cnt in rows:
        targets = ["oil", "gold"] if instrument == "both" else [instrument]
        for t in targets:
            result[t][direction] += cnt
            result[t]["total"] += cnt

    return result


async def get_source_reliability(source_name: str) -> float:
    """
    Return a confidence multiplier (0.7–1.3) based on historical signal rate.
    Signal rate = items with confidence >= 6 / total classified items for this source.
    Falls back to 1.0 if fewer than 5 classified items exist.
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN confidence >= 6 THEN 1 ELSE 0 END) as actionable
            FROM seen_items
            WHERE source_name = ? AND confidence IS NOT NULL
            """,
            (source_name,),
        )
        row = await cursor.fetchone()

    if not row or row[0] < 5:
        return 1.0  # not enough history

    total, actionable = row
    signal_rate = actionable / total
    return round(0.7 + 0.6 * signal_rate, 3)  # [0.7, 1.3]


async def get_items_since(since: datetime) -> list[dict]:
    """Return all items discovered on or after *since* (UTC-aware datetime)."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT source_name, title, url, summary, discovered_at,
                   instrument, direction, urgency, confidence, reasoning
            FROM seen_items
            WHERE discovered_at >= ? AND (ignored = 0 OR ignored IS NULL)
            ORDER BY discovered_at ASC
            """,
            (since.isoformat(),),
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def open_trade(
    instrument: str,
    direction: str,
    entry_price: float,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
) -> dict:
    opened_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO trades (instrument, direction, entry_price, sl_price, tp_price, opened_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (instrument, direction, entry_price, sl_price, tp_price, opened_at),
        )
        trade_id = cursor.lastrowid
        await db.commit()
    return {
        "id": trade_id, "instrument": instrument, "direction": direction,
        "entry_price": entry_price, "sl_price": sl_price, "tp_price": tp_price,
        "opened_at": opened_at,
    }


async def get_open_trades() -> list[dict]:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM trades WHERE closed_at IS NULL ORDER BY opened_at ASC"
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def close_trade(trade_id: int, close_price: float, outcome: str) -> dict:
    closed_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        trade = dict(await cursor.fetchone())

    entry = trade["entry_price"]
    if trade["direction"] == "long":
        pnl_pct = (close_price - entry) / entry * 100
    else:
        pnl_pct = (entry - close_price) / entry * 100

    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "UPDATE trades SET closed_at=?, close_price=?, outcome=?, pnl_pct=? WHERE id=?",
            (closed_at, close_price, outcome, round(pnl_pct, 3), trade_id),
        )
        await db.commit()

    return {**trade, "closed_at": closed_at, "close_price": close_price,
            "outcome": outcome, "pnl_pct": round(pnl_pct, 3)}


async def get_recent_trades(limit: int = 5) -> list[dict]:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM trades WHERE closed_at IS NOT NULL ORDER BY closed_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]