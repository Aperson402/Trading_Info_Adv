import hashlib
import logging
from datetime import datetime, timezone
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