"""
monitor.py — runs all sources concurrently and collects new items.
"""

import asyncio
import logging

import aiohttp

from config import REQUEST_TIMEOUT_SECONDS
from sources import ALL_SOURCES

logger = logging.getLogger(__name__)


async def run_all_sources() -> list[dict]:
    """
    Fetch all sources concurrently.
    Returns a flat list of new items across all sources.
    Individual source failures are caught and logged; others continue.
    """
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [source(session) for source in ALL_SOURCES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_new_items: list[dict] = []
    for source_fn, result in zip(ALL_SOURCES, results):
        name = source_fn.__name__
        if isinstance(result, Exception):
            logger.error("[%s] failed with exception: %s", name, result)
        elif isinstance(result, list):
            logger.info("[%s] returned %d new item(s)", name, len(result))
            all_new_items.extend(result)
        else:
            logger.warning("[%s] unexpected result type: %s", name, type(result))

    return all_new_items
