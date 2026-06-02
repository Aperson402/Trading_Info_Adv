"""
ib_orderbook.py — Interactive Brokers Level 2 order book snapshot.

Connects to a locally-running TWS or IB Gateway (read-only), requests
market depth for WTI or gold futures, and returns a formatted snapshot
for Claude to reason about support/resistance, walls, and order imbalance.

Requires:
  - IB Gateway or TWS running, with API enabled (port configured in .env)
  - An active market data subscription that includes CME/COMEX Level 2
    (typically "CME Real-Time" or "COMEX Real-Time" in IBKR's data shop)
"""

import asyncio
import logging
import statistics
from typing import Optional

from config import IB_HOST, IB_PORT, IB_CLIENT_ID

logger = logging.getLogger(__name__)

# Contracts to use for each instrument
_CONTRACTS = {
    "oil":  {"symbol": "CL", "exchange": "NYMEX", "currency": "USD", "secType": "FUT"},
    "gold": {"symbol": "GC", "exchange": "COMEX", "currency": "USD", "secType": "FUT"},
}

# Seconds to wait for depth data after subscription — increase if on a slow connection
_DEPTH_WAIT = 2.5


async def _async_fetch(instrument: str, num_rows: int) -> dict:
    """
    Fully async IB fetch — must be run via asyncio.run() in a fresh thread
    so it gets its own event loop (no conflict with the main app loop).
    """
    from ib_async import IB, Future

    spec = _CONTRACTS.get(instrument)
    if spec is None:
        return {"error": f"No IB contract configured for '{instrument}'"}

    ib = IB()
    try:
        await ib.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, readonly=True, timeout=10)

        contract = Future(
            symbol=spec["symbol"],
            exchange=spec["exchange"],
            currency=spec["currency"],
        )

        # qualifyContractsAsync resolves the front-month expiry automatically
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            return {"error": f"IB could not qualify {spec['symbol']} — check market hours or data subscription"}
        contract = qualified[0]

        ticker = ib.reqMktDepth(contract, numRows=num_rows, isSmartDepth=False)
        await asyncio.sleep(_DEPTH_WAIT)

        bids = list(ticker.domBids or [])
        asks = list(ticker.domAsks or [])

        ib.cancelMktDepth(contract, isSmartDepth=False)

        return {
            "instrument": instrument,
            "contract":   f"{contract.symbol} {contract.lastTradeDateOrContractMonth}",
            "bids":       [{"price": float(b.price), "size": int(b.size)} for b in bids],
            "asks":       [{"price": float(a.price), "size": int(a.size)} for a in asks],
        }

    except ConnectionRefusedError:
        return {"error": "IB Gateway / TWS not reachable — is it running and is the API enabled?"}
    except asyncio.TimeoutError:
        return {"error": "IB connection timed out — check IB_HOST/IB_PORT in .env"}
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def _fetch_depth_sync(instrument: str, num_rows: int = 5) -> dict:
    """
    Runs _async_fetch inside a fresh event loop.
    Called from an executor thread so it never touches the main app's loop.
    """
    return asyncio.run(_async_fetch(instrument, num_rows))


def _format_depth(data: dict) -> str:
    """Convert raw depth dict into a text block for Claude."""
    if "error" in data:
        return f"[Order Book] Error: {data['error']}"

    instrument = data["instrument"].upper()
    contract   = data.get("contract", "")
    bids       = data.get("bids", [])
    asks       = data.get("asks", [])

    if not bids and not asks:
        return (
            f"[Order Book — {instrument} {contract}]\n"
            "  No depth data received. Market may be closed, or Level 2 subscription is missing."
        )

    lines = [f"[Order Book — {instrument} {contract}]"]

    # Spread
    if bids and asks:
        best_bid = bids[0]["price"]
        best_ask = asks[0]["price"]
        spread   = best_ask - best_bid
        lines.append(f"  Best bid: {best_bid:.2f}  |  Best ask: {best_ask:.2f}  |  Spread: {spread:.2f}")
    else:
        lines.append("  (one side of book is empty)")

    # Imbalance
    total_bid = sum(b["size"] for b in bids)
    total_ask = sum(a["size"] for a in asks)
    total     = total_bid + total_ask
    if total > 0:
        bid_pct = total_bid / total * 100
        ask_pct = total_ask / total * 100
        if bid_pct >= 60:
            imbalance_note = f"BID-HEAVY ({bid_pct:.0f}% bid) — buy-side pressure"
        elif ask_pct >= 60:
            imbalance_note = f"ASK-HEAVY ({ask_pct:.0f}% ask) — sell-side pressure"
        else:
            imbalance_note = f"balanced ({bid_pct:.0f}% bid / {ask_pct:.0f}% ask)"
        lines.append(f"  Depth imbalance: {imbalance_note}  (bid {total_bid:,} / ask {total_ask:,} lots)")

    # Identify walls (levels with size >2x the mean size across all levels)
    all_sizes = [b["size"] for b in bids] + [a["size"] for a in asks]
    mean_size = statistics.mean(all_sizes) if all_sizes else 1
    wall_threshold = mean_size * 2.0

    bid_walls = [b for b in bids if b["size"] >= wall_threshold]
    ask_walls = [a for a in asks if a["size"] >= wall_threshold]

    if bid_walls:
        wall_str = "  |  ".join(f"{w['price']:.2f} ({w['size']:,})" for w in bid_walls)
        lines.append(f"  Bid walls (support): {wall_str}")
    if ask_walls:
        wall_str = "  |  ".join(f"{w['price']:.2f} ({w['size']:,})" for w in ask_walls)
        lines.append(f"  Ask walls (resistance): {wall_str}")

    # Full depth table
    lines.append("  --- BIDS ---                --- ASKS ---")
    n = max(len(bids), len(asks))
    for i in range(n):
        b_str = f"{bids[i]['price']:.2f}  x{bids[i]['size']:>5,}" if i < len(bids) else " " * 20
        a_str = f"{asks[i]['price']:.2f}  x{asks[i]['size']:>5,}" if i < len(asks) else ""
        lines.append(f"  {b_str:<22}  {a_str}")

    return "\n".join(lines)


async def fetch_order_book(instrument: str, num_rows: int = 5) -> str:
    """Async entry point — runs the sync IB fetch in an executor thread."""
    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(None, _fetch_depth_sync, instrument, num_rows)
        return _format_depth(data)
    except Exception as exc:
        logger.error("Order book fetch failed: %s", exc)
        return f"[Order Book] Unexpected error: {exc}"
