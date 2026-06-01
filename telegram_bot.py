"""
telegram_bot.py — formats and sends messages via python-telegram-bot.
Phase 2: alert format includes Claude classification.
"""

import html
import logging
from datetime import datetime, timezone

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

_MAX_MSG_LEN = 3800

# Emoji maps
_INSTRUMENT_EMOJI = {"oil": "🛢", "gold": "🥇", "both": "🛢🥇", "neither": "📰"}
_DIRECTION_EMOJI  = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪", "unclear": "🟡"}
_URGENCY_EMOJI    = {"breaking": "🚨", "developing": "📡", "routine": "📋"}


def _h(text: str) -> str:
    return html.escape(str(text))


def _fmt_item(item: dict) -> str:
    ts: datetime = item.get("timestamp") or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts_str = ts.strftime("%Y-%m-%d %H:%M UTC")

    source    = item.get("source_name", "Unknown")
    title     = item.get("title", "(no title)")
    url       = item.get("url", "")
    reasoning = item.get("reasoning", "")

    # Phase 2 enriched format
    if item.get("confidence"):
        instrument = item.get("instrument", "unclear")
        direction  = item.get("direction", "unclear")
        urgency    = item.get("urgency", "routine")
        confidence = item.get("confidence", 0)

        urg_emoji  = _URGENCY_EMOJI.get(urgency, "📡")
        dir_emoji  = _DIRECTION_EMOJI.get(direction, "⚪")

        header = (
            f"{urg_emoji} <b>{urgency.upper()} — "
            f"{dir_emoji} {direction.upper()} {instrument.upper()}</b>"
        )
        meta = (
            f"<b>Source:</b> {_h(source)}\n"
            f"<b>Confidence:</b> {confidence}/10\n"
        )
        body = f"{_h(title)}\n{url}"
        reasoning_line = f"\n💡 <i>{_h(reasoning)}</i>" if reasoning else ""

        # Market context block (matches market_context.py field names)
        mkt = item.get("market_context")
        market_block = ""
        if mkt:
            inst_label = mkt.get("instrument", "").upper()
            price      = mkt.get("price", "—")
            regime     = mkt.get("regime", "—")
            bias       = mkt.get("bias", "—")
            bb_pos     = mkt.get("bb_pos", "—")
            k          = mkt.get("k", "—")
            stoch_zone = mkt.get("stoch_zone", "—")
            htf_k      = mkt.get("htf_k", "—")
            rsi        = mkt.get("rsi", "—")
            fetched_at = mkt.get("fetched_at", "")
            squeeze    = mkt.get("squeeze", False)
            signal     = mkt.get("signal", "NONE")
            grade      = mkt.get("signal_grade", "—")
            sl         = mkt.get("suggested_sl")
            tp         = mkt.get("suggested_tp")

            squeeze_tag = "  ⚠️ <b>SQUEEZE</b>" if squeeze else ""
            if signal != "NONE":
                sig_emoji = "🟢" if signal == "LONG" else "🔴"
                sig_line = f"\n{sig_emoji} <b>{signal} [{grade}]</b>  SL {sl}  TP {tp}"
            else:
                sig_line = ""

            market_block = (
                f"\n\n<b>📊 {inst_label}  ${price}  {fetched_at}</b>"
                f"\nRegime: <b>{regime}</b>{squeeze_tag}  Bias: <b>{bias}</b>"
                f"\nBB: {bb_pos}  K={k} {stoch_zone}  HTF K={htf_k}  RSI={rsi}"
                f"{sig_line}"
            )

        return f"{header}\n{meta}{body}{reasoning_line}{market_block}\n<i>{_h(ts_str)}</i>"

    # Phase 1 fallback format (unclassified items, e.g. on API error)
    return (
        f"📡 <b>NEW — {_h(source)}</b>\n"
        f"{_h(title)}\n"
        f"{url}\n"
        f"<i>{_h(ts_str)}</i>"
    )


def _split_long_message(text: str) -> list[str]:
    if len(text) <= _MAX_MSG_LEN:
        return [text]
    parts: list[str] = []
    while text:
        if len(text) <= _MAX_MSG_LEN:
            parts.append(text)
            break
        split_at = text.rfind("\n", 0, _MAX_MSG_LEN)
        if split_at == -1:
            split_at = _MAX_MSG_LEN
        parts.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return parts


async def _send(bot: Bot, text: str) -> None:
    for part in _split_long_message(text):
        try:
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=part,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except TelegramError as exc:
            logger.error("Telegram send failed: %s", exc)


async def send_new_items(items: list[dict]) -> None:
    if not items:
        return
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured — skipping send.")
        return
    bot = Bot(token=TELEGRAM_TOKEN)
    for item in items:
        await _send(bot, _fmt_item(item))


async def send_morning_brief(
    items: list[dict],
    brief_text: str = "",
    oil_ctx: dict = None,
    gold_ctx: dict = None,
) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured — skipping morning brief.")
        return

    bot = Bot(token=TELEGRAM_TOKEN)
    today_str = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    header = f"☀️ <b>MORNING BRIEF — {_h(today_str)}</b>"

    if brief_text:
        # Sonnet synthesis — render as-is (plain text, no extra HTML needed)
        message = f"{header}\n\n{_h(brief_text)}"
    elif items:
        # Fallback: structured item list with classifications
        lines: list[str] = []
        for item in items:
            ts_raw = item.get("discovered_at", "")
            try:
                ts = datetime.fromisoformat(ts_raw)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                ts_str = ts.strftime("%H:%M UTC")
            except Exception:
                ts_str = "—"

            inst = item.get("instrument", "")
            dirn = item.get("direction", "")
            conf = item.get("confidence", "")
            tag  = f"[{inst.upper()} {dirn.upper()} {conf}/10] " if inst and inst != "neither" else ""
            lines.append(f"• {tag}<b>{_h(item.get('source_name','?'))}</b> {_h(item.get('title',''))} <i>{ts_str}</i>")

        message = (
            f"{header}\n\n"
            f"Good morning. {len(items)} overnight items:\n\n"
            + "\n".join(lines)
        )
    else:
        message = f"{header}\n\n<i>No relevant items in the last 12 hours.</i>"

    await _send(bot, message)


async def send_trade_opened(trade: dict) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    bot = Bot(token=TELEGRAM_TOKEN)
    emoji = "🛢" if trade["instrument"] == "oil" else "🥇"
    direction = trade["direction"].upper()
    sl = f"  SL: ${trade['sl_price']:,.2f}" if trade.get("sl_price") else ""
    tp = f"  TP: ${trade['tp_price']:,.2f}" if trade.get("tp_price") else ""
    await _send(bot,
        f"{emoji} <b>TRADE OPENED — {direction} {trade['instrument'].upper()}</b>\n"
        f"Entry: <b>${trade['entry_price']:,.2f}</b>{sl}{tp}"
    )


async def send_trade_closed(trade: dict) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    bot = Bot(token=TELEGRAM_TOKEN)
    emoji = "🛢" if trade["instrument"] == "oil" else "🥇"
    pnl = trade.get("pnl_pct", 0)
    outcome = trade.get("outcome", "manual")
    result_emoji = "✅" if pnl >= 0 else "❌"
    opened = datetime.fromisoformat(trade["opened_at"]).replace(tzinfo=timezone.utc)
    closed = datetime.fromisoformat(trade["closed_at"]).replace(tzinfo=timezone.utc)
    duration = closed - opened
    hours, rem = divmod(int(duration.total_seconds()), 3600)
    mins = rem // 60
    duration_str = f"{hours}h {mins}m" if hours else f"{mins}m"
    await _send(bot,
        f"{result_emoji}{emoji} <b>TRADE CLOSED — {trade['direction'].upper()} {trade['instrument'].upper()}</b>\n"
        f"Entry: ${trade['entry_price']:,.2f} → Exit: ${trade['close_price']:,.2f}  ({_h(outcome)})\n"
        f"P&amp;L: <b>{pnl:+.2f}%</b>  |  Duration: {duration_str}"
    )


async def send_trades_list(open_trades: list[dict], recent_trades: list[dict], current_prices: dict) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    bot = Bot(token=TELEGRAM_TOKEN)
    lines = ["📋 <b>TRADE JOURNAL</b>"]

    if open_trades:
        lines.append("\n<b>Open:</b>")
        for t in open_trades:
            emoji = "🛢" if t["instrument"] == "oil" else "🥇"
            cur = current_prices.get(t["instrument"])
            if cur:
                pnl = (cur - t["entry_price"]) / t["entry_price"] * 100
                if t["direction"] == "short":
                    pnl = -pnl
                pnl_str = f"  P&amp;L: {pnl:+.2f}%"
            else:
                pnl_str = ""
            sl = f"  SL ${t['sl_price']:,.2f}" if t.get("sl_price") else ""
            tp = f"  TP ${t['tp_price']:,.2f}" if t.get("tp_price") else ""
            lines.append(f"{emoji} {t['direction'].upper()} @ ${t['entry_price']:,.2f}{sl}{tp}{pnl_str}")
    else:
        lines.append("\n<i>No open trades.</i>")

    if recent_trades:
        lines.append("\n<b>Recent closed:</b>")
        for t in recent_trades:
            emoji = "🛢" if t["instrument"] == "oil" else "🥇"
            result = "✅" if (t.get("pnl_pct") or 0) >= 0 else "❌"
            lines.append(
                f"{result} {emoji} {t['direction'].upper()} {t['instrument'].upper()} "
                f"${t['entry_price']:,.2f}→${t['close_price']:,.2f}  "
                f"<b>{(t.get('pnl_pct') or 0):+.2f}%</b>"
            )

    await _send(bot, "\n".join(lines))


async def send_price_alert(instrument: str, price: float, target: float, direction: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    bot = Bot(token=TELEGRAM_TOKEN)
    emoji = "🛢" if instrument == "oil" else "🥇"
    crossed = "above" if direction == "above" else "below"
    message = (
        f"{emoji} <b>PRICE ALERT — {instrument.upper()}</b>\n"
        f"Price <b>${price:,.2f}</b> crossed {crossed} your target of <b>${target:,.2f}</b>"
    )
    await _send(bot, message)


async def send_advice(advice_text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    bot = Bot(token=TELEGRAM_TOKEN)
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    header = f"🎯 <b>ADVICE — {_h(now_str)}</b>"
    message = f"{header}\n\n{_h(advice_text)}"
    await _send(bot, message)


async def send_event_alert(event: dict, kind: str = "upcoming") -> None:
    """Send a pre-event warning or post-event result alert."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    bot = Bot(token=TELEGRAM_TOKEN)
    title = _h(event.get("title", ""))
    dt = event.get("datetime")
    when = dt.strftime("%H:%M UTC") if dt else "—"
    forecast = event.get("forecast", "")
    previous = event.get("previous", "")
    actual   = event.get("actual", "")

    if kind == "upcoming":
        lines = [f"⏰ <b>UPCOMING — {title}</b>  {when}"]
        if forecast: lines.append(f"Forecast: <b>{_h(forecast)}</b>")
        if previous: lines.append(f"Previous: {_h(previous)}")
    else:
        beat = ""
        try:
            if actual and forecast:
                a, f = float(actual.replace("%","").replace("K","000").replace("M","000000")), \
                       float(forecast.replace("%","").replace("K","000").replace("M","000000"))
                beat = "  ✅ <b>BEAT</b>" if a > f else "  ❌ <b>MISS</b>" if a < f else "  ➡️ <b>IN LINE</b>"
        except Exception:
            pass
        lines = [f"📊 <b>RESULT — {title}</b>{beat}"]
        if actual:   lines.append(f"Actual: <b>{_h(actual)}</b>")
        if forecast: lines.append(f"Forecast: {_h(forecast)}")
        if previous: lines.append(f"Previous: {_h(previous)}")

    await _send(bot, "\n".join(lines))


async def send_calendar(events: list[dict]) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    bot = Bot(token=TELEGRAM_TOKEN)
    today_str = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    header = f"📅 <b>ECONOMIC CALENDAR — {_h(today_str)}</b>\n<i>High-impact USD events, next 7 days</i>"

    if not events:
        await _send(bot, f"{header}\n\n<i>No high-impact events in the next 7 days.</i>")
        return

    lines = []
    for e in events:
        dt = e.get("datetime")
        when = dt.strftime("%a %d %b %H:%M UTC") if dt else e.get("time_str", "TBD")
        forecast = f"  f: {e['forecast']}" if e.get("forecast") else ""
        previous = f"  prev: {e['previous']}" if e.get("previous") else ""
        lines.append(f"• <b>{_h(when)}</b>  {_h(e['title'])}{_h(forecast)}{_h(previous)}")

    message = f"{header}\n\n" + "\n".join(lines)
    await _send(bot, message)


async def send_cot_update(cot_data: dict) -> None:
    """Send Friday COT positioning summary."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    bot = Bot(token=TELEGRAM_TOKEN)
    today_str = datetime.now(timezone.utc).strftime("%d %b %Y")

    def _fmt(pos: dict | None, inst: str) -> str:
        if not pos:
            return f"<b>{inst.upper()}</b>: data unavailable"
        net = pos["net_contracts"]
        direction = "NET LONG" if net > 0 else "NET SHORT"
        wow = pos["wow_change"]
        wow_str = (f"+{wow:,}" if wow >= 0 else f"{wow:,}")
        pct = pos["pct_rank"]
        label = pos["label"].upper()
        emoji = "🟢" if "long" in pos["label"] else "🔴" if "short" in pos["label"] else "⚪"
        crowd_warn = "  ⚠️ CROWDED" if pos["pct_rank"] >= 80 or pos["pct_rank"] <= 20 else ""
        return (
            f"{emoji} <b>{inst.upper()}</b> {direction} {abs(net):,} contracts"
            f"\n{pct:.0f}th percentile ({label}){crowd_warn}"
            f"\nWoW change: {wow_str} contracts"
            f"\nReport date: {pos['report_date']}"
        )

    oil_block  = _fmt(cot_data.get("oil"),  "oil")
    gold_block = _fmt(cot_data.get("gold"), "gold")

    message = (
        f"📊 <b>COT POSITIONING UPDATE — {_h(today_str)}</b>\n\n"
        f"{oil_block}\n\n"
        f"{gold_block}\n\n"
        f"<i>Source: CFTC Legacy Futures Only</i>"
    )
    await _send(bot, message)