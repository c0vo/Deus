"""
Telegram Message Formatters

Utility functions to format NewsArticle objects into visually appealing
HTML messages for Telegram, avoiding markdown parsing issues.
"""

from __future__ import annotations

from html import escape
from typing import Optional

from data.models import NewsArticle
from pipeline.grounded_answer import GroundedAnswer, classify_session_context

# Sent when nothing clears the briefing importance floor. A quiet day is signal,
# so the brief says so rather than going silent — silence is indistinguishable
# from the job having failed.
EMPTY_BRIEFING_TEXT = "📰 No high-impact stories in the last 24h."

# Per-article summary budget. Six articles at full classification_summary
# length plus long google_news URLs runs to ~5000 characters, over Telegram's
# 4096 limit; truncating here keeps the common case to a single message.
BRIEFING_SUMMARY_MAX = 220

# Telegram's hard limit is 4096. The margin absorbs the HTML markup that
# escaping can add after a chunk boundary has already been chosen.
TELEGRAM_CHUNK_LIMIT = 3800

# Per-field budgets for the daily stance note. A watchlist of ten tickers at
# four unbounded fields each runs well past the chunk limit; these keep the
# common case to one or two messages.
STANCE_THESIS_MAX = 400
STANCE_FIELD_MAX = 220

def escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram HTML parse mode."""
    return escape(str(text or ""), quote=True)

def _clip(text, limit: int) -> str:
    """Trim to a character budget BEFORE escaping, adding an ellipsis."""
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"

def chunk_html(text: str, limit: int = TELEGRAM_CHUNK_LIMIT) -> list[str]:
    """
    Split a message on paragraph boundaries so each part fits Telegram's limit.

    Splitting on blank lines keeps HTML tags intact, since the renderers here
    never open a tag in one paragraph and close it in the next.
    """
    chunks: list[str] = []
    chunk = ""

    for paragraph in text.split("\n\n"):
        # A single paragraph over the limit cannot be placed by the normal path,
        # so hard-split it. Without this the paragraph goes out whole and
        # Telegram rejects the send.
        if len(paragraph) > limit:
            if chunk.strip():
                chunks.append(chunk.strip())
                chunk = ""
            for i in range(0, len(paragraph), limit):
                chunks.append(paragraph[i : i + limit])
            continue

        if len(chunk) + len(paragraph) > limit and chunk.strip():
            chunks.append(chunk.strip())
            chunk = ""
        chunk += paragraph + "\n\n"

    if chunk.strip():
        chunks.append(chunk.strip())

    return chunks

def render_briefing(
    lanes: list[tuple[str, list[dict]]], title: str = "📰 Daily Market Briefing"
) -> str:
    """
    Render lane-grouped articles as a Telegram HTML brief.

    Takes the output of data.taxonomy.select_briefing_lanes. Shared by the
    scheduled 05:00 push and the /briefing command so the two cannot drift.
    """
    text = f"<b>{escape_html(title)}</b>\n\n"

    for label, articles in lanes:
        text += f"<b>--- {escape_html(label)} ---</b>\n"
        for row in articles:
            score = row.get("importance_score") or 0.0
            # classification_summary is null for articles ranked before they
            # were classified; falling through to summary avoids rendering the
            # literal string "None".
            summary = row.get("classification_summary") or row.get("summary") or ""
            # Truncate before escaping: escape_html can sextuple a character,
            # so trimming afterwards would not bound the result.
            if len(summary) > BRIEFING_SUMMARY_MAX:
                summary = summary[:BRIEFING_SUMMARY_MAX].rstrip() + "…"

            text += f"🔹 <b>{escape_html(row.get('headline'))}</b> ({score:.1f})\n"
            if summary:
                text += f"<i>{escape_html(summary)}</i>\n"
            text += f"<a href='{escape_html(row.get('url'))}'>Read more</a>\n\n"

    return text.rstrip() + "\n"

# Per-field budget inside a price alert. Three of these plus three cited
# sources, a technicals line and the headers fits one Telegram chunk with room
# to spare; without the cap a model writing six sentences into `cause` alone
# pushes the sources onto a second message where nobody reads them.
ALERT_FIELD_MAX = 420

# Cited sources shown in the message. The row keeps up to five; three is what
# fits before the message stops being scannable on a phone.
ALERT_SOURCE_LIMIT = 3

# Headers by alert kind. Keyed rather than branched so a new kind is one line.
_ALERT_HEADERS = {
    "price_drop": ("🩸", "PRICE DROP"),
    "price_move": ("🚀", "PRICE MOVE"),
    "volume": ("🌊", "VOLUME SPIKE"),
}


def _trim(text: str, limit: int = ALERT_FIELD_MAX) -> str:
    """Truncate before escaping — escaping can sextuple a character."""
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "…"


def render_price_alert(
    *,
    ticker: str,
    pct: float,
    price: float,
    kind: str = "price_move",
    answer: GroundedAnswer,
    index_context: Optional[dict[str, float]] = None,
    macro_lines: Optional[list[str]] = None,
    technical_line: str = "",
    vol_multiple: Optional[float] = None,
) -> str:
    """
    Render one price or volume alert as Telegram HTML.

    Structured so the reader can stop at any line and still have learned
    something: the move, then whether the market did the same thing, then
    whether anything was scheduled, then the cause — which is either cited or
    explicitly absent, never implied. The old message was a headline and two
    model-written sentences with nothing behind them.
    """
    emoji, label = _ALERT_HEADERS.get(kind, _ALERT_HEADERS["price_move"])
    safe_ticker = escape_html(ticker)
    session = classify_session_context(pct, index_context)

    text = f"{emoji} <b>{label}: {safe_ticker}</b>\n\n"

    if kind == "volume" and vol_multiple:
        text += (
            f"<b>{safe_ticker}</b> is trading at <b>{vol_multiple:.1f}x</b> its "
            f"20-session average volume, "
            f"{'up' if pct >= 0 else 'down'} <b>{abs(pct):.2f}%</b> to "
            f"<b>${price:,.2f}</b>.\n"
        )
    else:
        text += (
            f"<b>{safe_ticker}</b> is {'up' if pct >= 0 else 'down'} "
            f"<b>{abs(pct):.2f}%</b> today to <b>${price:,.2f}</b>.\n"
        )

    text += f"<b>Session:</b> {escape_html(session.line)}\n"

    for line in (macro_lines or [])[:2]:
        text += f"<b>Scheduled:</b> {escape_html(line)}\n"

    explanation = answer.explanation
    cause = _trim(explanation.cause if explanation else answer.text)
    if cause:
        text += f"\n<b>Cause:</b> {escape_html(cause)}\n"

    if explanation and explanation.sustainability.strip():
        text += f"<b>Read:</b> {escape_html(_trim(explanation.sustainability))}\n"

    if answer.sources:
        text += "\n<b>Sources:</b>\n"
        for i, source in enumerate(answer.sources[:ALERT_SOURCE_LIMIT], 1):
            title = escape_html(_trim(str(source.get("title") or "Untitled"), 120))
            label_bits = [
                str(source.get("source") or "").strip(),
                str(source.get("published_at") or "").strip(),
            ]
            meta = escape_html(", ".join(b for b in label_bits if b))
            url = str(source.get("url") or "").strip()
            if url:
                text += f"{i}. <a href=\"{escape_html(url)}\">{title}</a>"
            else:
                text += f"{i}. {title}"
            if meta:
                text += f" <i>({meta})</i>"
            text += "\n"
    else:
        # Said out loud rather than left as an empty section: "no sources" is the
        # honest reading of grounded_by='none', and an absent section reads like
        # a rendering bug instead.
        text += "\n<i>No dated source supports a cause for this move.</i>\n"

    if technical_line.strip():
        text += f"\n<b>Technicals:</b> {escape_html(_trim(technical_line, 200))}\n"

    if explanation and explanation.what_to_watch.strip():
        text += (
            f"\n<b>What to watch:</b> "
            f"{escape_html(_trim(explanation.what_to_watch))}\n"
        )

    return text.rstrip() + "\n"


def render_stance_message(rows, *, model_slug: str = "",
                          date: str = "") -> str:
    """
    Render the morning stance note.

    Takes the `StanceRow`s from `pipeline.daily_stance.DailyStanceEngine.compose`
    but does not import them: it reads flat attributes off each row, which keeps
    bot/ free of a pipeline dependency and lets the tests pass plain stand-ins.

    The header states the model, or says plainly that none is configured. The
    note it replaced printed "HOLD" in that situation, so an unset
    MODEL_DAILY_ADVISOR read exactly like a considered decision to hold every
    position — the one failure mode this header exists to make visible.
    """
    title = f"🎯 Daily Stance{f' — {date}' if date else ''}"
    text = f"<b>{escape_html(title)}</b>\n"
    if model_slug:
        text += f"<i>model: {escape_html(model_slug)}</i>\n\n"
    else:
        text += "<i>⚠️ MODEL_DAILY_ADVISOR not configured — facts only</i>\n\n"

    for row in rows:
        # A row with no arrow carries no real call (NO CALL / UNAVAILABLE):
        # there is nothing to compare against yesterday, so the slot shows a
        # failure marker instead of implying the view was held.
        arrow = getattr(row, "arrow", "") or "⚠️"
        action = str(getattr(row, "action", "") or "")
        conviction = str(getattr(row, "conviction", "") or "")

        # Truncate before escaping throughout: escape_html can sextuple a
        # character, so trimming afterwards would not bound the result.
        thesis = _clip(getattr(row, "thesis", ""), STANCE_THESIS_MAX)
        risk = _clip(getattr(row, "key_risk", ""), STANCE_FIELD_MAX)
        change = _clip(getattr(row, "what_would_change_my_mind", ""), STANCE_FIELD_MAX)
        evidence = [str(e) for e in (getattr(row, "evidence_used", None) or [])]

        head = f"{arrow} <b>{escape_html(getattr(row, 'ticker', '?'))}</b> {escape_html(action)}"
        if conviction:
            head += f" ({escape_html(conviction)})"
        if thesis:
            head += f" — {escape_html(thesis)}"
        text += head + "\n"

        if risk:
            text += f"<b>Risk:</b> {escape_html(risk)}\n"
        if evidence:
            text += f"<b>Evidence:</b> {escape_html(', '.join(evidence[:8]))}\n"
        if change:
            text += f"<b>Changes mind:</b> {escape_html(change)}\n"

        debate = getattr(row, "cached_debate", None) or {}
        if debate.get("direction"):
            detail = ", ".join(
                str(p) for p in (debate.get("conviction"), debate.get("date")) if p
            )
            line = escape_html(str(debate["direction"]))
            if detail:
                line += f" ({escape_html(detail)})"
            text += f"<b>5-day debate:</b> {line}\n"

        prev = getattr(row, "prev_action", None)
        if prev and prev != action:
            text += f"<i>was {escape_html(str(prev))}</i>\n"

        text += "\n"

    return text.rstrip() + "\n"

def format_article_alert(article: NewsArticle) -> str:
    """Format a single critical article for an immediate alert."""
    
    # Determine emoji based on event type and urgency
    urgency_emoji = {
        "critical": "🚨",
        "high": "⚠️",
        "medium": "🔔",
        "low": "📰"
    }.get(article.urgency, "📰")
    
    direction_emoji = {
        "bullish": "📈",
        "bearish": "📉",
        "neutral": "⚖️"
    }.get(article.suggested_direction, "")
    
    headline = escape_html(article.headline)
    summary = escape_html(article.classification_summary or article.summary[:200])
    
    text = f"<b>{urgency_emoji} BREAKING NEWS: {headline}</b>\n\n"
    text += f"<i>{summary}</i>\n\n"
    
    # Optional fields if they exist
    if article.sentiment_score is not None:
        text += f"<b>Sentiment:</b> {article.sentiment_score:.2f} {direction_emoji}\n"
    
    if article.affected_tickers:
        tickers = ", ".join(f"${escape_html(t)}" for t in article.affected_tickers)
        text += f"<b>Affected Tickers:</b> {tickers}\n"
        
    text += f"\n<a href='{escape_html(article.url)}'>Read Full Article</a>"

    return text


# Per-field budget for a weekly tip. A tip whose fields run long is a model
# ignoring "one to three sentences", and the message still has to chunk.
WEEKLY_TIP_FIELD_MAX = 420

# Section headings in the order the message reads: what history says, what is on
# the calendar, what to actually watch, then your own book.
WEEKLY_TIP_SEVERITY_MARKERS = {"warning": "🔴", "watch": "🟡", "info": "⚪"}

# Importance, as a marker rather than a number: 3 moves the whole tape, 2 often
# does, 1 is scheduling context (an expiration, a month end, a holiday). Lives
# here rather than in bot/commands.py so /macro and the weekly tip cannot drift
# into marking the same FOMC decision two different colours.
MACRO_IMPORTANCE_MARKERS = {3: "🔴", 2: "🟡", 1: "⚪"}

# Why the message is facts-only, in the reader's words rather than the log's.
# A digest that silently drops its tips looks like a quiet week instead of a
# misconfiguration, which is the same failure EMPTY_BRIEFING_TEXT avoids.
WEEKLY_TIP_STATUS_NOTES = {
    "not_configured": "⚠️ MODEL_WEEKLY_TIP not configured — facts only.",
    "failed": "⚠️ The tip model call failed — facts only.",
    "empty": "⚠️ No tip survived fact-checking against the data — facts only.",
}


def _clip_html(text, limit: int = WEEKLY_TIP_FIELD_MAX) -> str:
    """
    Truncate to `limit`, then escape.

    Same ordering as render_briefing and for the same reason: escape_html can
    sextuple a single character, so trimming afterwards would not bound the
    result.
    """
    raw = str(text or "").strip()
    if len(raw) > limit:
        raw = raw[: limit - 1].rstrip() + "…"
    return escape_html(raw)


def render_weekly_tip(facts: dict, tips: list) -> str:
    """
    Render the weekly tip as Telegram HTML.

    Four sections — Seasonality, Coming up, Watch, Your tickers — and every one
    of them is optional except the header: the message goes out whether or not
    the model ran, so each section is skipped when its facts are empty rather
    than printing a placeholder.

    Each tip and each list entry is its own paragraph, because `chunk_html`
    splits on blank lines: that is what lets a long week's message be sent as
    two valid messages instead of one Telegram rejects.

    `tips` is a list of `pipeline.weekly_tip.Tip`, read by attribute so the
    renderer does not import the pipeline just to type a parameter.
    """
    facts = facts or {}
    start = facts.get("period_start") or ""
    end = facts.get("period_end") or ""

    header = "<b>🧭 Weekly Tip</b>"
    if start and end:
        header += f" — {escape_html(start)} to {escape_html(end)}"
    text = header + "\n\n"

    note = WEEKLY_TIP_STATUS_NOTES.get(facts.get("tip_status") or "")
    if note and not tips:
        text += f"{note}\n\n"

    seasonality = facts.get("seasonality") or []
    if seasonality:
        text += "<b>--- Seasonality ---</b>\n\n"
        # Grouped by effect so one heading carries every ticker's line for it,
        # rather than repeating "September effect" once per ticker.
        grouped: dict[str, list[str]] = {}
        for row in seasonality:
            grouped.setdefault(str(row.get("name") or "Seasonality"), []).append(
                str(row.get("stat_line") or "")
            )
        for name, lines in grouped.items():
            block = f"<b>{_clip_html(name, 90)}</b>\n"
            for line in lines:
                if line:
                    block += f"• {_clip_html(line, 300)}\n"
            text += block + "\n"

    events = facts.get("events") or []
    earnings = facts.get("earnings") or []
    ipos = facts.get("ipos") or []
    if events or earnings or ipos:
        text += "<b>--- Coming up ---</b>\n\n"
        block = ""
        for row in events:
            marker = MACRO_IMPORTANCE_MARKERS.get(row.get("importance") or 1, "⚪")
            line = f"{marker} {_clip_html(row.get('date'), 12)} {_clip_html(row.get('name'), 120)}"
            if row.get("time_et"):
                line += f" — {_clip_html(row.get('time_et'), 8)} ET"
            if not row.get("confirmed"):
                line += " <i>(est.)</i>"
            block += line + "\n"
        for row in earnings:
            block += (
                f"📅 {_clip_html(row.get('date'), 12)} <b>{_clip_html(row.get('ticker'), 12)}</b>"
                f" {_clip_html(row.get('event_type'), 40)}\n"
            )
        for row in ipos:
            symbol = f" ({_clip_html(row.get('ticker'), 12)})" if row.get("ticker") else ""
            block += (
                f"🆕 {_clip_html(row.get('date'), 12)} {_clip_html(row.get('company_name'), 80)}"
                f"{symbol}\n"
            )
        text += block + "\n"

    warnings = facts.get("news_warnings") or {}
    themes = warnings.get("macro_themes") or []
    accelerating = warnings.get("accelerating_themes") or []
    headlines = warnings.get("headlines") or []
    if tips or themes or accelerating or headlines:
        text += "<b>--- Watch ---</b>\n\n"

        for tip in tips or []:
            severity = str(getattr(tip, "severity", "") or "info")
            marker = WEEKLY_TIP_SEVERITY_MARKERS.get(severity, "⚪")
            block = (
                f"{marker} <b>{_clip_html(getattr(tip, 'title', ''), 120)}</b> "
                f"<i>({escape_html(severity)})</i>\n"
            )
            precedent = _clip_html(getattr(tip, "precedent", ""))
            evidence = _clip_html(getattr(tip, "evidence", ""))
            action = _clip_html(getattr(tip, "action", ""))
            if precedent:
                block += f"<b>Precedent:</b> {precedent}\n"
            if evidence:
                block += f"<b>Evidence:</b> {evidence}\n"
            if action:
                block += f"<b>Action:</b> {action}\n"
            text += block + "\n"

        for row in themes:
            text += f"🌐 <b>{_clip_html(row.get('title'), 120)}</b>\n{_clip_html(row.get('explanation'))}\n\n"
        for row in accelerating:
            text += (
                f"📈 {_clip_html(row.get('label'), 160)} "
                f"<i>({row.get('acceleration')}x acceleration, "
                f"{row.get('articles_recent')} recent articles)</i>\n\n"
            )
        for row in headlines:
            text += (
                f"🔹 [{row.get('importance')}] {_clip_html(row.get('headline'), 200)} "
                f"<i>({_clip_html(row.get('source'), 40)})</i>\n\n"
            )

    performance = facts.get("performance") or []
    if performance:
        text += "<b>--- Your tickers ---</b>\n\n"
        block = ""
        for row in performance:
            pct = row.get("pct")
            if pct is None:
                continue
            emoji = "🟢" if pct >= 0 else "🔴"
            line = f"{emoji} <b>{_clip_html(row.get('ticker'), 12)}</b>: {pct:+.2f}% last week"
            if row.get("vs_spy") is not None:
                line += f" ({row['vs_spy']:+.2f}pp vs SPY)"
            block += line + "\n"
        text += block + "\n"

    return text.rstrip() + "\n"

