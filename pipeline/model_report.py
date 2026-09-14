"""
The per-horizon skill table for the pooled direction model.

One table, three places: the Sunday retrain report on Telegram, the bot's
/model command, and the console of scripts/manual/train_pooled_models.py. Each
reads `Database.get_latest_model_metrics()` rows (one per horizon) and renders
them here, so the three can never disagree about what a number means.

Every metric is from the confirm folds — the last walk-forward folds, which
the ship decision reads — not an in-sample fit. Below the table, every horizon
that serves the base rate says why: the ship conditions it failed, or why it
could not be measured.
"""

from __future__ import annotations

import html
import math
from typing import Any, Optional

from pipeline.predictor import HORIZON_LABELS

COLUMNS = ("Horizon", "Status", "AUC [90% CI]", "Brier skill", "Hi-conf acc (n)", "Base up",
           "Tickers/Rows")

LEGEND = (
    "AUC ranks tickers against each other on the same day: 0.50 = coin flip, 0.53+ = small real edge",
    "Brier skill > 0 = probabilities beat the base rate",
    "status prior = no measurable edge; serving the base rate",
    "not measurable = too few independent label windows to judge; serving the base rate",
)


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _note(row: dict) -> Optional[str]:
    config = row.get("config_json")
    if isinstance(config, dict) and config.get("note"):
        return str(config["note"])
    return str(row["note"]) if row.get("note") else None


def _not_measurable(row: dict) -> bool:
    """No folds could be built (no AUC, a note), or the confirm folds held too little evidence.

    The second case is `config_json["measurable"] is False`, written by
    model_training.metrics_row when the confirm folds hold fewer than
    MIN_CONFIRM_N_EFF independent label windows or no interval could be drawn:
    an AUC printed from them would read as a measurement it is not.
    """
    config = row.get("config_json")
    if isinstance(config, dict) and config.get("measurable") is False:
        return True
    return _number(row.get("auc_mean")) is None and bool(_note(row))


def _cells(row: dict) -> list[str]:
    horizon = row.get("horizon_days")
    try:
        label = HORIZON_LABELS.get(int(horizon), f"{int(horizon)}d")
    except (TypeError, ValueError):
        label = "?"

    status = str(row.get("status") or "n/a")

    auc, low, high = (_number(row.get(k)) for k in ("auc_mean", "auc_ci_low", "auc_ci_high"))
    if _not_measurable(row):
        auc_cell = "not measurable"
    elif auc is None:
        auc_cell = "n/a"
    elif low is not None and high is not None:
        auc_cell = f"{auc:.3f} [{low:.2f}-{high:.2f}]"
    else:
        auc_cell = f"{auc:.3f}"

    skill = _number(row.get("brier_skill_mean"))
    skill_cell = f"{skill:+.4f}" if skill is not None else "n/a"

    hi_acc = _number(row.get("hi_conf_acc"))
    hi_n = _number(row.get("hi_conf_n"))
    hi_n_text = f"{int(hi_n):,}" if hi_n is not None else "0"
    hi_cell = f"{hi_acc:.1%} ({hi_n_text})" if hi_acc is not None else f"n/a ({hi_n_text})"

    base = _number(row.get("prior_up_rate"))
    base_cell = f"{base:.1%}" if base is not None else "n/a"

    tickers = _number(row.get("n_tickers"))
    rows = _number(row.get("n_rows"))
    size_cell = (f"{int(tickers) if tickers is not None else 0}/"
                 f"{int(rows) if rows is not None else 0:,}")
    return [label, status, auc_cell, skill_cell, hi_cell, base_cell, size_cell]


def _table_lines(rows: list[dict]) -> list[str]:
    body = [_cells(r) for r in sorted(rows or [], key=lambda r: _number(r.get("horizon_days")) or 0)]
    widths = [max(len(COLUMNS[i]), *(len(line[i]) for line in body)) if body else len(COLUMNS[i])
              for i in range(len(COLUMNS))]

    def join(cells) -> str:
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()

    lines = [join(COLUMNS), "-+-".join("-" * w for w in widths)]
    lines += [join(line) for line in body]
    return lines


def _footer_lines(rows: list[dict], total_seconds: Optional[float]) -> list[str]:
    lines = []
    versions = sorted({int(v) for v in (_number(r.get("schema_version")) for r in rows or []) if v is not None})
    trained = sorted(str(r.get("created_at")) for r in rows or [] if r.get("created_at"))
    parts = []
    if versions:
        parts.append("schema v" + "/".join(str(v) for v in versions))
    if trained:
        parts.append(f"last trained {trained[-1][:16]} UTC")
    if total_seconds is not None and math.isfinite(total_seconds):
        minutes, seconds = divmod(int(round(total_seconds)), 60)
        parts.append(f"training took {minutes}m {seconds:02d}s")
    if parts:
        lines.append(", ".join(parts))
    for row in sorted(rows or [], key=lambda r: _number(r.get("horizon_days")) or 0):
        note = _note(row)
        if not note:
            continue
        label = HORIZON_LABELS.get(int(_number(row.get("horizon_days")) or 0), str(row.get("horizon_days")))
        # A prior row's note is the list of ship conditions it failed.
        verdict = "not measurable" if _not_measurable(row) else str(row.get("status") or "prior")
        lines.append(f"{label} {verdict}: {note}")
    return lines


def render_metrics_table_text(rows: list[dict]) -> str:
    """The skill table for a console: header, one line per horizon, legend."""
    if not rows:
        return "No model metrics recorded yet."
    lines = _table_lines(rows)
    lines.append("")
    lines += list(LEGEND)
    footer = _footer_lines(rows, None)
    if footer:
        lines.append("")
        lines += footer
    return "\n".join(lines)


def render_metrics_table_html(rows: list[dict], *, total_seconds: float | None = None,
                              failures: list[tuple[str, str]] | None = None) -> str:
    """The skill table as Telegram HTML: a <pre> table, the legend, then any failures.

    `failures` are (label, error) pairs for horizons that did not train this
    run; their previous row, if any, is what the table shows.
    """
    parts = ["<b>Direction model skill (walk-forward, confirm folds)</b>"]
    if rows:
        table = "\n".join(_table_lines(rows))
        parts.append(f"<pre>{html.escape(table)}</pre>")
        parts.append("\n".join(f"<i>{html.escape(line)}</i>" for line in LEGEND))
        footer = _footer_lines(rows, total_seconds)
        if footer:
            parts.append("\n".join(html.escape(line) for line in footer))
    else:
        parts.append("No model metrics recorded yet.")
        if total_seconds is not None and math.isfinite(total_seconds):
            minutes, seconds = divmod(int(round(total_seconds)), 60)
            parts.append(html.escape(f"training took {minutes}m {seconds:02d}s"))
    if failures:
        failed = "\n".join(f"- {html.escape(str(label))}: {html.escape(str(error)[:200])}"
                           for label, error in failures)
        parts.append(f"<b>Failed:</b>\n{failed}")
    return "\n\n".join(parts)
