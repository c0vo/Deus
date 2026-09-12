"""
A/B the batch-classifier request shape against the live model.

Why this exists: after the `item_N` + `Literal` enum schema shipped, most
batches of ten came back holding a single item — ~41 or ~300 output tokens
instead of ~1,500 — and nine of ten articles went unmatched, burning their
classification_attempts within fifteen minutes. The enum was added for a real
reason (models looping on `a1a1a1a1...`), so the question is not "revert" but
"which request shape returns all N without reopening the loop".

Eight variants, each the same ten real articles and the same prompt body that
`ArticleClassifier._classify_group` builds:

  A  item_N labels + Literal enum on `id`    (the shape that regressed)
  B  item_N labels, `id: str` required, no enum
  C  B + minItems/maxItems = N on the items array
  D  A + a prompt line stating N and the exact id list
  E  a1..aN labels, `id` required, no enum
  F  enum + minItems/maxItems + states N
  G  F + every field required                (what ships now; calls _batch_schema)
  H  a1..aN with `id: str = ""`, i.e. `required: []` — the true pre-W2 request

What it found, 3-4 runs each against deepseek-v4-flash-0731 and the nvidia
fallback: the model answers a json_schema `response_format` with the *minimum*
the schema permits, on both axes independently.

  - No `minItems` on the array ⇒ one object is legal ⇒ one object (A, B, D).
  - `required: ["id"]` ⇒ only `id` is demanded ⇒ only `id` comes back. So C, E
    and F return all ten ids and classify nothing: 316-340 output tokens of
    `{"id": "item_4", "urgency": "medium"}`. On the fallback slug A does this
    too, which is the `applied: 10` batch that changed no verdict.
  - Only G does both: 10/10 matched, 10/10 carrying a real event_type and
    summary, ~1,400 output tokens, no repetition loop.

H is the control that explains the "98% complete before the deploy" figure: an
empty `required` list demanded nothing, so the model fell back on the prompt for
the count — and still returned near-empty objects. The content half of this bug
predates the commit that exposed the count half.

Manual, not a test: it calls the live model. ~24 calls, ~$0.002 total.

    venv\\Scripts\\python.exe scripts/manual/ab_batch_classifier_schema.py
    ... --runs 3 --batch 10 --variants G,A --dump 600
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import create_model  # noqa: E402

from config.llm import complete, parse_json_list, salvage_json_array  # noqa: E402
from config.settings import settings  # noqa: E402
from pipeline.classifier import (  # noqa: E402
    BATCH_CLASSIFICATION_PROMPT,
    BatchClassifierResult,
    ClassifierResult,
    _batch_schema,
)

MODEL = "deepseek/deepseek-v4-flash-0731"
FALLBACK_MODEL = "nvidia/nemotron-3.5-lightning"
MAX_TOKENS_PER_ARTICLE = 400


# ── Article selection ────────────────────────────────────────────────────


def pick_articles(db_path: Path, count: int) -> list[dict]:
    """Read-only: real headlines and summaries, newest first."""
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, headline, summary FROM articles
            WHERE summary IS NOT NULL AND TRIM(summary) != ''
              AND headline IS NOT NULL AND TRIM(headline) != ''
              AND LENGTH(summary) > 120
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (int(count),),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ── The five request shapes ──────────────────────────────────────────────


def enum_schema(labels: list[str]) -> Any:
    """Variant A/D: `id` is a Literal of exactly this batch's labels."""
    model = create_model(
        "BatchClassifierResultLabelled",
        id=(Literal[tuple(labels)], ...),  # type: ignore[valid-type]
        __base__=ClassifierResult,
    )
    return list[model]  # type: ignore[valid-type]


def plain_schema() -> Any:
    """Variant B/C/E: `id` required but free text."""
    return list[BatchClassifierResult]


def all_required_schema(labels: list[str]) -> Any:
    """
    Variant G: enum on `id`, and every other field required too — which is now
    `pipeline.classifier._batch_schema` itself, so G re-measures shipped code
    rather than a copy of it that can drift away from it.

    Pydantic puts only default-less fields in `required`, so the schema as first
    deployed marked eight of nine properties optional — a conforming decoder may
    answer `{"id": "item_3"}` and be correct. Dropping the defaults makes the
    request ask for what the pipeline actually needs, while the *parsing* model
    keeps its defaults so a thin response still validates.
    """
    return _batch_schema(labels)


def optional_id_schema() -> Any:
    """Variant H: `id: str = ""`, which empties the schema's `required` list."""
    model = create_model(
        "BatchClassifierResultOptionalId",
        id=(str, ""),
        __base__=ClassifierResult,
    )
    return list[model]  # type: ignore[valid-type]


def build_prompt(payload: list[dict], labels: list[str], state_n: bool) -> str:
    """The shipped batch prompt, optionally with its count paragraph removed.

    There is one copy of the 4.4k-character guidance block and it lives in
    `pipeline.classifier`, so the variants that predate the count paragraph are
    reproduced by subtracting it rather than by keeping a second copy here. If
    this regex stops matching, the prompt was reworded: re-derive the paragraph
    rather than leaving the A/B silently comparing identical requests.
    """
    prompt = BATCH_CLASSIFICATION_PROMPT.format(
        articles_json=json.dumps(payload, ensure_ascii=False),
        count=len(labels),
        first_label=labels[0],
        last_label=labels[-1],
    )
    if state_n:
        return prompt

    stripped, subs = re.subn(
        r"There are \d+ articles above.*?no fewer\.\s*", "", prompt, flags=re.DOTALL
    )
    if subs != 1:
        raise SystemExit(
            "build_prompt: could not subtract the count paragraph from "
            "BATCH_CLASSIFICATION_PROMPT; the prompt was reworded."
        )
    return stripped


VARIANTS: dict[str, dict[str, Any]] = {
    "A": {"label": "item_{i}", "enum": True, "exact": False, "state_n": False,
          "desc": "item_N + enum (deployed)"},
    "B": {"label": "item_{i}", "enum": False, "exact": False, "state_n": False,
          "desc": "item_N, id:str, no enum"},
    "C": {"label": "item_{i}", "enum": False, "exact": True, "state_n": False,
          "desc": "B + min/maxItems=N"},
    "D": {"label": "item_{i}", "enum": True, "exact": False, "state_n": True,
          "desc": "A + prompt states N"},
    "E": {"label": "a{i}", "enum": False, "exact": False, "state_n": False,
          "desc": "a1..aN, no enum (pre-W2)"},
    # Not in the brief, but the obvious combination of the two winners and the
    # thing actually worth shipping if both help independently.
    "F": {"label": "item_{i}", "enum": True, "exact": True, "state_n": True,
          "desc": "enum + min/maxItems + states N"},
    # F plus every field required, because item *count* turned out not to be the
    # only thing the deployed schema left free.
    "G": {"label": "item_{i}", "enum": True, "exact": True, "state_n": True,
          "all_required": True, "desc": "F + all fields required"},
    # The *actual* pre-W2 request: `id: str = ""` gave it a default, so the
    # generated schema's `required` list was empty — nothing at all was demanded.
    # That is the shape that was 98% complete, and it is the control that tells
    # us a non-empty `required` list is what the model is reading as "emit this
    # and nothing more".
    "H": {"label": "a{i}", "enum": False, "exact": False, "state_n": False,
          "id_optional": True, "desc": "a1..aN, required:[] (true pre-W2)"},
}


async def run_one(variant: str, articles: list[dict], model: str = MODEL) -> dict:
    spec = VARIANTS[variant]
    n = len(articles)
    labels = [spec["label"].format(i=i) for i in range(1, n + 1)]

    payload = [
        {"id": key, "headline": a["headline"], "summary": a["summary"] or ""}
        for key, a in zip(labels, articles)
    ]
    prompt = build_prompt(payload, labels, spec["state_n"])
    if spec.get("all_required"):
        schema = all_required_schema(labels)
    elif spec.get("id_optional"):
        schema = optional_id_schema()
    elif spec["enum"]:
        schema = enum_schema(labels)
    else:
        schema = plain_schema()

    result: dict[str, Any] = {"variant": variant, "sent": n, "model": model}
    try:
        response = await complete(
            model=model,
            prompt=prompt,
            temperature=0.0,
            schema=schema,
            json_mode=False,
            max_tokens=MAX_TOKENS_PER_ARTICLE * n,
            reasoning="none",
            exact_items=n if spec["exact"] else None,
        )
    except Exception as e:  # live call; an outage must not kill the sweep
        result.update(returned=0, ids=[], out_tokens=0, error=f"{type(e).__name__}: {e}")
        return result

    text = (response.text or "").strip()
    usage = response.usage
    result["out_tokens"] = getattr(usage, "completion_tokens", 0) or 0
    result["in_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
    result["finish"] = response.finish_reason or ""
    result["cost"] = response.cost or 0.0

    salvaged = False
    try:
        items = parse_json_list(text)
    except Exception:
        items = salvage_json_array(text, key="items")
        salvaged = True
    result["salvaged"] = salvaged

    ids = [
        str(it.get("id")) for it in items
        if isinstance(it, dict) and it.get("id")
    ]
    result["returned"] = len(items)
    result["ids"] = ids
    result["matched"] = len(set(ids) & set(labels))
    # Does the runaway-repetition failure the enum was added for come back?
    result["looped"] = any(len(i) > 24 or i.count(labels[0]) > 1 for i in ids)

    # An item count is not the whole story. Every field but `id` has a default,
    # so it is absent from the schema's `required` list and a conforming decoder
    # may return `{"id": "item_3"}` and nothing else — ten of those parse, match,
    # and classify nothing. Count the items that carry real content.
    substantive = 0
    valid = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            ClassifierResult.model_validate({k: v for k, v in it.items() if k != "id"})
            valid += 1
        except Exception:
            pass
        if (it.get("classification_summary") or "").strip() and it.get("event_type"):
            substantive += 1
    result["substantive"] = substantive
    result["valid"] = valid
    result["text"] = text
    if not items:
        result["head"] = text[:160]
    return result


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--variants", default="A,B,C,D,E,F,G,H")
    ap.add_argument("--db", default=None, help="defaults to DB_PATH from .env")
    ap.add_argument(
        "--model", default=MODEL,
        help=f"OpenRouter slug to A/B; the fallback is {FALLBACK_MODEL}",
    )
    ap.add_argument(
        "--dump", type=int, default=0, metavar="CHARS",
        help="print this many chars of the first run's raw response per variant",
    )
    args = ap.parse_args()

    if not os.getenv("OPENROUTER_API_KEY") and not settings.openrouter_api_key:
        print("STOP: OPENROUTER_API_KEY is not set; cannot A/B against the live model.")
        return 2

    db_path = Path(args.db or settings.db_path)
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parents[2] / db_path
    if not db_path.exists():
        print(f"STOP: no database at {db_path}")
        return 2

    articles = pick_articles(db_path, args.batch)
    if len(articles) < args.batch:
        print(f"STOP: only {len(articles)} usable articles in {db_path}")
        return 2

    print(f"model   : {args.model}")
    print(f"db      : {db_path}")
    print(f"batch   : {len(articles)} articles, {args.runs} runs per variant")
    print(f"max_tok : {MAX_TOKENS_PER_ARTICLE * len(articles)}")
    print()
    for i, a in enumerate(articles, 1):
        print(f"  {i:2d}. {a['headline'][:88]}")
    print()

    wanted = [v.strip().upper() for v in args.variants.split(",") if v.strip()]
    rows: list[dict] = []
    for variant in wanted:
        if variant not in VARIANTS:
            continue
        print(f"── {variant}: {VARIANTS[variant]['desc']} " + "─" * 28)
        for run in range(1, args.runs + 1):
            r = await run_one(variant, articles, args.model)
            rows.append(r)
            if r.get("error"):
                print(f"   run {run}: ERROR {r['error']}")
                continue
            ids = ",".join(r["ids"][:12]) or "(none)"
            flags = []
            if r.get("salvaged"):
                flags.append("salvaged")
            if r.get("looped"):
                flags.append("LOOPED")
            if r.get("finish") not in ("stop", ""):
                flags.append(f"finish={r['finish']}")
            print(
                f"   run {run}: returned={r['returned']:2d}/{r['sent']} "
                f"matched={r['matched']:2d} with_content={r['substantive']:2d} "
                f"valid={r['valid']:2d} out_tokens={r['out_tokens']:5d} {' '.join(flags)}"
            )
            print(f"           ids=[{ids}]")
            if args.dump and run == 1:
                flat = " ".join(r["text"].split())
                print(f"           raw: {flat[:args.dump]}")
        print()

    print("=" * 94)
    print(
        f"{'var':<4}{'shape':<34}{'all N':>8}{'usable':>8}"
        f"{'avg items':>11}{'avg w/content':>14}{'avg out tok':>12}{'loops':>7}"
    )
    print("-" * 94)
    for variant in wanted:
        mine = [r for r in rows if r["variant"] == variant and not r.get("error")]
        if not mine:
            print(f"{variant:<4}{VARIANTS[variant]['desc']:<34}{'all errored':>8}")
            continue
        full = sum(1 for r in mine if r["matched"] == r["sent"])
        # The bar that matters: all N matched AND all N carrying real content.
        usable = sum(
            1 for r in mine if r["matched"] == r["sent"] and r["substantive"] == r["sent"]
        )
        avg_items = sum(r["returned"] for r in mine) / len(mine)
        avg_sub = sum(r["substantive"] for r in mine) / len(mine)
        avg_tok = sum(r["out_tokens"] for r in mine) / len(mine)
        loops = sum(1 for r in mine if r.get("looped"))
        print(
            f"{variant:<4}{VARIANTS[variant]['desc']:<34}"
            f"{str(full) + '/' + str(len(mine)):>8}{str(usable) + '/' + str(len(mine)):>8}"
            f"{avg_items:>11.1f}{avg_sub:>14.1f}{avg_tok:>12.0f}{loops:>7}"
        )
    total_cost = sum(r.get("cost") or 0.0 for r in rows)
    print("-" * 94)
    print(f"total measured cost: ${total_cost:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
