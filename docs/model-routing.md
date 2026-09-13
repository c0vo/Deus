# Deus model routing

Three ranked picks for all 21 `MODEL_*` settings in `.env`, priced per million tokens and weighted by
what each function actually needs.

Every model listed here was called through this project's own `config/llm.py` against a real schema
before it earned a place. Prices are OpenRouter's published per-million rates as of **16 Aug 2026**.

| | |
|---|---|
| Routable functions | 21 |
| Models verified working | 14 |
| Models rejected on test | 4 |
| Spent verifying | $0.011 |

---

## Read this first

### Every free model 404s on your account

All 19 advertised `:free` slugs — including the `openrouter/free` auto-router — return:

> `No endpoints available matching your guardrail restrictions and data policy.`

This is a **settings toggle, not availability**. Free tiers require allowing prompt logging at
<https://openrouter.ai/settings/privacy>. That means your prompts — which carry your watchlist,
positions and thesis reasoning — become training data.

Free models are therefore **excluded from every ranking below**. If you do enable the toggle, the
ceiling is 1,000 requests/day and 20/min, and the roster rotates weekly, so pin them only to
low-volume lanes.

### The three cheapest models on the price list do not work

| Model | Listed price | What happened |
|---|---|---|
| `inclusionai/ling-3.0-flash` | $0.021 / $0.063 | Provider returns HTTP 405 |
| `upstage/solar-pro4` | $0.03 / $0.12 | No routable endpoint (404) |
| `tencent/hy3` | $0.132 / $0.528 | Advertises structured output; failed to return parseable JSON against `list[TickerNote]` |

Advertised `supported_parameters` is not a guarantee. Probe before you commit a lane.

### Read the cost-shape tag, not just the price

The reasoning lanes bill thinking against the completion budget, so a debate turn emits up to 8,000
output tokens and a thesis decomposition up to 12,000. On those, **output price is the only number
that matters** — a model at $6/M out costs *20×* one at $0.28/M out for identical work, however
similar their input prices look.

---

## Verified working models

| Model | $/M in | $/M out | Schema | Reasoning | Context |
|---|---:|---:|:---:|:---:|---:|
| `nvidia/nemotron-3.5-lightning` | 0.10 | 0.25 | ✅ | — | 1M |
| `openai/gpt-5.6-luna` | 0.10 | 0.60 | ✅ | ✅ | 1.05M |
| `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | ✅ | ✅ | 1M |
| `google/gemini-3.5-flash-lite` | 0.30 | 2.50 | ✅ | ✅ | 1M |
| `google/gemini-3.7-flash` | 0.375 | 1.875 | ✅ | ✅ | 1M |
| `deepseek/deepseek-v4-pro-0813` | 0.435 | 0.87 | ✅ | ✅ | 1.05M |
| `qwen/qwen3.8-27b` | 0.45 | 3.20 | ✅ | ✅ | 262K |
| `google/gemini-3.6-flash` | 0.75 | 3.75 | ✅ | ✅ | 1M |
| `openai/gpt-5.6-terra` | 1.00 | 6.00 | ✅ | ✅ | 1M |
| `x-ai/grok-4.6` | 2.00 | 6.00 | ✅ | ✅ | 500K |
| `anthropic/claude-sonnet-5` | 2.00 | 10.00 | ✅ | ✅ | 1M |
| `moonshotai/kimi-k3` | 3.00 | 15.00 | ✅ | ✅ | 1M |
| `openai/gpt-5.6-sol` | 5.00 | 30.00 | ✅ | ✅ | 1M |
| `anthropic/claude-opus-5` | 5.00 | 25.00 | ✅ | ✅ | 1M |

`:batch` variants of the Google and OpenAI models are ~50% cheaper but are not usable here — this
pipeline needs synchronous responses.

---

## Ingest

Highest volume in the system — a cycle every 15 minutes. This is where model price compounds, and
where the quality floor sets everything downstream.

### `MODEL_CLASSIFIER` — input-heavy

Event type, sentiment, urgency and tickers, 10 articles per call. Runs with `reasoning="none"`.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | Your incumbent, and still the best extraction-per-dollar here. Cheapest output of any model that handles this reliably. |
| 2 | `nvidia/nemotron-3.5-lightning` | 0.10 | 0.25 | Cheapest verified option, 1M context. Has no reasoning mode at all — irrelevant here, since classification disables thinking anyway. |
| 3 | `openai/gpt-5.6-luna` | 0.10 | 0.60 | Same input price, stronger on messy headlines. Pick it if ticker extraction is missing names. |

### `MODEL_CLASSIFIER_FALLBACK` — negligible volume

Second attempt when the primary errors. Fires rarely, so price barely matters — vendor diversity does.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `google/gemini-3.5-flash-lite` | 0.30 | 2.50 | Different vendor from a DeepSeek primary, which is the entire job of this setting. Its price is irrelevant at fallback volume. |
| 2 | `openai/gpt-5.6-luna` | 0.10 | 0.60 | A third vendor and far cheaper, if you would rather the fallback also be frugal. |
| 3 | `nvidia/nemotron-3.5-lightning` | 0.10 | 0.25 | Cheapest, but weakest of the three at recovering the batch the primary just choked on. |

### `MODEL_REDDIT_SENTIMENT` — input-heavy

Sentiment over a WSB post and its comments. The easiest judgment call in the pipeline.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `nvidia/nemotron-3.5-lightning` | 0.10 | 0.25 | Cheapest verified, and sentiment on short social text does not reward a stronger model. |
| 2 | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | 40% more for better handling of sarcasm and meme phrasing, which WSB is made of. |
| 3 | `openai/gpt-5.6-luna` | 0.10 | 0.60 | Only if the comment threads you ingest are long — 1.05M context and cheap input. |

### `MODEL_REDDIT_SENTIMENT_FALLBACK` — negligible volume

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | Different vendor from an NVIDIA primary, and cheap enough to leave on permanently. |
| 2 | `google/gemini-3.5-flash-lite` | 0.30 | 2.50 | Widest vendor gap, if resilience matters more than the retry's cost. |
| 3 | *(leave empty)* | — | — | Perfectly valid. Reddit sentiment is the least load-bearing signal you ingest; a skipped batch costs almost nothing. |

### `MODEL_RANKER` — input-heavy

Importance 0–10 per article. Nothing reaches the daily brief or the alerts without it.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | **The single biggest saving available.** Ranking is almost all input tokens with an ~80-token output cap, and this is a fraction of the Gemini-lite output rate. |
| 2 | `nvidia/nemotron-3.5-lightning` | 0.10 | 0.25 | Cheapest. Watch the score distribution for a week — a ranker that drifts toward the middle quietly empties your brief. |
| 3 | `google/gemini-3.7-flash` | 0.375 | 1.875 | Best calibration judgment of the three. Worth it if scores feel noisy against your 0–10 anchors. |

### `MODEL_EXTRACT` — balanced

IPO details, calendar events, web-search summaries. Single-object JSON, moderate volume.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `nvidia/nemotron-3.5-lightning` | 0.10 | 0.25 | Field extraction against a fixed shape is the job cheap models are best at. Capped at 500 output tokens anyway. |
| 2 | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | Better at dates — IPO windows and earnings dates are where this lane actually fails. |
| 3 | `openai/gpt-5.6-luna` | 0.10 | 0.60 | Strongest at pulling a ticker out of prose that never states one. |

---

## Chat

You are sitting there waiting. Latency and answer quality outrank unit cost — the volume is whatever
you personally type.

### `MODEL_ROUTER` — negligible volume

One JSON field: shallow or complex. Runs before every chat turn.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `nvidia/nemotron-3.5-lightning` | 0.10 | 0.25 | Fastest cheap tier, and this call sits directly in front of your first token. A binary decision needs nothing more. |
| 2 | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | Marginally better at spotting a question that only looks simple. |
| 3 | `openai/gpt-5.6-luna` | 0.10 | 0.60 | Output price is irrelevant at ~10 tokens per call, so this is effectively free too. |

### `MODEL_CHAT_SHALLOW` — output-heavy

Fast prose answers, no reasoning budget.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | Cheapest output of anything that writes decent prose — and this lane is nearly all output. |
| 2 | `openai/gpt-5.6-luna` | 0.10 | 0.60 | Noticeably better written. Roughly 2× the cost of a reply, on a lane that costs fractions of a cent. |
| 3 | `google/gemini-3.5-flash-lite` | 0.30 | 2.50 | Your current default, and the most expensive of the three by an order of magnitude on output. |

### `MODEL_CHAT_COMPLEX` — output-heavy

Reasoning-tier answers over RAG plus live web context. Runs at `reasoning="medium"`.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `deepseek/deepseek-v4-pro-0813` | 0.435 | 0.87 | Half the output price of gemini-3.7-flash at a comparable reasoning tier. With medium effort billing thinking as completion, that gap is the whole cost of the lane. |
| 2 | `google/gemini-3.7-flash` | 0.375 | 1.875 | Cheaper input, better at holding a long RAG context together. The safer default if answers feel scattered. |
| 3 | `anthropic/claude-sonnet-5` | 2.00 | 10.00 | Best synthesis available. Defensible only because you are the only user — if chat is the feature you actually live in, this is where to spend. |

---

## Batch notes

Short structured JSON, once a day or on a trigger. Total spend here is rounding error — optimise for
whether you trust the output, not for price.

### `MODEL_TRENDING` — balanced

One-line note per trending ticker, `list[TickerNote]` schema, 24h cache.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | Verified against the list-schema envelope. One call covers every ticker, cached 24h — a handful of calls a day. |
| 2 | `nvidia/nemotron-3.5-lightning` | 0.10 | 0.25 | Also verified on this exact schema. Notes are blander. |
| 3 | `google/gemini-3.7-flash` | 0.375 | 1.875 | Best at saying something non-obvious about why a ticker is moving. |

### `MODEL_DAILY_ADVISOR` — balanced

HOLD or SELL per tracked ticker, once a day, straight into your Telegram brief.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `google/gemini-3.7-flash` | 0.375 | 1.875 | One call per day producing a call you may act on. This is the clearest case in the whole config for paying up. |
| 2 | `deepseek/deepseek-v4-pro-0813` | 0.435 | 0.87 | Comparable judgment at less than half the output price. |
| 3 | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | Fine if you read the brief as a prompt to look, not as advice. |

### `MODEL_MARKET_SCANNER` — output-heavy

Prose explanation attached to a ≥5% swing or an earnings whisper. Fires only on a trigger.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | Pure prose with no schema, and it lands on your phone. Cheapest good writer wins. |
| 2 | `openai/gpt-5.6-luna` | 0.10 | 0.60 | Better at the HTML-formatted, no-emoji house style this prompt asks for. |
| 3 | `google/gemini-3.7-flash` | 0.375 | 1.875 | Alerts are rare enough that the price never shows up on the dashboard. |

### `MODEL_SECTOR_ANALYZER` — balanced

A JSON map of ticker → one-sentence rationale for hot names.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `nvidia/nemotron-3.5-lightning` | 0.10 | 0.25 | The fallback string is "Surge in news mentions detected." — almost anything beats that, so buy cheap. |
| 2 | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | More specific rationales for barely more money. |
| 3 | `google/gemini-3.5-flash-lite` | 0.30 | 2.50 | Your current default; no reason to keep it here. |

### `MODEL_TREND_OUTLOOK` — balanced

Sector outlook and macro themes — two JSON calls at `reasoning="low"`.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | Handles the low reasoning budget and the JSON shape, at the lowest output price on the board. |
| 2 | `google/gemini-3.7-flash` | 0.375 | 1.875 | Broader macro knowledge, which is most of what this lane is doing. |
| 3 | `nvidia/nemotron-3.5-lightning` | 0.10 | 0.25 | Cheapest, but has no reasoning mode — the `low` budget is silently ignored. |

### `MODEL_PREDICTOR_NARRATIVE` — output-heavy

Plain-English rationale on an ML prediction — and the substitute predictor (`LlmPrediction` schema)
when no model exists for a ticker.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | Runs per ticker per day, so this is quietly one of your higher-volume lanes. Verified on the `LlmPrediction` schema. |
| 2 | `google/gemini-3.7-flash` | 0.375 | 1.875 | This lane also has to *be* the predictor when a ticker has no trained model. Worth upgrading if you see many `llm_only` rows. |
| 3 | `nvidia/nemotron-3.5-lightning` | 0.10 | 0.25 | Fine for narration, weakest of the three at the confidence calibration the prompt asks for. |

### `MODEL_REFLECTION` — balanced

Grades matured predictions into lessons. Feeds straight back into the next debate.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | Summarisation over a resolved outcome, batch-capped nightly. The code comment already says flash is the right tier. |
| 2 | `deepseek/deepseek-v4-pro-0813` | 0.435 | 0.87 | These lessons are re-injected into every future debate, so a shallow lesson compounds. Cheap upgrade for a nightly batch. |
| 3 | `google/gemini-3.7-flash` | 0.375 | 1.875 | Different vendor grading a DeepSeek-produced debate, which is a real argument against marking your own homework. |

---

## Reasoning

Lowest call volume, highest cost per call. Thinking is billed as completion, so this is where output
price decides your bill.

### `MODEL_DEBATE` — output-dominated

Bull and Bear, two rounds each, `reasoning="xhigh"`, up to 8,000 output tokens per turn, streamed to
the arena.

| # | Model | $/M in | $/M out | ~per debate | Why |
|---|---|---:|---:|---:|---|
| **1** | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | $0.009 | Four turns × 8k output. Your current default, and correctly chosen — nothing else comes close on output price. |
| 2 | `deepseek/deepseek-v4-pro-0813` | 0.435 | 0.87 | $0.028 | Genuinely sharper argument construction for 3× a figure that is still under three cents. |
| 3 | `openai/gpt-5.6-terra` | 1.00 | 6.00 | $0.19 | 21× the cheapest option. Only defensible if you run a handful of debates a week and read every word. |

### `MODEL_TRADER` — output-heavy

Synthesises the whole debate into BUY/SELL/HOLD plus conviction, via the `TraderAdvisory` schema at
`reasoning="high"`. Once per debate.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `deepseek/deepseek-v4-pro-0813` | 0.435 | 0.87 | One call per debate producing the verdict you read. Reasoning tier, schema-verified, and the cheapest output of any model at that tier. |
| 2 | `google/gemini-3.7-flash` | 0.375 | 1.875 | A second opinion from a different vendor than the debaters — worth something when the synthesiser is judging their arguments. |
| 3 | `anthropic/claude-sonnet-5` | 2.00 | 10.00 | The strongest synthesiser here. At one call per debate the absolute cost stays small, and this is the output you actually act on. |

### `MODEL_THESIS_REASONER` — output-dominated

Decomposes a theme into a bottleneck tree at `reasoning="xhigh"`, 12,000 output tokens. Its chain of
thought is streamed to the thesis page, **so the model must return reasoning**.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `deepseek/deepseek-v4-pro-0813` | 0.435 | 0.87 | Verified to stream reasoning deltas through the new facade, which this lane depends on — the thinking pane is empty without them. Roughly $0.010 per decomposition. |
| 2 | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | A third of the cost, but this is the one call whose entire value is the depth of the causal chain. Downgrade last. |
| 3 | `x-ai/grok-4.6` | 2.00 | 6.00 | ~$0.072 a run. Different reasoning style, which is worth something on a feature built to find non-obvious second-order names. |

### `MODEL_THESIS_EXTRACT` — balanced

Pulls company names out of search results into the nested `CandidateList` schema. Also names detected
themes. Up to 6 calls per thesis run.

| # | Model | $/M in | $/M out | Why |
|---|---|---:|---:|---|
| **1** | `google/gemini-3.7-flash` | 0.375 | 1.875 | Verified on the nested `CandidateList` schema, which is the most demanding shape in the codebase. Your current default and a good one. |
| 2 | `deepseek/deepseek-v4-flash-0731` | 0.14 | 0.28 | Also verified on nested schemas, at a seventh of the output price. The obvious saving if thesis runs get frequent. |
| 3 | `nvidia/nemotron-3.5-lightning` | 0.10 | 0.25 | Cheapest, but name-from-prose extraction is exactly where small models invent tickers. The quote-validation step catches those — at the cost of a wasted hop. |

### `MODEL_EMBEDDING` — pinned

Vectors for dedup, RAG and thesis grounding.

| # | Model | $/M | Why |
|---|---|---:|---|
| **1** | `google/gemini-embedding-001` | 0.15 | **Do not change this.** Verified cosine 1.000000 against your stored corpus through OpenRouter, so nothing needed re-embedding. Any other model produces a different vector space and silently miscalibrates dedup (0.70), theme clustering (0.30) and thesis grounding (0.65) at once. |
| 2 | — | — | There is no second choice that preserves the corpus. Switching means re-embedding every stored article and re-tuning three thresholds. |
| 3 | — | — | The embedder now discards any vector that is not 3072-wide, so a wrong slug shows up as articles that never embed rather than as a corrupted index. |

---

## Presets

Paste one into `.env` and adjust individual lanes from the tables above.

### Floor

Cheapest verified model everywhere; expect blander notes and a flatter importance curve.

```dotenv
# ingest
MODEL_CLASSIFIER=nvidia/nemotron-3.5-lightning
MODEL_CLASSIFIER_FALLBACK=deepseek/deepseek-v4-flash-0731
MODEL_REDDIT_SENTIMENT=nvidia/nemotron-3.5-lightning
MODEL_REDDIT_SENTIMENT_FALLBACK=
MODEL_RANKER=nvidia/nemotron-3.5-lightning
MODEL_EXTRACT=nvidia/nemotron-3.5-lightning
# chat
MODEL_ROUTER=nvidia/nemotron-3.5-lightning
MODEL_CHAT_SHALLOW=deepseek/deepseek-v4-flash-0731
MODEL_CHAT_COMPLEX=deepseek/deepseek-v4-pro-0813
# batch notes
MODEL_TRENDING=nvidia/nemotron-3.5-lightning
MODEL_DAILY_ADVISOR=deepseek/deepseek-v4-flash-0731
MODEL_MARKET_SCANNER=deepseek/deepseek-v4-flash-0731
MODEL_SECTOR_ANALYZER=nvidia/nemotron-3.5-lightning
MODEL_TREND_OUTLOOK=deepseek/deepseek-v4-flash-0731
MODEL_PREDICTOR_NARRATIVE=deepseek/deepseek-v4-flash-0731
MODEL_REFLECTION=deepseek/deepseek-v4-flash-0731
# reasoning
MODEL_DEBATE=deepseek/deepseek-v4-flash-0731
MODEL_TRADER=deepseek/deepseek-v4-flash-0731
MODEL_THESIS_REASONER=deepseek/deepseek-v4-flash-0731
MODEL_THESIS_EXTRACT=deepseek/deepseek-v4-flash-0731
# pinned
MODEL_EMBEDDING=google/gemini-embedding-001
```

### Balanced *(recommended)*

Cheap where volume is high, reasoning-tier where judgment compounds.

```dotenv
# ingest
MODEL_CLASSIFIER=deepseek/deepseek-v4-flash-0731
MODEL_CLASSIFIER_FALLBACK=google/gemini-3.5-flash-lite
MODEL_REDDIT_SENTIMENT=nvidia/nemotron-3.5-lightning
MODEL_REDDIT_SENTIMENT_FALLBACK=deepseek/deepseek-v4-flash-0731
MODEL_RANKER=deepseek/deepseek-v4-flash-0731
MODEL_EXTRACT=deepseek/deepseek-v4-flash-0731
# chat
MODEL_ROUTER=nvidia/nemotron-3.5-lightning
MODEL_CHAT_SHALLOW=deepseek/deepseek-v4-flash-0731
MODEL_CHAT_COMPLEX=deepseek/deepseek-v4-pro-0813
# batch notes
MODEL_TRENDING=deepseek/deepseek-v4-flash-0731
MODEL_DAILY_ADVISOR=google/gemini-3.7-flash
MODEL_MARKET_SCANNER=deepseek/deepseek-v4-flash-0731
MODEL_SECTOR_ANALYZER=nvidia/nemotron-3.5-lightning
MODEL_TREND_OUTLOOK=deepseek/deepseek-v4-flash-0731
MODEL_PREDICTOR_NARRATIVE=deepseek/deepseek-v4-flash-0731
MODEL_REFLECTION=deepseek/deepseek-v4-flash-0731
# reasoning
MODEL_DEBATE=deepseek/deepseek-v4-flash-0731
MODEL_TRADER=deepseek/deepseek-v4-pro-0813
MODEL_THESIS_REASONER=deepseek/deepseek-v4-pro-0813
MODEL_THESIS_EXTRACT=google/gemini-3.7-flash
# pinned
MODEL_EMBEDDING=google/gemini-embedding-001
```

### Sharp

Ingest stays cheap; the outputs you read and act on get the best model available.

```dotenv
# ingest — unchanged from Balanced, this is where volume lives
MODEL_CLASSIFIER=deepseek/deepseek-v4-flash-0731
MODEL_CLASSIFIER_FALLBACK=google/gemini-3.5-flash-lite
MODEL_REDDIT_SENTIMENT=deepseek/deepseek-v4-flash-0731
MODEL_REDDIT_SENTIMENT_FALLBACK=nvidia/nemotron-3.5-lightning
MODEL_RANKER=google/gemini-3.7-flash
MODEL_EXTRACT=deepseek/deepseek-v4-flash-0731
# chat
MODEL_ROUTER=nvidia/nemotron-3.5-lightning
MODEL_CHAT_SHALLOW=openai/gpt-5.6-luna
MODEL_CHAT_COMPLEX=anthropic/claude-sonnet-5
# batch notes
MODEL_TRENDING=google/gemini-3.7-flash
MODEL_DAILY_ADVISOR=google/gemini-3.7-flash
MODEL_MARKET_SCANNER=google/gemini-3.7-flash
MODEL_SECTOR_ANALYZER=deepseek/deepseek-v4-flash-0731
MODEL_TREND_OUTLOOK=google/gemini-3.7-flash
MODEL_PREDICTOR_NARRATIVE=google/gemini-3.7-flash
MODEL_REFLECTION=deepseek/deepseek-v4-pro-0813
# reasoning
MODEL_DEBATE=deepseek/deepseek-v4-pro-0813
MODEL_TRADER=anthropic/claude-sonnet-5
MODEL_THESIS_REASONER=deepseek/deepseek-v4-pro-0813
MODEL_THESIS_EXTRACT=google/gemini-3.7-flash
# pinned
MODEL_EMBEDDING=google/gemini-embedding-001
```

---

## Method and caveats

**How these were checked.** Every ranked model was called through this project's own `config/llm.py`
against a real schema (`list[TickerNote]`, the shape five call sites use) or with `reasoning="xhigh"`
for the reasoning tier. Models that errored or failed to produce parseable output were dropped rather
than listed with a caveat. Per-debate and per-thesis figures are computed from this project's
configured token budgets (`debate_max_output_tokens=8000`, `thesis_max_output_tokens=12000`), not
measured over a full run.

**Effectiveness is a judgment, not a benchmark.** The orderings weigh published capability, tier and
what each lane actually demands. They are not head-to-head evaluations on your corpus — the honest way
to settle any of these is to change one lane, watch `/api/usage` and the output for a week, and change
it back if it got worse.

**Everything here is one `.env` line.** No code change moves a function between providers, and unset
keys are reported by name at startup with the feature they disable.

**Prices move.** This snapshot is 16 Aug 2026. Re-check <https://openrouter.ai/models> before acting
on the narrower gaps — that volatility is the reason the routing is config rather than code.
