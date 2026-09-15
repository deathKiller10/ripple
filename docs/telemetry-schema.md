# Telemetry & Observability Schema

Deliverable for gate **G6** (100% trace coverage) and for the "Telemetry &
Observability Schema" item in §8 of the Theme 4 Guide.

One event type, JSONL, one line per event. The same stream is written to disk
(`traces/<session_id>.jsonl`) and pushed to the dashboard over the WebSocket,
so **what a judge sees on screen is exactly what is in the file** — the
dashboard computes nothing the engine did not emit.

## Envelope

Every line carries:

| field | type | meaning |
|---|---|---|
| `session_id` | string | dies with the session; never reused, never persisted |
| `seq` | int | monotonic within a session |
| `t_rel_s` | float | seconds since session start. In replay this is the *virtual* clock taken from transcript timestamps, so a run on a slow machine produces identical timings to a fast one |
| `type` | string | one of the event types below |
| `wall_clock_ms` | float | real time, for latency analysis |

Payload fields are present according to `type`; absent fields are omitted
rather than set to null.

## Event types

| type | emitted when | key payload |
|---|---|---|
| `session_started` | session opens | `detail.config`, `detail.index`, `detail.provider` |
| `chunk_received` | every transcript fragment | `chunk_text`, `prefix_text` |
| `controller_decision` | every fragment | `controller{decision, stability, novelty, drift, threshold, semantic_jump, reason}`, `latency_ms` |
| `compoundness_tested` | utterance end | `detail{is_compound, score, coordination, foci, topical_variance}` |
| `decomposed` | after the gate | `sub_queries`, `detail.method` (`single` · `zero-token` · `llm`) |
| `retrieval_started` | each search dispatched | `sub_query`, `trigger`, `intent_id` |
| `retrieval_completed` | results in | `retrieved[]` (citation strings), `latency_ms` |
| `fusion_completed` | budget allocated | `allocation{intent_id: n}`, `retrieved[]`, `rerank_scores[]`, `detail.starved_intents` |
| `pool_updated` | evidence state changes | `detail` — counts, evictions, supersessions |
| `constraint_detected` | late detail arrives | `detail{scope, value, invalidates[], raw_text}` |
| `claim_emitted` | a claim enters the answer | `claim_id`, `citations[]`, `detail.speculative` |
| `claim_preserved` | survives a refinement untouched | `claim_id`, `citations[]` |
| `claim_superseded` | invalidated by a constraint | `claim_id`, `detail.by` |
| `grounding_checked` | per claim, and pre-synthesis | `detail{lexical, semantic, grounded, reason}` or `detail{stage:"pre_synthesis_relevance", oov_rate, term_cover}` |
| `uncertainty_emitted` | abstention | `uncertainty` |
| `first_token` | first claim of a turn | `detail.speculative`, `detail.trigger` |
| `answer_version` | version committed | `answer_version`, `citations[]`, `detail{added, preserved, superseded, citations_changed_on_preserved, retrievals, ttft_rel_utterance_end}` |
| `utterance_end` | speaker finishes | `prefix_text` |
| `session_ended` | close | `cost`, `detail.pool` |

## Example

```json
{ "session_id": "dev_late_01", "seq": 14, "t_rel_s": 3.1,
  "type": "retrieval_started",
  "controller": { "decision": "RETRIEVE", "stability": 0.71,
                  "novelty": 0.80, "drift": 0.23, "threshold": 0.40,
                  "semantic_jump": false,
                  "reason": "stable_and_novel (s=0.71>=0.4, n=0.80>=0.3)" },
  "sub_query": "Can you tell me whether a display replacement is covered",
  "trigger": "provisional",
  "intent_id": "int_9f2c1ab4",
  "latency_ms": 11.2 }
```

## How coverage is *checked*, not claimed

`ripple/telemetry.py` exposes an `@instrumented` decorator that records which
engine stages ran. `tests/test_gates.py::test_trace_coverage_is_total` drives a
reference session and asserts that every required event type appeared. Add a
stage and forget to instrument it, and CI fails.

`evaluation/run_bench.py` computes the same number across a whole split and
reports it as the G6 figure, so the benchmark output and the test agree by
construction.

## What is deliberately *not* here

No user identifier, no device identifier, no cross-session key, and nothing
that would let two sessions be linked. The guide prohibits cross-session
profiling and persistent user tracking; `tests/test_gates.py::test_sessions_do_not_leak_into_each_other`
asserts that two identical sessions produce identical state, which they could
not if the second were influenced by the first.

## Cost accounting

`cost` carries `{prompt_tokens, completion_tokens, llm_calls, embed_calls,
retrieval_calls, rerank_calls, currency_cost}`. Prices live in
`config.py::CostTable` and are settable by environment variable, so a reported
cost-per-turn survives a price change without re-running anything. Tokens are
reported alongside money for the same reason.
