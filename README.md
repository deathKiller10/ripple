# Ripple — Streaming Live RAG

**Samsung PRISM GenAI Hackathon, 3rd Edition · Theme 04**

**Team** · Priyanshu Kundu · Souptik Hazra · Anushka Paul · Arpita Bhaumik
**Repo** · https://github.com/deathKiller10/ripple
**Submission tag** · `PRISM_GENAI_HACKATHON_Y2026` · due 25 Sep 2026

Retrieval that starts before the sentence lands, and an answer that updates
itself instead of starting over.

Ripple is an event-driven Streaming Live RAG engine. It decides *in retrieval
space* when a half-spoken request has stabilised enough to search, fans out
across the several questions hidden inside one utterance, and holds the answer
as a graph of individually-cited claims — so a detail arriving three seconds
later patches two sentences instead of restarting the search.

The reference application, **Ripple for Care**, grounds a support agent who is
listening to a customer. That framing matters: the assistant is not being
spoken to. There is no turn boundary to wait for and nobody to ask for
clarification, so full-duplex behaviour is the only mode that exists.

---

## Run it

```bash
docker compose up
# → http://localhost:8000
```

No API key. No GPU. No model download at run time — the corpus, the labels, the
scenarios and the index are all built into the image. This is gate **G1**, and
it is the gate most submissions lose.

Without Docker, two commands:

```bash
pip install -r requirements.txt
python scripts/setup.py
```

`setup.py` builds the corpus, the classifier labels, the benchmark scenarios
and the search index, runs the test suite, reports whether an LLM key is
configured, and prints what to run next. Safe to re-run at any time.

Then:

```bash
uvicorn ripple.server:app --port 8000     # → http://localhost:8000
```

### Using a real LLM

Everything above works with no API key. To get meaningful grounding and cost
numbers, create a file called `.env` in the repo root:

```
RIPPLE_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
RIPPLE_RPM=12
```

`.env` is git-ignored and loaded automatically. **Start with two systems** —
the full four-system run issues roughly 600 calls because B2 retrieves on
every chunk by design, and that will exhaust a free tier:

```bash
python -m evaluation.run_bench --split dev --provider gemini \
       --systems B1_static_rag,B3_ripple --out results_gemini
```

### The automated replay suite (no browser)

```bash
python -m ripple.replay --scenarios data/scenarios/dev.jsonl --out results
```

Writes, per scenario, the Theme 4 Guide's §4 structured output record with the
guide's field names verbatim — `retrieval_events`, `sub_queries`, `answer`,
`citations`, `uncertainty` — so a harness written against the guide can consume
our output with no adaptation. Ripple-specific extras are namespaced under
`ripple_ext` so the core record stays exactly as specified.

### Benchmark, calibration, tests

```bash
python -m evaluation.run_bench --split dev        # four systems, all six gates
python -m evaluation.calibrate                    # threshold sweeps (dev only)
python tests/test_gates.py                        # 9 property tests
```

---

## Current status — dev split, 40 scenarios, keyless provider

```
system                early retr false trig  multi-int   recall@k intent cov  fabricated continuity   TTFT med  retr/turn
B0_llm_only                0.000      0.000      0.000      0.000      0.000           0      1.000      0.000      0.000
B1_static_rag              0.000      1.000      0.000      0.835      0.768           0      1.000     +0.010      1.000
B2_naive_streaming         1.000      1.000      0.000      0.835      0.768           0      1.000     -2.069      4.039
B3_ripple                  0.884      0.000      0.889      0.960      0.772           0      1.000     -1.036      2.745

ACCEPTANCE GATES (Theme 4 Guide §5)
  [PASS] G2 early retrieval                   0.884 >= 0.80     38 of 43 eligible turns
  [PASS] G2 false triggers                    0.000             0 of 8 no-retrieval turns
  [PASS] G3 multi-intent identification       0.889 >= 0.70     8 of 9 compound turns
  [PASS] G4 citation support                  1.000 >= 0.85     see caveat below
  [PASS] G4 fabricated citations                  0 == 0        of 159 citations
  [PASS] G5 state continuity                  1.000 >= 1.00     over 6 refinement turns
  [PASS] G6 telemetry trace coverage          1.000 >= 1.00     asserted by test
```

**Read these with the caveats they deserve** — the full discussion is in
[`docs/evaluation-report.md`](docs/evaluation-report.md).

- `citation_support = 1.000` is **not an achievement** under the keyless stub
  provider, which answers by copying a sentence out of the chunk it cites. The
  stub exists to satisfy G1 on a machine with no API key, not to flatter G4.
  **The meaningful grounding number comes from a real-provider run**, and the
  report must name the provider.
- `llm_calls` and `cost_per_turn` are zero for the same reason, so cost is
  compared on `retrievals_per_turn` — which is real: naive streaming spends
  **49% more retrievals per turn** and triggers on **100%** of turns that
  needed no retrieval at all.
- **The held-out split has not been run.** Reserved for one execution after
  feature freeze. All calibration happened on `dev`.

### Five ablations, each changing exactly one variable

| Variant | TTFT med | recall@k | intent cov | retr/turn |
|---|---|---|---|---|
| **B3 full system** | **−1.036** | **0.960** | **0.772** | 2.75 |
| A1 controller OFF | −1.036 | 0.960 | 0.764 | **3.29** |
| A2 fusion = plain RRF | −1.036 | 0.960 | **0.718** | 2.75 |
| A3 speculation OFF | **0.000** | 0.960 | 0.797 | 2.75 |
| A4 add BM25 back | −1.036 | **0.893** | 0.720 | 2.75 |
| A5 reranker back ON | −1.036 | 0.960 | 0.726 | 2.75 |

**Two components were removed on this evidence.** BM25 *lowered* recall
(0.960 → 0.893) and won nothing on a 16-query part-number probe — our embedder
reads character n-grams, so the "hybrid" was two correlated lexical signals
competing. The lexical-semantic reranker *lowered* intent coverage
(0.772 → 0.726) for 90 ms per sub-query, because it re-sorted a ranking already
better than its own scoring. Both stay behind switches with a note on when to
re-measure. Deleting your own work on evidence is the parsimony rule applied to
yourself, and the guide grades exactly that.

A2 is the clearest positive result: recall@k is **identical** at 0.960 either
way, so every gold chunk was retrieved — plain RRF simply allocated the context
budget so that some sub-questions received none of it. That is sub-intent
starvation, and no standard retrieval metric can see it.

---

## Why this is Streaming Live RAG, mechanically

### M1 · Retrieval-space stability — the controller

Everyone else measures whether a partial utterance is *linguistically*
complete. That needs a model, costs tokens on every transcript chunk, and
answers the wrong question. The question that matters is: **will more words
change which documents come back?**

That is directly measurable. For each chunk we run a *shadow retrieval* — a
bare dense top-k lookup, no BM25, no reranking, no generation — and compare its
ranking to the previous chunk's with rank-biased overlap.

```
drift_t     = 1 - RBO(topk(prefix_t), topk(prefix_{t-1}))
stability_t = EMA(1 - drift_t)
novelty_t   = |topk(prefix_t) \ evidence_pool| / k

WAIT           stability < θ
RETRIEVE       stability ≥ θ  AND  novelty ≥ ν
RETRIEVE_MORE  semantic jump against active intents  →  parallel branch
SUPPRESS       presentation/social classifier fires  →  answer from state
```

Cost: one embedding plus one exact ANN query per chunk. **Zero tokens.** The
guide's parsimony rule is satisfied by construction rather than by assertion.

**θ is derived, not tuned.** `evaluation/calibrate.py` sweeps it against the
asymmetric cost `E = P(false trigger)·c_waste + P(late)·c_late`, with
`c_waste = 1` and `c_late = 175` reflecting that a wasted shadow retrieval
costs ~4 ms of CPU while a late one costs ~700 ms of user-visible silence. The
measured curve on dev:

```
  theta    early  false_trig    E[cost]
   0.40     0.85        0.00      26.92  <-- chosen
   0.56     0.62        0.00      67.31
   0.68     0.38        0.00     107.69
   0.92     0.15        0.00     148.08
```

An honest finding from that sweep: **θ trades early retrieval against wasted
shadow retrievals, not against false triggers.** The false-trigger rate is 0.00
at every θ, because false triggers are prevented upstream by the suppression
classifier, which is a different component entirely. The two halves of gate G2
are controlled by two different mechanisms.

### M2 · The claim graph — the state

Gates G4 (zero fabricated citations) and G5 (refine without restarting) fight
each other as long as the answer is a string: the only way to fold a late
constraint into a paragraph is to rewrite the paragraph, and a rewritten
paragraph's citations drift.

So the answer is not a string. It is a set of `Claim` objects, each carrying
its intent, its evidence IDs (which *are* the citation list) and the
assumptions it holds under. A late constraint becomes four deterministic steps,
only the last of which touches a model:

1. **Invalidate** — claims whose assumptions conflict are marked `SUPERSEDED`.
   Nothing is deleted; history is the audit trail and the version diff.
2. **Re-score** — pool evidence is re-ranked against the constraint. Chunks
   previously at rank 14 can surface *with no new retrieval at all*.
3. **Delta retrieve** — only for intents that lost a claim.
4. **Patch** — regenerate only the superseded claims.

Unaffected claims are never regenerated, so their text and citations *cannot*
drift. No model ever writes a citation string, which makes a fabricated
document ID unrepresentable rather than merely unlikely.

This also gives gate G5 the numeric definition Samsung left out:

> **state continuity** = (logically-unaffected claims whose text **and**
> citation set are byte-identical across versions) / (unaffected claims)

Measured, target 1.0, asserted in `tests/test_gates.py`.

**Scope tags come from the corpus, never from code.** A constraint is retrieved
like any query; whatever scope its top-scoring evidence declares is what it
activates, and it invalidates conflicting values of the same key. "Bought in
Dubai" activates `region:cross_border` from `DOC_WAR_03`, so it supersedes the
domestic warranty claim and leaves the device diagnosis standing. Point the
engine at a corpus about insurance riders and the same code does the same job.

### M3 · Coverage-budgeted fusion

RRF pools every sub-query's results into one ranked list. If the customer asked
three things and the budget is twelve chunks, RRF can return nine about the
display and zero about turnaround — every retrieval metric looks excellent
while a third of the answer silently disappears. **Sub-intent starvation is
invisible to recall@k.**

So we allocate rather than rank: a guaranteed floor per active intent, then
greedy marginal gain on reranker score, then MMR de-duplication across the
selection. Measured with a metric we had to define — per-intent grounded
coverage — because no standard one sees it.

### M4 · Signed TTFT

Time-to-first-token is reported **relative to end-of-utterance**, so it can be
negative. A measured median of **−0.52 s** means the answer is already
streaming while the customer is still speaking. Speculative synthesis fires on
a principled signal: *a new topic opening is the completion signal for the
previous one* — the speaker moving on is stronger evidence that a sub-question
is finished than any measure of the sentence itself.

---

## Architecture — the diagram is also a cost ledger

| Stage | Responsibility | Cost |
|---|---|---|
| Transcript feed | timestamped chunks over WebSocket; voice simulated from transcripts | 0 tokens |
| **1. Retrieval controller** (M1) | shadow retrieval, stability/novelty/jump → WAIT · RETRIEVE · RETRIEVE_MORE · SUPPRESS | **0 tokens**, ~4 ms |
| **2. Compoundness gate** | coordination + question foci + topical variance; single-intent utterances skip stage 3 entirely | **0 tokens** |
| **3. Sub-query extractor** | one structured call, only when compound, **concurrent** with the first retrieval so its latency never lands in TTFT | 1 call / compound turn |
| **4. Parallel retrieval** | FAISS flat IP, exact. Fanned out across sub-queries. BM25 removed on the evidence of ablation A4 | 0 tokens, ~8 ms |
| **5. Coverage-budgeted fusion** (M3) | per-intent floor, marginal-gain fill, MMR de-dup. Reranker removed on the evidence of ablation A5 | 0 tokens |
| **6. Claim synthesiser** (M2) | claims as structured objects bound to evidence IDs and assumptions | 1 call / version |
| **7. Abstention + grounding verifier** | corpus-vocabulary coverage test before synthesis; citation existence and support after | **0 tokens** |
| **8. Telemetry bus** | every stage emits; coverage asserted by test | 0 tokens |

**Deliberately not used:** LangChain, LlamaIndex, LangGraph, any agent
framework, Kafka, Kubernetes, Celery, a vector-database server, Redis. Session
state lives in process memory because the guide's session-bound rule makes
persistence a spec violation, not a missing feature.

---

## The abstention gate, and a hypothesis that failed

The corpus contains a deliberate coverage hole: nothing anywhere covers
screen-protector reimbursement or trade-in valuation. §1 of the guide requires
an explicit uncertainty indicator when evidence is insufficient.

Our first implementation measured **distribution shape** — the z-score of the
top result against the ranking's tail — on the theory that a covered query
produces a sharp peak and an uncovered one a flat ranking. Measured on dev, the
reverse happened: the screen-protector hole produced the **highest** standout
score of any query tested (z = 12.8), because the corpus contains a section
titled *"Screen protection accessories"* that matches the query's surface form
almost perfectly while answering nothing. Lexical confidence and topical
relevance came apart exactly where they needed to agree.

What works is a **corpus-vocabulary test**: the share of the question's content
words that appear nowhere in the corpus at all. If a question turns on a
concept the corpus has no word for, the corpus cannot answer it, however well
some passage matches the remaining words. Both signals are dimensionless and
derived from the corpus at index time, so they need no retuning when the
embedder, the reranker or the corpus changes — the property that matters for a
benchmark we are not allowed to see.

---

## Corpus

`data/corpus/care/` — the **Care Knowledge Pack**: 21 documents, 60 sections,
every section carrying an explicit `Doc_ID §Section` marker so citations are
exact by construction. It is **synthetic**, written for this project; it is not
Samsung material and every figure in it is invented.

It is a test instrument. Each document is present to exercise a mechanism:

| Property | How | Exercises |
|---|---|---|
| Multi-intent spread | one situation spans KB + warranty + policy + parts | G3, M3 |
| Late-constraint override | `DOC_WAR_03` overrides `DOC_WAR_01 §1–2` once purchase region is known | G5, M2 |
| Genuine contradiction | `DOC_POL_09 §1` supersedes `DOC_POL_07 §1`, with dates | evidence lifecycle |
| Lexical distractors | "flicker" in a monitor doc; "S24" in an accessory doc | prefix lock-in |
| Near-duplicates | `DOC_POL_12` restates `DOC_POL_11` with different SLAs | MMR dedup |
| **Deliberate hole** | nothing on screen-protector reimbursement or trade-in | abstention |

**Swapping corpora is a config change, not a refactor.** `RIPPLE_CORPUS` plus
an index rebuild. The ingestion adapter reads markdown or JSONL; adding a third
reader is ~20 lines. This is deliberate: the guide says the benchmark replay is
held-out and private, which implies the judges may point the engine at content
we have never indexed.

> **If Samsung supplied a Theme 4 corpus, use it.** It was not in our workspace.
> Point `RIPPLE_CORPUS` at it, rebuild the index, and re-label the scenarios.

---

## Evaluation discipline

`dev` (12 scenarios, 16 labelled turns) — all threshold calibration happens
here and nowhere else.
`heldout` (12 scenarios, 16 labelled turns) — **run once, after feature
freeze.** Never inspected while tuning.

Turn kinds make gate G2 measurable at all, which it is not without them:

| kind | needs retrieval | notes |
|---|---|---|
| `informational` | yes | G2 numerator |
| `refinement` | yes | delta search only; a full re-search counts against delta efficiency |
| `presentation` | **no** | a retrieval here is a **false trigger** — the half of G2 teams omit |
| `social` | **no** | greetings, hold messages |
| `uncoverable` | yes | correct answer is an uncertainty indicator |

---

## Two measurement bugs worth recording

Both were caught by the harness disagreeing with the mechanism, and both would
have led to fixing the wrong component.

**1. Early-retrieval rate of 8%.** Scenarios were *authored* as whole clauses
because that is readable, and replayed that way too. A single-chunk utterance
offers no opportunity to retrieve early — the first chunk to arrive is also the
last. The controller was fine; the input was wrong. Scenarios are now fragmented
at replay time into ~3 words per chunk at 2.9 words/sec, the granularity a
streaming recogniser actually emits, and the rate went to 85%.

**2. Ripple's recall appearing to be half the baseline's.** The harness
populated `retrieved_cites` for the static baseline but not for Ripple, so
recall fell back to the citation list — twelve retrieved chunks against three
cited claims. Recall must be computed over what a system *retrieved*, not what
it chose to cite. Fixed: 0.420 → 0.948. Anything that makes your own system
look bad in a way you cannot explain mechanically is a harness bug until proven
otherwise.

---

## Submission artifacts

| Deliverable | Where | Required by |
|---|---|---|
| Working prototype | this repo | main deck |
| README, reproducible setup, Docker | here, `Dockerfile`, `docker-compose.yml` | main deck |
| Presentation | `VITVellore_Ripple_Submission.pptx` | main deck |
| Demo video ≤ 5 min | *link to be added before submission* | both |
| **System architecture brief** | [`docs/architecture-brief.md`](docs/architecture-brief.md) | Theme 4 Guide §8 |
| **Benchmarking & evaluation report** | [`docs/evaluation-report.md`](docs/evaluation-report.md) | Theme 4 Guide §8 |
| **Telemetry & observability schema** | [`docs/telemetry-schema.md`](docs/telemetry-schema.md) | Theme 4 Guide §8 |
| Demo shot list | [`docs/demo-script.md`](docs/demo-script.md) | — |
| Jury Q&A preparation | [`docs/judge-qa.md`](docs/judge-qa.md) | — |
| AI usage disclosure log | [`docs/ai-log.md`](docs/ai-log.md) | disclosure form |

## Working on this

[`docs/architecture-brief.md`](docs/architecture-brief.md) is the design
rationale; [`docs/evaluation-report.md`](docs/evaluation-report.md) has the
measurements, five ablations and the analysed failures.

`AGENTS.md` holds the project rules — the hard constraints from Samsung, the
invariants that must not break, who owns which folders, and the evaluation
discipline. Claude Code, Antigravity and most agentic editors load it automatically.
`docs/kickoff-prompts.md` has a first-session prompt for each member.

## Layout

```
ripple/
  schemas.py          frozen contracts: telemetry, guide §4 record, scenarios
  config.py           every threshold, in one file, so calibrate.py can sweep it
  telemetry.py        the bus; `instrumented` is how G6 is checked not claimed
  engine.py           the orchestrator — transport-free
  server.py           FastAPI + WebSocket (a client of the engine)
  replay.py           headless CLI (the other client) — gate G1
  corpus/loader.py    ingestion adapter; sections are the retrieval unit
  retrieval/          embedders · hybrid index · fusion (M3) · rerank · relevance
  controller/         stability (M1) · compoundness + suppression · policy
  session/            pool (evidence lifecycle) · claim_graph (M2)
  synthesis/          providers · synthesizer · verifier
evaluation/           baselines (B0–B2) · metrics · run_bench · calibrate
data/                 corpus · scenarios · classifier labels (all generated)
frontend/             dashboard.html (zero-build) + React scaffold
docs/                 architecture brief · telemetry schema · AI disclosure log
tests/                9 property tests defending the report's claims
```

## Configuration

Everything has a working default; `.env` is optional. See `.env.example`.
The ones that matter: `RIPPLE_PROVIDER` (`stub` · `gemini` · `openai`),
`RIPPLE_EMBEDDER` (`tfidf-svd` · `bge-small`), `RIPPLE_THETA`, `RIPPLE_CORPUS`.

## Submission

Release tag **`PRISM_GENAI_HACKATHON_Y2026`** on the final commit. Everything
referenced by the deck, the demo video and the documentation must exist inside
that tagged commit.
