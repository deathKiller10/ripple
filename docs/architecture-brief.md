# Ripple — System Architecture Brief

**Samsung PRISM GenAI Hackathon, 3rd Edition · Theme 04 Streaming Live RAG**
Team: Priyanshu Kundu · Souptik Hazra · Anushka Paul · Arpita Bhaumik
Repository: https://github.com/deathKiller10/ripple

Deliverable for §8 of the Theme 4 Guide: design rationale, retrieval trigger
logic, query decomposition strategy, data provenance, trade-offs, and failure
mode mitigations.

---

## 1. The problem, stated precisely

A support agent is listening to a customer. The customer says:

> "My S24's screen is flickering after the update, the battery dies by 2pm, is
> any of this covered, and how long's a repair?"

Four questions in one breath. The agent has roughly 1.5 seconds after the
customer stops before the silence becomes conspicuous.

The structural point — and the reason this theme is not a chatbot problem — is
that **the customer is not addressing the assistant.** There is no submit
action, no turn boundary to wait for, and no way to ask for clarification. A
turn-based system waits for a query it will never receive. Full-duplex
behaviour here is not an optimisation; it is the only mode in which the system
exists at all.

Three consequences follow, and they are the three hard problems:

1. Retrieval must begin from a partial utterance, without knowing what comes
   next, and without spending a model call per fragment.
2. One utterance carries several questions that must each get answered, not
   blended.
3. A detail arriving seconds later must improve the answer rather than reset
   it — and must not break the citations already given.

---

## 2. Architecture

Single Python process. FastAPI serves both the WebSocket and the built
frontend, so `docker compose up` yields one container on one port. **The engine
is importable and transport-free**: the WebSocket server and the headless
replay CLI are both clients of it, and neither is a dependency of it. That
separation is what makes gate G1 achievable — an automated replay suite that
needs a browser is not automated.

The table below doubles as a cost ledger, because the guide grades
cost-to-performance and the claim "the controller is free" needs to be
checkable rather than asserted.

| Stage | Responsibility | Cost per invocation |
|---|---|---|
| Transcript feed | Timestamped chunks over WebSocket. Voice is simulated from transcripts, as the scope permits. | 0 tokens |
| **1 · Retrieval controller** | Shadow retrieval, stability / novelty / semantic-jump. Emits `WAIT` · `RETRIEVE` · `RETRIEVE_MORE` · `SUPPRESS`. | **0 tokens**, ~4 ms |
| **2 · Compoundness gate** | Coordination markers, question foci, topical variance. Single-intent utterances skip stage 3 entirely. | **0 tokens** |
| **3 · Sub-query extractor** | One structured call, only on compound utterances, running **concurrently** with the first retrieval so its latency never lands in TTFT. Zero-token fallback always available. | 1 call / compound turn |
| **4 · Parallel retrieval** | FAISS flat inner-product, exact. Fanned out with `asyncio` across sub-queries. | 0 tokens, ~8 ms |
| **5 · Coverage-budgeted fusion** | Per-intent floor, marginal-gain fill, MMR de-duplication. Writes into the session evidence pool with lifecycle states. | 0 tokens |
| **6 · Abstention gate** | Conservative pre-filter: does any single retrieved passage address this question? | **0 tokens** |
| **7 · Claim synthesiser** | Claims as structured objects bound to evidence IDs and assumptions. On refinement, patches only superseded claims. | 1 call / answer version |
| **8 · Grounding verifier** | Every cited ID must exist in the pool and the claim must be supported by it. Failures become uncertainty, not output. | **0 tokens** |
| **9 · Telemetry bus** | Every stage emits through one decorated emitter. Coverage asserted by test. | 0 tokens |

**Deliberately absent:** LangChain, LlamaIndex, LangGraph, any agent framework,
Kafka, Celery, Kubernetes, Redis, a vector-database server. Session state lives
in process memory because the guide's session-bound rule makes persistence a
spec violation rather than a missing feature.

---

## 3. Retrieval trigger logic

### 3.1 The reframe

Every system we know of measures whether a partial utterance is
*linguistically* complete. That requires a model, costs tokens on every
fragment, and answers a question we do not care about.

The question that matters is: **will more words change which documents come
back?** That is directly measurable, for free, by comparing the ranked result
sets of successive prefixes.

For each incoming chunk we run a **shadow retrieval** — a bare dense top-k
lookup, no reranking, no generation — and compare its ranking to the previous
chunk's using rank-biased overlap:

```
drift_t     = 1 − RBO(topk(prefix_t), topk(prefix_{t−1}))
stability_t = EMA(1 − drift_t)
novelty_t   = |topk(prefix_t) \ evidence_pool| / k

WAIT           stability < θ
RETRIEVE       stability ≥ θ  AND  novelty ≥ ν
RETRIEVE_MORE  semantic jump against active intents → parallel branch
SUPPRESS       presentation/social classifier fires → answer from state
```

Two properties are worth defending to a jury:

- A grammatically *incomplete* fragment can be stable. Once enough of the
  subject is present, the document set stops moving and the remaining words
  will not change it.
- A grammatically *complete* sentence can be unstable, if it sits on a boundary
  between document clusters.

RBO is chosen over Jaccard or Kendall's tau because it is top-weighted: a
change at rank 1 should register strongly, a change at rank 9 barely at all,
because rank 9 rarely reaches synthesis. `tests/test_gates.py` asserts this
property directly, and asserts that Jaccard cannot distinguish the two cases.

The **novelty** term is the budget controller in one number. Once the pool
already contains what the shadow retrieval found, a further retrieve buys
nothing, so it is suppressed even at maximum stability. Without it, a long
settled utterance would re-retrieve on every chunk: high stability is not by
itself a reason to search.

### 3.2 θ is derived, not chosen

`evaluation/calibrate.py` sweeps θ against the asymmetric cost

```
E[cost] = P(false trigger)·c_waste + P(late)·c_late
```

with `c_waste = 1` and `c_late = 175`, reflecting that a wasted shadow
retrieval costs ~4 ms of CPU while a retrieval that fires after the utterance
ends costs ~700 ms of user-visible silence. Measured on the 40-scenario dev
split:

| θ | early retrieval | false trigger | E[cost] |
|---|---|---|---|
| **0.40** | **0.88** | **0.00** | **20.35** |
| 0.48 | 0.84 | 0.00 | 28.49 |
| 0.56 | 0.72 | 0.00 | 48.84 |
| 0.68 | 0.53 | 0.00 | 81.39 |
| 0.92 | 0.14 | 0.00 | 150.58 |

An honest finding from that sweep: **θ trades early retrieval against wasted
shadow retrievals, not against false triggers.** The false-trigger rate is
0.00 at every θ, because false triggers are prevented upstream by the
suppression classifier — a different component. The two halves of gate G2 are
governed by two different mechanisms, which is not obvious until you sweep.

### 3.3 Suppression

Pitfall 4 in the guide is "ignoring presentation-only turns". Detection is a
169-example logistic regression over TF-IDF features, covering both
presentation turns ("give me that in two bullets") and social turns ("good
morning, my name is Priya, how can I help") — both require no retrieval, so
they share a label.

The classifier fits its **own** vectoriser rather than reusing the corpus
embedder. That was a measured mistake in an earlier version: a support corpus
contains none of these words, so a corpus-fitted embedding mapped every such
phrase to a near-zero vector and the classifier fired on everything.

Decision order puts `SUPPRESS` first and unconditional: a reformatting request
is maximally novel against the pool and would otherwise trigger. Thresholds
encode the asymmetry — suppressing an informational turn is a correctness
failure, while retrieving on a presentation turn is only a cost failure — so a
turn that asks for corpus content is never suppressed whatever the classifier
says.

---

## 4. Query decomposition strategy

Pitfall 5 is over-fragmentation. The usual response is a better splitter; that
is the wrong fix, because the problem is that the splitter runs at all on
simple utterances. So we **gate** it.

A **compoundness test** combines three orthogonal signals — coordination
markers followed by a request head, count of distinct question foci, and
topical variance between sliding-window embeddings of the utterance. Only
utterances that pass reach the extractor. Everything else retrieves once, with
zero LLM calls. Measured over-fragmentation on dev: **3.9%**.

Past the gate, extraction has two paths:

- **LLM path** (the submission path): one structured call returning a JSON list
  of self-contained sub-queries, run concurrently with the first retrieval.
- **Zero-token path** (keyless container, and the fallback whenever a call
  fails or is rate-limited): a syntactic splitter on coordination plus a
  TextTiling-style embedding segmenter that finds topic boundaries with no
  markers at all — which matters, because transcribed speech frequently runs
  two questions together with no "and" between them.

---

## 5. Evidence fusion

RRF pools every sub-query's results into one ranked list. That is correct when
sub-queries are paraphrases. It is wrong when they are *different questions*,
which is exactly the Theme 4 case: if the customer asked three things and the
budget is twelve chunks, RRF can return nine about the display and none about
turnaround. **Sub-intent starvation is invisible to recall@k** — each retrieval
looks excellent while a third of the answer silently disappears.

So we allocate rather than rank: a guaranteed floor per active intent, then
greedy marginal-gain fill, then MMR de-duplication across the selection. We
measure it with a metric we had to define, **per-intent grounded coverage**,
because no standard one sees the failure. Ablation A2: 0.772 with coverage
budgeting against 0.718 with plain RRF.

---

## 6. Session state and refinement

Gates G4 (zero fabricated citations) and G5 (refine without restarting) fight
each other as long as the answer is a string: the only way to fold a late
constraint into a paragraph is to rewrite it, and a rewritten paragraph's
citations drift.

So the answer is not a string. It is a set of `Claim` objects, each carrying
its intent, its **evidence IDs (which *are* the citation list)**, and the
assumptions it holds under. A late constraint becomes four deterministic steps,
only the last of which touches a model:

1. **Invalidate** — claims whose assumptions conflict are marked `SUPERSEDED`.
   Nothing is deleted; history is the audit trail and the version diff.
2. **Re-score** — pool evidence is re-ranked against the constraint. Chunks
   previously far down the ranking can surface *with no new retrieval at all*.
3. **Delta retrieve** — only for intents that lost a claim.
4. **Patch** — regenerate only the superseded claims.

Because unaffected claims are never regenerated, their text and citations
*cannot* drift. No model ever writes a citation string — they are derived from
evidence IDs, and any model-authored citation in prose is stripped — which
makes a fabricated document ID **unrepresentable** rather than merely unlikely.

This also gives gate G5 the numeric definition Samsung left out:

> **state continuity** = (logically-unaffected claims whose text **and**
> citation set are byte-identical across versions) / (unaffected claims)

Measured 1.000 over 6 refinement turns, and asserted in the test suite.

### Scope inference without domain rules

A constraint like "I bought it in Dubai" must be turned into a predicate. The
obvious implementation is a table of domain rules; that encodes our corpus into
application code and fails on the held-out private corpus the guide describes.

Instead **the corpus tells us what the constraint changes.** Claims inherit
scope tags from the evidence they rest on, taken from corpus front matter. A
constraint is retrieved like any query; whatever scope its highest-scoring
evidence declares is what it activates, and it invalidates conflicting values
of the same key. "Bought in Dubai" retrieves the cross-border policy, activates
`region:cross_border`, supersedes the domestic warranty claim, and leaves the
device diagnosis standing. Point the engine at a corpus about insurance riders
and the same code does the same job.

Two bugs in this mechanism are recorded in the source because both were
instructive: tagging unscoped documents `region:default` made every claim
region-scoped, so a purchase-region constraint superseded the entire answer
including the device diagnosis; and collecting tags from all probe evidence
activated `cross_border` and `domestic` together, which cancelled so that
nothing was ever invalidated. Scope must be *declared* by the corpus, and one
value per key wins, taken from the highest-scoring evidence.

---

## 7. Data provenance and corpus isolation

The corpus is the **Care Knowledge Pack**: 60 documents, 151 sections,
synthetic, written for this project. It is not Samsung material and every
figure in it is invented. Every section carries an explicit `Doc_ID §Section`
marker, so citations are exact by construction.

Sections are the retrieval unit. We do **not** re-chunk by token window: citing
"DOC_WAR_01 §2" is only meaningful if the retrieved span *is* §2, and
fixed-window chunking would make our citations approximate — failing G4 for a
reason that has nothing to do with the model.

Isolation is enforced mechanically, not by policy:

- The synthesis prompt receives only retrieved chunk text. No corpus content is
  inlined into prompt templates.
- The grounding verifier rejects any citation whose ID is absent from the pool,
  so parametric leakage becomes an emitted uncertainty rather than a confident
  sentence.
- No network call exists on the retrieval path. The only external call is the
  LLM provider, which is infrastructure, never a knowledge source.
- Session state is destroyed on disconnect. A test asserts that two sequential
  sessions with identical input produce identical state — cross-session
  non-profiling demonstrated rather than promised.

**The corpus is swappable by configuration.** `RIPPLE_CORPUS` plus an index
rebuild; the ingestion adapter reads markdown or JSONL and a third reader is
about twenty lines. This is deliberate: the guide says the benchmark replay is
held-out and private, which implies the judges may point the engine at content
we have never indexed.

---

## 8. Trade-offs we made, and what they cost

| Decision | Why | Measured cost |
|---|---|---|
| Exact FAISS flat index, not HNSW/IVF | A few hundred sections; approximation adds tuning surface and recall risk for no gain, and the controller queries the index on every chunk | none |
| No BM25 (removed) | Our embedder already reads character n-grams, so the "hybrid" was two correlated lexical signals competing | recall **improved** 0.893 → 0.960 |
| No reranker (removed) | Re-sorted a fused ranking better than its own scoring function | coverage **improved** 0.689 → 0.772 |
| Speculative synthesis ON | Produces negative TTFT: the answer starts before the sentence ends | coverage 0.797 → 0.772, a **2.5 point cost** for **1.04 s** of TTFT under the stub. With `gemini-3.5-flash-lite` (26 Sep 2026) the model's ~2 s reply time eats that head start except on multi-intent turns (median −3.17 s, 7 of 8 negative); overall median +1.80 s vs static RAG +2.07 s |
| Controller ON | Cuts retrievals 3.29 → 2.71 per turn and false triggers 1.00 → 0.00 | coverage 0.764 → 0.772 direction is favourable; the honest cost is complexity |
| TF-IDF+SVD embedder as default | No torch, no download, so the container starts on a clean machine — gate G1 | quality below a true semantic embedder; `bge-small` available behind a switch |

---

## 9. Failure modes and mitigations

| Failure mode | Mitigation | Status |
|---|---|---|
| Eager retrieval on noise (pitfall 1) | Stability + novelty gate; minimum prefix length | false-trigger rate 0.00 |
| Context loss on late constraints (pitfall 2) | Claim graph; unaffected claims never regenerated | state continuity 1.000 |
| Citation hallucination (pitfall 3) | Citations derived from evidence IDs; model-authored citations stripped; verifier checks existence and support | 0 fabricated of 159 |
| Ignoring presentation turns (pitfall 4) | Suppression classifier, decision-order first | 0 retrievals, 0 tokens on suppressed turns |
| Over-fragmenting sub-queries (pitfall 5) | Compoundness gate in front of the splitter | 3.9% over-fragmentation |
| Prefix lock-in on a distractor | Evidence pool with lifecycle states; provisional evidence demotable | analysed in the evaluation report |
| Corpus does not cover the question | Conservative retrieval gate + model instruction + verifier | precision 0.50, recall 0.50 on dev; see limitation below |
| Provider rate-limited or down | Client-side token bucket, exponential backoff, degradation to the zero-token path — never to an ungrounded guess | container runs with no key at all |

**Stated limitation.** The keyless `stub` provider cannot abstain on *near-miss*
holes — a question about reimbursing a screen protector, when the corpus has a
section about screen protectors that says nothing about reimbursement. That
judgement is semantic entailment. Two lexical heuristics were built, measured,
and found not to generalise (recorded in `ripple/retrieval/relevance.py`), so
abstention properly belongs to the model and the verifier, with the retrieval
gate as a cheap conservative first pass. With a real model
(`gemini-3.5-flash-lite`, dev split, 26 Sep 2026) Ripple's abstention precision
is 0.50 at recall 1.00, against 0.22 at recall 1.00 for static RAG.

---

## 10. What would change next

- **Embedder.** `bge-small` behind a switch, with ablation A4 re-run, since
  BM25 would likely earn its place again against a purely semantic embedder.
- **Cross-encoder reranking.** Available but unmeasured; no claim is made for
  it until ablation A5 is re-run with torch installed.
- **Learned controller.** The stability signal is currently unsupervised. With
  labelled retrieval-timing data a small classifier over (stability, novelty,
  prefix length, jump similarity) would likely beat a single threshold.
- **Corpus scale.** 151 chunks is enough to measure; production support
  knowledge bases are 10⁴–10⁵ sections, where the flat index decision would be
  revisited and HNSW would start to earn its tuning cost.
