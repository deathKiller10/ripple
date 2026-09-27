# Ripple — Benchmarking & Evaluation Report

**Samsung PRISM GenAI Hackathon, 3rd Edition · Theme 04 Streaming Live RAG**
Repository: https://github.com/deathKiller10/ripple

Deliverable for §8 of the Theme 4 Guide: quantitative comparison against a
baseline pipeline, **at least three analysed edge-case failures**, and **at
least two architectural ablations**. Five ablations are reported.

Reproduce everything here with:

```bash
python -m evaluation.run_bench --split dev --ablations
python -m evaluation.calibrate
python tests/test_gates.py
```

---

## 0. Read this before the numbers

**Provider.** Sections 2–5 come from the **`stub`** provider — the keyless
extractive path that exists so the container runs on a clean machine with no
API key (gate G1). **Section 2A reports a real-model run** (`gemini-3.5-flash-lite`,
26 Sep 2026) on the same dev split. Under the stub:

- `citation_support = 1.000` is **not a groundedness measure, under any
  model**. It is 1 − fabricated ÷ total citations, and Ripple derives citations
  from evidence IDs, so a fabricated one cannot be represented. It shows the
  claim graph works; it says nothing about whether the text is faithful. The
  groundedness figure that a model *can* move is **claim support** (§2A),
  read from the grounding verifier's verdicts in the trace.
- `llm_calls` and `cost_per_turn` are **zero**, so the cost comparison is made
  on `retrievals_per_turn`, which is real and provider-independent.
- **TTFT under the stub assumes a model that answers instantly.** It measures
  how early Ripple *starts* an answer — an upper bound on the head start, not
  a user-facing latency. §2A shows what a real model's reply time does to it.
- Abstention on *near-miss* holes is a stated limitation of this path
  (§4.3).

**Splits.** `dev` = 40 scenarios, 51 labelled turns. All threshold calibration
happened here and nowhere else. `heldout` = 34 scenarios, 43 labelled turns,
**run exactly once, after feature freeze (§2B).** The
split was declared in `data/scenarios/build_scenarios.py` before any tuning.

**Model naming.** Any figure from a real-provider run must be reported beside
the model that produced it (`report["model"]` in the benchmark JSON). A
groundedness number without a model name is not reproducible.

**Cost.** Tokens per turn are always measured. A *currency* figure is reported
only when `RIPPLE_PRICE_IN` / `RIPPLE_PRICE_OUT` are set from the provider's
current pricing page — the defaults are zero, because a rupee figure derived
from a price we guessed would be a fabricated number wearing a decimal point.

**Nothing here is estimated.** Every figure is produced by the commands above.
Where something has not been measured, it says so.

---

## 1. Systems compared

All four share the same index, corpus, embedder and provider. A comparison that
moves two variables at once measures nothing.

| | Description |
|---|---|
| **B0** LLM only | No retrieval. The grounding floor, and what the parametric model would say — which under the corpus-isolation rule must never reach the user. |
| **B1** Static RAG | One query at utterance end; full regeneration on every follow-up. What most submissions will be, and a genuinely strong grounding baseline. |
| **B2** Naive streaming | Retrieve on **every** transcript chunk, regenerate each time. The behaviour the guide names as pitfall 1, implemented deliberately so its cost can be priced rather than asserted. |
| **B3** Ripple | The full engine. |

## 2. Results — dev split, 40 scenarios (stub provider)

| system | early retr | false trig | multi-intent | recall@k | intent cov | cite supp | fabricated | continuity | TTFT median | retr/turn |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 LLM only | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 1.000 | 0.000 | 0.00 |
| B1 static RAG | 0.000 | 1.000 | 0.000 | 0.835 | 0.768 | 1.000 | 0 | 1.000 | +0.010 | 1.00 |
| B2 naive streaming | 1.000 | **1.000** | 0.000 | 0.835 | 0.768 | 1.000 | 0 | 1.000 | **−2.069** | **4.04** |
| **B3 Ripple** | **0.884** | **0.000** | **0.889** | **0.960** | **0.772** | 1.000 | **0** | **1.000** | **−1.036** | 2.75 |

### Acceptance gates (Theme 4 Guide §5)

| Gate | Criterion | Target | Measured | |
|---|---|---|---|---|
| G1 | Reproducibility | pass/fail | `docker compose up`, no key, no GPU; replay suite completes headless | **PASS** |
| G2 | Early retrieval | ≥ 80% | **0.884** (38 of 43 eligible turns) | **PASS** |
| G2′ | False triggers | low | **0.000** (0 of 8 no-retrieval turns) | **PASS** |
| G3 | Multi-intent identification | ≥ 70% | **0.889** (8 of 9 compound turns) | **PASS** |
| G4 | Factual grounding | ≥ 85% | Stub: citation validity only (*see §0*). Real model: **158 of 158 claims supported** (§2A) | **PASS** |
| G4 | Fabricated document IDs | 0 | **0 of 159 citations** | **PASS** |
| G5 | Session refinement | state continuity | **1.000** over 6 refinement turns | **PASS** |
| G6 | Telemetry | 100% trace coverage | **1.000**, asserted by test | **PASS** |

### The two readings that matter

**B2 shows what naive streaming actually costs.** It achieves a better TTFT
than Ripple (−2.07 s against −1.04 s) — it answers from the first fragment,
because it never waits for anything. It pays with a **100% false-trigger rate**
(it retrieves on every greeting and every "say that again") and **49% more
retrievals per turn**. With a real provider, where each of those retrievals
carries a regeneration, that multiplier lands directly on cost per turn.

**Ripple's recall (0.960) is the highest of any system, and that is
structural.** B1 and B2 issue a single query at each point and keep nothing;
Ripple accumulates evidence across the whole utterance in a session pool, so by
the time the utterance ends it holds gold chunks that a single end-of-utterance
query never saw.

---

## 2A. Real-model results — `gemini-3.5-flash-lite`, 26 Sep 2026

Dev split, **all 40 scenarios** (not a slice), B1 and B3. Google Gemini API,
free tier, client-side limit `RIPPLE_RPM=8` requests per minute (this sets
how long a run takes, not its results: queueing is excluded from TTFT). Run `2026-09-26T16:36:42Z`, code at commit `8e25364`. Zero failed
calls (B1 51 of 51, B3 129 of 129). Files: `results_gemini/run2_2026-09-26/`.

| | B1 static RAG | **B3 Ripple** |
|---|---|---|
| Claim support (grounding verifier) | not measured — B1 has no verifier | **158 of 158 (1.000)** |
| Median share of a claim's words found in its cited passage | — | 0.917 |
| Fabricated citations | 0 of 85 | 0 of 97 |
| Early retrieval / false triggers | 0.000 / 1.000 | **0.884 / 0.000** |
| Multi-intent identification | 0.000 | **0.889** |
| Recall@k | 0.835 | **0.942** |
| Per-intent coverage | **0.854** | 0.833 |
| LLM calls per turn | **1.00** | 2.53 |
| Tokens per turn | **1,097** | 1,576 (1.44×) |
| TTFT median / p90 | +2.070 s / +2.466 s | **+1.795 s** / +10.052 s |
| Turns answered before the utterance ended | 0% | 24% (8 of 33) |

All six gates PASS on this run. Currency cost is **not reported**: no price is
configured, and we do not use a guessed one (§0). Tokens per turn are measured.

**Groundedness.** Every claim the model wrote passed the verifier, which
checks that a claim's content appears in the passage it cites (word overlap
≥ 0.18, or embedding similarity ≥ 0.42) and drops it if not. The model
paraphrases rather than copies — median overlap 0.917, against 1.000 for the
stub — and still stayed inside its evidence. This is an automated proxy, not a
human judgement: a claim that reuses a passage's words but states a wrong
figure could pass. Reproduce with
`python -m evaluation.claim_support results_gemini/run2_2026-09-26/traces_dev`.

**Cost.** Ripple spends 1.44× the tokens and 2.5× the model calls of static
RAG per turn. That buys early retrieval, zero false triggers, multi-intent
decomposition and higher recall. It does **not** buy higher per-intent coverage
with this model: B1 0.854 against B3 0.833 (0.846 each in run 1). Under the
stub Ripple led by 0.004; with a real model that lead is gone, and we report
it as gone.

**Time to first token.** The stub's −1.036 s assumed an instant model. Gemini
took a median 2.08 s per call (20 speculative drafts; worst 25.5 s), which
consumes most of the head start. By scenario type (B3):

| type | turns | TTFT median | before utterance end |
|---|---|---|---|
| multi-intent | 8 | **−3.17 s** | 7 of 8 |
| distractor | 3 | +1.56 s | 0 of 3 |
| late constraint | 6 | +1.82 s | 0 of 6 |
| suppress | 5 | +1.89 s | 0 of 5 |
| single question | 9 | +4.09 s | 1 of 9 |
| conflict | 2 | +4.64 s | 0 of 2 |

Negative TTFT survives where the design says it should: when a customer asks
two things, Ripple answers the first while they are still asking the second.
On a single question it does not, and single-question turns are slower than
static RAG's overall median. The tail is worse too — p90 10.05 s against
2.47 s. The slowest turns trace to individual slow API responses (one call took
25.5 s), and Ripple makes 2.5× as many calls, so it meets more of them. We have
not separated slow responses from timeouts and retries.

How TTFT is measured here: trigger time on the transcript clock, plus our
processing and the model's reply time. Time spent queueing for the free-tier
rate limit (B1 229 s, B3 598 s over the run) is recorded and **excluded**,
because it measures our API plan, not the system. The call is non-streaming,
so this is time to the first *complete claim*, not the first streamed token.

**Run-to-run variation.** An earlier full run the same day (run 1, kept as
evidence) gave recall 0.954 and coverage 0.846 for B3, and 198 of 198 claims
supported. Differences of about 0.01 between runs are model non-determinism
at temperature 0.1. Run 1's TTFT is invalid (it included queueing).

**Three measurement bugs this run exposed, all fixed before the figures above
were taken.** Each was invisible under the stub, because a stub call is free
and instant:

1. A speculative draft that produced no usable claim was paid for but never
   counted in cost (commit `49e2242`).
2. Streaming systems stamped their first token when the model was *asked*,
   not when it *answered*; B1's stopwatch included the reply time, so the
   comparison flattered Ripple by the model's latency (`5c5635e`).
3. The model call then included time waiting in our own rate limiter — B1's
   7.45 s TTFT in run 1 was 60 s ÷ 8 requests per minute (`8e25364`).

---

## 2B. Held-out split — run once, after feature freeze

34 scenarios and 43 labelled turns that were never used for tuning or
inspected before this run. Keyless stub provider, all four systems, code at
commit `4a2ba02`, run `2026-09-27T16:01:30Z`. Files: `results_heldout/`.

| system | early retr | false trig | multi-intent | recall@k | intent cov | fabricated | continuity | TTFT median | retr/turn |
|---|---|---|---|---|---|---|---|---|---|
| B0 LLM only | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 1.000 | 0.000 | 0.00 |
| B1 static RAG | 0.000 | 1.000 | 0.000 | 0.867 | 0.704 | 0 | 1.000 | +0.006 | 1.00 |
| B2 naive streaming | 1.000 | 1.000 | 0.000 | 0.825 | 0.676 | 0 | 1.000 | −2.069 | 3.63 |
| **B3 Ripple** | **0.868** | **0.000** | **0.875** | **0.937** | **0.778** | **0** | **1.000** | **−1.036** | 2.49 |

**All six gates PASS on unseen data:** early retrieval 0.868 (33 of 38
eligible turns), false triggers 0 of 5, multi-intent 0.875 (7 of 8 compound
turns), 0 fabricated IDs of 135 citations, state continuity 1.000 over 6
refinement turns, trace coverage 1.000.

| B3 Ripple | dev (§2) | held-out | change |
|---|---|---|---|
| Early retrieval | 0.884 | 0.868 | −0.016 |
| False triggers | 0.000 | 0.000 | — |
| Multi-intent | 0.889 | 0.875 | −0.014 |
| Recall@k | 0.960 | 0.937 | −0.023 |
| Per-intent coverage | 0.772 | 0.778 | +0.006 |

The drop from dev to held-out is one to two points on every metric, which is
what calibrating on dev and not touching held-out should produce. Ripple keeps
its lead over both baselines on recall (0.937 vs 0.867 static, 0.825 naive)
and coverage (0.778 vs 0.704 and 0.676), and naive streaming still spends
46% more retrievals per turn (3.63 vs 2.49).

**Abstention did not generalise: 0 of 2.** Both held-out questions the corpus
cannot answer — a trade-in valuation and paying for a repair in instalments —
are *near-miss* holes: the corpus has related repair-payment and residual-value
text. Ripple answered both with correctly cited but non-responsive sentences
("Payment is collected on collection of the device") instead of declining.
Nothing was fabricated; the answers simply did not answer. This is the
keyless path's stated limitation (§4.3), now confirmed on unseen data. On dev
the two holes were clear misses (e.g. insurance), which the retrieval gate
catches; the dev abstention figure therefore overstated what the stub can do.

**Stub TTFT is identical on both splits** (−1.036 s, −2.069 s): replay chunks
arrive on a fixed synthetic cadence and the stub answers instantly, so the
median lands on the same chunk boundary. It measures where speculation fires,
not a latency (§0, §2A).

---

## 3. Ablations

Each variant changes **exactly one** config value on the real engine. They are
not separate baselines, so nothing else moves.

| Variant | TTFT med | early | recall@k | intent cov | retr/turn |
|---|---|---|---|---|---|
| **B3 full system** | **−1.036** | 0.884 | **0.960** | **0.772** | 2.75 |
| A1 controller OFF (retrieve every chunk) | −1.036 | 1.000 | 0.960 | 0.764 | **3.29** |
| A2 fusion = plain RRF | −1.036 | 0.884 | 0.960 | **0.718** | 2.75 |
| A3 speculative synthesis OFF | **0.000** | 0.884 | 0.960 | 0.797 | 2.75 |
| A4 add BM25 back (hybrid) | −1.036 | 0.884 | **0.893** | 0.720 | 2.75 |
| A5 reranker back ON | −1.036 | 0.884 | 0.960 | 0.726 | 2.75 |

**A1 — the controller.** Disabling it raises early retrieval to a trivial 1.000
(everything retrieves) and costs **20% more retrievals per turn**. Its real
value is not visible in this table: compare B2's false-trigger rate of 1.000
against B3's 0.000. The controller is what stops the system searching the
corpus when someone says good morning.

**A2 — coverage-budgeted fusion.** Per-intent grounded coverage falls from
**0.772 to 0.718** under plain RRF. This is sub-intent starvation, measured:
recall@k is *identical* at 0.960 in both configurations, so every gold chunk
was retrieved either way — RRF simply allocated the context budget so that some
sub-questions received none of it. No standard retrieval metric can see this,
which is why we had to define per-intent grounded coverage to report it.

**A3 — speculative synthesis.** The whole negative-TTFT effect: **−1.036 s with
it, 0.000 s without** — under the stub, whose model answers instantly. With a
real model the reply time is added to every draft, and the head start survives
at the median only on multi-intent turns (§2A). It costs 2.5 points of intent coverage (0.797 → 0.772),
because a draft written before the utterance ends is occasionally superseded.
That is a real trade and we report both sides. 74% of turns achieve a negative
TTFT.

**A4 — BM25, and why we removed it.** Adding the sparse half *lowers* recall
from **0.960 to 0.893**. Our default embedder is TF-IDF+SVD over word **and
character** n-grams, so it already matches exact identifiers; a 16-query probe
of part numbers and diagnostic codes (`GH82-S24U-DA1`, `DSP-114`, `BAT-207`…)
scores **16/16 hit@1 with and without BM25**. The "hybrid" was two correlated
lexical signals competing for the same slots, with the noisier one displacing
good hits. BM25 is off by default and available behind a switch — it would
likely earn its place against a purely semantic embedder such as `bge-small`,
and ablation A4 should be re-run if the embedder changes.

**A5 — the reranker, and why we removed that too.** Our lexical-semantic
reranker *lowers* intent coverage from **0.772 to 0.726**, changes neither
recall nor TTFT, and costs about 90 ms per sub-query. It was re-sorting a fused
ranking that was already better than its own scoring function. Removing a
component we built is the parsimony rule applied to ourselves. A true
cross-encoder is a different quality tier and remains available, but it has
**not** been measured, so no claim is made for it.

---

## 4. Analysed edge-case failures

### 4.1 The dominant failure mode: selection, not retrieval

The single most useful number in this report is the gap between two metrics:

```
recall@k              0.960     gold chunk reaches the evidence pool
per-intent coverage   0.772     gold chunk reaches the ANSWER
```

**Retrieval is not our bottleneck.** In 19% of sub-intents the right passage
was found and then not surfaced. Concrete instances from the dev run:

| Scenario | Sub-intent | Wanted | Answer cited instead |
|---|---|---|---|
| `dev_multi_01` | warranty coverage | `DOC_WAR_01 §3` | `DOC_KB_01 §1`, `DOC_KB_03 §1/§3` |
| `dev_multi_02` | battery part cost | `DOC_PRT_03 §2` | `DOC_KB_11 §1/§2/§3` |
| `dev_multi_08` | overheating while charging | `DOC_KB_11 §2/§3` | `DOC_KB_11 §1`, `DOC_KB_13 §1–3` |
| `dev_single_10` | technical confirmation report | `DOC_SVC_07 §1` | `DOC_SVC_07 §2` |

The pattern is consistent: the **symptom** documents crowd out the **policy**
documents. A sub-query like "is any of this covered" is lexically closer to the
KB article describing the fault than to the warranty clause answering the
question, and the last case shows it at section granularity — right document,
adjacent section.

Two honest observations. First, this is a *selection* problem, which is where
the reranker was supposed to help and measurably did not (A5) — our
lexical-semantic scorer has the same lexical bias as the retriever it was
re-sorting, so it reinforced the error rather than correcting it. A
cross-encoder, which scores query-document *relevance* rather than term
overlap, is the natural fix and is the highest-value next experiment. Second,
this is the failure mode most likely to improve with a real provider: the model
sees twelve chunks and chooses which to cite, and it is better at "this clause
answers the question, that symptom description does not" than any lexical
score.

### 4.2 Compound questions stated without an interrogative

`dev_multi_09`: *"Bluetooth keeps cutting out on his earbuds and wifi drops at
home as well."* Two questions. The compoundness gate originally scored it
single-intent because its coordination test required a request head after the
marker, and "…and wifi drops at home" states a second question as a **symptom**
rather than a question. The whole utterance became one intent, retrieval was
run on the blend, and — worse — the abstention gate then declared it uncovered,
because no single passage covers both Bluetooth and Wi-Fi.

Two fixes followed, both recorded in the source. A coordinated clause carrying
its own content words now counts at half weight even without an interrogative;
and **the abstention gate never judges an undecomposed compound question**,
because "no single passage covers all of this" is precisely the situation
decomposition exists to resolve. G3 moved 0.778 → **0.889**, over-fragmentation
unchanged at 0.039.

The residual failure, `dev_multi_07` — *"He bought it second hand and wants to
know if the warranty still applies and whether the protection plan came with
it"* — is a genuine miss: two questions about the same document family
(`DOC_WAR_09 §1` and `§2`), with low topical variance because they are about
the same topic. Distinguishing them needs semantics, not segmentation.

### 4.3 Abstention: two heuristics measured and discarded

The corpus contains deliberate holes. Getting abstention right took three
attempts and the first two failed in instructive ways.

**Attempt 1 — distribution shape.** Hypothesis: a covered query produces a
sharp peak in its ranking, an uncovered one a flat ranking. Measured, the
*reverse* held. The screen-protector-reimbursement hole produced the **highest**
standout score of any query tested (z = 12.8), because the corpus contains a
section titled "Screen protection accessories" that matches the query's surface
form almost perfectly while answering nothing. Peakedness measures how
distinctive the best match is, not whether it answers anything.

**Attempt 2 — corpus vocabulary.** Count the question's content words that
appear nowhere in the corpus. This separated cleanly at 21 documents and was
adopted. It then **broke when the corpus grew to 60 documents**: vocabulary
went from 658 to 1100 words, out-of-vocabulary rates fell across the board, and
holes began to look covered. A threshold fitted to one corpus size does not
survive another — disqualifying, given the guide's held-out private benchmark.

**What we do now.** A deliberately conservative retrieval-side pre-filter — does
any *single* passage address this question — with the real judgement left to
the model and the grounding verifier. On dev the pre-filter achieves
**precision 0.50, recall 0.50** on two uncoverable turns (small numbers; stated
as such). Its design target is precision: a wrong abstention destroys an answer
we could have given, while a missed one is caught downstream.

**Stated limitation.** The keyless stub cannot abstain on near-miss holes. A
test (`test_near_miss_holes_are_a_known_stub_limitation`) asserts the limitation
is exactly where we claim it is, and will fail if that ever changes — so the
caveat cannot silently become false.

### 4.4 Four bugs that only a larger test set revealed

The corpus and scenarios were tripled mid-project (21 → 60 documents, 16 → 51
labelled dev turns). Four real bugs surfaced immediately, each found by a number
disagreeing with the mechanism rather than by a crash.

1. **The controller skipped its own observation on short prefixes.** A minimum
   word-count guard returned `WAIT` *before* running the shadow retrieval, so
   the stability series started a chunk late. Every eligible turn that failed to
   retrieve early was two or three chunks long — the controller was still
   warming up when the speaker finished. Early retrieval **0.74 → 0.86**.
2. **The EMA was seeded at zero**, imposing a further warm-up lag of two to
   three chunks. Seeding with the first real observation removed it.
3. **Suppressed turns reported zero retrievals** even when one had already
   fired mid-utterance, hiding a false trigger from our own G2 figure — the
   exact half of that gate teams are tempted to omit.
4. **Speculative synthesis bypassed the abstention gate**, so a draft claim
   could cite a source for a question the corpus does not answer. A gate that
   guards only the slow path is not a gate.

### 4.5 A harness bug that made our own system look worse

Early runs reported Ripple's recall at roughly half the static baseline's
(0.420 vs 0.819). The engine was fine: the harness populated `retrieved_cites`
for B1 but not for B3, so the metric fell back to the citation list — twelve
retrieved chunks against three cited claims. Recall must be computed over what
a system *retrieved*, not what it chose to cite.

Worth recording as method: **a result that flatters or damns your system for a
reason you cannot explain mechanically is a harness bug until proven
otherwise.** Both directions of that rule fired during this project.

---

## 5. Threshold calibration

θ is derived rather than chosen. `evaluation/calibrate.py` sweeps it against
`E[cost] = P(false trigger)·c_waste + P(late)·c_late`, with c_waste = 1 and
c_late = 175 from the ~1000× asymmetry between a wasted 4 ms lookup and 700 ms
of user-visible silence.

| θ | early retrieval | false trigger | E[cost] |
|---|---|---|---|
| **0.40** | **0.88** | 0.00 | **20.35** |
| 0.48 | 0.84 | 0.00 | 28.49 |
| 0.56 | 0.72 | 0.00 | 48.84 |
| 0.62 | 0.60 | 0.00 | 69.19 |
| 0.68 | 0.53 | 0.00 | 81.39 |
| 0.80 | 0.40 | 0.00 | 105.81 |
| 0.92 | 0.14 | 0.00 | 150.58 |

The curve is monotone and the false-trigger column is flat, which is itself the
finding: **θ trades early retrieval against wasted shadow retrievals, not
against false triggers.** Those are prevented upstream by the suppression
classifier. The two halves of gate G2 are governed by different components — not
obvious until the sweep is run, and a reason to sweep rather than tune.

---

## 6. What has not been measured

Stated explicitly so no reader infers more than we tested.

- **The held-out split was run once, keyless only** (§2B). It has not been
  run with a real model.
- **One real model, one day.** §2A is `gemini-3.5-flash-lite` on the dev
  split, B1 and B3 only. B0, B2 and the ablations were not re-run with a real
  model, and no second model was tried.
- **Currency cost is not reported.** Tokens per turn are measured; turning
  them into money needs a published price, and none is configured.
- **Groundedness is an automated proxy.** Claim support uses word overlap and
  embedding similarity, not human judgement, and exists for B3 only.
- **TTFT is time to the first complete claim** (non-streaming calls), on a
  free-tier API whose response times vary; the tail in §2A is partly that.
- **`bge-small` embeddings are unmeasured.** Available behind a switch. Ablation
  A4 (BM25) would need re-running with it.
- **Cross-encoder reranking is unmeasured.** Available; no claim made.
- **Latency figures are wall-clock on one developer machine**, not a controlled
  benchmark. The replay harness uses a virtual clock taken from transcript
  timestamps, so early-retrieval figures and *stub* TTFT are
  machine-independent (real-model TTFT adds measured reply time, §2A), but
  the component timings quoted in the architecture brief (~4 ms controller,
  ~8 ms retrieval, ~90 ms reranker) are indicative only.
- **Corpus scale.** 151 chunks is enough to measure, but production support
  knowledge bases are 10⁴–10⁵ sections. The flat-index decision would be
  revisited there.
