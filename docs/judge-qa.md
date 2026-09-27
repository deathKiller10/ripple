# Jury Q&A preparation

Answers to the questions a Samsung R&D panel is most likely to ask, with the
measurement or the file that backs each one. Where we do not have an answer,
the answer is "we haven't measured that" — a jury forgives a gap far more
readily than a bluff, and one caught bluff costs every other claim its
credibility.

---

## The theme

**1. Why is this actually Streaming Live RAG and not a chatbot?**
Because the system is grounding a conversation it is not part of. The customer
addresses the agent, not us — there is no submit action, no turn boundary and
nobody to ask for clarification. Retrieval starts at 0.8 s into an utterance
that ends at 2.1 s. With a real model (`gemini-3.5-flash-lite`), when a
customer asks two things Ripple answers the first while they are still asking
the second: median first claim **3.17 s before the utterance ends** on
multi-intent turns, 7 of 8 negative. A turn-based system cannot produce a
negative number there, because it has not started. (On single questions a
~2 s model reply eats the head start; we report that too.)

**2. Why can't static RAG solve this?**
It is in the table (real model, `gemini-3.5-flash-lite`, dev split). B1 static
RAG: no early retrieval, no decomposition, recall 0.835, and every follow-up
regenerates the whole answer, so citations move each time. Ripple: early
retrieval 0.884 with zero false triggers, multi-intent 0.889, recall 0.942,
158 of 158 claims supported by their cited passage, and zero citation drift on
preserved claims — for 1.44× the tokens. Static RAG does keep slightly higher
per-intent coverage (0.854 vs 0.833) and a tighter latency tail; we say so.

**3. Why retrieve before the utterance ends at all — why not wait 400 ms?**
Because there is nothing to wait *for*. Endpoint detection assumes the speaker
is addressing you and will stop. Here the deadline is the customer's next
pause, which we cannot predict or request. And the loss is asymmetric: a wasted
shadow retrieval is ~4 ms of CPU and zero tokens; a late one is ~700 ms of
audible silence. We put that asymmetry into the threshold calibration rather
than into an opinion.

**4. What happens when the user's intent changes mid-utterance?**
Two mechanisms. A semantic jump against active intents opens a new branch
(`RETRIEVE_MORE`) rather than blending. And evidence fetched for an intent that
never materialises is demoted to `IRRELEVANT` in the pool and excluded from the
synthesis context — retrieval cancellation expressed where it actually matters.
Honest caveat: on a local index cancelling saves microseconds. Its value is
keeping stale evidence out of the context window, not saving compute, and we
say so rather than claiming a performance win.

---

## The controller

**5. How do you know when retrieval should happen?**
We stopped asking whether the sentence is complete and asked whether more words
would change which documents come back. Shadow retrieval on each fragment,
rank-biased overlap against the previous fragment, EMA. Retrieve when the result
set has stopped moving **and** it is novel against what the pool already holds.

**6. Why 0.40? Did you tune that on your demo?**
It is derived, and the curve is in `results/calibration.json`.
`E[cost] = P(false trigger)·c_waste + P(late)·c_late`, swept on the dev split
only. 0.40 minimises it at 0.88 early retrieval and 0.00 false triggers. A
finding from the sweep worth volunteering: **θ does not trade against false
triggers at all** — the false-trigger column is flat at every θ, because those
are prevented upstream by the suppression classifier. The two halves of gate G2
are governed by different components.

**7. How do you prevent retrieval on every token?**
The novelty term. High stability alone is not a reason to search — once the
pool already contains what the shadow retrieval found, further retrieves are
suppressed. Measured: 2.75 retrievals per turn against naive streaming's 4.04.

**8. How do you control cost?**
Arithmetic first: three fragments a second over a twenty-second utterance is
sixty controller decisions. One model call per decision is sixty per turn. So
**the controller contains no model call** — one embedding and one exact lookup,
about 4 ms. Model calls occur twice: once per *compound* utterance for
sub-query extraction, once per answer version. A single-intent turn costs one.

**9. Why not just send the whole transcript to a large model?**
Three reasons. You cannot send a transcript that has not finished, so TTFT
stays positive. You get no provenance, and gate G4 requires every assertion
traced to a corpus chunk. And the corpus-isolation rule forbids answering from
parametric memory, which is exactly what that design would do.

---

## Decomposition and fusion

**10. How does multi-intent decomposition work, and how do you avoid
over-splitting?**
A compoundness gate runs *before* the splitter — coordination markers, question
foci, and topical variance between sliding-window embeddings. Single-intent
utterances never reach the extractor, so they cost zero model calls. Measured:
multi-intent 0.889, over-fragmentation 0.039.

**11. Why not just use RRF?**
We did, and measured it. Ablation A2: **recall@k is identical at 0.960** with
plain RRF — every gold chunk was retrieved either way — but per-intent coverage
falls 0.772 → 0.718. RRF pools, so one loud sub-question takes the whole budget
and another gets none of it. No standard retrieval metric can see that, which
is why we had to define per-intent grounded coverage to report it.

---

## Grounding and refinement

**12. How do you handle late-arriving details without restarting?**
Invalidate, re-score, delta-retrieve, patch. Only the last step touches a model.
Claims whose assumptions conflict are marked superseded; pool evidence is
re-ranked against the constraint, so chunks already retrieved can surface with
no new search; only intents that *lost* a claim are re-queried.

**13. How do citations stay correct after refinement?**
Structurally, not by prompting. Unaffected claims are never regenerated, so
their text and citations cannot move — measured state continuity **1.000** over
six refinement turns. And no model ever writes a citation string: citations are
derived from evidence IDs, and any the model writes into prose are stripped.
A fabricated document ID is **unrepresentable**, not merely unlikely. Measured:
**0 fabricated of 159 citations**.

**14. What happens when retrieved documents conflict?**
The corpus declares supersession in front matter, and the pool marks the older
chunk `SUPERSEDED` with the newer one recorded. Two dated contradiction pairs
exist in the corpus specifically to exercise this. The answer surfaces the
current figure and may cite both — we do not silently pick.

**15. How is session memory implemented?**
In-process objects, destroyed on disconnect. No Redis, no database — the
session-bound rule makes persistence a spec violation, not a missing feature.
`test_sessions_do_not_leak_into_each_other` asserts two identical sessions
produce identical state, which they could not if one influenced the other.

---

## Rigour

**16. What is your actual technical contribution? Which part is novel?**
Three things, in order of how much I would defend them:
(a) deciding *when* to retrieve in **retrieval space** rather than language
space, which makes the controller free;
(b) representing the answer as a claim graph, which is what makes G4 and G5
stop fighting each other;
(c) treating fusion across sub-queries as **budget allocation** rather than
ranking.
Plus one small metric contribution: signed TTFT relative to end-of-utterance.

**17. What did you measure, and can you reproduce it?**
`python -m evaluation.run_bench --split dev --ablations` and
`python -m evaluation.calibrate`. Four systems, six gates, five ablations, 11
property tests. Replay uses a virtual clock from transcript timestamps, so the
timings are machine-independent — a slow laptop produces the same numbers.

**18. Did you overfit to your own test set?**
The dev/held-out split was declared in the scenario builder before any tuning,
all calibration happened on dev, and held-out was run exactly once after
feature freeze: all six gates pass on 34 unseen scenarios, one to two points
below dev (recall 0.937 vs 0.960, early retrieval 0.868 vs 0.884). The one
thing that did not generalise is keyless abstention — 0 of 2 near-miss holes —
and the report says so.
A build-time check fails if any gold citation does not exist in the corpus.

**19. What are the limitations?**
Four, and they are in the report. Grounding of 1.000 is an artefact of the
keyless provider, which copies the sentence it cites — the real number needs a
model run. The keyless path cannot abstain on near-miss holes. Selection, not
retrieval, is our bottleneck: 0.960 recall against 0.772 coverage. And the
corpus is 151 chunks, which is enough to measure and not enough to claim scale.

**20. Anything you tried that didn't work?**
Two abstention heuristics. Distribution shape failed instructively — the
deliberate coverage hole produced the *highest* standout score of any query
(z = 12.8), because a section titled "Screen protection accessories" matches
the surface form perfectly while answering nothing. Corpus-vocabulary overlap
worked at 21 documents and **broke at 60**, which is disqualifying for a
held-out corpus. Both are documented in `relevance.py`. We also deleted BM25
and our own reranker after measuring both making things worse.

---

## Product and scale

**21. Would a real user want this?**
The agent is the user, and the metric they are judged on is handle time.
Grounding arrives before the customer stops, with every claim one click from
the clause it rests on.

**22. Why would Samsung care?**
Samsung runs support centres. Beyond that, the engine is corpus-agnostic and
contains no domain rules — scope comes from corpus metadata — so field service,
insurance adjudication and regulatory Q&A run on it unchanged.

**23. How does it scale?**
Retrieval is the only part that changes: at 10⁴–10⁵ sections the flat index
would become HNSW, and that is a tuning cost we would then be paying for a
reason. The controller, the claim graph and fusion are all independent of corpus
size. Sessions are independent and in-process, so horizontal scaling is trivial.

**24. What stops another team building this?**
Nothing in principle — but the claim graph is a one-way door. Build refinement
as string rewriting and converting to claim-level state is a rewrite of
synthesis, state and the UI. And the parts that are hard to retrofit are the
discipline, not the code: a declared held-out split, a derived threshold with a
curve, and published negative results.

**25. What happens when the stream is noisy, or retrieval is simply wrong?**
Noise: stability is an EMA over rank overlap, so a single corrupted fragment
moves it slightly rather than tripping a retrieval. We have **not** run an
injected-ASR-noise experiment, and I will not claim robustness we have not
measured — it is the next experiment after the cross-encoder.
Wrong retrieval: the grounding verifier rejects claims their cited chunk does
not support, and the answer degrades to an explicit uncertainty rather than a
confident wrong answer. That is the designed failure direction.

---

## If you are stuck

- **You don't know:** "We haven't measured that." Then say what you would
  measure and how long it would take. That answer scores.
- **They found a real flaw:** agree, say what it would take to fix, and say
  where it sits in the ordering on the What's Next slide. Arguing with a
  correct criticism costs more than the flaw does.
- **They push on the stub provider:** do not defend the 1.000. Volunteer that
  citation validity is 1.000 by construction under any model, explain why the
  keyless path exists (gate G1 on their machine), and give the real number:
  with `gemini-3.5-flash-lite` on the full dev split, 158 of 158 model-written
  claims passed the grounding verifier (automated check, not human review).
- **They push on latency:** the stub's −1.04 s assumes an instant model. With
  a real model the median is +1.80 s (static RAG +2.07 s); it goes negative on
  multi-intent turns (−3.17 s). The p90 tail is worse than static RAG's
  (10.1 s vs 2.5 s) because Ripple makes 2.5× the calls on a free-tier API.
