"""Tests that defend the claims we make to the jury.

These are not coverage tests. Each one asserts a property the evaluation report
states as a fact, so that the report cannot quietly drift away from the code.
Run: python -m pytest tests/ -q      (or: python tests/test_gates.py)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ripple.config import Config                        # noqa: E402
from ripple.controller.stability import jaccard, rbo    # noqa: E402
from ripple.engine import RippleEngine                  # noqa: E402
from ripple.schemas import Decision, EventType          # noqa: E402

CFG = Config()


def _engine():
    if not os.path.exists(os.path.join(CFG.index_path, "chunks.jsonl")):
        from ripple.retrieval.index import build_and_save
        build_and_save(CFG.corpus_path, CFG.index_path, CFG.embedder)
    return RippleEngine(CFG)


# ---------------------------------------------------------------------------
# M1 -- stability primitives
# ---------------------------------------------------------------------------


def test_rbo_bounds_and_top_weighting():
    a = ["x1", "x2", "x3", "x4", "x5"]
    assert rbo(a, a) == 1.0
    assert rbo(a, ["y1", "y2", "y3", "y4", "y5"]) == 0.0

    # A swap at rank 1 must cost more than a swap at rank 5. This top-weighting
    # is the entire reason RBO was chosen over Jaccard, which cannot tell the
    # two cases apart.
    swap_top = ["z", "x2", "x3", "x4", "x5"]
    swap_tail = ["x1", "x2", "x3", "x4", "z"]
    assert rbo(a, swap_top) < rbo(a, swap_tail)
    assert jaccard(a, swap_top) == jaccard(a, swap_tail)


# ---------------------------------------------------------------------------
# G6 -- telemetry coverage is measured, not asserted
# ---------------------------------------------------------------------------


REQUIRED = {
    EventType.SESSION_STARTED, EventType.CHUNK_RECEIVED,
    EventType.CONTROLLER_DECISION, EventType.COMPOUNDNESS_TESTED,
    EventType.DECOMPOSED, EventType.RETRIEVAL_STARTED,
    EventType.RETRIEVAL_COMPLETED, EventType.FUSION_COMPLETED,
    EventType.POOL_UPDATED, EventType.CLAIM_EMITTED,
    EventType.GROUNDING_CHECKED, EventType.ANSWER_VERSION,
    EventType.UTTERANCE_END,
}


def test_trace_coverage_is_total():
    e = _engine()
    s = e.session(session_id="t_cov", sink_dir=None)
    for t, txt in [(0.0, "My Galaxy S24 screen"), (0.5, "keeps flickering after"),
                   (1.0, "the update and I"), (1.5, "want to know if"),
                   (2.0, "the repair is covered")]:
        s.on_chunk(t, txt)
    s.end_utterance(2.6)
    seen = {ev.type for ev in s.bus.events()}
    missing = {r.value for r in REQUIRED} - seen
    assert not missing, f"stages emitting no telemetry: {sorted(missing)}"
    s.close()


# ---------------------------------------------------------------------------
# G2 -- retrieval must begin before the utterance ends
# ---------------------------------------------------------------------------


def test_retrieval_fires_before_utterance_end():
    e = _engine()
    s = e.session(session_id="t_early", sink_dir=None)
    fired_at = None
    chunks = [(0.0, "I need to know"), (0.5, "whether a display repair"),
              (1.0, "is covered under warranty"), (1.5, "on a ten month"),
              (2.0, "old handset please")]
    for t, txt in chunks:
        out = s.on_chunk(t, txt)
        if out.decision in (Decision.RETRIEVE, Decision.RETRIEVE_MORE) \
                and fired_at is None:
            fired_at = t
    assert fired_at is not None, "controller never triggered retrieval"
    assert fired_at < chunks[-1][0], (
        f"retrieval fired at {fired_at}s, not before the last chunk")
    s.close()


# ---------------------------------------------------------------------------
# G4 -- a fabricated document ID must be unrepresentable
# ---------------------------------------------------------------------------


def test_citations_are_always_real_corpus_identifiers():
    e = _engine()
    corpus = {c.cite for c in e.index.chunks}
    s = e.session(session_id="t_cite", sink_dir=None)
    for t, txt in [(0.0, "what are the exclusions"), (0.6, "on the standard"),
                   (1.2, "warranty for liquid damage")]:
        s.on_chunk(t, txt)
    r = s.end_utterance(1.8)
    assert r.citations, "expected at least one citation"
    for c in r.citations:
        assert c in corpus, f"citation {c!r} is not a corpus identifier"
    s.close()


# ---------------------------------------------------------------------------
# G5 -- refine, do not restart
# ---------------------------------------------------------------------------


def test_late_constraint_preserves_unaffected_claims_exactly():
    e = _engine()
    s = e.session(session_id="t_refine", sink_dir=None)
    for t, txt in [(0.0, "My S24 screen flickers"), (0.5, "after the update and"),
                   (1.0, "I want to know"), (1.5, "if a repair is"),
                   (2.0, "covered under warranty")]:
        s.on_chunk(t, txt)
    v1 = s.end_utterance(2.6)
    assert v1.version == 1
    before = {c.id: c.signature() for c in s.graph.active_claims()}

    for t, txt in [(6.0, "Oh I should mention"), (6.5, "I bought it in"),
                   (7.0, "Dubai while travelling")]:
        s.on_chunk(t, txt)
    v2 = s.end_utterance(7.6)

    assert v2.kind == "refinement", f"expected refinement, got {v2.kind}"
    cs = v2.change_summary
    assert cs["preserved"] > 0, "a constraint superseded the entire answer"
    assert cs["citations_changed_on_preserved"] == 0, (
        "citations drifted on claims the constraint did not affect -- "
        "this is exactly the G4/G5 conflict the claim graph exists to prevent")

    for cid in cs.get("preserved_claim_ids", []):
        assert s.graph.claims[cid].signature() == before[cid], (
            f"preserved claim {cid} is not byte-identical")

    # And it must be cheaper than a restart.
    assert v2.retrievals_this_turn < v1.retrievals_this_turn, (
        "refinement issued as many retrievals as the original turn")
    s.close()


# ---------------------------------------------------------------------------
# Pitfall 4 -- presentation turns must cost nothing
# ---------------------------------------------------------------------------


def test_presentation_turn_costs_nothing():
    e = _engine()
    s = e.session(session_id="t_suppress", sink_dir=None)
    for t, txt in [(0.0, "what is the turnaround"), (0.6, "for a display repair")]:
        s.on_chunk(t, txt)
    s.end_utterance(1.2)
    before_retrievals = s.index.calls

    out = s.on_chunk(5.0, "please repeat that in two bullets")
    assert out.decision == Decision.SUPPRESS, out.reason
    r = s.end_utterance(5.6)
    assert r.kind == "suppressed"
    assert r.retrievals_this_turn == 0
    assert r.cost.llm_calls == 0
    assert s.index.calls == before_retrievals, "a suppressed turn hit the index"
    assert r.citations, "suppression discarded the prior citations"
    s.close()


# ---------------------------------------------------------------------------
# Pitfall 5 -- a single question must not be fragmented
# ---------------------------------------------------------------------------


def test_single_intent_is_not_decomposed():
    e = _engine()
    s = e.session(session_id="t_single", sink_dir=None)
    for t, txt in [(0.0, "what proof of purchase"), (0.6, "do we accept")]:
        s.on_chunk(t, txt)
    r = s.end_utterance(1.2)
    assert len(r.sub_queries) == 1, (
        f"single question split into {len(r.sub_queries)}: {r.sub_queries}")
    s.close()


# ---------------------------------------------------------------------------
# Corpus isolation and session-bound state
# ---------------------------------------------------------------------------


def test_sessions_do_not_leak_into_each_other():
    """The guide prohibits cross-session profiling. Two identical sessions must
    produce identical state -- if the second were influenced by the first, it
    would not."""
    e = _engine()
    chunks = [(0.0, "is a loaner device"), (0.6, "available for this repair")]

    def run(sid):
        s = e.session(session_id=sid, sink_dir=None)
        for t, txt in chunks:
            s.on_chunk(t, txt)
        r = s.end_utterance(1.2)
        out = (sorted(r.citations), r.version, len(s.graph.claims))
        s.close()
        return out

    assert run("iso_a") == run("iso_b")


def test_abstains_on_an_out_of_domain_question():
    """The conservative retrieval-side gate must catch a question whose subject
    the corpus does not discuss at all."""
    e = _engine()
    s = e.session(session_id="t_hole", sink_dir=None)
    for t, txt in [(0.0, "how do i file an"), (0.5, "insurance claim with the"),
                   (1.0, "customer's own insurer")]:
        s.on_chunk(t, txt)
    r = s.end_utterance(1.6)
    assert r.uncertainty, "no uncertainty indicator on an out-of-domain question"
    assert not r.citations, f"cited {r.citations} for a question with no support"
    s.close()


def test_never_wrongly_abstains_on_a_covered_question():
    """The gate's precision is what matters more than its recall.

    A wrong abstention throws away an answer we could have given; a missed one
    is caught downstream by the synthesiser and the grounding verifier. So this
    test is the strict one, and the gate's threshold is set to keep it green.
    """
    e = _engine()
    covered = [
        "what proof of purchase do we accept",
        "how long does a display repair take",
        "is a loaner device available",
        "what are the exclusions for liquid damage",
        "what is the warranty on a completed repair",
        "what documentation does an imported device need",
    ]
    from ripple.retrieval.relevance import assess

    for q in covered:
        hits = e.index.search(q, k=20)
        v = assess(q, [h.chunk for h in hits])
        assert v.sufficient, (
            f"wrongly abstained on a covered question: {q!r} "
            f"(coverage={v.max_passage_coverage:.2f}, "
            f"missing={v.unmatched_terms})")


def test_near_miss_holes_are_a_known_stub_limitation():
    """Documents, rather than hides, where the keyless path falls short.

    The retrieval-side gate is deliberately conservative and does not catch a
    hole whose subject the corpus *does* discuss without answering the actual
    question -- reimbursing a screen protector, when there is a section about
    screen protectors. That judgement is semantic entailment and belongs to the
    model, not to a lexical heuristic; two lexical heuristics were measured
    failing at it (see ripple/retrieval/relevance.py).

    This test asserts the limitation is still exactly where we say it is. If it
    starts failing because a near-miss hole IS caught, that is good news -- but
    the evaluation report's honesty caveat must then be updated to match.
    """
    e = _engine()
    from ripple.retrieval.relevance import assess

    q = "do we reimburse the cost of the screen protector"
    hits = e.index.search(q, k=20)
    v = assess(q, [h.chunk for h in hits])
    assert v.sufficient, (
        "the pre-filter now catches a near-miss hole -- update the limitation "
        "documented in relevance.py and in the evaluation report")
    # The signal is visible even though it is not actionable lexically: the
    # unmatched terms are precisely the substantive ask.
    assert "reimburs" in " ".join(v.unmatched_terms), v.unmatched_terms


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}\n        {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}\n        {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
