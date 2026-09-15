"""The three baselines Ripple is measured against.

All four systems share the same index, the same embedder, the same reranker and
the same provider, so every reported difference is attributable to the streaming
architecture rather than to a better retriever or a bigger model. That control
is the point; a comparison that changes two things at once measures nothing.

  B0  LLM only          no retrieval at all. Establishes the grounding floor
                        and shows what the parametric model would say --
                        which, under the guide's corpus-isolation rule, is
                        exactly what must never reach the user.

  B1  Static RAG        one query at utterance end, full regeneration on every
                        follow-up. This is what most submissions will be, and
                        it is a genuinely strong baseline on grounding. It
                        loses on time-to-first-token and on refinement cost.

  B2  Naive streaming   retrieve on EVERY transcript chunk, regenerate the
                        whole answer each time. The behaviour the guide names
                        as pitfall 1. Implemented deliberately and honestly so
                        that its cost can be priced rather than asserted.

  B3  Ripple            the full engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ripple.config import Config
from ripple.engine import RippleEngine
from ripple.retrieval.rerank import apply as apply_rerank
from ripple.schemas import Cost, Scenario


@dataclass
class TurnRecord:
    kind: str
    answer: str
    citations: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    retrieved_cites: list[str] = field(default_factory=list)
    uncertainty: str = ""
    retrievals: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    currency_cost: float = 0.0
    ttft_rel_end: float | None = None
    fired_before_end: bool = False
    change_summary: dict = field(default_factory=dict)
    wall_ms: float = 0.0


@dataclass
class SystemRun:
    system: str
    scenario_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    shadow_calls: int = 0
    retrieval_calls: int = 0


def _utterances(scenario: Scenario):
    """Group transcript chunks into utterances."""
    out, cur = [], []
    for ch in scenario.chunks:
        cur.append(ch)
        if ch.is_final:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------------------
# B0 -- no retrieval
# ---------------------------------------------------------------------------


def run_b0(engine: RippleEngine, scenario: Scenario) -> SystemRun:
    run = SystemRun("B0_llm_only", scenario.scenario_id)
    for utt in _utterances(scenario):
        text = " ".join(c.text for c in utt)
        t_end = utt[-1].t
        t0 = time.perf_counter()
        # No corpus access by construction. With the stub provider there is
        # nothing to extract from, so the answer is empty -- which is the
        # honest representation of "a model with no grounding and no licence
        # to invent". With an API provider this is a genuine ungrounded
        # completion.
        res = engine.provider.complete(
            f"Answer the support agent's question in two sentences.\n\n{text}",
            max_tokens=200)
        run.turns.append(TurnRecord(
            kind="answer", answer=(res.text or "").strip(), citations=[],
            sub_queries=[text], retrievals=0,
            llm_calls=res.cost.llm_calls,
            prompt_tokens=res.cost.prompt_tokens,
            completion_tokens=res.cost.completion_tokens,
            currency_cost=res.cost.currency_cost,
            ttft_rel_end=0.0, fired_before_end=False,
            wall_ms=(time.perf_counter() - t0) * 1000))
    return run


# ---------------------------------------------------------------------------
# B1 -- static RAG
# ---------------------------------------------------------------------------


def run_b1(engine: RippleEngine, scenario: Scenario) -> SystemRun:
    from ripple.session.pool import EvidencePool
    from ripple.synthesis.synthesizer import ClaimSynthesizer

    cfg = engine.cfg
    run = SystemRun("B1_static_rag", scenario.scenario_id)
    synth = ClaimSynthesizer(engine.provider, cfg.synthesis.max_claims)
    history: list[str] = []

    for utt in _utterances(scenario):
        text = " ".join(c.text for c in utt)
        t_end = utt[-1].t
        t0 = time.perf_counter()
        history.append(text)
        # Static RAG has no session state worth the name: a follow-up is
        # handled by concatenating history and searching again from scratch.
        query = " ".join(history[-2:])
        hits = engine.index.search(query, k=cfg.retrieval.top_k)
        hits = apply_rerank(engine.reranker, query, hits,
                            cfg.retrieval.rerank_candidates)
        pool = EvidencePool()
        items = [pool.add(h.chunk, h.score, "single", t_end, "final",
                          provisional=False)
                 for h in hits[: cfg.retrieval.context_budget]]
        res = synth.synthesize(query, items)
        cites: list[str] = []
        for c in res.claims:
            for x in c.citations:
                if x not in cites:
                    cites.append(x)
        run.turns.append(TurnRecord(
            kind="answer",
            answer=" ".join(c.text for c in res.claims),
            citations=cites, sub_queries=[query],
            retrieved_cites=[h.cite for h in hits[:cfg.retrieval.context_budget]],
            uncertainty=res.uncertainty, retrievals=1,
            llm_calls=res.cost.llm_calls,
            prompt_tokens=res.cost.prompt_tokens,
            completion_tokens=res.cost.completion_tokens,
            currency_cost=res.cost.currency_cost,
            # Retrieval starts only once the utterance is complete, so the
            # first token can never precede the end of the utterance.
            ttft_rel_end=(time.perf_counter() - t0), fired_before_end=False,
            change_summary={"regenerated": True},
            wall_ms=(time.perf_counter() - t0) * 1000))
    run.retrieval_calls = len(run.turns)
    return run


# ---------------------------------------------------------------------------
# B2 -- naive streaming
# ---------------------------------------------------------------------------


def run_b2(engine: RippleEngine, scenario: Scenario) -> SystemRun:
    from ripple.session.pool import EvidencePool
    from ripple.synthesis.synthesizer import ClaimSynthesizer

    cfg = engine.cfg
    run = SystemRun("B2_naive_streaming", scenario.scenario_id)
    synth = ClaimSynthesizer(engine.provider, cfg.synthesis.max_claims)

    for utt in _utterances(scenario):
        prefix = ""
        retrievals = 0
        llm_calls = pt = ct = 0
        money = 0.0
        first_token_t = None
        last = None
        t_end = utt[-1].t
        t0 = time.perf_counter()

        for ch in utt:
            prefix = (prefix + " " + ch.text).strip()
            if len(prefix.split()) < 2:
                continue
            # Pitfall 1, implemented on purpose: retrieve on every chunk,
            # regenerate the whole answer every time, keep nothing.
            hits = engine.index.search(prefix, k=cfg.retrieval.top_k)
            hits = apply_rerank(engine.reranker, prefix, hits,
                                cfg.retrieval.rerank_candidates)
            retrievals += 1
            pool = EvidencePool()
            items = [pool.add(h.chunk, h.score, "single", ch.t, "chunk",
                              provisional=False)
                     for h in hits[: cfg.retrieval.context_budget]]
            res = synth.synthesize(prefix, items)
            llm_calls += res.cost.llm_calls
            pt += res.cost.prompt_tokens
            ct += res.cost.completion_tokens
            money += res.cost.currency_cost
            if res.claims and first_token_t is None:
                first_token_t = ch.t
            last = res

        cites: list[str] = []
        if last:
            for c in last.claims:
                for x in c.citations:
                    if x not in cites:
                        cites.append(x)
        last_retrieved = [h.cite for h in hits[: cfg.retrieval.context_budget]]
        run.turns.append(TurnRecord(
            kind="answer",
            answer=" ".join(c.text for c in last.claims) if last else "",
            citations=cites, retrieved_cites=last_retrieved,
            sub_queries=[prefix],
            uncertainty=last.uncertainty if last else "",
            retrievals=retrievals, llm_calls=llm_calls,
            prompt_tokens=pt, completion_tokens=ct, currency_cost=money,
            ttft_rel_end=((first_token_t - t_end)
                          if first_token_t is not None else None),
            fired_before_end=retrievals > 0,
            change_summary={"regenerated_times": retrievals},
            wall_ms=(time.perf_counter() - t0) * 1000))
    return run


# ---------------------------------------------------------------------------
# B3 -- Ripple
# ---------------------------------------------------------------------------


def run_b3(engine: RippleEngine, scenario: Scenario, speculative: bool = True,
           sink_dir: str | None = None, **overrides) -> SystemRun:
    from ripple.schemas import Decision

    run = SystemRun("B3_ripple", scenario.scenario_id)
    s = engine.session(session_id=f"b3_{scenario.scenario_id}",
                       speculative=speculative, sink_dir=sink_dir)
    for k, v in overrides.items():
        setattr(s, k, v)

    fired = False
    t0 = time.perf_counter()
    for ch in scenario.chunks:
        out = s.on_chunk(ch.t, ch.text, ch.speaker, ch.is_final)
        if out.decision in (Decision.RETRIEVE, Decision.RETRIEVE_MORE):
            fired = True
        if ch.is_final:
            before = s.turn_retrievals
            r = s.end_utterance(ch.t + 0.001)
            # MEASUREMENT FIX. Recall must be computed over what the system
            # RETRIEVED, not over what it chose to cite. The first version left
            # this empty for Ripple while populating it for the static
            # baseline, so metrics.py fell back to the citation list -- a dozen
            # retrieved chunks for B1 against three cited claims for B3 -- and
            # reported Ripple's recall as roughly half the baseline's. The
            # engine was fine; the comparison was not. Anything that makes your
            # own system look bad in a way you cannot explain mechanically is a
            # harness bug until proven otherwise.
            retrieved = [it.cite for it in s.pool.usable()]
            run.turns.append(TurnRecord(
                kind=r.kind, answer=r.answer, citations=r.citations,
                retrieved_cites=retrieved,
                sub_queries=r.sub_queries, uncertainty=r.uncertainty,
                retrievals=r.retrievals_this_turn,
                llm_calls=r.cost.llm_calls,
                prompt_tokens=r.cost.prompt_tokens,
                completion_tokens=r.cost.completion_tokens,
                currency_cost=r.cost.currency_cost,
                ttft_rel_end=r.ttft_rel_utterance_end,
                fired_before_end=fired, change_summary=r.change_summary,
                wall_ms=(time.perf_counter() - t0) * 1000))
            fired = False
            t0 = time.perf_counter()
    run.shadow_calls = s.index.shadow_calls
    s.close()
    return run


RUNNERS = {
    "B0_llm_only": run_b0,
    "B1_static_rag": run_b1,
    "B2_naive_streaming": run_b2,
    "B3_ripple": run_b3,
}
