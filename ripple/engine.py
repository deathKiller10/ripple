"""The Ripple engine.

Transport-free by design. The WebSocket server and the replay CLI are both
thin clients of this object; neither is a dependency of it. That is what makes
gate G1 achievable -- an automated replay suite that needs a browser is not
automated.

One session is one conversation. Session state lives in this object's fields
and dies with it. There is no database, no cache keyed by user, and no
persistence across sessions, because the guide's session-bound state rule makes
persistence a spec violation rather than a missing feature.

Turn lifecycle
--------------
on_chunk(t, text)              per transcript fragment; the controller decides
                               WAIT / RETRIEVE / RETRIEVE_MORE / SUPPRESS and
                               early retrieval populates the evidence pool
end_utterance(t)               resolve the turn: suppression, refinement, or a
                               full decomposed answer
output_record()                the guide's section 4 JSON, verbatim field names
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .config import Config
from .controller.compound import (PresentationClassifier, assess_compoundness,
                                  load_presentation_examples)
from .controller.policy import ControllerOutput, RetrievalController
from .retrieval import fusion as fusion_mod
from .retrieval.index import Hit, HybridIndex
from .retrieval.rerank import apply as apply_rerank
from .retrieval.rerank import build_reranker
from .retrieval.relevance import assess_question as assess_relevance
from .retrieval.relevance import build_vocabulary
from .schemas import (Cost, Decision, EventType, GuideRetrievalEvent,
                      OutputRecord, Trigger, new_id)
from .session.claim_graph import ClaimGraph, Constraint
from .session.pool import EvidencePool
from .synthesis.providers import build_provider
from .synthesis.synthesizer import (ClaimSynthesizer, SubQueryExtractor,
                                    infer_constraint, looks_like_constraint,
                                    scope_tags)
from .synthesis.verifier import GroundingVerifier, uncertainty_for
from .telemetry import TelemetryBus, instrumented


@dataclass
class TurnResult:
    kind: str                       # "answer" | "refinement" | "suppressed"
    version: int
    answer: str
    citations: list[str]
    uncertainty: str
    change_summary: dict = field(default_factory=dict)
    retrievals_this_turn: int = 0
    sub_queries: list[str] = field(default_factory=list)
    cost: Cost = field(default_factory=Cost)
    ttft_rel_utterance_end: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "version": self.version, "answer": self.answer,
            "citations": list(self.citations), "uncertainty": self.uncertainty,
            "change_summary": dict(self.change_summary),
            "retrievals_this_turn": self.retrievals_this_turn,
            "sub_queries": list(self.sub_queries),
            "cost": self.cost.to_dict(),
            "ttft_rel_utterance_end": self.ttft_rel_utterance_end,
        }


class RippleSession:
    def __init__(self, index: HybridIndex, cfg: Config, session_id: str,
                 bus: Optional[TelemetryBus] = None,
                 reranker=None, provider=None,
                 speculative: bool = True,
                 on_event: Optional[Callable] = None):
        self.index = index
        self.cfg = cfg
        self.session_id = session_id
        self.bus = bus or TelemetryBus(session_id, sink_dir=cfg.telemetry_path)
        if on_event:
            self.bus.subscribe(on_event)

        self.pool = EvidencePool()
        self.graph = ClaimGraph()

        pres = PresentationClassifier()
        examples = load_presentation_examples("data/labels/presentation.jsonl")
        if examples:
            try:
                pres.fit(examples, lambda ts: self.index.embedder.encode(ts))
            except Exception:
                pass
        self.controller = RetrievalController(index, cfg.controller, pres)

        self.reranker = (reranker if reranker is not None
                         else build_reranker(index, cfg.retrieval.reranker))
        self.provider = provider if provider is not None else build_provider(
            cfg.synthesis, cfg.cost)
        self.extractor = SubQueryExtractor(
            self.provider, encode=lambda ts: self.index.embedder.encode(ts))
        self.synthesizer = ClaimSynthesizer(self.provider, cfg.synthesis.max_claims)
        self.verifier = GroundingVerifier(index, cfg.synthesis.grounding_min_overlap)
        # Corpus vocabulary drives the abstention gate. Built once per index,
        # cached on the index object so sessions are cheap.
        if not hasattr(index, '_vocabulary'):
            index._vocabulary = build_vocabulary(index.chunks)
        self.vocabulary = index._vocabulary
        self.speculative = speculative

        # per-utterance state
        self.prefix = ""
        self.last_trigger_prefix = ""
        self.suppressed = False
        self.turn_retrievals = 0
        self.turn_cost = Cost()
        self.first_token_t: Optional[float] = None
        self.utterance_index = 0

        # guide output record accumulation
        self.guide_events: list[GuideRetrievalEvent] = []
        self.all_sub_queries: list[str] = []

        self.bus.emit(EventType.SESSION_STARTED,
                      detail={"config": cfg.to_dict(),
                              "index": index.stats(),
                              "provider": self.provider.name,
                              "reranker": getattr(self.reranker, "name", "?"),
                              "speculative_synthesis": speculative})

    # ---------------------------------------------------------------- utils

    def _intent_vectors(self) -> list[np.ndarray]:
        vecs = []
        for i in self.graph.active_intents():
            try:
                vecs.append(self.index.encode_query(i.focus()))
            except Exception:
                pass
        return vecs

    def _assumption_universe(self) -> set[str]:
        u: set[str] = set()
        for c in self.graph.active_claims():
            u.update(c.assumptions)
        return u

    # ------------------------------------------------------------- ingestion

    @instrumented("on_chunk")
    def on_chunk(self, t: float, text: str, speaker: str = "customer",
                 is_final: bool = False) -> ControllerOutput:
        """One transcript fragment. This is the hot path: it must stay free of
        LLM calls."""
        self.bus.set_virtual_clock(t)
        delta = text.strip()
        self.prefix = (self.prefix + " " + delta).strip()
        self.bus.emit(EventType.CHUNK_RECEIVED, chunk_text=delta,
                      prefix_text=self.prefix,
                      detail={"speaker": speaker, "is_final": is_final})

        t0 = time.perf_counter()
        out = self.controller.decide(
            prefix=self.prefix,
            delta_text=self.prefix[len(self.last_trigger_prefix):].strip(),
            pool_ids=self.pool.ids(),
            intent_vecs=self._intent_vectors(),
            is_turn_start=(self.prefix == delta),
        )
        latency = (time.perf_counter() - t0) * 1000.0
        self.bus.emit(EventType.CONTROLLER_DECISION,
                      controller=out.to_trace(self.cfg.controller.theta),
                      prefix_text=self.prefix, latency_ms=round(latency, 3))

        if out.decision == Decision.SUPPRESS:
            # Not sticky. Suppression blocks retrieval for THIS chunk, which is
            # where the cost saving comes from, but the resolution path is
            # decided at end_utterance against the complete utterance. A turn
            # that starts "say that again..." and continues "...and what does
            # the policy say about imports" must not stay suppressed.
            self.suppressed = True
            return out
        self.suppressed = False

        if out.decision in (Decision.RETRIEVE, Decision.RETRIEVE_MORE):
            self._early_retrieve(t, out)

        return out

    @instrumented("early_retrieve")
    def _early_retrieve(self, t: float, out: ControllerOutput) -> None:
        """Speculative retrieval on a partial utterance.

        The output is a warm evidence pool, not an answer. Chunks land as
        PROVISIONAL and are promoted only when a claim actually rests on them.
        """
        query = self.prefix
        source = ("jump" if out.decision == Decision.RETRIEVE_MORE
                  else "decomposition")
        focus = self.prefix[len(self.last_trigger_prefix):].strip() or self.prefix
        intent = self.graph.open_intent(query, t, source=source,
                                        focus_text=focus)

        # A new topic opening is the completion signal for the previous one.
        # The speaker moving on is stronger evidence that the earlier
        # sub-question is finished than any measure of the sentence itself,
        # so this is where speculative synthesis pays off.
        if out.decision == Decision.RETRIEVE_MORE and self.speculative:
            self._speculate_on_closed_intents(t, exclude=intent.id)

        self.bus.emit(EventType.RETRIEVAL_STARTED, sub_query=query,
                      trigger=out.trigger.value if out.trigger else None,
                      intent_id=intent.id,
                      controller=out.to_trace(self.cfg.controller.theta))
        self.guide_events.append(
            GuideRetrievalEvent(timestamp_s=t, query=query,
                                trigger=out.trigger.value if out.trigger else "provisional")
        )

        t0 = time.perf_counter()
        hits = self._search(query)
        for h in hits:
            self.pool.add(h.chunk, h.score, intent.id, t,
                          out.trigger.value if out.trigger else "provisional",
                          provisional=True)
        self.turn_retrievals += 1
        self.last_trigger_prefix = self.prefix

        self.bus.emit(EventType.RETRIEVAL_COMPLETED, sub_query=query,
                      intent_id=intent.id,
                      retrieved=[h.cite for h in hits[:10]],
                      latency_ms=round((time.perf_counter() - t0) * 1000.0, 3))
        self.bus.emit(EventType.POOL_UPDATED, detail=self.pool.counts())

        if self.speculative and self.first_token_t is None:
            self._maybe_speculate(t, intent, hits)

    def _relevance_ok(self, question: str, items) -> bool:
        """Abstention gate, applied to the SPECULATIVE path too.

        Speculation runs before the utterance ends and originally called the
        synthesiser directly, which meant a draft claim could assert something
        the end-of-turn abstention gate would have blocked -- and because the
        draft was already emitted and cited, the turn ended with a citation on
        a question the corpus does not answer. A gate that only guards the slow
        path is not a gate.
        """
        if not items:
            return False
        rel = assess_relevance(question, [i.chunk for i in items],
                               split_fn=self._split_for_gate)
        self.bus.emit(EventType.GROUNDING_CHECKED,
                      detail={"stage": "pre_speculation_relevance",
                              **rel.to_dict()})
        if not rel.sufficient:
            self.bus.emit(EventType.UNCERTAINTY_EMITTED, uncertainty=rel.reason)
        return rel.sufficient

    @instrumented("speculate_closed")
    def _speculate_on_closed_intents(self, t: float, exclude: str) -> None:
        """Answer an intent the speaker has already moved on from.

        This is what makes time-to-first-token negative relative to end of
        utterance: by the time the customer starts their second question, the
        answer to the first is already streaming. It is also low risk, because
        we only do it for a topic the speaker has demonstrably finished.
        """
        for intent in self.graph.active_intents():
            if intent.id == exclude:
                continue
            if any(c.intent_id == intent.id for c in self.graph.active_claims()):
                continue
            items = self._rank_pool_for(intent.focus(), intent.id)[:3]
            if not self._relevance_ok(intent.focus(), items):
                continue
            call0 = time.perf_counter()
            res = self.synthesizer.synthesize(intent.focus(), items)
            call_s = time.perf_counter() - call0
            # A call that produced no usable claim was still PAID FOR. Recording
            # cost only after the claims check made every discarded draft free
            # in our own accounts -- invisible with the zero-cost stub, and an
            # understated cost-per-turn the moment a real model is used.
            self.turn_cost = self.turn_cost + res.cost
            self.bus.add_cost(res.cost)
            if not res.claims:
                continue
            sc = res.claims[0]
            claim = self.graph.add_claim(
                sc.text, intent.id, sc.evidence_ids, sc.citations,
                assumptions=sc.assumptions,
                version=len(self.graph.versions) + 1,
                confidence=sc.confidence, note="speculative draft")
            self.pool.promote(sc.evidence_ids)
            if self.first_token_t is None:
                # t + the model's own reply time. See _maybe_speculate.
                self.first_token_t = t + call_s
                self.bus.emit(EventType.FIRST_TOKEN, claim_id=claim.id,
                              detail={"speculative": True,
                                      "trigger": "topic_closed_by_speaker",
                                      "intent": intent.focus()})
            self.bus.emit(EventType.CLAIM_EMITTED, claim_id=claim.id,
                          citations=sc.citations, intent_id=intent.id,
                          detail={"speculative": True, "text": sc.text})
            break

    @instrumented("speculate")
    def _maybe_speculate(self, t: float, intent, hits: list[Hit]) -> None:
        """Begin answering before the utterance ends.

        This is what makes time-to-first-token negative relative to end of
        utterance, which is the headline number in the evaluation report. We
        only speculate when one retrieval is decisively well matched, so the
        draft claim is unlikely to be thrown away.
        """
        if len(hits) < 2 or len(intent.text.split()) < 6:
            return
        conf, margin = self._semantic_confidence(intent.text, hits[:6])
        if conf < 0.30 or margin < 0.06:
            return  # ambiguous top-1; more words are worth waiting for

        items = [self.pool.get(h.chunk.chunk_id) for h in hits[:3]]
        items = [i for i in items if i]
        if not self._relevance_ok(intent.text, items):
            return
        call0 = time.perf_counter()
        res = self.synthesizer.synthesize(intent.text, items)
        call_s = time.perf_counter() - call0
        # Cost first: a discarded draft was still paid for (see above).
        self.turn_cost = self.turn_cost + res.cost
        self.bus.add_cost(res.cost)
        if not res.claims:
            return
        sc = res.claims[0]
        claim = self.graph.add_claim(
            sc.text, intent.id, sc.evidence_ids, sc.citations,
            assumptions=sc.assumptions, version=len(self.graph.versions) + 1,
            confidence=sc.confidence, note="speculative draft",
        )
        self.pool.promote(sc.evidence_ids)
        # THE FIRST TOKEN EXISTS WHEN THE MODEL HAS ANSWERED, not when we
        # asked it. `t` is the transcript time the draft was triggered; the
        # call itself takes real seconds. Stamping `t` alone left the model's
        # latency out of Ripple's TTFT while the B1 baseline's stopwatch
        # included it -- invisible with the microsecond stub, and worth ~2 s
        # of flattery with a real model. The speech keeps arriving while the
        # call runs, so trigger time + call time is when the answer lands.
        self.first_token_t = t + call_s
        self.bus.emit(EventType.FIRST_TOKEN, claim_id=claim.id,
                      detail={"speculative": True, "intent": intent.text,
                              "model_seconds": round(call_s, 3)})
        self.bus.emit(EventType.CLAIM_EMITTED, claim_id=claim.id,
                      citations=sc.citations, intent_id=intent.id,
                      detail={"speculative": True, "text": sc.text})

    # ------------------------------------------------------------ resolution

    @instrumented("end_utterance")
    def end_utterance(self, t: float) -> TurnResult:
        self.bus.set_virtual_clock(t)
        # Wall-clock start of end-of-utterance work, so an answer produced
        # here is stamped t + (retrieval + model time), the same stopwatch
        # the B1 baseline uses.
        self._end_wall0 = time.perf_counter()
        self.bus.emit(EventType.UTTERANCE_END, prefix_text=self.prefix)
        utterance = self.prefix.strip()
        self.utterance_index += 1

        # Re-test on the COMPLETE utterance. Per-chunk suppression is a cost
        # decision taken under uncertainty; this is the correctness decision
        # taken with the full sentence in hand.
        final_suppress, p_sup = self.controller.presentation.predict(utterance)
        self.bus.emit(EventType.CONTROLLER_DECISION,
                      controller={"decision": "SUPPRESS" if final_suppress
                                  else "RESOLVE",
                                  "stability": 1.0, "novelty": 0.0, "drift": 0.0,
                                  "threshold": self.cfg.controller.theta,
                                  "semantic_jump": False,
                                  "reason": f"utterance_final_check p={p_sup:.2f}"},
                      prefix_text=utterance)

        if final_suppress:
            # With no prior answer this is a social turn -- a greeting or a
            # hold message. Nothing to restructure and nothing to retrieve.
            result = self._resolve_suppressed(t, utterance)
        elif (self.graph.versions
              and looks_like_constraint(utterance, has_prior_answer=True)):
            result = self._resolve_refinement(t, utterance)
        else:
            result = self._resolve_answer(t, utterance)

        self._reset_utterance()
        return result

    def _reset_utterance(self) -> None:
        self.prefix = ""
        self.last_trigger_prefix = ""
        self.suppressed = False
        self.turn_retrievals = 0
        self.turn_cost = Cost()
        self.first_token_t = None
        self.controller.reset_utterance()

    # -- path 1: suppression ----------------------------------------------

    @instrumented("resolve_suppressed")
    def _resolve_suppressed(self, t: float, utterance: str) -> TurnResult:
        """Guide example 3. No vector search, no synthesis call, prior
        citations retained, none invented."""
        claims = self.graph.active_claims()
        answer = self._restructure(utterance, claims)
        citations = self.graph.all_citations()
        self.bus.emit(EventType.ANSWER_VERSION,
                      answer_version=self.graph.versions[-1].version
                      if self.graph.versions else 0,
                      citations=citations,
                      detail={"retrieval_required": False,
                              "reason": "presentation_restructure",
                              "retrievals": 0, "llm_calls": 0})
        # Report what ACTUALLY happened, not what the resolution path did.
        # A turn resolved by suppression can still have triggered a speculative
        # retrieval earlier in the utterance, before the controller had heard
        # enough to recognise it as presentation or social. Reporting zero
        # there would hide a real false trigger from our own gate G2 figure,
        # which is precisely the half of that gate teams are tempted to omit.
        return TurnResult(
            kind="suppressed",
            version=self.graph.versions[-1].version if self.graph.versions else 0,
            answer=answer, citations=citations, uncertainty="",
            change_summary={"reason": "presentation_restructure",
                            "retrievals": self.turn_retrievals,
                            "early_retrieval_before_suppression":
                                self.turn_retrievals > 0},
            retrievals_this_turn=self.turn_retrievals, cost=self.turn_cost,
        )

    def _restructure(self, request: str, claims) -> str:
        """Deterministic re-presentation of existing claims.

        Zero tokens. The claim graph already holds the answer as discrete
        assertions, so reformatting is a rendering choice rather than a
        generation task -- which is only true because the answer was never a
        string in the first place.
        """
        r = request.lower()
        want_bullets = any(w in r for w in ("bullet", "list", "points", "steps"))
        n = None
        for word, val in (("one", 1), ("two", 2), ("three", 3), ("four", 4)):
            if word in r:
                n = val
        for m in ("1", "2", "3", "4", "5"):
            if m in r:
                n = int(m)
        selected = claims[:n] if n else claims
        if want_bullets or n:
            return "\n".join(
                f"- {c.text} " + " ".join(f"[{x}]" for x in c.citations)
                for c in selected
            )
        return " ".join(
            f"{c.text} " + " ".join(f"[{x}]" for x in c.citations)
            for c in selected
        )

    # -- path 2: refinement (guide example 2) ------------------------------

    @instrumented("resolve_refinement")
    def _resolve_refinement(self, t: float, utterance: str) -> TurnResult:
        """Late-arriving detail. Refine, do not restart.

        Four steps, only the last of which touches a model: invalidate,
        re-score, delta-retrieve, patch.
        """
        # -- step 0: what does this constraint activate in the corpus? ------
        probe = self._search(utterance, k=6)
        probe_items = []
        con_intent = self.graph.open_intent(utterance, t, source="constraint")
        for h in probe:
            probe_items.append(
                self.pool.add(h.chunk, h.score, con_intent.id, t,
                              Trigger.DELTA.value, provisional=True))
        self.turn_retrievals += 1
        self.guide_events.append(
            GuideRetrievalEvent(timestamp_s=t, query=utterance,
                                trigger=Trigger.DELTA.value))
        self.bus.emit(EventType.RETRIEVAL_COMPLETED, sub_query=utterance,
                      trigger=Trigger.DELTA.value, intent_id=con_intent.id,
                      retrieved=[h.cite for h in probe[:8]])

        constraint = infer_constraint(utterance, probe_items,
                                      self._assumption_universe(), t)
        self.bus.emit(EventType.CONSTRAINT_DETECTED,
                      detail=constraint.to_dict())

        # -- step 1: invalidate --------------------------------------------
        partition = self.graph.apply_constraint(constraint)
        for cid in partition["affected_claim_ids"]:
            self.bus.emit(EventType.CLAIM_SUPERSEDED, claim_id=cid,
                          detail={"by": constraint.scope + "=" + constraint.value})
        for cid in partition["preserved_claim_ids"]:
            self.bus.emit(EventType.CLAIM_PRESERVED, claim_id=cid,
                          citations=self.graph.claims[cid].citations)

        # -- step 2: corpus-declared supersession in the pool ---------------
        for older, newer, how in self.pool.detect_supersession():
            self.bus.emit(EventType.POOL_UPDATED,
                          detail={"superseded": older, "by": newer, "how": how})

        # -- step 3: delta retrieval, only for intents that lost a claim ----
        needing = self.graph.intents_needing_delta(partition)
        delta_queries = []
        for intent in needing:
            q = f"{intent.text} {utterance}".strip()
            delta_queries.append(q)
            self.bus.emit(EventType.RETRIEVAL_STARTED, sub_query=q,
                          trigger=Trigger.DELTA.value, intent_id=intent.id)
            self.guide_events.append(
                GuideRetrievalEvent(timestamp_s=t, query=q,
                                    trigger=Trigger.DELTA.value))
            hits = self._search(q)
            for h in hits:
                self.pool.add(h.chunk, h.score, intent.id, t,
                              Trigger.DELTA.value, provisional=False)
            self.turn_retrievals += 1
            self.bus.emit(EventType.RETRIEVAL_COMPLETED, sub_query=q,
                          intent_id=intent.id,
                          retrieved=[h.cite for h in hits[:8]])

        # -- step 4: patch only the superseded claims ----------------------
        rejected = []
        for intent in needing:
            items = self._rank_pool_for(intent.text + " " + utterance, intent.id)
            res = self.synthesizer.synthesize(intent.text, items,
                                              constraints=[utterance])
            self.turn_cost = self.turn_cost + res.cost
            self.bus.add_cost(res.cost)
            for sc in res.claims:
                # The new claim carries the constraint's scope, so a further
                # constraint on the same key supersedes it in turn.
                claim = self.graph.add_claim(
                    sc.text, intent.id, sc.evidence_ids, sc.citations,
                    assumptions=sc.assumptions,
                    version=len(self.graph.versions) + 1,
                    confidence=sc.confidence, note="patched by constraint",
                )
                self.pool.promote(sc.evidence_ids)
                self.bus.emit(EventType.CLAIM_EMITTED, claim_id=claim.id,
                              citations=sc.citations, intent_id=intent.id,
                              detail={"text": sc.text, "patched": True})

        # Also answer the constraint turn itself when it carries a question.
        kept, verdicts = self.verifier.verify_all(self.graph.active_claims(),
                                                  self.pool)
        for v in verdicts:
            self.bus.emit(EventType.GROUNDING_CHECKED, claim_id=v.claim_id,
                          detail=v.to_dict())
        rejected = [c for c in self.graph.active_claims() if not c.grounded]
        for c in rejected:
            c.status = "SUPERSEDED"
        unc = uncertainty_for(rejected, self.graph.intents)
        if unc:
            self.bus.emit(EventType.UNCERTAINTY_EMITTED, uncertainty=unc)

        version = self.graph.commit_version(t, uncertainty=unc)
        answer = self.graph.render()
        citations = self.graph.all_citations()
        self.bus.emit(EventType.ANSWER_VERSION, answer_version=version.version,
                      citations=citations,
                      detail={**version.change_summary,
                              "retrievals": self.turn_retrievals,
                              "delta_queries": delta_queries})
        return TurnResult(
            kind="refinement", version=version.version, answer=answer,
            citations=citations, uncertainty=unc,
            change_summary=version.change_summary,
            retrievals_this_turn=self.turn_retrievals,
            sub_queries=delta_queries, cost=self.turn_cost,
        )

    # -- path 3: full answer ----------------------------------------------

    @instrumented("resolve_answer")
    def _resolve_answer(self, t: float, utterance: str) -> TurnResult:
        if not utterance:
            return TurnResult("answer", 0, "", [], "")

        # -- compoundness gate ---------------------------------------------
        comp = assess_compoundness(
            utterance, encode=lambda ts: self.index.embedder.encode(ts))
        self.bus.emit(EventType.COMPOUNDNESS_TESTED, detail=comp.to_dict())

        dec = self.extractor.extract(utterance, comp.is_compound)
        self.turn_cost = self.turn_cost + dec.cost
        self.bus.add_cost(dec.cost)
        self.bus.emit(EventType.DECOMPOSED, sub_queries=dec.sub_queries,
                      detail={"method": dec.method,
                              "compound": comp.is_compound})
        for q in dec.sub_queries:
            if q not in self.all_sub_queries:
                self.all_sub_queries.append(q)

        # -- retire the provisional intents opened on partial prefixes ------
        # Those were placeholders. The real intents are the sub-queries.
        for i in self.graph.active_intents():
            if i.source in ("decomposition", "jump") and i.text not in dec.sub_queries:
                self.graph.abandon_intent(i.id)

        intents = [self.graph.open_intent(q, t, "decomposition")
                   for q in dec.sub_queries]

        # -- retrieve per sub-query, reusing the warm pool -----------------
        per_intent: dict[str, list[Hit]] = {}
        for intent in intents:
            pooled = self.pool.for_intent(intent.id)
            need_search = True
            if pooled and len(pooled) >= self.cfg.retrieval.per_intent_floor:
                # Early retrieval already covered this intent. This is the
                # payoff of treating early retrieval as a cache.
                need_search = False
            if need_search:
                self.bus.emit(EventType.RETRIEVAL_STARTED, sub_query=intent.text,
                              trigger=Trigger.MULTI_INTENT.value,
                              intent_id=intent.id)
                self.guide_events.append(
                    GuideRetrievalEvent(timestamp_s=t, query=intent.text,
                                        trigger=Trigger.MULTI_INTENT.value))
                hits = self._search(intent.text)
                for h in hits:
                    self.pool.add(h.chunk, h.score, intent.id, t,
                                  Trigger.MULTI_INTENT.value, provisional=False)
                self.turn_retrievals += 1
                self.bus.emit(EventType.RETRIEVAL_COMPLETED,
                              sub_query=intent.text, intent_id=intent.id,
                              retrieved=[h.cite for h in hits[:10]])
            hits = self._pool_hits_for(intent.text, intent.id)
            hits = apply_rerank(self.reranker, intent.text, hits,
                                self.cfg.retrieval.rerank_candidates)
            per_intent[intent.id] = hits

        # -- M3: coverage-budgeted fusion ----------------------------------
        vectors = self._vectors_for(per_intent)
        if self.cfg.retrieval.fusion == "rrf":
            # Ablation A2: plain pooled RRF across sub-queries, the standard
            # approach. Same candidates, same reranking, same budget -- only
            # the allocation rule differs.
            fr = fusion_mod.rrf_pool(
                per_intent, budget=self.cfg.retrieval.context_budget,
                rrf_k=self.cfg.retrieval.rrf_k)
        else:
            fr = fusion_mod.coverage_budgeted(
                per_intent, budget=self.cfg.retrieval.context_budget,
                per_intent_floor=self.cfg.retrieval.per_intent_floor,
                mmr_lambda=self.cfg.retrieval.mmr_lambda, vectors=vectors)
        self.bus.emit(EventType.FUSION_COMPLETED, allocation=fr.allocation,
                      retrieved=[h.cite for h in fr.selected],
                      rerank_scores=[round(getattr(h, "rerank_score", 0.0) or 0.0, 4)
                                     for h in fr.selected],
                      detail={"starved_intents": fr.starved_intents,
                              "dropped_duplicates": len(fr.dropped_duplicates)})

        # -- evict speculative evidence for intents that never materialised -
        evicted = self.pool.evict_unused_provisional({i.id for i in intents})
        if evicted:
            self.bus.emit(EventType.POOL_UPDATED,
                          detail={"evicted_irrelevant": evicted})

        for older, newer, how in self.pool.detect_supersession():
            self.bus.emit(EventType.POOL_UPDATED,
                          detail={"superseded": older, "by": newer, "how": how})

        # -- synthesise per intent from its allocated slice ----------------
        selected_by_intent: dict[str, list] = {i.id: [] for i in intents}
        for h in fr.selected:
            item = self.pool.get(h.chunk.chunk_id)
            if not item:
                continue
            for iid in item.fetched_for.split(","):
                if iid in selected_by_intent:
                    selected_by_intent[iid].append(item)

        constraint_texts = [c.raw_text for c in self.graph.constraints]
        uncovered: list[str] = []
        for intent in intents:
            items = selected_by_intent.get(intent.id) or self._rank_pool_for(
                intent.text, intent.id)
            if not items:
                continue

            # Abstention gate. Runs BEFORE synthesis, because the cheapest way
            # to avoid an ungrounded answer is not to generate one. See
            # retrieval/relevance.py for why this is a vocabulary test rather
            # than a score threshold.
            rel = assess_relevance(intent.text, [i.chunk for i in items],
                                   split_fn=self._split_for_gate)
            self.bus.emit(EventType.GROUNDING_CHECKED, intent_id=intent.id,
                          detail={"stage": "pre_synthesis_relevance",
                                  **rel.to_dict()})
            if not rel.sufficient:
                uncovered.append(intent.text)
                self.bus.emit(EventType.UNCERTAINTY_EMITTED,
                              intent_id=intent.id, uncertainty=rel.reason)
                continue

            res = self.synthesizer.synthesize(intent.text, items,
                                              constraints=constraint_texts)
            self.turn_cost = self.turn_cost + res.cost
            self.bus.add_cost(res.cost)
            for sc in res.claims:
                claim = self.graph.add_claim(
                    sc.text, intent.id, sc.evidence_ids, sc.citations,
                    assumptions=sc.assumptions,
                    version=len(self.graph.versions) + 1,
                    confidence=sc.confidence)
                self.pool.promote(sc.evidence_ids)
                if self.first_token_t is None:
                    self.first_token_t = t + (time.perf_counter()
                                              - getattr(self, "_end_wall0",
                                                        time.perf_counter()))
                    self.bus.emit(EventType.FIRST_TOKEN, claim_id=claim.id,
                                  detail={"speculative": False})
                self.bus.emit(EventType.CLAIM_EMITTED, claim_id=claim.id,
                              citations=sc.citations, intent_id=intent.id,
                              detail={"text": sc.text})

        # -- drop speculative drafts superseded by the real answer ---------
        self._retire_superseded_speculation()

        # -- G4: verify -----------------------------------------------------
        kept, verdicts = self.verifier.verify_all(self.graph.active_claims(),
                                                  self.pool)
        for v in verdicts:
            self.bus.emit(EventType.GROUNDING_CHECKED, claim_id=v.claim_id,
                          detail=v.to_dict())
        rejected = [c for c in self.graph.active_claims() if not c.grounded]
        for c in rejected:
            c.status = "SUPERSEDED"

        unanswered = [i for i in intents
                      if not any(c.intent_id == i.id
                                 for c in self.graph.active_claims())
                      and i.text not in uncovered]
        unc_parts = []
        u1 = uncertainty_for(rejected, self.graph.intents)
        if u1:
            unc_parts.append(u1)
        if unanswered:
            topics = "; ".join(i.text for i in unanswered)
            unc_parts.append(
                f"The corpus contains no passage addressing: {topics}. "
                f"No claim is made about it.")
        if uncovered:
            topics = "; ".join(uncovered)
            unc_parts.append(
                f"The corpus does not cover: {topics}. It contains no passage "
                f"on this subject, so no claim is made and no citation is "
                f"offered.")
        unc = " ".join(unc_parts)
        if unc:
            self.bus.emit(EventType.UNCERTAINTY_EMITTED, uncertainty=unc)

        version = self.graph.commit_version(t, uncertainty=unc)
        answer = self.graph.render()
        citations = self.graph.all_citations()
        ttft = (self.first_token_t - t) if self.first_token_t is not None else None
        self.bus.emit(EventType.ANSWER_VERSION, answer_version=version.version,
                      citations=citations,
                      detail={**version.change_summary,
                              "retrievals": self.turn_retrievals,
                              "ttft_rel_utterance_end": ttft})
        return TurnResult(
            kind="answer", version=version.version, answer=answer,
            citations=citations, uncertainty=unc,
            change_summary=version.change_summary,
            retrievals_this_turn=self.turn_retrievals,
            sub_queries=dec.sub_queries, cost=self.turn_cost,
            ttft_rel_utterance_end=ttft,
        )

    # -- helpers -----------------------------------------------------------

    def _semantic_confidence(self, query: str,
                             hits: list[Hit]) -> tuple[float, float]:
        """How decisively does one chunk match this intent?

        Deliberately measured in EMBEDDING space rather than on the reranker's
        score. Reranker scores live on different scales per backend, and an
        earlier version keyed speculation off them -- which meant switching the
        reranker off silently switched speculative synthesis off too, and
        ablation A5 was measuring two changes at once. Cosine similarity is
        comparable across configurations, so the ablation now isolates the
        reranker alone.

        Returns (top similarity, relative margin over the runner-up).
        """
        if len(hits) < 2:
            return 0.0, 0.0
        try:
            qv = self.index.encode_query(query)
            sims = []
            for h in hits:
                row = self.index.chunks.index(self.index.by_id[h.chunk.chunk_id])
                sims.append(float(np.dot(qv, self.index.matrix[row])))
        except Exception:
            return 0.0, 0.0
        sims.sort(reverse=True)
        top, second = sims[0], sims[1]
        if top <= 0:
            return top, 0.0
        return top, (top - second) / (abs(top) + 1e-6)

    def _split_for_gate(self, text: str) -> list[str]:
        """Zero-token split used only by the abstention gate."""
        from .controller.compound import assess_compoundness, best_split

        enc = lambda ts: self.index.embedder.encode(ts)   # noqa: E731
        if not assess_compoundness(text, encode=enc).is_compound:
            return [text]
        return best_split(text, enc)

    def _search(self, query: str, k: int | None = None) -> list[Hit]:
        """Single entry point for retrieval, so ablation switches apply
        everywhere rather than to whichever call site was remembered."""
        r = self.cfg.retrieval
        return self.index.search(query, k=k or r.top_k, rrf_k=r.rrf_k,
                                 dense_weight=r.dense_weight,
                                 dense_only=r.dense_only)

    def _pool_hits_for(self, query: str, intent_id: str) -> list[Hit]:
        items = self.pool.for_intent(intent_id)
        hits = [Hit(chunk=i.chunk, score=i.score, rank=n)
                for n, i in enumerate(sorted(items, key=lambda x: -x.score))]
        return hits

    def _rank_pool_for(self, query: str, intent_id: str) -> list:
        hits = self._pool_hits_for(query, intent_id)
        hits = apply_rerank(self.reranker, query, hits,
                            self.cfg.retrieval.rerank_candidates)
        out = []
        for h in hits[: self.cfg.retrieval.context_budget]:
            item = self.pool.get(h.chunk.chunk_id)
            if item:
                item.rerank_score = getattr(h, "rerank_score", None)
                out.append(item)
        return out

    def _vectors_for(self, per_intent: dict[str, list[Hit]]) -> dict:
        ids = {h.chunk.chunk_id for hits in per_intent.values() for h in hits}
        out = {}
        for cid in ids:
            try:
                row = self.index.chunks.index(self.index.by_id[cid])
                out[cid] = self.index.matrix[row]
            except Exception:
                pass
        return out

    def _retire_superseded_speculation(self) -> None:
        """A speculative draft and a real claim on the same intent are
        duplicates. Keep the real one."""
        by_intent: dict[str, list] = {}
        for c in self.graph.active_claims():
            by_intent.setdefault(c.intent_id, []).append(c)
        for group in by_intent.values():
            drafts = [c for c in group if c.note == "speculative draft"]
            reals = [c for c in group if c.note != "speculative draft"]
            if drafts and reals:
                for d in drafts:
                    d.status = "SUPERSEDED"
                    d.note = "speculative draft replaced by final synthesis"

    # ------------------------------------------------------------- outputs

    def output_record(self) -> OutputRecord:
        """The Theme 4 Guide section 4 record, with their field names."""
        unc = self.graph.versions[-1].uncertainty if self.graph.versions else ""
        return OutputRecord(
            retrieval_events=list(self.guide_events),
            sub_queries=list(self.all_sub_queries),
            answer=self.graph.render(),
            citations=self.graph.all_citations(),
            uncertainty=unc,
            ripple_ext={
                "session_id": self.session_id,
                "answer_versions": [v.to_dict() for v in self.graph.versions],
                "claims": [c.to_dict() for c in self.graph.claims.values()],
                "evidence_pool": self.pool.snapshot(),
                "pool_counts": self.pool.counts(),
                "cost": self.bus.cost.to_dict(),
                "index_stats": self.index.stats(),
            },
        )

    def close(self) -> None:
        self.bus.emit(EventType.SESSION_ENDED, cost=self.bus.cost.to_dict(),
                      detail={"pool": self.pool.counts(),
                              "versions": len(self.graph.versions)})
        self.bus.close()


class RippleEngine:
    """Factory. Holds the index and the models; sessions are cheap."""

    def __init__(self, cfg: Optional[Config] = None, index=None):
        self.cfg = cfg or Config()
        self.index = index or HybridIndex.load(self.cfg.index_path,
                                               self.cfg.embedder)
        self.reranker = build_reranker(self.index, self.cfg.retrieval.reranker)
        self.provider = build_provider(self.cfg.synthesis, self.cfg.cost)

    def session(self, session_id: Optional[str] = None, speculative: bool = True,
                on_event=None, sink_dir: Optional[str] = None) -> RippleSession:
        sid = session_id or new_id("ses")
        bus = TelemetryBus(sid, sink_dir=sink_dir or self.cfg.telemetry_path)
        return RippleSession(self.index, self.cfg, sid, bus=bus,
                             reranker=self.reranker, provider=self.provider,
                             speculative=speculative, on_event=on_event)
