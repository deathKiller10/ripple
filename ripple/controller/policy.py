"""The retrieval controller: WAIT / RETRIEVE / RETRIEVE_MORE / SUPPRESS.

This is the component the Theme 4 Guide calls the "Retrieval Controller" and
it contains no LLM call. That is the whole point of the design: a transcript
arriving at three chunks per second over a twenty-second utterance produces
sixty controller decisions, and a team that spends one LLM call per decision
spends sixty per turn. The guide's parsimony rule prices that.

Decision order matters and is deliberate:

  1. SUPPRESS beats everything. A presentation-only turn must never retrieve,
     regardless of how unstable or novel it looks -- "give me that in two
     bullets" is maximally novel against the pool and would otherwise trigger.
  2. Utterances shorter than the minimum word count WAIT unconditionally. This
     is a near-free guard against pitfall 1 (eager retrieval on noise).
  3. A semantic jump with an already-populated pool means a second topic has
     opened: RETRIEVE_MORE, dispatching a parallel branch.
  4. Otherwise retrieve when stability has cleared theta AND the shadow result
     set is novel against the pool. Both conditions are required: high
     stability alone would re-fire on every subsequent chunk of a long, settled
     utterance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import ControllerConfig
from ..schemas import Decision, Trigger
from .compound import PresentationClassifier
from .stability import StabilityState, novelty, semantic_jump


@dataclass
class ControllerOutput:
    decision: Decision
    trigger: Optional[Trigger]
    stability: float
    drift: float
    novelty: float
    semantic_jump: bool
    jump_similarity: float
    reason: str
    shadow_ids: list[str]

    def to_trace(self, threshold: float) -> dict:
        return {
            "decision": self.decision.value,
            "stability": round(self.stability, 4),
            "novelty": round(self.novelty, 4),
            "drift": round(self.drift, 4),
            "threshold": threshold,
            "semantic_jump": self.semantic_jump,
            "reason": self.reason,
        }


class RetrievalController:
    def __init__(self, index, cfg: ControllerConfig,
                 presentation: Optional[PresentationClassifier] = None):
        self.index = index
        self.cfg = cfg
        self.presentation = presentation or PresentationClassifier()
        self.state = StabilityState(ema_alpha=cfg.ema_alpha, rbo_p=cfg.rbo_p)
        self.fired_once = False

    def reset_utterance(self) -> None:
        """Called at each utterance boundary. Stability is a within-utterance
        signal; carrying it across a turn boundary would let a settled previous
        question suppress the next one."""
        self.state.reset()
        self.fired_once = False

    def decide(
        self,
        prefix: str,
        delta_text: str,
        pool_ids: set[str],
        intent_vecs: list[np.ndarray],
        is_turn_start: bool = False,
    ) -> ControllerOutput:
        cfg = self.cfg

        # -- 1. presentation-only turns short-circuit everything ------------
        if is_turn_start or self.fired_once is False:
            is_pres, p_pres = self.presentation.predict(prefix)
        else:
            is_pres, p_pres = self.presentation.predict(prefix)
        if is_pres:
            return ControllerOutput(
                decision=Decision.SUPPRESS, trigger=None, stability=0.0,
                drift=0.0, novelty=0.0, semantic_jump=False, jump_similarity=0.0,
                reason=f"presentation_restructure (p={p_pres:.2f})",
                shadow_ids=[],
            )

        # -- 2. shadow retrieval: the only work done on every chunk ---------
        #
        # This runs BEFORE the minimum-length guard, deliberately. An earlier
        # version returned WAIT on a short prefix without observing anything,
        # which meant the stability series started one chunk late and a short
        # utterance ended before the controller had two observations to compare.
        # The guard exists to stop us FIRING on a fragment, not to stop us
        # LOOKING at one -- and looking is nearly free, which is the whole
        # premise of retrieval-space stability.
        shadow_ids = self.index.shadow(prefix, cfg.shadow_k)
        stability, drift = self.state.update(shadow_ids)
        nov = novelty(shadow_ids, pool_ids)

        words = prefix.split()
        if len(words) < cfg.min_prefix_words:
            return ControllerOutput(
                decision=Decision.WAIT, trigger=None, stability=stability,
                drift=drift, novelty=nov, semantic_jump=False,
                jump_similarity=0.0,
                reason=f"prefix_too_short ({len(words)}<{cfg.min_prefix_words})",
                shadow_ids=shadow_ids,
            )

        # -- 3. has a second topic opened? ---------------------------------
        jumped, jump_sim = False, 1.0
        if delta_text and len(delta_text.split()) >= 3 and intent_vecs:
            try:
                dv = self.index.encode_query(delta_text)
                jumped, jump_sim = semantic_jump(dv, intent_vecs, cfg.jump_similarity)
            except Exception:
                jumped, jump_sim = False, 1.0

        # A semantic jump is NOT gated on novelty. That was a bug worth
        # recording: a new sub-question deserves its own retrieval and its own
        # slice of the context budget even when the pool happens to already
        # contain some of the chunks it would return, because fusion allocates
        # per intent. Gating it on novelty made the second intent invisible
        # precisely when the first retrieval had been broad.
        if jumped and stability >= cfg.theta * 0.7:
            return ControllerOutput(
                decision=Decision.RETRIEVE_MORE, trigger=Trigger.MULTI_INTENT,
                stability=stability, drift=drift, novelty=nov,
                semantic_jump=True, jump_similarity=jump_sim,
                reason=(f"new_intent_branch (sim={jump_sim:.2f}<"
                        f"{cfg.jump_similarity})"),
                shadow_ids=shadow_ids,
            )

        # -- 4. the main gate ----------------------------------------------
        if cfg.always_retrieve:
            # Ablation A1: the controller is disabled and every chunk
            # retrieves. Everything downstream is unchanged.
            self.fired_once = True
            return ControllerOutput(
                decision=Decision.RETRIEVE, trigger=Trigger.PROVISIONAL,
                stability=stability, drift=drift, novelty=nov,
                semantic_jump=False, jump_similarity=jump_sim,
                reason="ablation:always_retrieve (controller disabled)",
                shadow_ids=shadow_ids,
            )

        if stability >= cfg.theta and nov >= cfg.nu:
            self.fired_once = True
            return ControllerOutput(
                decision=Decision.RETRIEVE, trigger=Trigger.PROVISIONAL,
                stability=stability, drift=drift, novelty=nov,
                semantic_jump=False, jump_similarity=jump_sim,
                reason=(f"stable_and_novel (s={stability:.2f}>={cfg.theta}, "
                        f"n={nov:.2f}>={cfg.nu})"),
                shadow_ids=shadow_ids,
            )

        if stability >= cfg.theta:
            reason = f"stable_but_pool_covers_it (n={nov:.2f}<{cfg.nu})"
        else:
            reason = f"unstable (s={stability:.2f}<{cfg.theta})"
        return ControllerOutput(
            decision=Decision.WAIT, trigger=None, stability=stability,
            drift=drift, novelty=nov, semantic_jump=False,
            jump_similarity=jump_sim, reason=reason, shadow_ids=shadow_ids,
        )

    def force_final(self, prefix: str, pool_ids: set[str]) -> ControllerOutput:
        """Utterance ended without the gate ever firing.

        A safety net, and a metric in its own right: every time this fires we
        failed to retrieve early, which is precisely the denominator of gate
        G2. The evaluation harness counts these.
        """
        shadow_ids = self.index.shadow(prefix, self.cfg.shadow_k)
        nov = novelty(shadow_ids, pool_ids)
        return ControllerOutput(
            decision=Decision.RETRIEVE, trigger=Trigger.FINAL,
            stability=self.state.stability, drift=self.state.drift, novelty=nov,
            semantic_jump=False, jump_similarity=0.0,
            reason="utterance_end_fallback (early retrieval did not fire)",
            shadow_ids=shadow_ids,
        )
