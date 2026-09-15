"""M1 -- Retrieval-space stability.

The central idea of the project, stated precisely:

  Everyone else measures whether a partial utterance is *linguistically*
  complete. That needs a model, costs tokens on every transcript chunk, and
  answers a question we do not care about. The question we care about is
  whether more words would change *which documents come back*. That is
  directly measurable, for free, by comparing the ranked result sets of
  successive prefixes.

So for each incoming chunk we run a shadow retrieval -- a bare dense top-k
lookup, no BM25, no reranking, no generation -- and compare its ranking to the
previous chunk's with rank-biased overlap. Low drift means the prefix has
stopped moving the answer set: it is retrieval-stable, and further waiting buys
nothing.

Two properties follow that are worth defending to a jury:

  * A grammatically incomplete fragment can be stable. "I need to plan a
    customer workshop in Pune for 30" already pins the document set; the
    remaining words will not move it. Language-space stability would say wait.
  * A grammatically complete sentence can be unstable, if it sits on a
    boundary between document clusters. Language-space stability would say go.

Cost: one embedding plus one exact ANN query per chunk. Single-digit
milliseconds on CPU, zero tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def rbo(list_a: list[str], list_b: list[str], p: float = 0.9) -> float:
    """Rank-biased overlap, extrapolated (Webber, Moffat & Zobel 2010).

    Chosen over Jaccard or Kendall's tau because it is top-weighted and handles
    non-conjoint lists. A change at rank 1 should register strongly; a change at
    rank 9 should barely register, because rank 9 rarely reaches synthesis.
    Returns 1.0 for identical rankings, 0.0 for disjoint.
    """
    if not list_a and not list_b:
        return 1.0
    if not list_a or not list_b:
        return 0.0

    sa: set[str] = set()
    sb: set[str] = set()
    depth = max(len(list_a), len(list_b))
    agreement_sum = 0.0
    overlap_at_depth = 0.0

    for d in range(depth):
        if d < len(list_a):
            sa.add(list_a[d])
        if d < len(list_b):
            sb.add(list_b[d])
        overlap = len(sa & sb)
        agreement = overlap / (d + 1)
        agreement_sum += (p ** d) * agreement
        overlap_at_depth = overlap

    rbo_min = (1 - p) * agreement_sum
    # extrapolation term for the unseen tail
    ext = (overlap_at_depth / depth) * (p ** depth)
    return float(min(1.0, rbo_min + ext))


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


@dataclass
class StabilityState:
    """Per-utterance stability tracker."""

    ema_alpha: float = 0.55
    rbo_p: float = 0.90

    prev_ids: Optional[list[str]] = None
    stability: float = 0.0
    drift: float = 1.0
    history: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        self.prev_ids = None
        self.stability = 0.0
        self.drift = 1.0

    def update(self, ids: list[str]) -> tuple[float, float]:
        """Feed one shadow-retrieval result. Returns (stability, drift)."""
        if self.prev_ids is None:
            # First observation of an utterance: no evidence of stability yet.
            self.prev_ids = ids
            self.drift = 1.0
            self.stability = 0.0
            self.history.append({"drift": 1.0, "stability": 0.0})
            return self.stability, self.drift

        inst_stability = rbo(ids, self.prev_ids, self.rbo_p)
        self.drift = 1.0 - inst_stability
        # EMA so that a single noisy chunk does not trip a retrieval, and so
        # that sustained agreement across several chunks is what fires it.
        self.stability = (
            self.ema_alpha * inst_stability + (1 - self.ema_alpha) * self.stability
        )
        self.prev_ids = ids
        self.history.append(
            {"drift": round(self.drift, 4), "stability": round(self.stability, 4)}
        )
        return self.stability, self.drift


def novelty(candidate_ids: list[str], pool_ids: set[str]) -> float:
    """Fraction of a shadow result set not already in the evidence pool.

    This is the budget controller in one number. Once the pool already holds
    what the shadow retrieval found, a further retrieve buys nothing, so we
    suppress it even at maximum stability. Without this term a long, stable
    utterance would re-retrieve on every chunk -- high stability is not by
    itself a reason to search.
    """
    if not candidate_ids:
        return 0.0
    unseen = [c for c in candidate_ids if c not in pool_ids]
    return len(unseen) / len(candidate_ids)


def semantic_jump(new_vec: np.ndarray, intent_vecs: list[np.ndarray],
                  threshold: float) -> tuple[bool, float]:
    """Has the speaker opened a new topic rather than continuing the old one?

    Compares the embedding of the text added since the last retrieval against
    every currently-active intent. If it resembles none of them, a new intent
    branch is declared and a parallel retrieval is dispatched -- this is the
    RETRIEVE_MORE decision, and it is how a second question arriving mid-
    utterance gets its own search instead of being blended into the first.
    """
    if not intent_vecs or new_vec is None:
        return False, 0.0
    sims = [float(np.dot(new_vec, v)) for v in intent_vecs]
    best = max(sims)
    return best < threshold, best
