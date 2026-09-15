"""M3 -- Coverage-budgeted fusion.

Reciprocal Rank Fusion pools every sub-query's results into one ranked list.
That is correct when the sub-queries are paraphrases of one question. It is
wrong when they are *different questions*, which is exactly the Theme 4 case.

Concretely: the customer asks three things, the context budget is twelve
chunks, and RRF hands back nine chunks about the display fault and zero about
turnaround time. Every retrieval metric looks excellent -- recall@10 per
sub-query is fine, the fused ranking is well ordered -- and yet a third of the
answer silently disappears. Sub-intent starvation is invisible to standard
retrieval metrics, which is why it survives in so many systems.

The objective is not "rank these chunks". It is "allocate a scarce context
budget across n questions". So:

    1. floor      every active intent is guaranteed `per_intent_floor` slots
    2. fill       remaining slots go to whichever intent's next-best chunk has
                  the highest reranker score (greedy marginal gain)
    3. dedup      MMR across the whole selection, because two intents often
                  retrieve the same policy section and paying for it twice
                  wastes the budget

We measure the difference with a metric we have to define ourselves --
per-intent grounded coverage -- because no off-the-shelf metric sees it.
Ablation A2 is exactly this function against plain RRF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .index import Hit


@dataclass
class FusionResult:
    selected: list[Hit]
    allocation: dict[str, int]          # intent_id -> chunks granted
    dropped_duplicates: list[str]
    starved_intents: list[str]

    def to_dict(self) -> dict:
        return {
            "allocation": dict(self.allocation),
            "selected": [h.cite for h in self.selected],
            "dropped_duplicates": list(self.dropped_duplicates),
            "starved_intents": list(self.starved_intents),
        }


def rrf_pool(per_intent: dict[str, list[Hit]], budget: int,
             rrf_k: int = 60) -> FusionResult:
    """BASELINE, used by ablation A2. Standard RRF over the pooled lists."""
    scores: dict[str, float] = {}
    keep: dict[str, Hit] = {}
    for hits in per_intent.values():
        for rank, h in enumerate(hits):
            cid = h.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            if cid not in keep or h.score > keep[cid].score:
                keep[cid] = h
    order = sorted(scores.items(), key=lambda kv: -kv[1])[:budget]
    selected = [keep[cid] for cid, _ in order]

    alloc: dict[str, int] = {k: 0 for k in per_intent}
    chosen = {h.chunk.chunk_id for h in selected}
    for intent_id, hits in per_intent.items():
        alloc[intent_id] = sum(1 for h in hits if h.chunk.chunk_id in chosen)
    starved = [i for i, n in alloc.items() if n == 0]
    return FusionResult(selected, alloc, [], starved)


def _mmr_ok(candidate_vec, chosen_vecs, lam: float) -> bool:
    """Reject a candidate that is near-identical to something already chosen.

    lam is the relevance/diversity trade-off: at lam=1.0 nothing is rejected.
    """
    if candidate_vec is None or not chosen_vecs:
        return True
    sim = max(float(np.dot(candidate_vec, v)) for v in chosen_vecs)
    return sim <= lam


def coverage_budgeted(
    per_intent: dict[str, list[Hit]],
    budget: int,
    per_intent_floor: int = 2,
    mmr_lambda: float = 0.72,
    vectors: dict[str, np.ndarray] | None = None,
) -> FusionResult:
    """Allocate `budget` chunks across intents with a guaranteed floor."""
    intents = [i for i, hits in per_intent.items() if hits]
    if not intents:
        return FusionResult([], {}, [], list(per_intent.keys()))

    vectors = vectors or {}
    selected: list[Hit] = []
    chosen_ids: set[str] = set()
    chosen_vecs: list[np.ndarray] = []
    dropped: list[str] = []
    alloc: dict[str, int] = {i: 0 for i in per_intent}
    cursor: dict[str, int] = {i: 0 for i in intents}

    def try_take(intent_id: str) -> bool:
        hits = per_intent[intent_id]
        while cursor[intent_id] < len(hits):
            h = hits[cursor[intent_id]]
            cursor[intent_id] += 1
            cid = h.chunk.chunk_id
            if cid in chosen_ids:
                # Already selected for another intent. It counts toward this
                # intent's coverage too -- the chunk answers both -- but does
                # not consume another budget slot.
                alloc[intent_id] += 1
                dropped.append(cid)
                continue
            v = vectors.get(cid)
            if not _mmr_ok(v, chosen_vecs, mmr_lambda):
                dropped.append(cid)
                continue
            selected.append(h)
            chosen_ids.add(cid)
            if v is not None:
                chosen_vecs.append(v)
            alloc[intent_id] += 1
            return True
        return False

    # -- 1. floor: round-robin so no intent can be starved by a loud one ----
    floor = max(1, min(per_intent_floor, max(1, budget // max(1, len(intents)))))
    for _ in range(floor):
        for intent_id in intents:
            if len(selected) >= budget:
                break
            try_take(intent_id)

    # -- 2. fill: greedy marginal gain on the reranker score ---------------
    while len(selected) < budget:
        best_intent, best_score = None, -1e9
        for intent_id in intents:
            idx = cursor[intent_id]
            hits = per_intent[intent_id]
            if idx >= len(hits):
                continue
            nxt = hits[idx]
            s = nxt.rerank_score if hasattr(nxt, "rerank_score") else None
            s = s if s is not None else nxt.score
            if s > best_score:
                best_intent, best_score = intent_id, s
        if best_intent is None:
            break
        if not try_take(best_intent):
            # exhausted this intent; loop continues with the others
            if all(cursor[i] >= len(per_intent[i]) for i in intents):
                break

    starved = [i for i, n in alloc.items() if n == 0]
    return FusionResult(selected, alloc, dropped, starved)


def per_intent_coverage(selected: list[Hit], gold: dict[str, list[str]]) -> dict:
    """Our evaluation metric: did every sub-intent get at least one of its gold
    chunks into the final context?

    `gold` maps sub-intent -> list of gold citation strings. Returns the
    fraction covered plus the list of starved sub-intents, which is the number
    ablation A2 reports.
    """
    chosen = {h.cite for h in selected}
    covered, missed = 0, []
    for intent, cites in gold.items():
        if any(c in chosen for c in cites):
            covered += 1
        else:
            missed.append(intent)
    total = max(1, len(gold))
    return {
        "coverage": covered / total,
        "covered": covered,
        "total": total,
        "starved": missed,
    }
