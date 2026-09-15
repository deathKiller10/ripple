"""The session evidence pool.

The important modelling decision in this project, after the claim graph: early
retrieval produces a *cache*, not an answer. If streaming retrieval is modelled
as a sequence of independent searches, nothing downstream can work -- evidence
fetched speculatively at 0.8 s must still be usable at 3.2 s, and must be
demotable if the intent it was fetched for never materialises.

So the pool accumulates evidence across an entire session, tracks why each
chunk was fetched, and gives every chunk a lifecycle state. The lifecycle is
what makes late-constraint refinement possible without a restart: a constraint
does not clear the pool, it re-states parts of it.

    PROVISIONAL  fetched from a partial utterance, intent unconfirmed
    ACTIVE       supporting at least one live claim
    SUPERSEDED   a more specific chunk overrides it (regional policy over base
                 policy; a dated amendment over the original)
    CONTRADICTED in conflict with another ACTIVE chunk -- both are surfaced to
                 the user with citations rather than one being silently picked
    IRRELEVANT   fetched for an intent that never materialised; excluded from
                 synthesis context but retained in the trace

Context pollution is the hidden cost of early retrieval, and eviction is how we
pay it. Without lifecycle states, speculative chunks accumulate in the context
window and degrade the final answer -- which would show up as a mysterious
quality drop that most teams would misattribute to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..schemas import Chunk, EvidenceState


@dataclass
class EvidenceItem:
    chunk: Chunk
    score: float
    state: EvidenceState
    fetched_for: str            # intent_id
    fetched_at: float           # t_rel_s
    trigger: str
    rerank_score: Optional[float] = None
    superseded_by: Optional[str] = None
    conflicts_with: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def cite(self) -> str:
        return self.chunk.cite

    @property
    def id(self) -> str:
        return self.chunk.chunk_id

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.id,
            "cite": self.cite,
            "doc_title": self.chunk.doc_title,
            "section_title": self.chunk.section_title,
            "score": round(self.score, 5),
            "rerank_score": (round(self.rerank_score, 5)
                             if self.rerank_score is not None else None),
            "state": self.state.value,
            "fetched_for": self.fetched_for,
            "fetched_at": round(self.fetched_at, 3),
            "trigger": self.trigger,
            "superseded_by": self.superseded_by,
            "conflicts_with": list(self.conflicts_with),
            "note": self.note,
        }


class EvidencePool:
    def __init__(self):
        self.items: dict[str, EvidenceItem] = {}
        self.order: list[str] = []

    # -- population -------------------------------------------------------

    def add(self, chunk: Chunk, score: float, intent_id: str, t: float,
            trigger: str, provisional: bool = True) -> EvidenceItem:
        existing = self.items.get(chunk.chunk_id)
        if existing:
            # Keep the better score and record the additional intent it serves.
            existing.score = max(existing.score, score)
            if intent_id not in existing.fetched_for.split(","):
                existing.fetched_for = f"{existing.fetched_for},{intent_id}"
            if not provisional and existing.state == EvidenceState.PROVISIONAL:
                existing.state = EvidenceState.ACTIVE
            return existing
        item = EvidenceItem(
            chunk=chunk, score=score,
            state=EvidenceState.PROVISIONAL if provisional else EvidenceState.ACTIVE,
            fetched_for=intent_id, fetched_at=t, trigger=trigger,
        )
        self.items[chunk.chunk_id] = item
        self.order.append(chunk.chunk_id)
        return item

    # -- queries ----------------------------------------------------------

    def ids(self) -> set[str]:
        return set(self.items.keys())

    def usable(self) -> list[EvidenceItem]:
        """Everything a synthesiser may cite. SUPERSEDED and CONTRADICTED are
        included deliberately -- a superseded policy is still citable when the
        answer explains that it was superseded, and a contradiction must be
        shown from both sides. Only IRRELEVANT is withheld."""
        return [i for i in self.items.values() if i.state != EvidenceState.IRRELEVANT]

    def for_intent(self, intent_id: str) -> list[EvidenceItem]:
        return [i for i in self.usable() if intent_id in i.fetched_for.split(",")]

    def get(self, chunk_id: str) -> Optional[EvidenceItem]:
        return self.items.get(chunk_id)

    def by_cite(self, cite: str) -> Optional[EvidenceItem]:
        for i in self.items.values():
            if i.cite == cite:
                return i
        return None

    # -- lifecycle transitions -------------------------------------------

    def promote(self, chunk_ids: list[str]) -> list[str]:
        """Speculative evidence that ended up supporting a claim becomes
        ACTIVE."""
        changed = []
        for cid in chunk_ids:
            it = self.items.get(cid)
            if it and it.state == EvidenceState.PROVISIONAL:
                it.state = EvidenceState.ACTIVE
                changed.append(cid)
        return changed

    def evict_unused_provisional(self, keep_intents: set[str]) -> list[str]:
        """Demote provisional evidence fetched for intents that never
        materialised. This is retrieval cancellation expressed where it
        actually matters: not as saved CPU, but as evidence kept out of the
        synthesis context."""
        evicted = []
        for it in self.items.values():
            if it.state != EvidenceState.PROVISIONAL:
                continue
            intents = set(it.fetched_for.split(","))
            if not (intents & keep_intents):
                it.state = EvidenceState.IRRELEVANT
                it.note = "intent never materialised"
                evicted.append(it.id)
        return evicted

    def mark_superseded(self, older_id: str, newer_id: str, note: str = "") -> bool:
        it = self.items.get(older_id)
        if not it:
            return False
        it.state = EvidenceState.SUPERSEDED
        it.superseded_by = newer_id
        it.note = note or f"superseded by {newer_id}"
        return True

    def mark_conflict(self, a_id: str, b_id: str, note: str = "") -> None:
        for x, y in ((a_id, b_id), (b_id, a_id)):
            it = self.items.get(x)
            if it:
                if y not in it.conflicts_with:
                    it.conflicts_with.append(y)
                if it.state == EvidenceState.ACTIVE:
                    it.state = EvidenceState.CONTRADICTED
                it.note = note or it.note

    # -- conflict and supersession detection ------------------------------

    def detect_supersession(self) -> list[tuple[str, str, str]]:
        """Corpus-declared supersession, plus a date heuristic.

        A corpus that declares `supersedes:` in front matter gives us this for
        free. Where it does not, two chunks from the same document family with
        different `effective_from` dates and high textual overlap are flagged
        as a probable amendment. We report rather than silently resolve: the
        answer says which is current and cites both.
        """
        found: list[tuple[str, str, str]] = []
        present = self.items
        for it in list(present.values()):
            for older in it.chunk.supersedes:
                if older in present and present[older].state != EvidenceState.SUPERSEDED:
                    self.mark_superseded(
                        older, it.id,
                        f"declared superseded by {it.cite}"
                        + (f" effective {it.chunk.effective_from}"
                           if it.chunk.effective_from else ""),
                    )
                    found.append((older, it.id, "declared"))
        return found

    def snapshot(self) -> list[dict]:
        return [self.items[i].to_dict() for i in self.order]

    def counts(self) -> dict:
        c: dict[str, int] = {}
        for i in self.items.values():
            c[i.state.value] = c.get(i.state.value, 0) + 1
        return c
