"""M2 -- The claim graph.

The load-bearing idea. Gate G4 (factual grounding, zero fabricated citations)
and gate G5 (session refinement without restart) fight each other as long as
the answer is a string: the only way to fold a late constraint into a paragraph
is to rewrite the paragraph, and a rewritten paragraph's citations drift.

So the answer is not a string. It is an ordered set of Claims, each of which
carries:

    text            one assertion, one sentence
    intent_id       which sub-question it answers
    evidence_ids    the chunks that support it -- this IS the citation list
    assumptions     the constraint scope under which it holds

A late-arriving constraint is then a predicate over that set, and refinement is
four deterministic steps, only the last of which involves a model:

    1. INVALIDATE   claims whose assumptions conflict with the predicate are
                    marked SUPERSEDED. Nothing is deleted; history is the audit
                    trail and the source of the version diff.
    2. RESCORE      pool evidence is re-ranked against the constraint-augmented
                    sub-query. Chunks previously at rank 14 can surface with no
                    new retrieval at all.
    3. DELTA        retrieve only for intents whose coverage is now
                    insufficient. Usually one sub-query, not four.
    4. PATCH        regenerate only the superseded claims.

Because unaffected claims are never regenerated, their text and their citations
cannot drift. Citation preservation stops being something we prompt for and
becomes something the data structure guarantees. We never instruct a model to
"keep the citations".

That also gives gate G5 the numeric definition Samsung left out:

    state continuity = (unaffected claims whose text AND citation set are
                        byte-identical across versions) / (unaffected claims)

Target 1.0, machine-checked in evaluation/metrics.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..schemas import ClaimStatus, new_id


@dataclass
class Intent:
    id: str
    text: str                       # the retrieval query (usually the full prefix)
    status: str = "active"          # active | resolved | abandoned
    opened_at: float = 0.0
    source: str = "decomposition"   # decomposition | jump | constraint
    # The text span that OPENED this intent, as opposed to the whole prefix
    # used to query. Jump detection compares against this: an accumulating
    # prefix gets broader with every word and eventually resembles everything,
    # which makes a genuinely new topic look like a continuation.
    focus_text: str = ""

    def focus(self) -> str:
        return self.focus_text or self.text

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "status": self.status,
                "opened_at": round(self.opened_at, 3), "source": self.source,
                "focus_text": self.focus_text}


@dataclass
class Constraint:
    """A late-arriving detail, normalised into a predicate."""

    id: str
    scope: str                  # e.g. "purchase_region"
    value: str                  # e.g. "non_domestic"
    raw_text: str = ""
    arrived_at: float = 0.0
    # Assumption tags that this constraint invalidates. A claim carrying any
    # of these is superseded.
    invalidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "scope": self.scope, "value": self.value,
                "raw_text": self.raw_text, "arrived_at": round(self.arrived_at, 3),
                "invalidates": list(self.invalidates)}


@dataclass
class Claim:
    id: str
    text: str
    intent_id: str
    evidence_ids: list[str]           # chunk_ids
    citations: list[str]              # "DOC_X §N" -- derived, never authored
    assumptions: list[str] = field(default_factory=list)
    born_in_version: int = 1
    status: str = ClaimStatus.ACTIVE.value
    superseded_by: Optional[str] = None
    confidence: float = 1.0
    grounded: bool = True
    note: str = ""

    def signature(self) -> tuple:
        """What must stay identical for a claim to count as preserved."""
        return (self.text.strip(), tuple(sorted(self.citations)))

    def to_dict(self) -> dict:
        return {
            "id": self.id, "text": self.text, "intent_id": self.intent_id,
            "evidence_ids": list(self.evidence_ids),
            "citations": list(self.citations),
            "assumptions": list(self.assumptions),
            "born_in_version": self.born_in_version, "status": self.status,
            "superseded_by": self.superseded_by,
            "confidence": round(self.confidence, 3), "grounded": self.grounded,
            "note": self.note,
        }


@dataclass
class AnswerVersion:
    version: int
    claim_ids: list[str]
    created_at: float
    uncertainty: str = ""
    change_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"version": self.version, "claim_ids": list(self.claim_ids),
                "created_at": round(self.created_at, 3),
                "uncertainty": self.uncertainty,
                "change_summary": dict(self.change_summary)}


class ClaimGraph:
    def __init__(self):
        self.intents: dict[str, Intent] = {}
        self.claims: dict[str, Claim] = {}
        self.constraints: list[Constraint] = []
        self.versions: list[AnswerVersion] = []

    # -- intents ----------------------------------------------------------

    def open_intent(self, text: str, t: float, source: str = "decomposition",
                    focus_text: str = "") -> Intent:
        for i in self.intents.values():
            if i.text.strip().lower() == text.strip().lower():
                return i
        intent = Intent(id=new_id("int"), text=text.strip(), opened_at=t,
                        source=source, focus_text=(focus_text or text).strip())
        self.intents[intent.id] = intent
        return intent

    def active_intents(self) -> list[Intent]:
        return [i for i in self.intents.values() if i.status == "active"]

    def abandon_intent(self, intent_id: str) -> None:
        if intent_id in self.intents:
            self.intents[intent_id].status = "abandoned"

    # -- claims -----------------------------------------------------------

    def add_claim(self, text: str, intent_id: str, evidence_ids: list[str],
                  citations: list[str], assumptions: list[str] | None = None,
                  version: int = 1, grounded: bool = True,
                  confidence: float = 1.0, note: str = "") -> Claim:
        # Overlapping sub-intents routinely retrieve the same passage, which
        # would otherwise put the same sentence in the answer two or three
        # times. Merge instead: keep one claim and union its evidence, so the
        # citation list records that two sub-questions rest on it.
        norm = " ".join(text.strip().lower().split())
        for existing in self.active_claims():
            if " ".join(existing.text.lower().split()) == norm:
                for e in evidence_ids:
                    if e not in existing.evidence_ids:
                        existing.evidence_ids.append(e)
                for cit in citations:
                    if cit not in existing.citations:
                        existing.citations.append(cit)
                for a in (assumptions or []):
                    if a not in existing.assumptions:
                        existing.assumptions.append(a)
                return existing
        c = Claim(
            id=new_id("clm"), text=text.strip(), intent_id=intent_id,
            evidence_ids=list(evidence_ids), citations=list(citations),
            assumptions=list(assumptions or []), born_in_version=version,
            grounded=grounded, confidence=confidence, note=note,
        )
        self.claims[c.id] = c
        return c

    def active_claims(self) -> list[Claim]:
        return [c for c in self.claims.values()
                if c.status == ClaimStatus.ACTIVE.value]

    # -- refinement -------------------------------------------------------

    def apply_constraint(self, constraint: Constraint) -> dict:
        """Step 1 of refinement: INVALIDATE.

        Returns the partition of the current answer into claims that survive
        untouched and claims that must be regenerated. This partition is the
        whole refinement contract -- everything downstream reads it.
        """
        self.constraints.append(constraint)
        affected: list[Claim] = []
        preserved: list[Claim] = []

        inv = set(a.lower() for a in constraint.invalidates)
        for c in self.active_claims():
            tags = set(a.lower() for a in c.assumptions)
            if tags & inv:
                affected.append(c)
            else:
                preserved.append(c)

        for c in affected:
            c.status = ClaimStatus.SUPERSEDED.value
            c.note = f"superseded by constraint {constraint.scope}={constraint.value}"

        return {
            "constraint": constraint.to_dict(),
            "affected_claim_ids": [c.id for c in affected],
            "affected_intents": sorted({c.intent_id for c in affected}),
            "preserved_claim_ids": [c.id for c in preserved],
        }

    def intents_needing_delta(self, partition: dict) -> list[Intent]:
        """Step 3: which intents actually need a fresh search.

        Only intents that lost a claim. An intent whose claims all survived is
        not re-queried, which is the measured difference between refining and
        restarting.
        """
        ids = set(partition["affected_intents"])
        return [self.intents[i] for i in ids if i in self.intents]

    # -- versions ---------------------------------------------------------

    def commit_version(self, t: float, uncertainty: str = "") -> AnswerVersion:
        prev = self.versions[-1] if self.versions else None
        active = self.active_claims()
        active_ids = [c.id for c in active]

        summary: dict = {}
        if prev is None:
            summary = {"added": len(active_ids), "preserved": 0, "superseded": 0,
                       "citations_changed_on_preserved": 0}
        else:
            prev_ids = set(prev.claim_ids)
            prev_sigs = {
                cid: self.claims[cid].signature()
                for cid in prev.claim_ids if cid in self.claims
            }
            preserved = [cid for cid in active_ids if cid in prev_ids]
            added = [cid for cid in active_ids if cid not in prev_ids]
            superseded = [cid for cid in prev.claim_ids
                          if cid in self.claims
                          and self.claims[cid].status == ClaimStatus.SUPERSEDED.value]
            drifted = [
                cid for cid in preserved
                if self.claims[cid].signature() != prev_sigs.get(cid)
            ]
            summary = {
                "added": len(added),
                "preserved": len(preserved),
                "superseded": len(superseded),
                "citations_changed_on_preserved": len(drifted),
                "added_claim_ids": added,
                "preserved_claim_ids": preserved,
                "superseded_claim_ids": superseded,
            }

        v = AnswerVersion(
            version=len(self.versions) + 1, claim_ids=active_ids,
            created_at=t, uncertainty=uncertainty, change_summary=summary,
        )
        self.versions.append(v)
        return v

    # -- rendering --------------------------------------------------------

    def render(self, version: Optional[int] = None) -> str:
        """Assemble the prose answer from claims.

        Citations are appended from `claim.citations`, which is derived from
        evidence IDs. No model ever writes a citation string, so a fabricated
        document ID is not merely unlikely -- it is unrepresentable.
        """
        if version is None:
            claims = self.active_claims()
        else:
            v = next((x for x in self.versions if x.version == version), None)
            if v is None:
                return ""
            claims = [self.claims[c] for c in v.claim_ids if c in self.claims]

        by_intent: dict[str, list[Claim]] = {}
        for c in claims:
            by_intent.setdefault(c.intent_id, []).append(c)

        parts = []
        for intent_id, group in by_intent.items():
            for c in group:
                cites = " ".join(f"[{x}]" for x in c.citations)
                parts.append(f"{c.text} {cites}".strip())
        return " ".join(parts)

    def all_citations(self, version: Optional[int] = None) -> list[str]:
        if version is None:
            claims = self.active_claims()
        else:
            v = next((x for x in self.versions if x.version == version), None)
            claims = [self.claims[c] for c in v.claim_ids] if v else []
        seen, out = set(), []
        for c in claims:
            for cite in c.citations:
                if cite not in seen:
                    seen.add(cite)
                    out.append(cite)
        return out

    def snapshot(self) -> dict:
        return {
            "intents": [i.to_dict() for i in self.intents.values()],
            "claims": [c.to_dict() for c in self.claims.values()],
            "constraints": [c.to_dict() for c in self.constraints],
            "versions": [v.to_dict() for v in self.versions],
        }
