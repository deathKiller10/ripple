"""Grounding verifier -- gate G4.

Two failure modes to catch, and they need different tests.

1. FABRICATED IDENTIFIER. A claim cites DOC_999 §7, which does not exist.
   Structurally impossible in Ripple, because citations are derived from
   evidence IDs rather than written by a model (see synthesizer
   `_strip_inline_citations`). We check anyway: a guarantee you do not test is
   a guarantee you do not have, and the check is what lets the evaluation
   report state "zero fabricated IDs" as a measurement.

2. UNSUPPORTED CLAIM. The cited chunk exists but does not say what the claim
   says. This is the real risk and the one G4's 85% threshold targets. We test
   it two ways and take the stronger signal:
     * lexical containment -- what share of the claim's content words appear in
       the cited chunk. Cheap, and catches wholesale invention.
     * embedding cosine between claim and chunk. Catches faithful paraphrase
       that lexical overlap would wrongly reject.

A claim that fails both is not repaired and not silently dropped from the
record: it is withheld from the answer and converted into an explicit
uncertainty statement. That is the behaviour section 1 of the guide asks for --
"returning explicit uncertainty indicators when evidence is insufficient" --
and it is why abstention is a feature here rather than a failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_STOP = set("""a an the of to in on for and or but with without is are was were
be been being as at by from that this these those it its their his her our your
my you we they he she i not no nor so if then than there here which who whom
whose what when where why how any some all each both few more most other such
only own same too very can will just should now must may might shall do does
did done have has had also into over under about per""".split())

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]*")


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower())
            if w not in _STOP and len(w) > 2}


@dataclass
class GroundingVerdict:
    claim_id: str
    grounded: bool
    lexical: float
    semantic: float
    missing_citations: list[str]
    fabricated_citations: list[str]
    reason: str

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id, "grounded": self.grounded,
            "lexical": round(self.lexical, 4), "semantic": round(self.semantic, 4),
            "missing_citations": list(self.missing_citations),
            "fabricated_citations": list(self.fabricated_citations),
            "reason": self.reason,
        }


class GroundingVerifier:
    def __init__(self, index, min_overlap: float = 0.18,
                 min_semantic: float = 0.42):
        self.index = index
        self.min_overlap = min_overlap
        self.min_semantic = min_semantic

    def verify(self, claim, pool) -> GroundingVerdict:
        fabricated, texts = [], []
        for cite in claim.citations:
            item = pool.by_cite(cite)
            if item is None:
                # Not in the session pool -- is it even in the corpus?
                if cite in getattr(self.index, "by_cite", {}):
                    texts.append(self.index.by_cite[cite].text)
                else:
                    fabricated.append(cite)
            else:
                texts.append(item.chunk.text)

        if fabricated:
            return GroundingVerdict(
                claim.id, False, 0.0, 0.0, [], fabricated,
                f"cites identifier(s) not present in the corpus: {fabricated}",
            )
        if not texts:
            return GroundingVerdict(
                claim.id, False, 0.0, 0.0, list(claim.citations), [],
                "claim carries no resolvable citation",
            )

        support = " ".join(texts)
        cw = content_words(claim.text)
        sw = content_words(support)
        lexical = (len(cw & sw) / len(cw)) if cw else 0.0

        semantic = 0.0
        try:
            v_claim = self.index.encode_query(claim.text)
            v_sup = self.index.encode_query(support)
            semantic = float(np.dot(v_claim, v_sup))
        except Exception:
            semantic = 0.0

        grounded = lexical >= self.min_overlap or semantic >= self.min_semantic
        reason = (
            "supported"
            if grounded
            else (f"claim content not found in cited passage "
                  f"(lexical={lexical:.2f}<{self.min_overlap}, "
                  f"semantic={semantic:.2f}<{self.min_semantic})")
        )
        return GroundingVerdict(claim.id, grounded, lexical, semantic, [], [],
                                reason)

    def verify_all(self, claims, pool) -> tuple[list, list[GroundingVerdict]]:
        kept, verdicts = [], []
        for c in claims:
            v = self.verify(c, pool)
            verdicts.append(v)
            c.grounded = v.grounded
            if v.grounded:
                kept.append(c)
            else:
                c.note = v.reason
        return kept, verdicts


def uncertainty_for(rejected: list, intents: dict) -> str:
    """Turn withheld claims into the explicit uncertainty indicator."""
    if not rejected:
        return ""
    topics = []
    for c in rejected:
        it = intents.get(c.intent_id)
        label = it.text if it else c.intent_id
        if label not in topics:
            topics.append(label)
    if len(topics) == 1:
        return (f"The corpus did not contain evidence sufficient to answer: "
                f"{topics[0]}. No claim is made about it.")
    joined = "; ".join(topics)
    return (f"The corpus did not contain evidence sufficient to answer the "
            f"following, and no claim is made about them: {joined}.")
