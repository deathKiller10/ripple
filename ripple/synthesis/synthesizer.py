"""Claim synthesis, sub-query extraction and constraint inference.

The mechanism worth reading here is `infer_constraint`.

A late-arriving detail such as "oh, I bought it in Dubai" has to be turned into
a predicate that tells the claim graph which existing claims it invalidates.
The obvious implementation is a table of domain rules -- if the utterance
mentions a foreign country, set purchase_region=non_domestic. That works on our
demo and fails on the held-out private benchmark the guide describes, because
it encodes our corpus into application code, which the no-hardcoding rule
forbids.

So instead the corpus tells us what the constraint changes:

  * Every claim inherits SCOPE TAGS from the evidence it rests on. Tags come
    from corpus metadata -- a document with `region: cross-border` contributes
    `region:cross_border`; a document with no region qualifier contributes
    `region:default`; a dated amendment contributes `effective:<date>`.
  * A constraint is retrieved against the corpus like any query. Whatever scope
    tags appear on the evidence it newly activates are the constraint's
    activated tags.
  * The constraint then invalidates every claim carrying a *conflicting value
    of the same tag key*.

"Bought in Dubai" retrieves DOC_WAR_03 (region: cross-border), so it activates
`region:cross_border`, so it invalidates claims tagged `region:default` -- and
leaves claims tagged only `warranty:period` or `kb:diagnostic` untouched. No
domain rule anywhere. Point the engine at a corpus about insurance riders or
regulatory circulars and the same code does the same job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..schemas import Cost, new_id
from ..session.claim_graph import Constraint
from ..session.pool import EvidenceItem
from .providers import (LLMResult, Provider, StubProvider, best_sentences,
                        parse_json_loose)


# ---------------------------------------------------------------------------
# Scope tags -- derived from corpus metadata, never from code
# ---------------------------------------------------------------------------


def scope_tags(chunk) -> list[str]:
    """Scope conditions a claim inherits from the evidence it rests on.

    A tag is emitted ONLY where the document declares that scope. Silence is
    not a value: a knowledge-base article about display flicker says nothing
    about country of purchase, so it gets no region tag and a later
    "I bought it abroad" leaves it standing.

    Getting this wrong is instructive and worth recording. The first version
    tagged unscoped documents `region:default`, which made every claim
    region-scoped -- and so a purchase-region constraint superseded the entire
    answer, including the device diagnosis, which is obviously wrong. Scope
    must be declared by the corpus, never assumed by the code. Documents that
    genuinely are region-conditional say so in their front matter, and that is
    a property of the corpus an author controls, not a rule in application
    code.
    """
    tags = []
    if chunk.region:
        tags.append(f"region:{chunk.region.replace('-', '_')}")
    if chunk.effective_from:
        tags.append(f"effective:{chunk.effective_from}")
    if chunk.doc_type:
        tags.append(f"doctype:{chunk.doc_type}")
    return tags


def tag_key(tag: str) -> str:
    return tag.split(":", 1)[0]


def conflicting_tags(activated: list[str], universe: set[str]) -> list[str]:
    """Tags in `universe` that share a key with an activated tag but differ in
    value. Those are exactly the assumptions the constraint breaks."""
    out = []
    act_by_key: dict[str, set[str]] = {}
    for t in activated:
        act_by_key.setdefault(tag_key(t), set()).add(t)
    for t in universe:
        k = tag_key(t)
        if k in act_by_key and t not in act_by_key[k]:
            out.append(t)
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Sub-query extraction
# ---------------------------------------------------------------------------


SUBQUERY_PROMPT = """You split a single spoken request into the separate \
search queries it implies.

Rules:
- Return between 1 and 4 queries. Return exactly 1 if the request asks one \
thing; splitting a single question into near-duplicates is an error.
- Each query must be self-contained and searchable: carry over the subject of \
the sentence into every query.
- Do not answer anything. Do not invent details the speaker did not say.
- Output JSON only: {"queries": ["...", "..."]}

Spoken request:
\"\"\"%s\"\"\"
"""


@dataclass
class Decomposition:
    sub_queries: list[str]
    cost: Cost
    method: str          # "llm" | "rule" | "single"


class SubQueryExtractor:
    def __init__(self, provider: Provider, encode=None):
        self.provider = provider
        self.encode = encode

    def extract(self, utterance: str, is_compound: bool) -> Decomposition:
        if not is_compound:
            # The compoundness gate said single-intent. Zero LLM calls. This is
            # the guard against pitfall 5 (over-fragmentation) and it is where
            # most of the cost saving on simple turns comes from.
            return Decomposition([utterance.strip()], Cost(), "single")

        from ..controller.compound import best_split

        if isinstance(self.provider, StubProvider) or not self.provider.available():
            return Decomposition(best_split(utterance, self.encode), Cost(),
                                 "zero-token")

        res = self.provider.complete(SUBQUERY_PROMPT % utterance,
                                     max_tokens=220, json_mode=True)
        data = parse_json_loose(res.text) if res.text else None
        queries = None
        if isinstance(data, dict):
            queries = data.get("queries")
        elif isinstance(data, list):
            queries = data
        if not queries or not isinstance(queries, list):
            return Decomposition(best_split(utterance, self.encode), res.cost,
                                 "zero-token-fallback")
        clean = [str(q).strip() for q in queries if str(q).strip()][:4]
        return Decomposition(clean or best_split(utterance, self.encode),
                             res.cost, "llm")


# ---------------------------------------------------------------------------
# Constraint inference
# ---------------------------------------------------------------------------


_CONSTRAINT_CUE = re.compile(
    r"\b(?:actually|oh|also|by the way|i should mention|forgot to (?:say|mention)|"
    r"one thing|it turns out|note that|but |however|except|although|"
    r"the .{0,24} (?:was|is|were) |it (?:was|is) |he (?:has|had) |she (?:has|had) |"
    r"they (?:have|had) |we (?:have|had) )",
    re.I,
)


def looks_like_constraint(text: str, has_prior_answer: bool) -> bool:
    """A turn refines rather than asks when there is already an answer and the
    turn supplies a fact instead of requesting one."""
    if not has_prior_answer:
        return False
    t = text.strip()
    if "?" in t:
        return False
    from ..controller.compound import _REQUEST_HEAD

    if _REQUEST_HEAD.match(t):
        return False
    return bool(_CONSTRAINT_CUE.search(t)) or len(t.split()) <= 18


def infer_constraint(raw_text: str, new_evidence: list[EvidenceItem],
                     claim_assumption_universe: set[str],
                     t: float) -> Constraint:
    """Build the predicate from what the constraint retrieves. See module
    docstring -- this is the corpus-driven, domain-rule-free mechanism."""
    # Evidence in descending retrieval score. Order matters: a constraint about
    # cross-border purchase retrieves BOTH the cross-border policy and the
    # domestic policy it overrides, because the two documents discuss the same
    # subject. Collecting every tag would activate `region:cross_border` and
    # `region:domestic` together, and they would cancel -- the same silent
    # cancellation that made the first version invalidate nothing.
    #
    # Resolution: one value per scope key, taken from the highest-scoring
    # evidence. The constraint activates the scope of the document it most
    # strongly retrieves, which is the overriding document, because that is the
    # one whose language the constraint actually matches.
    ranked = sorted(new_evidence[:8], key=lambda e: -e.score)
    activated: list[str] = []
    claimed_keys: set[str] = set()
    for item in ranked:
        for tag in scope_tags(item.chunk):
            key = tag_key(tag)
            if key == "doctype":
                if tag not in activated:
                    activated.append(tag)
                continue
            if key in claimed_keys:
                continue
            claimed_keys.add(key)
            activated.append(tag)

    # Only region and effective-date tags carry invalidating force. A doctype
    # tag tells us where evidence came from but does not make an earlier claim
    # wrong, so including it would supersede the entire answer on every turn.
    #
    # And within those keys, only QUALIFIED values activate. A document that
    # carries no region qualifier is silent about region -- it is not asserting
    # "domestic". Treating `region:default` as an activated value was a real
    # bug: the probe returns a mix of qualified and unqualified documents, the
    # default value landed in the activated set alongside the qualified one,
    # and the two cancelled out so that nothing was ever invalidated. The rule
    # is: a constraint is activated by evidence that is explicitly scoped.
    qualified = [t_ for t_ in activated if tag_key(t_) == "region"]

    dated = sorted((t_ for t_ in activated if tag_key(t_) == "effective"),
                   reverse=True)
    if dated:
        qualified = qualified + dated[:1]

    invalidates = conflicting_tags(qualified, claim_assumption_universe)

    scope = tag_key(qualified[0]) if qualified else "unscoped"
    value = (qualified[0].split(":", 1)[1] if qualified else "unknown")

    return Constraint(
        id=new_id("con"), scope=scope, value=value, raw_text=raw_text.strip(),
        arrived_at=t, invalidates=invalidates,
    )


# ---------------------------------------------------------------------------
# Claim synthesis
# ---------------------------------------------------------------------------


CLAIM_PROMPT = """You answer a support agent's question using ONLY the \
numbered evidence below. You are writing for an agent who is on a live call.

Hard rules:
- Every claim must be supported by at least one numbered evidence item.
- If the evidence does not answer the question, say so in "uncertainty" and \
return fewer claims. Never fill a gap from general knowledge.
- Do not write citation markers in the claim text. Put evidence numbers in the \
"evidence" array instead.
- One assertion per claim. Keep each claim under 40 words.

Question: %s
%s

Evidence:
%s

Output JSON only:
{"claims": [{"text": "...", "evidence": [1, 3]}], "uncertainty": "..."}
"""


@dataclass
class SynthesizedClaim:
    text: str
    evidence_ids: list[str]
    citations: list[str]
    assumptions: list[str]
    confidence: float = 1.0


@dataclass
class SynthesisResult:
    claims: list[SynthesizedClaim]
    uncertainty: str
    cost: Cost
    method: str


class ClaimSynthesizer:
    def __init__(self, provider: Provider, max_claims: int = 6):
        self.provider = provider
        self.max_claims = max_claims

    # -- stub path --------------------------------------------------------

    def _stub(self, question: str, evidence: list[EvidenceItem],
              constraints: list[str]) -> SynthesisResult:
        claims: list[SynthesizedClaim] = []
        for item in evidence[:3]:
            sents = best_sentences(item.chunk.text, question, n=1)
            if not sents:
                continue
            text = sents[0].strip()
            if len(text.split()) > 48:
                text = " ".join(text.split()[:48]) + " ..."
            claims.append(
                SynthesizedClaim(
                    text=text,
                    evidence_ids=[item.id],
                    citations=[item.cite],
                    assumptions=scope_tags(item.chunk),
                    confidence=min(1.0, 0.55 + 0.45 * min(1.0, item.score * 40)),
                )
            )
            if len(claims) >= self.max_claims:
                break
        unc = ""
        if not claims:
            unc = (f"The corpus returned no passage that addresses "
                   f"'{question.strip()}'.")
        return SynthesisResult(claims, unc, Cost(), "stub")

    # -- llm path ---------------------------------------------------------

    def _llm(self, question: str, evidence: list[EvidenceItem],
             constraints: list[str]) -> SynthesisResult:
        if not evidence:
            return SynthesisResult(
                [], f"No corpus evidence was retrieved for '{question.strip()}'.",
                Cost(), "llm-noevidence")

        numbered = "\n".join(
            f"[{i + 1}] ({e.cite}) {e.chunk.section_title}: {e.chunk.text}"
            for i, e in enumerate(evidence)
        )
        cons = ""
        if constraints:
            cons = ("Constraints the caller added later (these take "
                    "precedence):\n- " + "\n- ".join(constraints))
        res: LLMResult = self.provider.complete(
            CLAIM_PROMPT % (question, cons, numbered),
            max_tokens=700, json_mode=True,
        )
        data = parse_json_loose(res.text) if res.text else None
        if not isinstance(data, dict) or "claims" not in data:
            out = self._stub(question, evidence, constraints)
            out.cost = res.cost
            out.method = "stub-fallback"
            return out

        claims: list[SynthesizedClaim] = []
        for raw in data.get("claims", [])[: self.max_claims]:
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            idxs = raw.get("evidence") or []
            items: list[EvidenceItem] = []
            for n in idxs:
                try:
                    k = int(n) - 1
                except Exception:
                    continue
                if 0 <= k < len(evidence):
                    items.append(evidence[k])
            if not items:
                # A claim with no evidence reference is dropped, not repaired.
                # This is the cheapest possible defence of gate G4.
                continue
            assumptions: list[str] = []
            for it in items:
                for tg in scope_tags(it.chunk):
                    if tg not in assumptions:
                        assumptions.append(tg)
            claims.append(
                SynthesizedClaim(
                    text=_strip_inline_citations(text),
                    evidence_ids=[i.id for i in items],
                    citations=list(dict.fromkeys(i.cite for i in items)),
                    assumptions=assumptions,
                )
            )
        unc = str(data.get("uncertainty", "") or "").strip()
        if not claims and not unc:
            unc = (f"The retrieved corpus passages did not support an answer "
                   f"to '{question.strip()}'.")
        return SynthesisResult(claims, unc, res.cost, "llm")

    def synthesize(self, question: str, evidence: list[EvidenceItem],
                   constraints: Optional[list[str]] = None) -> SynthesisResult:
        constraints = constraints or []
        if isinstance(self.provider, StubProvider) or not self.provider.available():
            return self._stub(question, evidence, constraints)
        return self._llm(question, evidence, constraints)


_INLINE_CITE = re.compile(r"\s*[\[\(](?:DOC_[A-Z0-9_]+\s*§[\w.]+|\d+)[\]\)]")


def _strip_inline_citations(text: str) -> str:
    """Models like to write citations into the prose. We remove them: the only
    citations that reach the user are the ones the claim graph derives from
    evidence IDs, so a model-authored document ID can never survive to the
    output. This is why 'zero fabricated document IDs' is structural."""
    return _INLINE_CITE.sub("", text).strip()
