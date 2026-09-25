"""Abstention: a cheap retrieval-side pre-filter, and why it is only that.

Section 1 of the Theme 4 Guide requires "explicit uncertainty indicators when
evidence is insufficient", and our corpus contains deliberate coverage holes to
force the case. Getting this right took three attempts, two of which failed.
The whole sequence is recorded because the conclusion — *where* abstention
belongs in the pipeline — is the useful part, and because the evaluation report
reports negative results rather than only the final design.

ATTEMPT 1 — DISTRIBUTION SHAPE. Score the top result against the tail of its
own ranking; a covered query should peak, an uncovered one should be flat.
Measured on the dev split, the reverse happened. Asking about screen-protector
reimbursement — a deliberate hole — produced the *highest* standout of any
query tested (z = 12.8), because the corpus contains a section titled "Screen
protection accessories" that matches the query's surface form almost perfectly
while answering nothing. Peakedness measures how distinctive the best match is,
not whether it answers anything.

ATTEMPT 2 — CORPUS VOCABULARY. Count the share of the question's content words
that appear nowhere in the corpus. This separated cleanly at 21 documents and
was adopted. It then broke the moment the corpus grew to 60 documents: the
vocabulary went from 658 words to 1100, out-of-vocabulary rates fell across the
board, and holes started looking covered. A threshold fitted to one corpus size
does not survive another — which is disqualifying here, because the guide's
benchmark replay is held-out and private and may use a corpus we have never
seen.

ATTEMPT 3 — WHAT WE DO NOW, AND WHERE. The distinguishing judgement for
"does the corpus have a passage about *reimbursing* screen protectors, given it
clearly has one about screen protectors" is semantic entailment. No cheap
lexical proxy generalises to it; we tried two and measured both failing. So
abstention is decided in two places, with the weight on the second:

  1. THIS MODULE — a deliberately CONSERVATIVE pre-filter. It fires only when
     almost none of the question's topical content appears in any single
     retrieved passage, which catches wholly out-of-domain questions
     (insurance procedure, financing) at essentially zero false-positive rate.
     It is a cheap first pass, not the mechanism. It will miss near-miss holes
     and that is intended: a wrong abstention costs a real answer, while a
     missed one is caught downstream.

  2. THE SYNTHESISER AND VERIFIER — the real mechanism. The prompt instructs
     the model to decline when the evidence does not answer, and
     `synthesis/verifier.py` then rejects any claim its cited chunk does not
     support. Together these make abstention a judgement about *meaning*,
     verified structurally.

KNOWN LIMITATION, stated plainly because the evaluation report states it: the
keyless `stub` provider cannot abstain on near-miss holes. It answers by
copying the best-matching sentence, so a passage that is topically adjacent but
unresponsive will be returned as if it were an answer. The stub exists to make
gate G1 achievable on a machine with no API key, not to produce quality
numbers. Abstention figures worth reporting come from a real provider run.

THE MEASURE. For each retrieved chunk, what share of the question's *topical*
terms appear in that one chunk — then take the maximum. Co-occurrence in a
single passage is the point: a corpus can contain "screen protector" in one
document and "reimbursement" in another and still have no passage that answers
a question about both. Light verbs and generic nouns are excluded, because
"how long does a display repair take" is a question about displays and repairs,
not about taking.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..synthesis.verifier import content_words

_SUFFIXES = ("ements", "ement", "ations", "ation", "ings", "ing", "ers",
             "ies", "ed", "es", "ly", "al", "s")

# Words that carry no topical content in a support question. Excluding them is
# what lets "how long does a display repair take" be recognised as a question
# about {display, repair} rather than {display, repair, long, take}. This list
# is generic English, not corpus-specific, so it transfers with the code.
_LIGHT = set("""
take takes taken taking long give gives given get gets got getting make makes
made want wants know knows need needs tell tells say says said ask asks come
comes go goes going put puts use uses used using help helps offer offers
offered available possible able thing things way ways lot bit kind sort
actually really please maybe okay yes sure right well back still even again
much many more less good bad better best new old big small little happen
happens work works working look looks looking find finds see sees show shows
mean means like likes itself himself herself themselves myself against over
under about between during through within without across around toward towards
into onto upon per via plus minus customer customers phone device devices
handset unit units something anything nothing everything someone anyone
""".split())


def stem(word: str) -> str:
    """Deliberately crude suffix stripping.

    A real stemmer would be better but adds a dependency for a signal this
    coarse. The only job is to stop "reimburse" and "reimbursement" counting as
    different concepts.
    """
    for suf in _SUFFIXES:
        if len(word) > len(suf) + 3 and word.endswith(suf):
            return word[: -len(suf)]
    return word


def topical_terms(text: str) -> set[str]:
    """Content words with light verbs and generic nouns removed.

    Hyphenated tokens contribute BOTH the whole form and its parts. Without
    this, a question about "the walk in turnaround" never matched a passage
    about "walk-in repairs", and the abstention gate declared a well-covered
    question uncoverable. Writers hyphenate inconsistently and speakers do not
    hyphenate at all, so the two forms have to be the same concept.
    """
    out = set()
    for w in content_words(text):
        forms = [w] + (w.split("-") if "-" in w else [])
        for f in forms:
            if len(f) < 3:
                continue
            st = stem(f)
            if st in _LIGHT or f in _LIGHT:
                continue
            out.add(st)
    return out


def build_vocabulary(chunks) -> set[str]:
    """Kept for the telemetry payload and for the evaluation report; no longer
    used as a decision signal (see ATTEMPT 2 above)."""
    vocab: set[str] = set()
    for c in chunks:
        for w in content_words(f"{c.doc_title} {c.section_title} {c.text}"):
            vocab.add(stem(w))
    return vocab


@dataclass
class RelevanceVerdict:
    max_passage_coverage: float
    best_cite: str
    query_terms: list[str]
    unmatched_terms: list[str]
    sufficient: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "max_passage_coverage": round(self.max_passage_coverage, 4),
            "best_cite": self.best_cite,
            "query_terms": list(self.query_terms),
            "unmatched_terms": list(self.unmatched_terms),
            "sufficient": self.sufficient,
            "reason": self.reason,
        }


def assess_question(query: str, top_chunks, split_fn=None,
                    **kw) -> RelevanceVerdict:
    """Assess a question, splitting it first if it is compound.

    The single-passage measure is only meaningful for a SINGLE question. Asked
    "bluetooth keeps cutting out and wifi drops at home", no one passage covers
    both -- the corpus answers each in its own article -- and the gate declared
    a fully-covered compound question uncoverable. That is the precise
    situation decomposition exists to resolve, so the gate must see the parts,
    never the whole.

    Returns the most favourable verdict across the parts: a compound question
    is abstained on only when NO part of it is covered.
    """
    if split_fn is None:
        return assess(query, top_chunks, **kw)
    try:
        parts = split_fn(query)
    except Exception:
        parts = [query]
    if len(parts) <= 1:
        return assess(query, top_chunks, **kw)
    best = None
    for part in parts:
        v = assess(part, top_chunks, **kw)
        if v.sufficient:
            return v
        if best is None or v.max_passage_coverage > best.max_passage_coverage:
            best = v
    return best or assess(query, top_chunks, **kw)


def assess(query: str, top_chunks, vocabulary: set[str] | None = None,
           min_passage_coverage: float = 0.34, top_n: int = 10,
           **_legacy) -> RelevanceVerdict:
    """Conservative pre-filter for ONE question. Use assess_question() when the
    text may be compound. `vocabulary` is accepted and ignored; it is retained
    so callers written against attempt 2 keep working."""
    qw = topical_terms(query)
    if not qw:
        qw = {stem(w) for w in content_words(query)}
    if not qw:
        return RelevanceVerdict(1.0, "", [], [], True,
                                "no topical content to assess")

    best, best_cite, best_missing = 0.0, "", sorted(qw)
    for c in list(top_chunks)[:top_n]:
        cw = topical_terms(f"{c.section_title} {c.text}")
        cov = len(qw & cw) / len(qw)
        if cov > best:
            best, best_cite, best_missing = cov, c.cite, sorted(qw - cw)

    if best < min_passage_coverage:
        return RelevanceVerdict(
            best, best_cite, sorted(qw), best_missing, False,
            f"no retrieved passage addresses this question: the closest "
            f"({best_cite or 'none'}) covers {best:.0%} of its topical terms, "
            f"missing {best_missing}")
    return RelevanceVerdict(
        best, best_cite, sorted(qw), best_missing, True,
        f"{best_cite} covers {best:.0%} of the question's topical terms")
