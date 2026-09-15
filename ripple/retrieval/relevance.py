"""Abstention gate: does the corpus have anything to say about this question?

Section 1 of the Theme 4 Guide requires "explicit uncertainty indicators when
evidence is insufficient", and our corpus contains a deliberate coverage hole
to force the case. Getting this right turned out to be the least obvious part
of the system, so the reasoning is recorded here in full -- the failed
hypothesis included, because the evaluation report reports both.

WHAT DOES NOT WORK
------------------
*Result count.* A dense index always returns k results. Never fires.

*Absolute similarity threshold.* Does not transfer across embedders, rerankers
or corpora. The guide's benchmark is held-out and private, so a threshold
fitted to our corpus is a threshold fitted to the wrong corpus.

*Distribution shape (z-score of top-1 against the tail).* This was our first
implementation and it failed instructively. We predicted that a covered query
produces a sharp peak and an uncovered one a flat ranking. Measured on the dev
split, the reverse happened: asking about screen-protector reimbursement -- a
deliberate hole -- produced the HIGHEST standout score of any query tested
(z = 12.8), because the corpus contains a section titled "Screen protection
accessories" that matches the query's surface form almost perfectly while
answering nothing. Lexical confidence and topical relevance came apart exactly
where we needed them to agree. Peakedness measures how distinctive the best
match is, not whether it answers anything.

WHAT WORKS
----------
The corpus's vocabulary is the corpus's conceptual coverage. If a question
turns on a concept the corpus has no word for, the corpus cannot answer it,
however well some passage matches the remaining words.

    oov_rate   share of the question's content words (lightly stemmed) that
               appear NOWHERE in the corpus vocabulary
    term_cover share of the question's content words present in the top few
               retrieved chunks

Measured on the dev split: covered questions sit at oov_rate <= 0.25, holes at
0.25 to 0.60. The single overlapping band at 0.25 is resolved by term_cover,
which is low for holes and moderate for covered questions whose missing words
are ordinary verbs ("take", "need").

Both numbers are dimensionless and derived from the corpus at index time, so
they need no retuning when the embedder, the reranker or the corpus changes.
That is the property that matters for a benchmark we are not allowed to see.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..synthesis.verifier import content_words

_SUFFIXES = ("ements", "ement", "ations", "ation", "ings", "ing", "ers",
             "ies", "ed", "es", "ly", "al", "s")


def stem(word: str) -> str:
    """Deliberately crude suffix stripping.

    A real stemmer would be better but adds a dependency for a signal this
    coarse. The only job here is to stop "reimburse" and "reimbursement"
    counting as different concepts.
    """
    for suf in _SUFFIXES:
        if len(word) > len(suf) + 3 and word.endswith(suf):
            return word[: -len(suf)]
    return word


def build_vocabulary(chunks) -> set[str]:
    """Called once at index load. The corpus defines what is knowable."""
    vocab: set[str] = set()
    for c in chunks:
        text = f"{c.doc_title} {c.section_title} {c.text}"
        for w in content_words(text):
            vocab.add(stem(w))
    return vocab


@dataclass
class RelevanceVerdict:
    oov_rate: float
    term_cover: float
    oov_terms: list[str]
    sufficient: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "oov_rate": round(self.oov_rate, 4),
            "term_cover": round(self.term_cover, 4),
            "oov_terms": list(self.oov_terms),
            "sufficient": self.sufficient,
            "reason": self.reason,
        }


def assess(query: str, top_chunks, vocabulary: set[str],
           oov_hard: float = 0.40, oov_soft: float = 0.20,
           cover_floor: float = 0.30, top_n: int = 3) -> RelevanceVerdict:
    qw = {stem(w) for w in content_words(query)}
    if not qw:
        return RelevanceVerdict(0.0, 1.0, [], True, "no content words to assess")

    oov = sorted(w for w in qw if w not in vocabulary)
    oov_rate = len(oov) / len(qw)

    seen: set[str] = set()
    for c in list(top_chunks)[:top_n]:
        seen |= {stem(w) for w in content_words(f"{c.section_title} {c.text}")}
    term_cover = len(qw & seen) / len(qw)

    if oov_rate >= oov_hard:
        return RelevanceVerdict(
            oov_rate, term_cover, oov, False,
            f"the corpus has no vocabulary for {oov} -- these concepts do not "
            f"appear anywhere in it")
    if oov_rate >= oov_soft and term_cover <= cover_floor:
        return RelevanceVerdict(
            oov_rate, term_cover, oov, False,
            f"the corpus has no vocabulary for {oov}, and the retrieved "
            f"passages cover only {term_cover:.0%} of the question")
    return RelevanceVerdict(
        oov_rate, term_cover, oov, True,
        f"corpus vocabulary covers the question (oov={oov_rate:.0%}, "
        f"retrieved terms={term_cover:.0%})")
