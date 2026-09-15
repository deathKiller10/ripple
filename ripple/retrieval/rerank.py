"""Reranking, applied to at most `rerank_candidates` chunks and never to a
shadow retrieval.

That restriction is the whole cost argument. The controller runs a retrieval on
every transcript chunk; if reranking sat on that path the engine would spend
~90 ms per chunk and the parsimony claim would collapse. Reranking runs once
per sub-query, on the final candidate set only.

Two backends:

  cross-encoder  ms-marco-MiniLM-L-6-v2. Best quality. Needs torch, so it is
                 opt-in and baked into the Docker image rather than downloaded
                 at run time.
  lexical-semantic  Query-term coverage, IDF-weighted, blended with embedding
                 cosine and a small prior for exact identifier matches. No
                 torch, no download. This is the default so that the keyless
                 container path still reranks rather than silently skipping a
                 pipeline stage.

Ablation A5 reports whether the cross-encoder earns its latency. If it does
not, we remove it and say so -- an honest negative result on one component is
worth more than an unexamined dependency.
"""

from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np

from .index import Hit, tokenize


class LexicalSemanticReranker:
    name = "lexical-semantic"

    def __init__(self, index):
        self.index = index
        self._idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        n = len(self.index.chunks)
        df: dict[str, int] = {}
        for c in self.index.chunks:
            for t in set(tokenize(f"{c.doc_title} {c.section_title} {c.text}")):
                df[t] = df.get(t, 0) + 1
        return {t: math.log((n + 1) / (d + 0.5)) for t, d in df.items()}

    def score(self, query: str, hits: list[Hit]) -> list[float]:
        q_terms = [t for t in tokenize(query)]
        q_set = set(q_terms)
        q_weight = sum(self._idf.get(t, 1.0) for t in q_set) or 1.0

        try:
            qv = self.index.encode_query(query)
        except Exception:
            qv = None

        out = []
        for h in hits:
            body = f"{h.chunk.doc_title} {h.chunk.section_title} {h.chunk.text}"
            terms = set(tokenize(body))
            covered = sum(self._idf.get(t, 1.0) for t in (q_set & terms))
            coverage = covered / q_weight

            sem = 0.0
            if qv is not None:
                try:
                    i = self.index.by_id[h.chunk.chunk_id]
                    row = self.index.chunks.index(i)
                    sem = float(np.dot(qv, self.index.matrix[row]))
                except Exception:
                    sem = 0.0

            # Exact identifier match (part numbers, diagnostic codes, doc ids)
            # is a strong signal that dense similarity understates.
            ident = 0.0
            for t in q_set:
                if re.search(r"\d", t) and len(t) >= 4 and t in terms:
                    ident += 0.12
            out.append(float(0.55 * coverage + 0.45 * max(0.0, sem) + min(0.3, ident)))
        return out


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self._m = CrossEncoder(model_name, device="cpu", max_length=384)

    def score(self, query: str, hits: list[Hit]) -> list[float]:
        pairs = [(query, f"{h.chunk.section_title}. {h.chunk.text}") for h in hits]
        raw = self._m.predict(pairs, show_progress_bar=False)
        return [float(1.0 / (1.0 + math.exp(-s))) for s in raw]


class NoOpReranker:
    name = "none"

    def score(self, query: str, hits: list[Hit]) -> list[float]:
        return [h.score for h in hits]


def build_reranker(index, kind: Optional[str] = None):
    kind = (kind or "auto").lower()
    if kind in ("none", "off", "0"):
        return NoOpReranker()
    if kind in ("cross-encoder", "ce"):
        try:
            return CrossEncoderReranker()
        except Exception:
            return LexicalSemanticReranker(index)
    if kind in ("lexical", "lexical-semantic"):
        return LexicalSemanticReranker(index)
    # auto: prefer the cross-encoder when torch is importable
    try:
        import torch  # noqa: F401

        return CrossEncoderReranker()
    except Exception:
        return LexicalSemanticReranker(index)


def apply(reranker, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
    """Score the top `top_n` candidates and reorder. Anything below `top_n` is
    left in its fused order rather than discarded, so recall is unchanged."""
    if not hits:
        return hits
    head, tail = hits[:top_n], hits[top_n:]
    scores = reranker.score(query, head)
    for h, s in zip(head, scores):
        setattr(h, "rerank_score", float(s))
    head.sort(key=lambda h: -getattr(h, "rerank_score", 0.0))
    for i, h in enumerate(head):
        h.rank = i
    for h in tail:
        if not hasattr(h, "rerank_score"):
            setattr(h, "rerank_score", None)
    return head + tail
