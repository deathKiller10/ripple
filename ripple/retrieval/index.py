"""Hybrid index: FAISS flat inner-product + BM25, merged with RRF.

Why a flat (exact) FAISS index rather than HNSW or IVF: the corpus is a few
hundred to a few thousand sections. Exact search over that is sub-millisecond,
and an approximate index would add tuning surface and a recall risk for no
measurable gain. Choosing the simpler structure is a defensible engineering
answer under the guide's parsimony rule, not a shortcut -- and it matters
doubly here because the controller runs a shadow retrieval on *every*
transcript chunk, so retrieval latency sits on the hot path.

Why hybrid: dense retrieval smears exact identifiers. A customer saying
"GH82-S24-DA1" or an agent saying "DSP-114" needs lexical matching. BM25
catches those; the dense side catches paraphrase. RRF merges them without
needing score calibration between two incomparable scales.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from dataclasses import dataclass

import numpy as np

from ..schemas import Chunk
from .embedders import build_embedder, load_embedder

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    source: str = "hybrid"     # "dense" | "sparse" | "hybrid"

    @property
    def cite(self) -> str:
        return self.chunk.cite


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder, matrix: np.ndarray,
                 bm25, faiss_index=None):
        self.chunks = chunks
        self.embedder = embedder
        self.matrix = matrix
        self.bm25 = bm25
        self._faiss = faiss_index
        self.by_id = {c.chunk_id: c for c in chunks}
        self.by_cite = {c.cite: c for c in chunks}
        # retrieval call counter -- feeds the cost ledger
        self.calls = 0
        self.shadow_calls = 0

    # -- construction -----------------------------------------------------

    @staticmethod
    def build(chunks: list[Chunk], embedder_kind: str = "tfidf-svd"):
        from rank_bm25 import BM25Okapi

        texts = [f"{c.doc_title} {c.section_title} {c.text}" for c in chunks]
        embedder = build_embedder(embedder_kind, corpus_texts=texts)
        if hasattr(embedder, "encode"):
            matrix = embedder.encode(texts)
        matrix = np.ascontiguousarray(matrix.astype("float32"))

        faiss_index = None
        try:
            import faiss

            faiss_index = faiss.IndexFlatIP(matrix.shape[1])
            faiss_index.add(matrix)
        except Exception:
            faiss_index = None  # numpy fallback keeps the engine runnable

        bm25 = BM25Okapi([tokenize(t) for t in texts])
        return HybridIndex(chunks, embedder, matrix, bm25, faiss_index)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "chunks.jsonl"), "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(os.path.join(path, "matrix.npy"), self.matrix)
        self.embedder.save(os.path.join(path, "embedder.pkl"))
        with open(os.path.join(path, "bm25.pkl"), "wb") as fh:
            pickle.dump(self.bm25, fh)

    @staticmethod
    def load(path: str, embedder_kind: str = "tfidf-svd") -> "HybridIndex":
        chunks = []
        with open(os.path.join(path, "chunks.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                d.pop("cite", None)
                chunks.append(Chunk(**d))
        matrix = np.load(os.path.join(path, "matrix.npy"))
        embedder = load_embedder(embedder_kind, os.path.join(path, "embedder.pkl"))
        with open(os.path.join(path, "bm25.pkl"), "rb") as fh:
            bm25 = pickle.load(fh)
        faiss_index = None
        try:
            import faiss

            faiss_index = faiss.IndexFlatIP(matrix.shape[1])
            faiss_index.add(np.ascontiguousarray(matrix.astype("float32")))
        except Exception:
            pass
        return HybridIndex(chunks, embedder, matrix, bm25, faiss_index)

    # -- encoding ---------------------------------------------------------

    def encode_query(self, text: str) -> np.ndarray:
        enc = self.embedder.encode
        try:
            v = enc([text], is_query=True)     # type: ignore[call-arg]
        except TypeError:
            v = enc([text])
        return v[0]

    # -- search -----------------------------------------------------------

    def dense(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        q = np.ascontiguousarray(qvec.reshape(1, -1).astype("float32"))
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, min(k, len(self.chunks)))
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        sims = (self.matrix @ q[0])
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]

    def sparse(self, text: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(text))
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def shadow(self, text: str, k: int) -> list[str]:
        """Dense-only, IDs-only lookup used by the controller on every chunk.

        Deliberately cheap: no BM25, no reranking, no object construction
        beyond a list of ids. This is the operation whose cost lets us claim
        the controller is free.
        """
        self.shadow_calls += 1
        qv = self.encode_query(text)
        return [self.chunks[i].chunk_id for i, _ in self.dense(qv, k)]

    def search(self, text: str, k: int = 20, rrf_k: int = 60,
               dense_weight: float = 0.5,
               dense_only: bool = False) -> list[Hit]:
        """Full hybrid retrieval for one sub-query.

        `dense_only` is ablation A4: drop the BM25 half. Everything else --
        depth, fusion constant, reranking downstream -- is untouched, so the
        measured difference is attributable to lexical matching alone.
        """
        self.calls += 1
        depth = max(k * 2, 30)
        qv = self.encode_query(text)
        d_hits = self.dense(qv, depth)
        s_hits = [] if dense_only else self.sparse(text, depth)

        fused: dict[int, float] = {}
        dw = 1.0 if dense_only else dense_weight
        for rank, (i, _) in enumerate(d_hits):
            fused[i] = fused.get(i, 0.0) + dw / (rrf_k + rank + 1)
        for rank, (i, _) in enumerate(s_hits):
            fused[i] = fused.get(i, 0.0) + (1 - dense_weight) / (rrf_k + rank + 1)

        d_ids = {i for i, _ in d_hits}
        s_ids = {i for i, _ in s_hits}
        order = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        out = []
        for rank, (i, score) in enumerate(order):
            src = "hybrid" if i in d_ids and i in s_ids else (
                "dense" if i in d_ids else "sparse")
            out.append(Hit(chunk=self.chunks[i], score=float(score), rank=rank,
                           source=src))
        return out

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "documents": len({c.doc_id for c in self.chunks}),
            "dim": int(self.matrix.shape[1]),
            "faiss": self._faiss is not None,
            "retrieval_calls": self.calls,
            "shadow_calls": self.shadow_calls,
        }


def build_and_save(corpus_path: str, index_path: str,
                   embedder_kind: str = "tfidf-svd") -> HybridIndex:
    from ..corpus.loader import corpus_stats, load_corpus

    t0 = time.perf_counter()
    chunks = load_corpus(corpus_path)
    idx = HybridIndex.build(chunks, embedder_kind)
    idx.save(index_path)
    stats = corpus_stats(chunks)
    stats["build_seconds"] = round(time.perf_counter() - t0, 2)
    stats["embedder"] = embedder_kind
    stats["dim"] = int(idx.matrix.shape[1])
    with open(os.path.join(index_path, "manifest.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    return idx
