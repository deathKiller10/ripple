"""Embedding backends.

Two implementations, chosen by RIPPLE_EMBEDDER:

  tfidf-svd   TF-IDF over word and character n-grams, reduced with truncated
              SVD and L2-normalised. Pure scikit-learn: no torch, no model
              download, no network. This is the DEFAULT because gate G1 says
              the container must launch on a clean machine with one command --
              a 2 GB torch wheel and a HuggingFace download at first run is
              exactly how teams fail that gate. It is a real dense embedding
              (LSA), not a placeholder.

  bge-small   BAAI/bge-small-en-v1.5 via sentence-transformers. Better
              retrieval quality; used for the headline numbers. The Dockerfile
              bakes the weights into the image so it still needs no network at
              run time.

Both satisfy the same protocol, so the controller's stability signal, the
index and the evaluation harness are indifferent to which is active. Ablation
A6 in the evaluation report is exactly the delta between them.
"""

from __future__ import annotations

import os
import pickle
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class TfidfSvdEmbedder:
    """Latent semantic embedding fitted on the corpus itself.

    Fitting on the corpus is legitimate and in fact desirable here: the corpus
    is the entire permitted knowledge source (corpus isolation), so a
    representation derived from it introduces no outside knowledge. It also
    makes the embedding adapt to a swapped-in corpus automatically.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._word = None
        self._char = None
        self._svd = None
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfSvdEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._word = TfidfVectorizer(
            lowercase=True, sublinear_tf=True, ngram_range=(1, 2),
            min_df=1, max_df=0.85, stop_words="english",
        )
        # Character n-grams catch part numbers, build strings and model codes
        # (GH82-S24-DA1, DSP-114) that word tokenisation destroys. In a support
        # corpus these are exactly the high-value query terms.
        self._char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True,
        )
        import scipy.sparse as sp

        Xw = self._word.fit_transform(texts)
        Xc = self._char.fit_transform(texts)
        X = sp.hstack([Xw, Xc]).tocsr()
        n_comp = min(self.dim, X.shape[0] - 1, X.shape[1] - 1)
        n_comp = max(2, n_comp)
        self._svd = TruncatedSVD(n_components=n_comp, random_state=0)
        self._svd.fit(X)
        self.dim = n_comp
        self._fitted = True
        return self

    def _raw(self, texts: list[str]):
        import scipy.sparse as sp

        Xw = self._word.transform(texts)
        Xc = self._char.transform(texts)
        return sp.hstack([Xw, Xc]).tocsr()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfSvdEmbedder.fit() must run before encode()")
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        V = self._svd.transform(self._raw(texts)).astype("float32")
        return _l2(V)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {"word": self._word, "char": self._char, "svd": self._svd,
                 "dim": self.dim}, fh)

    @staticmethod
    def load(path: str) -> "TfidfSvdEmbedder":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        e = TfidfSvdEmbedder(dim=d["dim"])
        e._word, e._char, e._svd, e._fitted = d["word"], d["char"], d["svd"], True
        return e


class SentenceTransformerEmbedder:
    """bge-small-en-v1.5. Query prefixing follows the model card."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model_name, device="cpu")
        self.dim = self._m.get_sentence_embedding_dimension()
        self.model_name = model_name

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        return self  # pre-trained; nothing to fit

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        if is_query:
            texts = [
                "Represent this sentence for searching relevant passages: " + t
                for t in texts
            ]
        v = self._m.encode(texts, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
        return v.astype("float32")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"model_name": self.model_name}, fh)

    @staticmethod
    def load(path: str) -> "SentenceTransformerEmbedder":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return SentenceTransformerEmbedder(d["model_name"])


def build_embedder(kind: str, corpus_texts: list[str] | None = None):
    kind = (kind or "tfidf-svd").lower()
    if kind in ("tfidf-svd", "tfidf", "lsa"):
        e = TfidfSvdEmbedder()
        if corpus_texts:
            e.fit(corpus_texts)
        return e
    if kind in ("bge-small", "bge", "st", "sentence-transformers"):
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embedder {kind!r}. Use 'tfidf-svd' or 'bge-small'.")


def load_embedder(kind: str, path: str):
    kind = (kind or "tfidf-svd").lower()
    if kind in ("tfidf-svd", "tfidf", "lsa"):
        return TfidfSvdEmbedder.load(path)
    return SentenceTransformerEmbedder.load(path)
