"""Single source of tunable constants.

Every threshold that appears in a paper-style claim ("we retrieve at theta =
0.68") lives here and nowhere else, so that evaluation/calibrate.py can sweep
it and the README can point at one file. Nothing in the engine may hard-code a
threshold inline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass
class ControllerConfig:
    # --- M1: retrieval-space stability -------------------------------------
    # theta: stability above which the prefix is considered retrieval-stable.
    # NOT hand-tuned. Derived by evaluation/calibrate.py from the asymmetric
    # cost curve; see docs/architecture-brief.md section 3.
    theta: float = field(default_factory=lambda: _f("RIPPLE_THETA", 0.40))

    # nu: novelty floor. Even at high stability we do not re-retrieve if the
    # shadow result set is already in the pool. This is the budget controller.
    nu: float = field(default_factory=lambda: _f("RIPPLE_NU", 0.30))

    # EMA smoothing on the stability signal. Higher alpha = more reactive.
    ema_alpha: float = field(default_factory=lambda: _f("RIPPLE_EMA_ALPHA", 0.55))

    # RBO persistence parameter. 0.9 weights the top of the ranking heavily,
    # which is what we care about: a change at rank 1 matters, rank 9 does not.
    rbo_p: float = field(default_factory=lambda: _f("RIPPLE_RBO_P", 0.90))

    # Shadow retrieval depth. Deliberately small -- this runs on every chunk.
    shadow_k: int = field(default_factory=lambda: _i("RIPPLE_SHADOW_K", 10))

    # Minimum prefix length (words) before any retrieval may fire. Guards
    # against pitfall 1 (eager retrieval on noise) at near-zero cost.
    min_prefix_words: int = field(default_factory=lambda: _i("RIPPLE_MIN_WORDS", 4))

    # A new intent branch is declared when the *new* text since the last
    # trigger is this dissimilar from every active intent.
    jump_similarity: float = field(default_factory=lambda: _f("RIPPLE_JUMP_SIM", 0.42))

    # --- ablation switch A1 -------------------------------------------------
    # Retrieve on every chunk, bypassing the stability policy entirely. This
    # is the pitfall the guide names first; having it as a switch means the
    # ablation isolates the CONTROLLER and nothing else, unlike comparing
    # against the naive-streaming baseline, which also changes state handling.
    always_retrieve: bool = field(
        default_factory=lambda: os.environ.get("RIPPLE_ALWAYS_RETRIEVE") == "1")

    # Cost asymmetry used by the calibration sweep. A wasted shadow retrieval
    # costs ~4ms of CPU; a late retrieval costs ~700ms of user-visible silence.
    c_waste: float = field(default_factory=lambda: _f("RIPPLE_C_WASTE", 1.0))
    c_late: float = field(default_factory=lambda: _f("RIPPLE_C_LATE", 175.0))


@dataclass
class RetrievalConfig:
    top_k: int = field(default_factory=lambda: _i("RIPPLE_TOP_K", 20))
    rerank_candidates: int = field(default_factory=lambda: _i("RIPPLE_RERANK_N", 30))
    rrf_k: int = field(default_factory=lambda: _i("RIPPLE_RRF_K", 60))
    dense_weight: float = field(default_factory=lambda: _f("RIPPLE_DENSE_W", 0.5))
    use_reranker: bool = field(
        default_factory=lambda: os.environ.get("RIPPLE_RERANK", "1") == "1"
    )

    # --- M3: coverage-budgeted fusion --------------------------------------
    # Total chunks handed to synthesis.
    context_budget: int = field(default_factory=lambda: _i("RIPPLE_BUDGET", 12))
    # Guaranteed floor per active intent. This is the anti-starvation term.
    per_intent_floor: int = field(default_factory=lambda: _i("RIPPLE_FLOOR", 2))
    # MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
    mmr_lambda: float = field(default_factory=lambda: _f("RIPPLE_MMR", 0.72))

    # --- ablation switches --------------------------------------------------
    # A2: "coverage" = per-intent floor + marginal gain + MMR (M3);
    #     "rrf" = plain pooled reciprocal rank fusion across sub-queries.
    fusion: str = field(
        default_factory=lambda: os.environ.get("RIPPLE_FUSION", "coverage"))
    # A4: the sparse half of the hybrid.
    #
    # DEFAULT DEPENDS ON THE EMBEDDER, because the measurement says it should.
    # `tfidf-svd` is built on word AND character n-grams, so it already matches
    # exact identifiers -- part numbers, diagnostic codes, firmware builds.
    # Measured on the dev split, adding BM25 on top of it LOWERED recall from
    # 0.960 to 0.893 and won nothing on a 16-query identifier probe (16/16
    # hit@1 either way): two correlated lexical signals competing for the same
    # slots, with the noisier one displacing good hits. So BM25 is off by
    # default here, and one dependency leaves the hot path.
    #
    # `bge-small` is a purely semantic embedder with no character n-grams and
    # no defence against an unseen part number, so the two signals are genuinely
    # complementary there and the hybrid is kept. Re-run ablation A4 after any
    # embedder change rather than assuming this still holds.
    dense_only: bool = field(default_factory=lambda: (
        os.environ.get("RIPPLE_DENSE_ONLY",
                       "1" if os.environ.get("RIPPLE_EMBEDDER", "tfidf-svd")
                       in ("tfidf-svd", "tfidf", "lsa") else "0") == "1"))
    # A5: reranker backend -- "none" | "lexical" | "cross-encoder" | "auto".
    #
    # DEFAULT IS OFF, because the measurement said so and we believed it.
    # Our lexical-semantic reranker was built to re-sort the final candidate
    # set. Measured on the dev split it LOWERED per-intent coverage from 0.736
    # to 0.689, left recall and time-to-first-token unchanged, and cost about
    # 90 ms per sub-query. It re-sorted a fused ranking that was already
    # better than its own scoring function, so it could only do harm.
    #
    # Removing a stage we built is the parsimony rule applied to ourselves:
    # the guide grades cost-to-performance, and a component that costs latency
    # and returns nothing is exactly what that grade is for.
    #
    # `cross-encoder` (ms-marco-MiniLM, needs torch) is a genuinely different
    # quality tier and remains available. It has NOT been measured here, so it
    # is not the default and no claim is made for it -- re-run ablation A5
    # before switching it on.
    reranker: str = field(
        default_factory=lambda: os.environ.get("RIPPLE_RERANKER", "none"))


@dataclass
class SynthesisConfig:
    provider: str = field(
        default_factory=lambda: os.environ.get("RIPPLE_PROVIDER", "stub")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("RIPPLE_MODEL", "gemini-2.0-flash")
    )
    max_claims: int = field(default_factory=lambda: _i("RIPPLE_MAX_CLAIMS", 6))
    # Grounding verifier: minimum lexical overlap between a claim and the chunk
    # it cites before the claim is accepted.
    grounding_min_overlap: float = field(
        default_factory=lambda: _f("RIPPLE_GROUND_MIN", 0.18)
    )
    # Free-tier friendly: requests per minute ceiling, enforced client-side.
    rpm_limit: int = field(default_factory=lambda: _i("RIPPLE_RPM", 12))


@dataclass
class CostTable:
    """Prices are config, not code, so the reported cost-per-turn survives a
    price change. Defaults are Gemini 2.0 Flash free-tier-equivalent list
    prices in INR per 1M tokens as of Sep 2026."""

    prompt_per_1m: float = field(default_factory=lambda: _f("RIPPLE_PRICE_IN", 8.3))
    completion_per_1m: float = field(default_factory=lambda: _f("RIPPLE_PRICE_OUT", 33.2))

    def compute(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.prompt_per_1m / 1_000_000
            + completion_tokens * self.completion_per_1m / 1_000_000
        )


@dataclass
class Config:
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    cost: CostTable = field(default_factory=CostTable)

    embedder: str = field(
        default_factory=lambda: os.environ.get("RIPPLE_EMBEDDER", "tfidf-svd")
    )
    corpus_path: str = field(
        default_factory=lambda: os.environ.get("RIPPLE_CORPUS", "data/corpus/care")
    )
    index_path: str = field(
        default_factory=lambda: os.environ.get("RIPPLE_INDEX", ".index")
    )
    telemetry_path: str = field(
        default_factory=lambda: os.environ.get("RIPPLE_TELEMETRY", "traces")
    )
    session_ttl_s: int = field(default_factory=lambda: _i("RIPPLE_TTL", 1800))

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = Config()
