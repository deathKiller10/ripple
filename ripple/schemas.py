"""
Frozen contracts for Ripple.

FROZEN ON DAY 0. Every member codes against this file. Changes require
agreement from all four members because the dashboard, the replay CLI and the
evaluation harness all deserialise these shapes independently.

Three distinct schemas live here:

1. TelemetryEvent  -- our internal observability record (gate G6).
2. OutputRecord    -- the Samsung Theme 4 Guide section 4 structured output
                      event record, emitted VERBATIM in the guide's field
                      names so a held-out private harness written against the
                      guide can consume our output with no adaptation.
3. Scenario        -- the benchmark replay input format.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Literal, Optional
import json
import time
import uuid


# --------------------------------------------------------------------------
# Controller vocabulary
# --------------------------------------------------------------------------


class Decision(str, Enum):
    """The four controller decisions named in the Theme 4 Guide section 2."""

    WAIT = "WAIT"
    RETRIEVE = "RETRIEVE"
    RETRIEVE_MORE = "RETRIEVE_MORE"
    SUPPRESS = "SUPPRESS"


class Trigger(str, Enum):
    """Why a retrieval fired. Mirrors the guide's `trigger` field values."""

    PROVISIONAL = "provisional"   # early, fired on a partial utterance
    MULTI_INTENT = "multi_intent"  # fan-out across decomposed sub-queries
    DELTA = "delta"                # narrow re-query after a late constraint
    FINAL = "final"                # fired at utterance end (baseline behaviour)


class EvidenceState(str, Enum):
    """Lifecycle of a chunk inside the session evidence pool.

    PROVISIONAL  fetched speculatively from a partial utterance, not yet
                 confirmed to belong to a real intent
    ACTIVE       supporting at least one live claim
    SUPERSEDED   replaced by a later, more specific chunk (e.g. a regional
                 policy overriding a base policy)
    CONTRADICTED in conflict with another ACTIVE chunk; both are surfaced
    IRRELEVANT   fetched for an intent that never materialised; evicted from
                 the synthesis context but kept in the trace
    """

    PROVISIONAL = "PROVISIONAL"
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    CONTRADICTED = "CONTRADICTED"
    IRRELEVANT = "IRRELEVANT"


class ClaimStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"


class EventType(str, Enum):
    """Every telemetry event type. G6 requires 100% trace coverage, so this
    enum is the checklist: tests/test_telemetry_coverage.py asserts that every
    public engine method emits at least one of these."""

    SESSION_STARTED = "session_started"
    CHUNK_RECEIVED = "chunk_received"
    CONTROLLER_DECISION = "controller_decision"
    COMPOUNDNESS_TESTED = "compoundness_tested"
    DECOMPOSED = "decomposed"
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    RETRIEVAL_CANCELLED = "retrieval_cancelled"
    FUSION_COMPLETED = "fusion_completed"
    POOL_UPDATED = "pool_updated"
    CONSTRAINT_DETECTED = "constraint_detected"
    CLAIM_EMITTED = "claim_emitted"
    CLAIM_SUPERSEDED = "claim_superseded"
    CLAIM_PRESERVED = "claim_preserved"
    GROUNDING_CHECKED = "grounding_checked"
    UNCERTAINTY_EMITTED = "uncertainty_emitted"
    ANSWER_VERSION = "answer_version"
    FIRST_TOKEN = "first_token"
    UTTERANCE_END = "utterance_end"
    SESSION_ENDED = "session_ended"


# --------------------------------------------------------------------------
# 1. Telemetry  (gate G6)
# --------------------------------------------------------------------------


@dataclass
class ControllerTrace:
    """The controller's reasoning, exposed so the dashboard can draw the
    stability curve and so a judge can audit why a retrieval fired."""

    decision: str
    stability: float
    novelty: float
    drift: float
    threshold: float
    semantic_jump: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Cost:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    embed_calls: int = 0
    retrieval_calls: int = 0
    rerank_calls: int = 0
    currency_cost: float = 0.0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
            embed_calls=self.embed_calls + other.embed_calls,
            retrieval_calls=self.retrieval_calls + other.retrieval_calls,
            rerank_calls=self.rerank_calls + other.rerank_calls,
            currency_cost=self.currency_cost + other.currency_cost,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TelemetryEvent:
    """One line of the JSONL trace. This is the schema documented in
    docs/telemetry-schema.md and shipped as the G6 deliverable."""

    session_id: str
    seq: int
    t_rel_s: float                  # seconds since session start
    type: str                       # EventType value
    wall_clock_ms: float = field(default_factory=lambda: time.time() * 1000)

    # optional payload -- present depending on `type`
    chunk_text: Optional[str] = None
    prefix_text: Optional[str] = None
    controller: Optional[dict] = None
    trigger: Optional[str] = None
    intent_id: Optional[str] = None
    sub_query: Optional[str] = None
    sub_queries: Optional[list[str]] = None
    retrieved: Optional[list[str]] = None      # ["DOC_KB_14 §2", ...]
    rerank_scores: Optional[list[float]] = None
    allocation: Optional[dict] = None          # intent_id -> n chunks granted
    claim_id: Optional[str] = None
    citations: Optional[list[str]] = None
    answer_version: Optional[int] = None
    uncertainty: Optional[str] = None
    cost: Optional[dict] = None
    latency_ms: Optional[float] = None
    detail: Optional[dict] = None

    def to_json(self) -> str:
        d = {k: v for k, v in asdict(self).items() if v is not None}
        return json.dumps(d, ensure_ascii=False)


# --------------------------------------------------------------------------
# 2. Samsung Theme 4 Guide, section 4 output record  -- FIELD NAMES ARE THEIRS
# --------------------------------------------------------------------------


@dataclass
class GuideRetrievalEvent:
    """Exactly the guide's shape:
    { "timestamp_s": 0.8, "query": "...", "trigger": "provisional" }
    """

    timestamp_s: float
    query: str
    trigger: str

    def to_dict(self) -> dict:
        return {
            "timestamp_s": round(self.timestamp_s, 3),
            "query": self.query,
            "trigger": self.trigger,
        }


@dataclass
class OutputRecord:
    """The guide's section 4 "Structured Output Event Record".

    Do NOT rename these fields. A private held-out harness written against the
    guide should be able to parse our replay output directly. Ripple-specific
    extras live under `ripple_ext` so the core record stays exactly as
    specified.
    """

    retrieval_events: list[GuideRetrievalEvent] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    uncertainty: str = ""

    # namespaced extension -- never required by the guide
    ripple_ext: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "retrieval_events": [e.to_dict() for e in self.retrieval_events],
            "sub_queries": list(self.sub_queries),
            "answer": self.answer,
            "citations": list(self.citations),
            "uncertainty": self.uncertainty,
            "ripple_ext": self.ripple_ext,
        }


# --------------------------------------------------------------------------
# 3. Corpus
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    """One retrievable unit. `cite` is the canonical citation string and is the
    ONLY form allowed to appear in an answer."""

    doc_id: str          # "DOC_WAR_01"
    section: str         # "2"
    section_title: str
    text: str
    chunk_id: str        # "DOC_WAR_01#2"
    doc_title: str = ""
    doc_type: str = ""
    effective_from: Optional[str] = None   # ISO date, drives supersession
    supersedes: list[str] = field(default_factory=list)  # chunk_ids
    region: Optional[str] = None

    @property
    def cite(self) -> str:
        return f"{self.doc_id} §{self.section}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cite"] = self.cite
        return d


# --------------------------------------------------------------------------
# 4. Benchmark scenario format
# --------------------------------------------------------------------------


@dataclass
class TranscriptChunk:
    t: float        # seconds from session start
    text: str
    speaker: str = "customer"   # "customer" | "agent"
    is_final: bool = False      # marks end of an utterance


@dataclass
class GoldTurn:
    """Gold labels for one utterance inside a scenario.

    `needs_retrieval` is the G2 eligibility label -- the taxonomy is documented
    in docs/eligibility-taxonomy.md. Without it G2 is unmeasurable, which is
    why we define it explicitly rather than assuming it.
    """

    utterance_index: int
    needs_retrieval: bool
    turn_kind: str                  # "informational" | "refinement" |
                                    # "presentation" | "social" | "uncoverable"
    gold_sub_intents: list[str] = field(default_factory=list)
    gold_doc_ids: dict = field(default_factory=dict)   # sub_intent -> [cite]
    claims_that_must_not_change: list[str] = field(default_factory=list)
    expect_uncertainty: bool = False


@dataclass
class Scenario:
    scenario_id: str
    split: str                      # "dev" | "heldout"
    chunks: list[TranscriptChunk]
    gold: list[GoldTurn]
    notes: str = ""

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        return Scenario(
            scenario_id=d["scenario_id"],
            split=d["split"],
            chunks=[TranscriptChunk(**c) for c in d["chunks"]],
            gold=[GoldTurn(**g) for g in d["gold"]],
            notes=d.get("notes", ""),
        )

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "split": self.split,
            "chunks": [asdict(c) for c in self.chunks],
            "gold": [asdict(g) for g in self.gold],
            "notes": self.notes,
        }


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
