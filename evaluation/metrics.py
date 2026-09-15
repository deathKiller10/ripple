"""Metrics, including the two Samsung left for us to define.

Every gate in section 5 of the Theme 4 Guide is computed here, plus the two
metrics that no off-the-shelf harness provides:

  PER-INTENT GROUNDED COVERAGE  did the answer address every gold sub-intent
                                with at least one correct citation? Sub-intent
                                starvation is invisible to recall@k, because
                                each individual retrieval looks fine while a
                                whole sub-answer is missing from the output.

  STATE CONTINUITY              of the claims a late constraint does NOT
                                logically affect, what fraction survived with
                                byte-identical text AND an identical citation
                                set? This is our numeric reading of gate G5,
                                which the guide states only as "verified state
                                continuity".

An honesty note that belongs in the evaluation report, not just in code:
groundedness measured against the STUB provider is close to meaningless,
because the stub answers by copying sentences out of the chunk it cites, so a
claim is grounded by construction. The stub exists for gate G1 (a container
that runs with no API key), not to flatter gate G4. Groundedness numbers worth
reporting come from a run with a real provider, and the report must say which
provider produced them.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

_CITE = re.compile(r"[A-Z][A-Z0-9_]*\s*§[\w.]+")


def norm_cite(c: str) -> str:
    return re.sub(r"\s+", " ", c).strip()


def doc_of(cite: str) -> str:
    return norm_cite(cite).split("§")[0].strip()


# ---------------------------------------------------------------------------


@dataclass
class GateResults:
    early_retrieval_rate: float = 0.0
    early_eligible: int = 0
    early_hits: int = 0

    false_trigger_rate: float = 0.0
    no_retrieval_turns: int = 0
    false_triggers: int = 0

    multi_intent_accuracy: float = 0.0
    compound_turns: int = 0
    multi_intent_hits: int = 0
    over_fragmentation_rate: float = 0.0

    citation_support: float = 0.0
    fabricated_citations: int = 0
    total_citations: int = 0

    state_continuity: float = 1.0
    continuity_turns: int = 0

    trace_coverage: float = 0.0

    recall_at_k: float = 0.0
    per_intent_coverage: float = 0.0

    abstention_precision: float = 0.0
    abstention_recall: float = 0.0

    ttft_median: float | None = None
    ttft_p90: float | None = None
    negative_ttft_rate: float = 0.0

    retrievals_per_turn: float = 0.0
    llm_calls_per_turn: float = 0.0
    tokens_per_turn: float = 0.0
    cost_per_turn: float = 0.0
    delta_efficiency: float = 0.0

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def evaluate(runs: list, scenarios: dict, corpus_cites: set[str],
             trace_coverage: float = 0.0) -> GateResults:
    """`runs` are SystemRun objects for ONE system across all scenarios."""
    g = GateResults()
    ttfts: list[float] = []
    recalls: list[float] = []
    coverages: list[float] = []
    continuities: list[float] = []
    delta_ratios: list[float] = []
    over_frag = 0
    n_turns = 0
    abst_tp = abst_fp = abst_fn = 0

    for run in runs:
        sc = scenarios[run.scenario_id]
        gold_by_idx = {gt.utterance_index: gt for gt in sc.gold}

        for idx, turn in enumerate(run.turns):
            n_turns += 1
            gold = gold_by_idx.get(idx)

            # -- cost ---------------------------------------------------
            g.retrievals_per_turn += turn.retrievals
            g.llm_calls_per_turn += turn.llm_calls
            g.tokens_per_turn += turn.prompt_tokens + turn.completion_tokens
            g.cost_per_turn += turn.currency_cost

            # -- citations (G4) -----------------------------------------
            for c in turn.citations:
                g.total_citations += 1
                if norm_cite(c) not in corpus_cites:
                    g.fabricated_citations += 1

            if gold is None:
                continue

            # -- G2 -----------------------------------------------------
            if gold.needs_retrieval:
                g.early_eligible += 1
                if turn.fired_before_end:
                    g.early_hits += 1
            else:
                g.no_retrieval_turns += 1
                if turn.retrievals > 0 or turn.fired_before_end:
                    g.false_triggers += 1

            # -- G3 -----------------------------------------------------
            n_gold_intents = len(gold.gold_sub_intents)
            if n_gold_intents >= 2:
                g.compound_turns += 1
                if len(turn.sub_queries) >= 2:
                    g.multi_intent_hits += 1
            elif n_gold_intents == 1 and len(turn.sub_queries) > 1:
                # Splitting a single-intent question is pitfall 5.
                over_frag += 1

            # -- retrieval recall and per-intent coverage ---------------
            if gold.gold_doc_ids:
                answered = {norm_cite(c) for c in turn.citations}
                retrieved = {norm_cite(c) for c in
                             (turn.retrieved_cites or turn.citations)}
                hit_docs = 0
                covered_intents = 0
                total_gold = 0
                for intent, cites in gold.gold_doc_ids.items():
                    want = {norm_cite(c) for c in cites}
                    total_gold += len(want)
                    hit_docs += len(want & retrieved)
                    if want & answered:
                        covered_intents += 1
                if total_gold:
                    recalls.append(hit_docs / total_gold)
                coverages.append(covered_intents / max(1, len(gold.gold_doc_ids)))

            # -- abstention ---------------------------------------------
            abstained = bool(turn.uncertainty) and not turn.citations
            if gold.expect_uncertainty:
                if abstained or turn.uncertainty:
                    abst_tp += 1
                else:
                    abst_fn += 1
            elif abstained:
                abst_fp += 1

            # -- G5 -----------------------------------------------------
            if gold.turn_kind == "refinement":
                cs = turn.change_summary or {}
                preserved = cs.get("preserved", 0)
                drifted = cs.get("citations_changed_on_preserved", 0)
                if preserved or drifted:
                    continuities.append(
                        (preserved - drifted) / max(1, preserved))
                    g.continuity_turns += 1
                # delta efficiency: retrievals used vs a full restart, which
                # would re-query every active intent
                prior = run.turns[idx - 1] if idx else None
                full = max(1, len(prior.sub_queries) if prior else 1)
                delta_ratios.append(min(1.0, turn.retrievals / full))

            # -- TTFT ---------------------------------------------------
            if turn.ttft_rel_end is not None:
                ttfts.append(turn.ttft_rel_end)

    # -- finalise ----------------------------------------------------------
    g.early_retrieval_rate = g.early_hits / max(1, g.early_eligible)
    g.false_trigger_rate = g.false_triggers / max(1, g.no_retrieval_turns)
    g.multi_intent_accuracy = g.multi_intent_hits / max(1, g.compound_turns)
    g.over_fragmentation_rate = over_frag / max(1, n_turns)
    g.citation_support = (1.0 - g.fabricated_citations /
                          max(1, g.total_citations))
    g.recall_at_k = statistics.fmean(recalls) if recalls else 0.0
    g.per_intent_coverage = statistics.fmean(coverages) if coverages else 0.0
    g.state_continuity = statistics.fmean(continuities) if continuities else 1.0
    g.delta_efficiency = (statistics.fmean(delta_ratios)
                          if delta_ratios else 0.0)
    g.abstention_precision = abst_tp / max(1, abst_tp + abst_fp)
    g.abstention_recall = abst_tp / max(1, abst_tp + abst_fn)
    g.trace_coverage = trace_coverage

    if ttfts:
        g.ttft_median = statistics.median(ttfts)
        g.ttft_p90 = sorted(ttfts)[max(0, int(0.9 * len(ttfts)) - 1)]
        g.negative_ttft_rate = sum(1 for x in ttfts if x < 0) / len(ttfts)

    n = max(1, n_turns)
    g.retrievals_per_turn /= n
    g.llm_calls_per_turn /= n
    g.tokens_per_turn /= n
    g.cost_per_turn /= n
    return g


GATE_TARGETS = {
    "G2 early retrieval": ("early_retrieval_rate", 0.80, ">="),
    "G3 multi-intent identification": ("multi_intent_accuracy", 0.70, ">="),
    "G4 citation support": ("citation_support", 0.85, ">="),
    "G4 fabricated citations": ("fabricated_citations", 0, "=="),
    "G5 state continuity": ("state_continuity", 1.00, ">="),
    "G6 telemetry trace coverage": ("trace_coverage", 1.00, ">="),
}


def gate_report(results: GateResults) -> list[dict]:
    out = []
    d = results.to_dict()
    for name, (field_name, target, op) in GATE_TARGETS.items():
        value = d.get(field_name, 0)
        if op == ">=":
            ok = value >= target
        else:
            ok = value == target
        out.append({"gate": name, "value": value, "target": target,
                    "op": op, "pass": ok})
    return out
