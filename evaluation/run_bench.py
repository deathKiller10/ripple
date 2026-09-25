#!/usr/bin/env python3
"""Benchmark runner: four systems, one harness, plus the ablations.

    python -m evaluation.run_bench --split dev
    python -m evaluation.run_bench --split heldout --ablations

Everything is held constant across systems except the streaming architecture:
same index, same embedder, same reranker, same provider. A comparison that
changes two things at once measures nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.baselines import RUNNERS, run_b3                 # noqa: E402
from evaluation.metrics import evaluate, gate_report, norm_cite  # noqa: E402
from ripple.config import Config                                 # noqa: E402
from ripple.engine import RippleEngine                           # noqa: E402
from ripple.schemas import EventType, Scenario                   # noqa: E402


def load_scenarios(path: str) -> list[Scenario]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(Scenario.from_dict(json.loads(line)))
    return out


# G6: the set of engine stages that MUST appear in a trace. The coverage number
# is computed, not asserted -- see ripple/telemetry.py `instrumented`.
REQUIRED_STAGES = {
    "on_chunk", "early_retrieve", "end_utterance", "resolve_answer",
}
REQUIRED_EVENTS = {
    EventType.SESSION_STARTED, EventType.CHUNK_RECEIVED,
    EventType.CONTROLLER_DECISION, EventType.COMPOUNDNESS_TESTED,
    EventType.DECOMPOSED, EventType.RETRIEVAL_STARTED,
    EventType.RETRIEVAL_COMPLETED, EventType.FUSION_COMPLETED,
    EventType.POOL_UPDATED, EventType.CLAIM_EMITTED,
    EventType.GROUNDING_CHECKED, EventType.ANSWER_VERSION,
    EventType.UTTERANCE_END, EventType.SESSION_ENDED,
}


def measure_trace_coverage(trace_dir: str) -> float:
    """Fraction of required event types observed across all traces."""
    seen: set[str] = set()
    if not os.path.isdir(trace_dir):
        return 0.0
    for fn in os.listdir(trace_dir):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(trace_dir, fn), encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        seen.add(json.loads(line)["type"])
                    except Exception:
                        pass
    want = {e.value for e in REQUIRED_EVENTS}
    return len(want & seen) / max(1, len(want))


def run_system(name, engine, scenarios, sink_dir=None, **kw):
    runs = []
    t0 = time.perf_counter()
    for sc in scenarios:
        if name == "B3_ripple":
            runs.append(run_b3(engine, sc, sink_dir=sink_dir, **kw))
        else:
            runs.append(RUNNERS[name](engine, sc))
    return runs, time.perf_counter() - t0


def fmt(v, width=9):
    if v is None:
        return " " * (width - 1) + "-"
    if isinstance(v, float):
        return f"{v:>{width}.3f}"
    return f"{v:>{width}}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evaluation.run_bench")
    ap.add_argument("--split", default="dev", choices=["dev", "heldout"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--systems", default="B0_llm_only,B1_static_rag,"
                                         "B2_naive_streaming,B3_ripple")
    args = ap.parse_args(argv)

    cfg = Config()
    if args.provider:
        cfg.synthesis.provider = args.provider
    if args.embedder:
        cfg.embedder = args.embedder

    if not os.path.exists(os.path.join(cfg.index_path, "chunks.jsonl")):
        from ripple.retrieval.index import build_and_save
        build_and_save(cfg.corpus_path, cfg.index_path, cfg.embedder)

    engine = RippleEngine(cfg)
    corpus_cites = {norm_cite(c.cite) for c in engine.index.chunks}
    path = f"data/scenarios/{args.split}.jsonl"
    scenarios = load_scenarios(path)
    by_id = {s.scenario_id: s for s in scenarios}

    print(f"split={args.split}  scenarios={len(scenarios)}  "
          f"provider={engine.provider.name}  embedder={cfg.embedder}  "
          f"reranker={getattr(engine.reranker, 'name', '?')}")
    print()

    os.makedirs(args.out, exist_ok=True)
    trace_dir = os.path.join(args.out, f"traces_{args.split}")
    report: dict = {"split": args.split, "provider": engine.provider.name,
                    "embedder": cfg.embedder,
                    "reranker": getattr(engine.reranker, "name", "?"),
                    "config": cfg.to_dict(), "systems": {}, "ablations": {}}

    rows = []
    for name in args.systems.split(","):
        sink = trace_dir if name == "B3_ripple" else None
        runs, secs = run_system(name, engine, scenarios, sink_dir=sink)
        cov = measure_trace_coverage(trace_dir) if name == "B3_ripple" else 0.0
        res = evaluate(runs, by_id, corpus_cites, trace_coverage=cov)
        report["systems"][name] = res.to_dict()
        report["systems"][name]["wall_seconds"] = round(secs, 2)
        rows.append((name, res))

    # ---------------- comparison table ----------------
    cols = [
        ("early retr", "early_retrieval_rate"),
        ("false trig", "false_trigger_rate"),
        ("multi-int", "multi_intent_accuracy"),
        ("recall@k", "recall_at_k"),
        ("intent cov", "per_intent_coverage"),
        ("cite supp", "citation_support"),
        ("fabricated", "fabricated_citations"),
        ("continuity", "state_continuity"),
        ("TTFT med", "ttft_median"),
        ("retr/turn", "retrievals_per_turn"),
        ("llm/turn", "llm_calls_per_turn"),
        ("tok/turn", "tokens_per_turn"),
    ]
    print(f"{'system':<21}" + "".join(f"{c[0]:>11}" for c in cols))
    print("-" * (21 + 11 * len(cols)))
    for name, res in rows:
        d = res.to_dict()
        line = f"{name:<21}"
        for _, key in cols:
            v = d.get(key)
            line += fmt(v, 11)
        print(line)

    # ---------------- gates ----------------
    print()
    ripple = dict(rows).get("B3_ripple")
    if ripple:
        print("ACCEPTANCE GATES (Theme 4 Guide section 5)")
        print("-" * 62)
        gates = gate_report(ripple)
        report["gates"] = gates
        for row in gates:
            status = "PASS" if row["pass"] else "FAIL"
            print(f"  [{status}] {row['gate']:<34} "
                  f"{row['value']:>8} {row['op']} {row['target']}")

    # ---------------- ablations ----------------
    # Each variant changes EXACTLY ONE thing and shares everything else --
    # same index, same corpus, same provider, same scenarios. A comparison
    # that moves two variables at once measures nothing, which is why these
    # are config switches on the real engine rather than separate baselines.
    if args.ablations:
        print()
        print("ABLATIONS  (each changes one variable; all else held constant)")
        print("-" * 78)

        def variant(**over):
            c = Config()
            if args.provider:
                c.synthesis.provider = args.provider
            if args.embedder:
                c.embedder = args.embedder
            for path, val in over.items():
                group, _, attr = path.partition(".")
                setattr(getattr(c, group), attr, val)
            return c

        variants = {
            "A1  controller OFF (retrieve every chunk)":
                variant(**{"controller.always_retrieve": True}),
            "A2  fusion = plain RRF (no coverage budget)":
                variant(**{"retrieval.fusion": "rrf"}),
            "A3  speculative synthesis OFF":
                ("speculative_off", variant()),
            "A4  add BM25 back (hybrid retrieval)":
                variant(**{"retrieval.dense_only": False}),
            "A5  reranker back ON (lexical-semantic)":
                variant(**{"retrieval.reranker": "lexical"}),
        }

        base = dict(rows).get("B3_ripple")
        cols = [("TTFT med", "ttft_median"), ("early", "early_retrieval_rate"),
                ("recall@k", "recall_at_k"),
                ("intent cov", "per_intent_coverage"),
                ("retr/turn", "retrievals_per_turn"),
                ("tok/turn", "tokens_per_turn")]
        print(f"{'variant':<44}" + "".join(f"{c[0]:>12}" for c in cols))
        if base:
            bd = base.to_dict()
            print(f"{'B3  full system (reference)':<44}"
                  + "".join(fmt(bd.get(k), 12) for _, k in cols))
        print("-" * 78)

        for label, spec in variants.items():
            spec_cfg = spec[1] if isinstance(spec, tuple) else spec
            spec_cfg.index_path = cfg.index_path
            eng = RippleEngine(spec_cfg, index=engine.index)
            speculative = not (isinstance(spec, tuple)
                               and spec[0] == "speculative_off")
            runs = [run_b3(eng, sc, speculative=speculative)
                    for sc in scenarios]
            res = evaluate(runs, by_id, corpus_cites)
            report["ablations"][label] = res.to_dict()
            d = res.to_dict()
            print(f"{label:<44}" + "".join(fmt(d.get(k), 12) for _, k in cols))

    out_path = os.path.join(args.out, f"benchmark_{args.split}.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
