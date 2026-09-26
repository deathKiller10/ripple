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


def run_system(name, engine, scenarios, sink_dir=None, progress=True, **kw):
    """Run one system across all scenarios.

    Prints per-scenario progress. An earlier version printed nothing until a
    whole system finished, so a rate-limited run sat silent for ten minutes
    and was indistinguishable from a hang. Silence is not a neutral default
    for anything that takes minutes.
    """
    runs = []
    t0 = time.perf_counter()
    n = len(scenarios)
    for i, sc in enumerate(scenarios, 1):
        if name == "B3_ripple":
            runs.append(run_b3(engine, sc, sink_dir=sink_dir, **kw))
        else:
            runs.append(RUNNERS[name](engine, sc))
        if progress:
            elapsed = time.perf_counter() - t0
            eta = (elapsed / i) * (n - i)
            ok_calls = getattr(engine.provider, "calls_made", None)
            bad = getattr(engine.provider, "calls_failed", 0)
            extra = f"  {ok_calls} llm calls" if ok_calls else ""
            # Show failures too. Reporting only successes is how a run that was
            # failing 94% of its calls looked merely slow.
            if bad:
                extra += f", {bad} FAILED"
                err = getattr(engine.provider, "last_error", "")
                if err:
                    extra += f" ({err.split(';')[0][:60]})"
            sys.stderr.write(
                f"\r  {name:<20} {i:>3}/{n}  {elapsed:6.1f}s elapsed"
                f"  ~{eta:5.1f}s left{extra}      ")
            sys.stderr.flush()
    if progress:
        sys.stderr.write("\n")
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
    # A real-model run is bounded by free-tier quota, not by patience. A
    # documented slice of the split beats no real-model numbers at all -- and
    # beats a full run that dies at scenario 31 and reports nothing. Slicing is
    # deterministic (first N of a fixed file) and the count is recorded in the
    # report, so the table can say what it actually measured.
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N scenarios (0 = all). Records "
                         "the count in the report so a slice cannot be "
                         "mistaken for the full split.")
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
    full_n = len(scenarios)
    if args.limit and args.limit < full_n:
        scenarios = scenarios[:args.limit]
    by_id = {s.scenario_id: s for s in scenarios}

    model_note = (f"  model={cfg.synthesis.model}"
                  if engine.provider.name != "stub" else "")
    slice_note = (f" (SLICE of {full_n} -- not the full split)"
                  if len(scenarios) < full_n else "")
    print(f"split={args.split}  scenarios={len(scenarios)}{slice_note}  "
          f"provider={engine.provider.name}{model_note}  "
          f"embedder={cfg.embedder}  "
          f"reranker={getattr(engine.reranker, 'name', '?')}")
    wanted = (cfg.synthesis.provider or "stub").lower()

    if engine.provider.name != "stub" and hasattr(engine.provider, "preflight"):
        print("  checking the provider with one call ...", end="", flush=True)
        ok, detail = engine.provider.preflight()
        # A RATE LIMIT IS NOT A CONFIGURATION ERROR. The old code treated any
        # preflight failure as fatal and told the user to go fix something,
        # which is wrong and slightly insulting advice when the actual problem
        # is that check_key.py used the minute's quota ten seconds earlier.
        # Wait it out once; a quota that resets in 30s is not worth a re-run.
        if not ok and "rate limit" in detail.lower():
            print(" rate limited")
            for wait in (20, 40):
                print(f"  free-tier quota is busy; waiting {wait}s and "
                      f"retrying ...", end="", flush=True)
                time.sleep(wait)
                ok, detail = engine.provider.preflight()
                if ok or "rate limit" not in detail.lower():
                    break
                print(" still limited")
        if ok:
            print(f" ok  (model replied {detail!r})")
        elif getattr(engine.provider, "quota_exhausted", ""):
            print(" DAILY QUOTA EXHAUSTED")
            print()
            print("!" * 74)
            print(f"  This key has no generation quota left today:")
            print(f"     {engine.provider.quota_exhausted}")
            print()
            print("  Nothing was run, and waiting will not help -- a per-day")
            print("  quota resets at midnight Pacific time, not in a minute.")
            print()
            print("  Three ways forward:")
            print("    1. Run the keyless version now. It measures the")
            print("       architecture (gates G2/G3/G5/G6 do not need an LLM):")
            print("         python -m evaluation.run_bench --split dev")
            print("    2. Run a SMALL real-model slice when quota returns, and")
            print("       say in the report that it is a slice:")
            print("         python -m evaluation.run_bench --split dev \\")
            print("                --provider gemini --limit 8 \\")
            print("                --systems B1_static_rag,B3_ripple \\")
            print("                --out results_gemini")
            print("    3. Use a different key or project.")
            print("!" * 74)
            return 2
        else:
            print(" FAILED")
            print()
            print("!" * 74)
            print(f"  The {wanted} provider rejected a test call:")
            print(f"     {detail}")
            print()
            print("  Nothing was run. Fix it and try again.")
            print()
            print("  Run this for a proper diagnosis \u2014 it separates a")
            print("  rejected key from a wrong model name, which fail")
            print("  identically here but need opposite fixes:")
            print("     python scripts/check_key.py")
            print("!" * 74)
            return 2

    if wanted != "stub" and engine.provider.name == "stub":
        env_var = "GEMINI_API_KEY" if wanted == "gemini" else "OPENAI_API_KEY"
        print()
        print("!" * 74)
        print(f"  You asked for provider={wanted!r} but no key was found, so this")
        print(f"  run is using the KEYLESS STUB. Its groundedness and cost")
        print(f"  figures are meaningless (see docs/evaluation-report.md #0).")
        print()
        print(f"  Set {env_var} in a .env file in the repo root, then re-run.")
        print("!" * 74)
    print()

    os.makedirs(args.out, exist_ok=True)
    trace_dir = os.path.join(args.out, f"traces_{args.split}")
    report: dict = {"split": args.split, "provider": engine.provider.name,
                    # Only name a model when a model was actually used. On a
                    # stub run this recorded whatever RIPPLE_MODEL happened to
                    # say, which is an attribution to a model that was never
                    # called -- precisely the kind of claim the evaluation
                    # report exists to keep out.
                    "model": (cfg.synthesis.model
                              if engine.provider.name != "stub" else None),
                    "scenarios_used": len(scenarios),
                    "scenarios_in_split": full_n,
                    "is_slice": len(scenarios) < full_n,
                    "embedder": cfg.embedder,
                    "reranker": getattr(engine.reranker, "name", "?"),
                    "config": cfg.to_dict(), "systems": {}, "ablations": {}}

    rows = []
    for name in args.systems.split(","):
        sink = trace_dir if name == "B3_ripple" else None
        before_ok = getattr(engine.provider, "calls_made", 0)
        before_bad = getattr(engine.provider, "calls_failed", 0)
        runs, secs = run_system(name, engine, scenarios, sink_dir=sink)
        cov = measure_trace_coverage(trace_dir) if name == "B3_ripple" else 0.0
        res = evaluate(runs, by_id, corpus_cites, trace_coverage=cov)
        report["systems"][name] = res.to_dict()
        report["systems"][name]["wall_seconds"] = round(secs, 2)
        good = getattr(engine.provider, "calls_made", 0) - before_ok
        bad = getattr(engine.provider, "calls_failed", 0) - before_bad
        report["systems"][name]["llm_calls_ok"] = good
        report["systems"][name]["llm_calls_failed"] = bad

        # STOP rather than tabulate garbage. If most calls to the model failed,
        # every groundedness and cost number below is computed over empty
        # answers, and printing it anyway is how a broken run becomes a slide.
        if bad and bad > good:
            print()
            print("!" * 74)
            print(f"  ABORTING. {name} made {good} successful and {bad} FAILED")
            print("  LLM calls. Any table printed from this would be measuring")
            print("  empty answers, not the system.")
            print()
            print(f"     last error: "
                  f"{getattr(engine.provider, 'last_error', '')[:300]}")
            print()
            print("  Diagnose with:  python scripts/check_key.py")
            print("!" * 74)
            return 2
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

    if engine.provider.name != "stub" and not cfg.cost.configured:
        print()
        print("  NOTE: tokens per turn are measured; the currency column is 0")
        print("        because no prices are configured. Set RIPPLE_PRICE_IN")
        print("        and RIPPLE_PRICE_OUT from the provider's pricing page")
        print(f"        for {cfg.synthesis.model} to get a cost-per-turn figure.")

    out_path = os.path.join(args.out, f"benchmark_{args.split}.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
