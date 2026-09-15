"""Headless replay CLI -- gate G1.

    python -m ripple.replay --scenarios data/scenarios/dev.jsonl --out results/

The guide says the benchmark replay evaluation is "held-out and private" and
that the automated replay suite must complete "without manual intervention".
That means the entry point cannot be a browser. This module drives the engine
in-process from timestamped transcripts and writes, per scenario:

  * the guide's section 4 output record, with their field names verbatim, so a
    private harness written against the guide can consume it unmodified
  * the full Ripple telemetry trace as JSONL

Replay uses a virtual clock taken from the transcript timestamps rather than
wall time, so a run on a slow machine produces byte-identical timings to a run
on a fast one. Reproducibility means the numbers repeat, not just that the
program starts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .config import Config
from .engine import RippleEngine
from .schemas import Scenario


def load_scenarios(path: str) -> list[Scenario]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Scenario.from_dict(json.loads(line)))
    return out


def run_scenario(engine: RippleEngine, scenario: Scenario, out_dir: str,
                 speculative: bool = True, realtime: float = 0.0) -> dict:
    trace_dir = os.path.join(out_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)
    session = engine.session(session_id=scenario.scenario_id,
                             speculative=speculative, sink_dir=trace_dir)

    turns: list[dict] = []
    wall_start = time.perf_counter()
    for chunk in scenario.chunks:
        if realtime:
            target = wall_start + chunk.t * realtime
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)
        session.on_chunk(chunk.t, chunk.text, speaker=chunk.speaker,
                         is_final=chunk.is_final)
        if chunk.is_final:
            result = session.end_utterance(chunk.t + 0.001)
            turns.append(result.to_dict())

    # A scenario whose last chunk was not flagged final still gets resolved.
    if session.prefix.strip():
        last_t = scenario.chunks[-1].t if scenario.chunks else 0.0
        turns.append(session.end_utterance(last_t + 0.001).to_dict())

    record = session.output_record()
    record.ripple_ext["turns"] = turns
    record.ripple_ext["scenario_id"] = scenario.scenario_id
    record.ripple_ext["split"] = scenario.split
    record.ripple_ext["wall_seconds"] = round(time.perf_counter() - wall_start, 3)
    session.close()

    os.makedirs(os.path.join(out_dir, "records"), exist_ok=True)
    with open(os.path.join(out_dir, "records", f"{scenario.scenario_id}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(record.to_dict(), fh, indent=2, ensure_ascii=False)
    return record.to_dict()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="ripple.replay",
        description="Replay timestamped transcripts through the Ripple engine.")
    ap.add_argument("--scenarios", required=True,
                    help="JSONL file of scenarios (see ripple/schemas.py)")
    ap.add_argument("--out", default="results",
                    help="output directory for records and traces")
    ap.add_argument("--corpus", default=None, help="override corpus path")
    ap.add_argument("--index", default=None, help="override index path")
    ap.add_argument("--provider", default=None,
                    help="stub | gemini | openai (default: RIPPLE_PROVIDER)")
    ap.add_argument("--embedder", default=None, help="tfidf-svd | bge-small")
    ap.add_argument("--no-speculative", action="store_true",
                    help="disable speculative synthesis (ablation A3b)")
    ap.add_argument("--realtime", type=float, default=0.0,
                    help="playback speed multiplier; 0 = as fast as possible")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild-index", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config()
    if args.corpus:
        cfg.corpus_path = args.corpus
    if args.index:
        cfg.index_path = args.index
    if args.provider:
        cfg.synthesis.provider = args.provider
    if args.embedder:
        cfg.embedder = args.embedder

    if args.rebuild_index or not os.path.exists(
            os.path.join(cfg.index_path, "chunks.jsonl")):
        from .retrieval.index import build_and_save

        print(f"[ripple] building index from {cfg.corpus_path} ...",
              file=sys.stderr)
        build_and_save(cfg.corpus_path, cfg.index_path, cfg.embedder)

    engine = RippleEngine(cfg)
    scenarios = load_scenarios(args.scenarios)
    if args.limit:
        scenarios = scenarios[: args.limit]

    os.makedirs(args.out, exist_ok=True)
    print(f"[ripple] provider={engine.provider.name} "
          f"embedder={cfg.embedder} "
          f"reranker={getattr(engine.reranker, 'name', '?')} "
          f"scenarios={len(scenarios)}", file=sys.stderr)

    all_records = []
    t0 = time.perf_counter()
    for i, sc in enumerate(scenarios, 1):
        rec = run_scenario(engine, sc, args.out,
                           speculative=not args.no_speculative,
                           realtime=args.realtime)
        all_records.append(rec)
        print(f"[{i}/{len(scenarios)}] {sc.scenario_id}: "
              f"{len(rec['retrieval_events'])} retrievals, "
              f"{len(rec['sub_queries'])} sub-queries, "
              f"{len(rec['citations'])} citations"
              + (", UNCERTAIN" if rec["uncertainty"] else ""),
              file=sys.stderr)

    summary_path = os.path.join(args.out, "records.jsonl")
    with open(summary_path, "w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[ripple] {len(all_records)} scenarios in "
          f"{time.perf_counter() - t0:.1f}s -> {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
