#!/usr/bin/env python3
"""Threshold calibration -- DEV SPLIT ONLY.

Two sweeps, both run against data/scenarios/dev.jsonl and never against the
held-out split:

  1. ABSTENTION GATE.  Choose (oov_hard, oov_soft, cover_floor) so that the
     gate fires on `uncoverable` turns and stays silent on `informational`
     ones. Reported as a confusion matrix, not a single number, because the
     two error types have very different costs: a missed abstention produces
     an ungrounded answer and fails gate G4, while a spurious abstention only
     loses an answer we could have given.

  2. RETRIEVAL THRESHOLD theta.  Choose the controller's stability threshold
     by minimising the asymmetric cost

         E[cost] = P(false trigger) * c_waste + P(late) * c_late

     with c_waste = 1 and c_late = 175 from config.py, reflecting that a
     wasted shadow retrieval costs about 4 ms of CPU while a retrieval that
     fires after the utterance ends costs about 700 ms of user-visible
     silence. This is why the README can answer "why 0.68?" with a curve
     rather than a shrug.

Run:  python -m evaluation.calibrate
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ripple.config import Config                      # noqa: E402
from ripple.engine import RippleEngine                # noqa: E402
from ripple.retrieval.rerank import apply as apply_rerank  # noqa: E402
from ripple.retrieval.relevance import assess, build_vocabulary  # noqa: E402
from ripple.schemas import Decision, Scenario         # noqa: E402


def load(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(Scenario.from_dict(json.loads(line)))
    return out


# ---------------------------------------------------------------------------
# 1. abstention gate
# ---------------------------------------------------------------------------


def sweep_abstention(engine, scenarios) -> dict:
    vocab = build_vocabulary(engine.index.chunks)
    samples = []   # (query, should_abstain)
    for sc in scenarios:
        for g in sc.gold:
            if g.turn_kind not in ("informational", "uncoverable"):
                continue
            should = g.turn_kind == "uncoverable"
            for q in (g.gold_sub_intents or [" ".join(
                    c.text for c in sc.chunks)]):
                samples.append((q, should))

    rows = []
    for q, should in samples:
        hits = engine.index.search(q, k=20)
        hits = apply_rerank(engine.reranker, q, hits, 20)
        chunks = [h.chunk for h in hits]
        rows.append((q, should, chunks))

    best, best_score = None, -1e9
    grid = []
    for oov_hard in (0.30, 0.40, 0.50, 0.60, 0.70):
        for oov_soft in (0.15, 0.20, 0.25, 0.34):
            for cover_floor in (0.20, 0.30, 0.40, 0.50):
                tp = fp = tn = fn = 0
                for q, should, chunks in rows:
                    v = assess(q, chunks, vocab, oov_hard=oov_hard,
                               oov_soft=oov_soft, cover_floor=cover_floor)
                    abstained = not v.sufficient
                    if should and abstained:
                        tp += 1
                    elif should and not abstained:
                        fn += 1
                    elif not should and abstained:
                        fp += 1
                    else:
                        tn += 1
                # A missed abstention is the expensive error: it yields an
                # answer with no support, which is exactly what gate G4 fails.
                score = 4.0 * tp - 6.0 * fn - 1.0 * fp
                grid.append(dict(oov_hard=oov_hard, oov_soft=oov_soft,
                                 cover_floor=cover_floor, tp=tp, fp=fp,
                                 tn=tn, fn=fn, score=score))
                if score > best_score:
                    best_score, best = score, grid[-1]
    return {"best": best, "n_samples": len(rows),
            "grid": sorted(grid, key=lambda r: -r["score"])[:10]}


# ---------------------------------------------------------------------------
# 2. controller threshold
# ---------------------------------------------------------------------------


def sweep_theta(cfg: Config, scenarios) -> dict:
    """Replay every scenario at a range of theta and price the outcome."""
    from ripple.retrieval.index import HybridIndex

    index = HybridIndex.load(cfg.index_path, cfg.embedder)
    results = []
    for theta in [0.40, 0.48, 0.56, 0.62, 0.68, 0.74, 0.80, 0.86, 0.92]:
        c = Config()
        c.controller.theta = theta
        eng = RippleEngine(c, index=index)
        early_hits = eligible = false_triggers = no_retrieval_turns = 0
        shadow_total = retrieval_total = 0

        for sc in scenarios:
            s = eng.session(session_id=f"cal_{theta}_{sc.scenario_id}",
                            speculative=False, sink_dir=None)
            gold_by_idx = {g.utterance_index: g for g in sc.gold}
            idx, fired_before_end = 0, False
            for ch in sc.chunks:
                out = s.on_chunk(ch.t, ch.text, ch.speaker, ch.is_final)
                if out.decision in (Decision.RETRIEVE, Decision.RETRIEVE_MORE):
                    fired_before_end = True
                if ch.is_final:
                    g = gold_by_idx.get(idx)
                    if g is not None:
                        if g.needs_retrieval:
                            eligible += 1
                            if fired_before_end:
                                early_hits += 1
                        else:
                            no_retrieval_turns += 1
                            if fired_before_end:
                                false_triggers += 1
                    s.end_utterance(ch.t + 0.001)
                    idx += 1
                    fired_before_end = False
            shadow_total += s.index.shadow_calls
            retrieval_total += s.turn_retrievals
            s.close()

        early_rate = early_hits / max(1, eligible)
        ft_rate = false_triggers / max(1, no_retrieval_turns)
        expected_cost = (ft_rate * cfg.controller.c_waste
                         + (1 - early_rate) * cfg.controller.c_late)
        results.append({
            "theta": theta,
            "early_retrieval_rate": round(early_rate, 4),
            "false_trigger_rate": round(ft_rate, 4),
            "eligible_turns": eligible,
            "no_retrieval_turns": no_retrieval_turns,
            "expected_cost": round(expected_cost, 3),
        })
    best = min(results, key=lambda r: r["expected_cost"])
    return {"curve": results, "best_theta": best["theta"], "best": best}


def main():
    cfg = Config()
    engine = RippleEngine(cfg)
    dev = load("data/scenarios/dev.jsonl")
    assert all(s.split == "dev" for s in dev), "calibration must use dev only"

    print("=" * 72)
    print("1. ABSTENTION GATE SWEEP (dev split)")
    print("=" * 72)
    ab = sweep_abstention(engine, dev)
    b = ab["best"]
    print(f"samples: {ab['n_samples']}")
    print(f"best: oov_hard={b['oov_hard']} oov_soft={b['oov_soft']} "
          f"cover_floor={b['cover_floor']}")
    print(f"  correctly abstained (tp)={b['tp']}  missed (fn)={b['fn']}  "
          f"spurious (fp)={b['fp']}  correctly answered (tn)={b['tn']}")
    print("\n  top configurations:")
    for r in ab["grid"][:5]:
        print(f"    hard={r['oov_hard']} soft={r['oov_soft']} "
              f"floor={r['cover_floor']}: tp={r['tp']} fn={r['fn']} "
              f"fp={r['fp']} tn={r['tn']}")

    print()
    print("=" * 72)
    print("2. CONTROLLER THRESHOLD SWEEP (dev split)")
    print("=" * 72)
    th = sweep_theta(cfg, dev)
    print(f"{'theta':>7} {'early':>8} {'false_trig':>11} {'E[cost]':>10}")
    for r in th["curve"]:
        mark = "  <-- chosen" if r["theta"] == th["best_theta"] else ""
        print(f"{r['theta']:>7.2f} {r['early_retrieval_rate']:>8.2f} "
              f"{r['false_trigger_rate']:>11.2f} {r['expected_cost']:>10.2f}{mark}")
    print(f"\nchosen theta = {th['best_theta']} "
          f"(c_waste={cfg.controller.c_waste}, c_late={cfg.controller.c_late})")

    os.makedirs("results", exist_ok=True)
    with open("results/calibration.json", "w") as fh:
        json.dump({"abstention": ab, "theta": th}, fh, indent=2)
    print("\nwritten to results/calibration.json")


if __name__ == "__main__":
    main()
