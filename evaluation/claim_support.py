"""Claim support: the groundedness figure that a real model can actually move.

`citation_support` in metrics.py is 1 - fabricated/total citations. In Ripple a
citation is derived from an evidence ID, so a fabricated citation cannot be
represented and that figure is 1.000 under ANY model -- it proves the claim
graph works, not that the model wrote faithful text. It must not be reported
as groundedness.

This module reads the verdicts the engine's GroundingVerifier already records
in the trace (`grounding_checked` events with a `grounded` field): for every
claim the synthesiser produced, whether its content is supported by the
passage it cites (lexical overlap >= grounding_min_overlap, or embedding
similarity >= min_semantic). That depends on what the model wrote, so it is
the number to report beside a model name. It is an automated proxy, not a
human judgement: a claim that reuses the passage's words but states a wrong
figure can still pass.

    python -m evaluation.claim_support results_gemini/run2_2026-09-26/traces_dev
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys


def measure_claim_support(trace_dir: str) -> dict:
    supported = rejected = 0
    lexical: list[float] = []
    for path in sorted(glob.glob(os.path.join(trace_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("type") != "grounding_checked":
                    continue
                d = ev.get("detail") or {}
                # Relevance gates carry a `stage`; claim verdicts do not.
                if d.get("stage") is not None or "grounded" not in d:
                    continue
                lexical.append(float(d.get("lexical", 0.0)))
                if d["grounded"]:
                    supported += 1
                else:
                    rejected += 1
    n = supported + rejected
    return {
        "claims_checked": n,
        "claims_supported": supported,
        "claims_rejected": rejected,
        "claim_support_rate": round(supported / n, 4) if n else None,
        "median_lexical_overlap": (round(statistics.median(lexical), 4)
                                   if lexical else None),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m evaluation.claim_support <trace_dir>")
    print(json.dumps(measure_claim_support(sys.argv[1]), indent=2))
