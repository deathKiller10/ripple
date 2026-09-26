# Real-model run 1 — 26 Sep 2026

- Model: `gemini-3.5-flash-lite` (Google Gemini API, free tier)
- Split: dev, all 40 scenarios (not a slice)
- Systems: B1_static_rag, B3_ripple
- Code: commit 5c5635e
- Calls: B1 51 ok / 0 failed; B3 128 ok / 0 failed

## Valid from this run

Citation support, fabricated citations, recall@k, per-intent coverage,
early retrieval, false triggers, multi-intent accuracy, abstention,
LLM calls per turn, tokens per turn. None of these depend on wall-clock
time.

## NOT valid from this run: TTFT

`ttft_median` / `ttft_p90` in `benchmark_dev.json` include time spent
queueing for the free-tier rate limit (8 rpm; median 7.5 s per call,
worst 263 s). B1's 7.451 s is almost exactly 60 s / 8 rpm. Fixed in
commit 8e25364; real-model TTFT comes from a later run.
