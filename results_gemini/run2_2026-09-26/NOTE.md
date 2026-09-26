# Real-model run 2 — 26 Sep 2026 (the run to report)

- Model: `gemini-3.5-flash-lite` (Google Gemini API, free tier, 8 rpm client limit)
- Run: `2026-09-26T16:36:42Z` (from `run_utc` in `benchmark_dev.json`)
- Split: dev, all 40 scenarios (not a slice)
- Systems: B1_static_rag, B3_ripple
- Code: commit 8e25364 (TTFT excludes rate-limit queueing)
- Calls: B1 51 ok / 0 failed; B3 129 ok / 0 failed
- Time spent queueing for the rate limit (excluded from TTFT, reported
  here): B1 229 s, B3 598 s

Reproduce the groundedness figure from the saved traces:

    python -m evaluation.claim_support results_gemini/run2_2026-09-26/traces_dev

TTFT is measured with a non-streaming API call, so it is the time to the
first complete claim, not the first streamed token.

Run 1 (`../run1_2026-09-26`) is kept as evidence; its TTFT is invalid.
