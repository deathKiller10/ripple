# Dashboard

Two implementations, on purpose.

**`dashboard.html`** — zero-build, no Node, no npm. `ripple/server.py` serves it
directly and the Dockerfile copies it in. This is what ships and what gate G1
depends on: a judge running `docker compose up` gets a working UI without a
JavaScript toolchain anywhere in the image.

**`src/`** — the Vite + React + Tailwind migration target (Member C). When
`npm run build` has produced `frontend/dist/`, the server prefers it
automatically; no server change is needed. Keep `dashboard.html` working
regardless — it is the fallback that keeps the container buildable.

```bash
npm install
npm run dev     # proxies /api and /ws to the Python server on :8000
npm run build   # → dist/, picked up by the server on next start
```

## What the UI must show (Theme 4 Guide §8 demo-video checklist)

| Panel | Must render |
|---|---|
| Left | live transcript, fragment by fragment, speaker-attributed |
| Centre | stability curve with the θ line, and every controller decision on a clock |
| Centre | sub-queries, retrieved document IDs, rerank scores, budget allocation per intent |
| Right | claims with citations, answer version, and the v1→v2 diff (preserved / superseded / added / **citation drift**) |
| Right | evidence pool with lifecycle states, and the uncertainty indicator when it fires |
| KPIs | signed TTFT, retrievals this turn, LLM calls this turn |

The citation-drift counter is the one a judge should be able to read at a
glance: it must say **0** on every refinement, and that is the visible proof of
gate G5.

Nothing in the UI computes anything. Every number comes from a telemetry event,
so what is on screen is exactly what is in `traces/<session>.jsonl`.
