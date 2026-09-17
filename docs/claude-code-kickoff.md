# Claude Code kickoff prompts

One prompt per person for the first session. After this, `CLAUDE.md` is loaded
automatically every time, so you never need to re-explain the project.

**Before your first prompt**, in your clone:

```bash
cd ripple
pip install -r requirements.txt
python data/corpus/care/build_corpus.py
python data/labels/build_labels.py
python data/scenarios/build_scenarios.py
python -c "from ripple.retrieval.index import build_and_save; build_and_save('data/corpus/care','.index','tfidf-svd')"
python tests/test_gates.py          # must print 9/9 passed
claude
```

If `9/9 passed` does not appear, stop and fix that before anything else.
Everything below assumes a green baseline.

---

## Priyanshu — B, the engine

> Read CLAUDE.md first. I own the controller, session state and synthesis:
> `ripple/controller/`, `ripple/session/`, `ripple/synthesis/`, `ripple/engine.py`.
> Do not edit files outside those without telling me.
>
> First job: wire up the real Gemini provider and find out what breaks. My key
> is in `.env` as `GEMINI_API_KEY`, with `RIPPLE_PROVIDER=gemini` and
> `RIPPLE_RPM=12`. Right now everything has only ever run against the `stub`
> provider, which answers by copying sentences out of the chunk it cites — so
> our groundedness number is meaningless and our token cost is zero.
>
> Run `python -m evaluation.run_bench --split dev --provider gemini` and show me
> what changes versus the stub run. I expect sub-query extraction to improve a
> lot, because the zero-token splitter merges questions that have no "and"
> between them. I also expect groundedness to fall below 1.0 for the first
> time, which is the honest number.
>
> Rules: the controller must stay free of LLM calls, no model may ever write a
> citation string, and the keyless stub path must keep working. Tell me before
> you touch `ripple/schemas.py`.

---

## Souptik — A, retrieval and the corpus

> Read CLAUDE.md first. I own `ripple/retrieval/`, `ripple/corpus/` and
> `data/corpus/`. Do not edit files outside those without telling me.
>
> Two jobs, in order.
>
> **1. Grow the corpus.** `data/corpus/care/build_corpus.py` currently generates
> 21 documents and 60 sections. Take it to roughly 60 documents and 300+
> sections in the same style, with the same `Doc_ID §Section` structure and
> front matter. Read the docstring at the top — each document exists to exercise
> a specific mechanism, and new ones must too. Keep every engineered property:
> the cross-border override, the dated contradiction pair, the lexical
> distractors, the near-duplicates, and the deliberate coverage holes. **Do not
> fill the holes** — nothing may cover screen-protector reimbursement or
> trade-in valuation, because our abstention demo depends on them.
>
> **2. Compare embedders.** Install `requirements-quality.txt` and benchmark
> `bge-small` against the current `tfidf-svd` on `--split dev`. Report recall@k
> and per-intent coverage for both. If bge-small wins clearly, make it the
> quality path but keep tfidf-svd as the keyless default so the container still
> builds without torch.
>
> Everything is synthetic and invented. Nothing may be copied from real Samsung
> material.

---

## Anushka — C, the dashboard

> Read CLAUDE.md first, then `frontend/README.md`. I own `frontend/` and nothing
> else.
>
> `frontend/dashboard.html` works today and is what the Docker container serves.
> `frontend/src/` is a React + Vite + Tailwind scaffold with the shell and the
> WebSocket hook already written. My job is to finish the React version without
> ever breaking the plain HTML one, because that is the fallback the container
> depends on.
>
> Build these four, in this order:
>
> 1. **StabilityChart** — the signature graphic. The stability curve over time,
>    a dashed θ threshold line, and a coloured dot at every controller decision
>    (RETRIEVE, RETRIEVE_MORE, SUPPRESS). A judge should see the curve cross the
>    line and a retrieval fire *while the transcript is still arriving*.
> 2. **VersionDiff** — pills showing preserved / superseded / added, and a
>    **citation drift** counter that must read 0 on every refinement. That zero
>    is gate G5, visible at a glance.
> 3. **EvidencePool** — retrieved chunks colour-coded by lifecycle state.
> 4. **CitationPopover** — click any `DOC_X §N` and fetch
>    `/api/corpus?cite=...` so the judge can read the exact passage a claim
>    rests on.
>
> Start the backend with `uvicorn ripple.server:app --port 8000`, then
> `npm install && npm run dev` — the Vite proxy is already configured.
>
> One rule: the UI computes nothing. Every number on screen must come from a
> telemetry event, so the screen and `traces/*.jsonl` can never disagree.

---

## Arpita — D, evaluation and reproducibility

> Read CLAUDE.md first. I own `evaluation/`, `data/scenarios/`, `tests/`,
> `Dockerfile` and `docker-compose.yml`. Do not edit files outside those without
> telling me.
>
> Three jobs, in order.
>
> **1. Prove the container works on a machine that has never seen this project.**
> `docker compose up` on a clean checkout, no `.env`, no network after build.
> This is gate G1 and it is pass/fail. Do it now, not on day eight. Fix whatever
> breaks and write down what you did.
>
> **2. Grow the scenarios.** `data/scenarios/build_scenarios.py` has 12 dev and
> 12 held-out. Take each to about 40, keeping the same balance of turn kinds —
> informational, refinement, presentation, social, uncoverable — and read the
> docstring for what each kind means. Gold labels must be real: check every
> `gold_doc_ids` entry actually exists in the corpus. **Do not look at the
> held-out results while writing them.**
>
> **3. Finish the ablations.** `evaluation/run_bench.py --ablations` has three
> stubs that do not yet isolate what they claim. Make A2 genuinely swap
> coverage-budgeted fusion for plain RRF, and add A4 for dense-only versus
> hybrid retrieval. Each ablation must change exactly one thing.
>
> Never invent a number. If something has not been measured, write "not
> measured yet".

---

## Daily rhythm

Each person, end of day, in the group chat: what you finished, what broke, what
you need from someone else. Two minutes, not a meeting.

Before every push:

```bash
python tests/test_gates.py                  # 9/9
python -m evaluation.run_bench --split dev  # six gates PASS
```

If either goes red, fix it before pushing. A red main branch blocks three people.

## Freeze — 21 September

After the freeze, the only permitted changes are bug fixes, documentation and
measurement. Days 22–24 are: run the held-out split once, run the ablations,
write the evaluation report and the six-page architecture brief, record the
demo video, populate the deck, complete the AI disclosure form, and tag
`PRISM_GENAI_HACKATHON_Y2026`.

Submit on the 25th by 18:00, not 23:59.
