# Ripple — project rules for any AI coding tool

Read this before changing anything. It is the shared contract for four people
working on one repo in eight days.

This file applies whatever tool you use — Claude Code, Antigravity, Cursor.
Most agentic editors read `AGENTS.md` automatically; if yours has a "rules",
"memory" or "system prompt" setting instead, point it at this file, or paste
the sections down to **Who owns what** into it once.

## What this is

An event-driven **Streaming Live RAG engine** for the Samsung PRISM GenAI
Hackathon, Theme 04. It starts retrieving before the speaker finishes their
sentence, splits one natural utterance into the several questions it implies,
and holds the answer as a graph of individually-cited claims so a late-arriving
detail patches only the affected claims instead of restarting the search.

Reference application: **Ripple for Care** — grounding a support agent who is
listening to a customer. The agent is not being spoken to. There is no turn
boundary to wait for and nobody to ask for clarification. That is why
full-duplex behaviour is the only mode that exists here.

**Team** · Priyanshu Kundu · Souptik Hazra · Anushka Paul · Arpita Bhaumik
**Repo** · https://github.com/deathKiller10/ripple
**Due** · 30 Sep 2026 (target 29 Sep) · **Feature freeze 28 Sep** · Tag `PRISM_GENAI_HACKATHON_Y2026`

---

## Hard rules from Samsung — these are disqualifiers, not preferences

1. **Corpus isolation.** Every factual claim traces to the provided corpus. No
   web access, no third-party knowledge base, and **no unindexed parametric
   model memory**. If retrieval comes back thin, the system says so. It does
   not fill the gap from what the model already knows.
2. **No hardcoding, no precomputation.** The benchmark replay is *held-out and
   private*. Never embed prompts, queries, canned answers or benchmark-specific
   phrasing in application code. If a change would behave differently on a
   corpus we have never seen, it is wrong.
3. **Citations are `[DOC_ID §Section]`**, always real, never authored by a model.
4. **Session-bound state only.** Cross-session profiling and persistent user
   tracking are prohibited. Adding Redis or a database for "memory" is a spec
   violation, not a feature.
5. **Architectural parsimony.** Multi-agent frameworks are graded on
   cost-to-performance. Every component must justify its latency and compute.

## Invariants — break one and the submission breaks

- **The controller contains no LLM call.** `ripple/controller/` runs on every
  transcript chunk. Three chunks a second over a twenty-second utterance is
  sixty decisions; an LLM call per decision is sixty calls per turn. This is the
  whole cost argument. Do not put a model on this path.
- **No model ever writes a citation string.** Citations are derived from
  evidence IDs in `session/claim_graph.py`. This is what makes a fabricated
  document ID *unrepresentable* rather than merely unlikely. Never ask a model
  to "keep the citations" or to include them in prose.
- **Unaffected claims are never regenerated.** That is the only reason their
  citations cannot drift, which is the only reason gates G4 and G5 stop
  fighting each other.
- **`ripple/engine.py` must stay transport-free.** It must never import
  `server.py`, FastAPI, or anything web. The WebSocket server and the replay
  CLI are both *clients* of the engine. If the engine ever depends on the
  dashboard, the headless replay dies and gate G1 goes with it.
- **The default path needs no API key and no GPU.** `docker compose up` on a
  clean machine must work with `RIPPLE_PROVIDER=stub` and
  `RIPPLE_EMBEDDER=tfidf-svd`. The judge will not have your `.env`.
- **Thresholds live in `ripple/config.py` only.** Never hard-code a number
  inline. If you change a threshold, re-run `python -m evaluation.calibrate`
  and say what moved.

## Shared files — do not edit alone

- `ripple/schemas.py` — the contract between all four workstreams. Changing it
  silently breaks the dashboard, the replay CLI and the evaluation harness at
  once. Agree in the group chat first.
- `ripple/config.py` — coordinate, and re-run calibration after any change.
- `CLAUDE.md` — this file.

## Who owns what

This section originally split the folders across four members. In practice
Priyanshu Kundu built and owns all of the code; the teammates' contributions
(reproduction, the demo video, the disclosure sign-off) are recorded under
**Team and contributions** in the README.

## Commands

```bash
# first-time setup
pip install -r requirements.txt
python data/corpus/care/build_corpus.py
python data/labels/build_labels.py
python data/scenarios/build_scenarios.py
python -c "from ripple.retrieval.index import build_and_save; build_and_save('data/corpus/care','.index','tfidf-svd')"

# run
uvicorn ripple.server:app --port 8000     # dashboard at localhost:8000
docker compose up                          # the judge's path — must always work

# before every commit
python tests/test_gates.py                 # 11 property tests, must stay 11/11
python -m evaluation.run_bench --split dev # all six gates, must stay PASS

# after changing any threshold
python -m evaluation.calibrate
```

## Evaluation discipline — non-negotiable

- **Calibrate on `dev` only.** `data/scenarios/heldout.jsonl` is run **once**,
  after feature freeze on 28 Sep. Do not open it, do not tune against it, do
  not "just check" it. It is the only credible answer to *"did you overfit to
  your own demo?"*.
- **Never invent a number.** Every figure in the README, the deck or the report
  must come from an actual run. If something has not been measured, write "not
  measured yet".
- **A result that flatters us for a reason we cannot explain mechanically is a
  harness bug until proven otherwise.** Two have already been caught this way
  and are documented at the bottom of the README. Expect more.
- Known caveat to preserve honestly: with `RIPPLE_PROVIDER=stub`, groundedness
  is ~1.0 by construction, because the stub answers by copying sentences out of
  the chunk it cites. Grounding numbers worth reporting come from a real
  provider run, and the report must name the provider.

## Secrets

The Gemini key goes in a local `.env` file, which `.gitignore` already
excludes. **Never commit it, never paste it into a chat, never put it in a
prompt.** If it leaks, revoke it in Google AI Studio and make a new one.

```
RIPPLE_PROVIDER=gemini
GEMINI_API_KEY=...        # local only
RIPPLE_RPM=12             # free tier: keep the rate limiter on
```

## Working with a free-tier agent

Most of the team is on a free tier with limited requests. That changes how you
should work, not what you should build.

- **Ask for one thing at a time.** "Add the StabilityChart component" beats
  "finish the dashboard". A small task that lands is worth more than a big one
  that half-lands and burns your quota.
- **Never let it refactor broadly.** If it offers to "clean up" or "improve"
  files you did not ask about, say no. Every unrequested change is a merge
  conflict for someone else and a request you cannot afford.
- **Read the diff before accepting.** Especially watch for: an LLM call sneaking
  into `ripple/controller/`, a citation string being written by a model, a new
  dependency, or a hard-coded threshold. Those four break the submission.
- **Run the tests yourself** rather than asking the agent to. It is faster and
  it does not cost a request:
  `python tests/test_gates.py` then `python -m evaluation.run_bench --split dev`.
- **When you run out of requests**, keep going by hand. Everything in this repo
  is ordinary Python; the agent is an accelerator, not a dependency.
- If a tool insists on writing its own rules file (`.cursorrules`,
  `.antigravity/`, etc.), make it a one-line pointer to `AGENTS.md` rather than
  a second copy of the rules. Two rule files drift apart within a day.

## Style

- Python 3.11, standard library first. **Adding a dependency requires a reason
  written in the pull request** — the parsimony rule is graded.
- Explicitly forbidden: LangChain, LlamaIndex, LangGraph, any agent framework,
  Kafka, Celery, Kubernetes, a vector-database server, Redis.
- Comments explain *why*, especially where a simpler-looking option was
  rejected. A jury will ask "why not X?" about half of these decisions and the
  answer should already be in the code.
- Keep the failure stories. `retrieval/relevance.py` documents a hypothesis we
  tested and discarded; the README documents two measurement bugs. Do not tidy
  these away — they are evidence of method, and they are what separates a
  measured project from a demo.

## What matters most, in order

1. It runs on the judge's machine with one command. (30% of the rubric by proxy)
2. Measured numbers against real baselines, with a declared held-out split.
3. One mechanism you can name and defend, not three half-built ones.
4. A demo where the judge *sees* retrieval fire before the sentence ends.

Teams lose this not by building too little, but by still building on day eight.
When in doubt after 28 Sep, the answer is no.
