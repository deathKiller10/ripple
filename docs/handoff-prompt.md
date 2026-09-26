# Continue the Ripple project (Samsung PRISM GenAI Hackathon)

Paste everything below the line into a new Claude conversation, with the folder
`C:\Priyanshu\Study\VIT Vellore\Semester_3\Samsung Prism` connected.

---

I'm continuing work on a hackathon project from a previous conversation that got
too long. Everything you need is below. Please read it fully before doing
anything, then do the **Immediate next step** at the end.

## Who and what

I'm Priyanshu Kundu (VIT Vellore). Team: me, Souptik Hazra, Anushka Paul,
Arpita Bhaumik — but **I'm building all of it myself**; my teammates don't have
Claude Code, so don't plan work for them.

- **Competition:** Samsung PRISM GenAI Hackathon, 3rd Edition 2026-27,
  **Theme 04 "Streaming Live RAG"**
- **Submission deadline: 30 September 2026** (I want to be done by the 29th)
- **Today it is 26 September** — four days left
- Project name: **Ripple** (concept: "Ripple for Care", a customer-support
  assistant)

## Where the code is

- **My machine, the connected folder:**
  `C:\Priyanshu\Study\VIT Vellore\Semester_3\Samsung Prism\ripple`
  **This is the only copy that exists. Treat it as the source of truth.**
- GitHub repo exists but **nothing has been pushed yet**:
  `https://github.com/deathKiller10/ripple.git`
- My folder is **not** a git repository. The full 16-commit history lives in
  `Samsung Prism\_transfer\ripple-history.bundle`, and
  `ripple-sync.zip` beside it is a snapshot of all 132 files. The commit
  messages document why each design decision was made, so they're worth keeping.
- I'm on **Windows / PowerShell**. Python is installed and dependencies are
  already set up. I run commands myself in PowerShell, in the `ripple` folder.

**How to work on it:** edit files directly in my connected folder. Don't ask me
to copy-paste code. If you need to run the benchmark, tell me the command and
I'll run it in PowerShell and paste the output back — the API key is on my
machine, not yours.

## Project rules (important)

`AGENTS.md` in the repo root is the canonical rule file — **read it first**. Key
standing rules:

- **Never** put my API key in a chat, a prompt, or a commit. It goes in the
  local `.env` only, which `.gitignore` excludes. (I leaked one earlier and
  revoked it.)
- **No fabricated numbers.** If a price, a metric or a figure isn't measured,
  it stays at zero or absent and the report says so. Cost prices default to 0.0
  for exactly this reason.
- Whichever model produced a number must be named beside that number.
- Deliberately **no** LangChain / LlamaIndex / LangGraph / agent frameworks /
  Kafka / Celery / Kubernetes / Redis / vector-DB servers. Stack is Python 3.11,
  FastAPI + WebSockets, FAISS flat IP, scikit-learn TF-IDF+SVD embeddings, no
  torch by default. Dependency count is graded — keep it small.
- Components get **deleted when measurement says they don't earn their place**.
  BM25 and the reranker were both removed on evidence; don't reintroduce them.

## How I like to be helped

- **Explain things in simple words.** I'm a student, not a senior engineer.
- **Tell me explicitly what I need to do**, as numbered steps with exact
  commands.
- When something breaks, tell me plainly what went wrong and whether it was your
  mistake. Don't make me debug your code.
- Don't hand me homework (e.g. "go set this environment variable") if the code
  can work it out itself.

## The four technical ideas the project is built on

1. **M1 — retrieval-space stability.** The controller decides when to retrieve
   from a partial utterance using Rank-Biased Overlap between successive prefix
   result sets — not by judging whether the sentence "looks finished". Costs zero
   LLM tokens.
2. **M2 — claim graph.** The answer is stored as `Claim` objects bound to
   evidence IDs, so citations are *derived*, not authored by the model. A
   fabricated citation is structurally impossible.
3. **M3 — coverage-budgeted fusion.** Per-intent floor + marginal gain + MMR,
   instead of plain RRF. Fixes "sub-intent starvation", which recall@k cannot see.
4. **M4 — signed TTFT.** Time-to-first-token measured relative to end-of-
   utterance, so it can be **negative** (we answered before they finished
   speaking).

## Current measured state — keyless stub provider, dev split, 40 scenarios

| system | early retr | false trig | multi-intent | recall@k | intent cov | TTFT med | retr/turn |
|---|---|---|---|---|---|---|---|
| B1 static RAG | 0.000 | 1.000 | 0.000 | 0.835 | 0.768 | +0.010 | 1.00 |
| B2 naive streaming | 1.000 | 1.000 | 0.000 | 0.835 | 0.768 | −2.069 | 4.04 |
| **B3 Ripple** | **0.884** | **0.000** | **0.889** | **0.960** | **0.772** | **−1.036** | 2.75 |

**All six acceptance gates G1–G6 PASS.** These numbers are reproducible — if a
stub dev run gives anything different, something regressed.

Everything is built: engine, controller, claim graph, retrieval, evaluation
harness, four baselines, five ablations, corpus (60 docs / 151 chunks),
scenarios (40 dev / 34 held-out), frontend, Dockerfile, all of `docs/`, and a
12-slide `VITVellore_Ripple_Submission.pptx` from Samsung's template.

## The one thing that isn't finished

**Groundedness and cost are still placeholders**, because every number above
came from the keyless stub. I need one run against a real model to fill those
two columns. That has been a long fight, now nearly won:

- Provider is Gemini. `.env` has `RIPPLE_PROVIDER=gemini`, my key, and
  `RIPPLE_MODEL=gemini-3.8-flash`.
- `gemini-3.8-flash` is a **reasoning model**: it spends output tokens thinking
  before it writes anything. With a small `maxOutputTokens` it returns HTTP 200
  and **empty text**. That silently produced a run where 51 turns made only 3
  successful calls and a preflight that printed `ok (model replied '')`.
- Fixed across three commits: thinking disabled with a 1024-token floor; one
  automatic escalation to 4096 tokens if the answer still comes back empty, and
  the working budget is then remembered for the rest of the run; an empty answer
  is now a failure with a real diagnosis; failures are counted and shown in the
  progress line; and the run aborts rather than tabulating measurements of empty
  answers.
- Then it hit `rate_limited`, because running `scripts/check_key.py` immediately
  before the benchmark used up the free tier's per-minute quota. Also fixed:
  Google's 429 says whether the quota resets in a **minute** or at **midnight**,
  and those need opposite responses — the code now reads that, honours the
  server's retry delay, slows its own request rate when corrected, and gives up
  immediately on a daily cap instead of retrying for hours.
- `scripts/check_key.py` last reported: **`RESULT: WORKING. The model replied
  'ok'.`** So the key, the project quota and the model are all fine.

### Immediate next step

I have **not run this yet**. It's the first thing to do:

```powershell
python -m evaluation.run_bench --split dev --provider gemini --limit 8 --systems B1_static_rag,B3_ripple --out results_gemini
```

Two deliberate choices: **`--limit 8`** (a real-model run is limited by free-tier
quota, not patience — 8 scenarios is enough to fill groundedness and cost, and
the table and JSON both label it `SLICE of 40` so it can't be mistaken for the
full split), and **do not run `check_key.py` first** — it passed already, and it
eats the same quota the benchmark needs.

Wait for me to paste the output, then tell me what it means and what's next.

## Remaining checklist before submission

1. Real-model slice above → put the groundedness and cost figures into
   `docs/evaluation-report.md` and the deck, **naming the model and the date**
   beside them, and labelling the slice as a slice.
2. Run the **held-out split once only**, after feature freeze. Not before.
3. Push to GitHub and **verify `.env` is not in the repo**. My folder isn't a git
   repo yet — the history bundle in `_transfer\` needs restoring, or a fresh
   repo initialised. I'd like the commit history preserved if that's practical.
4. Record a **demo video, 5 minutes max**, and put the link in the README. The
   deck's checklist slide already claims "Linked in the repo README" — that
   claim is currently **false** and must be made true.
5. Fill in the **AI disclosure form** from `docs/ai-log.md`.
6. Tag the repo `PRISM_GENAI_HACKATHON_Y2026`.
7. Submit by 30 September, aiming for the 29th.
8. Optional: set `RIPPLE_PRICE_IN` / `RIPPLE_PRICE_OUT` from Google's current
   pricing page to get a real cost-per-turn figure instead of zero. Only use
   real published prices.

## One warning from last time

Sending individual files to my folder went wrong: two files silently stayed on
older versions, and one of them still contained **invented INR prices** that had
already been removed for being made up. My folder has now been fully re-synced
and verified file-by-file against the tested code. If you change code, change it
in my folder and **verify the change landed** — don't assume a file copy worked.
