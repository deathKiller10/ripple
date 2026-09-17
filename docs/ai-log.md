# AI Usage Disclosure Log

Maintained from day 0. This log feeds directly into the official
**AI Usage Disclosure Form** (`LangAI3.0_AI_Disclosure.docx`), which asks, per
feature: name · self-generated / AI-generated / both · AI tool used · prompt
used · output summary · our modifications.

**Keep this current as you work.** It cannot be reconstructed accurately on day
nine, and the form asks for the prompts, which nobody remembers a week later.

---

## Team details (fill in before submission)

- Team name: `________` (register as `CollegeName_TeamName`)
- Project name: **Ripple**
- Institution: VIT Vellore
- Submission date: 25 Sep 2026
- Members: Priyanshu Kundu · Souptik Hazra · Anushka Paul · Arpita Bhaumik
- Repository: https://github.com/deathKiller10/ripple
- Did your team use AI in developing this project? **Yes**

## Purpose of AI usage

| Purpose | Used | Notes |
|---|---|---|
| Idea generation / brainstorming | Yes | Concept scoring across 12 candidate products against the official rubric; selection of the Care-Call framing |
| Code generation or assistance | Yes | Initial scaffold of the engine, retrieval layer, evaluation harness |
| UI / UX design | Yes | Dashboard layout and the stability-curve visualisation |
| Content creation | Yes | The synthetic Care Knowledge Pack corpus and the benchmark scenarios |
| Data analysis | Yes | Threshold calibration sweeps and their interpretation |
| Testing / debugging | Yes | Property tests; diagnosis of the two measurement bugs recorded in the README |
| Other | — | |

---

## Feature origin classification

Add a row per feature. `Both` means AI produced a draft that the team then
changed materially — say what changed, because that is the part the form
actually cares about.

| # | Feature | Origin | Tool | Prompt (summary) | Output summary | Our modifications |
|---|---|---|---|---|---|---|
| 1 | Competition analysis & concept selection | Both | Claude (Opus 5) | Analyse the official PRISM deck + Theme 4 Guide; generate and score 12 product concepts against Samsung's published rubric | 12 concepts with a weighted scoring matrix; recommended the Care-Call Copilot | Team chose the concept; corrected the timeline (10 days, not 14) |
| 2 | M1 retrieval-space stability controller | Both | Claude (Opus 5) | Design a controller that decides when to retrieve on a partial utterance without any LLM call | RBO-based drift/stability/novelty policy with four decisions | *(record what you change)* |
| 3 | M2 claim graph & delta refinement | Both | Claude (Opus 5) | Represent the answer so a late constraint patches only affected claims and citations cannot drift | Intent/Evidence/Claim/AnswerVersion graph with a four-step refinement | |
| 4 | M3 coverage-budgeted fusion | Both | Claude (Opus 5) | Fix sub-intent starvation that RRF hides | Per-intent floor + marginal-gain fill + MMR | |
| 5 | Care Knowledge Pack corpus | AI-generated | Claude (Opus 5) | Author a synthetic support corpus engineered for multi-intent, late-constraint override, contradiction, distractors and one deliberate coverage hole | 21 documents / 60 sections with `Doc_ID §Section` markers | Region scoping added after the over-invalidation bug |
| 6 | Benchmark scenarios & gold labels | Both | Claude (Opus 5) | Write dev/held-out scenarios with a G2 eligibility taxonomy | 24 scenarios, 32 labelled turns | Fragmentation at replay time added after the 8% early-retrieval bug |
| 7 | Abstention gate | Both | Claude (Opus 5) | Detect questions the corpus cannot answer | z-score approach proposed, **measured, and rejected**; replaced with a corpus-vocabulary test | The failed hypothesis is documented rather than hidden |
| 8 | Evaluation harness & baselines | Both | Claude (Opus 5) | Implement B0–B2 baselines and all six gate metrics | Four-system harness with calibration sweeps | Recall measurement bug found and fixed |
| 9 | Dashboard | Both | Claude (Opus 5) | Three-column live view: transcript, controller timeline, answer with version diff | Zero-build HTML dashboard over the WebSocket | |
| 10 | Docker / reproducible setup | Both | Claude (Opus 5) | One command, clean machine, no key, no GPU | Multi-stage Dockerfile with index baked in at build time | |

---

## Ethical & compliance

- AI usage complies with the hackathon guidelines. **Yes**
- No proprietary or copyrighted data misused. **Agreed.** The corpus is
  synthetic and written for this project; it is not Samsung material and every
  figure in it is invented. No external knowledge source is used at inference
  time — see the corpus-isolation section of the architecture brief.

## Sign-off

- Team representative: `________`
- Role: `________`
- Date: `________`
