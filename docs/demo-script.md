# Five-minute demo video — shot list

The Theme 4 Guide §8 names six things the video must show: **early retrieval
triggering · multi-intent decomposition · late-detail refinement · presentation
query suppression · citation traceability · runtime telemetry.** All six are
below, in one continuous call.

**Record, don't perform live.** The submission is a video file. Save the live
run for 15 October, and rehearse that one against the replay CLI with the
keyless provider so nothing on stage depends on a network.

## Before you record

```bash
git pull && docker compose up --build       # verify the judge's path works
python tests/test_gates.py                  # 11/11
python -m evaluation.run_bench --split dev --ablations
```

**Record with the keyless provider**, so the video costs no quota and every
run looks the same. In PowerShell, in the `ripple` folder (this overrides
`.env` for that window only):

```powershell
$env:RIPPLE_PROVIDER="stub"
uvicorn ripple.server:app --port 8000
```

Only pick scenarios whose name starts with `dev_`. The list also shows
`heldout_` scenarios; do not open them — that split is reserved.

Open `http://localhost:8000`. Set the speed selector to **1× realtime** — the
whole point is that a judge sees the clock. Full screen, 1080p, no browser
chrome, no notifications. Record audio separately if your mic is poor; a clean
voiceover over a clean screen recording beats both done at once.

---

## 0:00 – 0:30 · The gap

**Screen:** the dashboard, idle. Scenario `dev_multi_01` selected.

> "A customer calls support. They say this —" *(play the first utterance, stop
> the moment they finish)* "— four questions in one breath. The agent has about
> a second and a half before the silence gets awkward.
>
> Every assistant built for this waits for the customer to finish and *then*
> starts searching. And notice: the customer isn't talking to the assistant.
> There's no send button, no pause, nobody to ask to repeat. A turn-based
> system is waiting for a query that never arrives."

## 0:30 – 2:00 · Beat 1 — early retrieval and multi-intent

**Screen:** hit Replay. Let it run at 1×. Say nothing for the first four
seconds — let the transcript fill and the stability curve climb.

> "Watch the middle column. That line is the controller asking one question on
> every fragment: *if they say more words, will different documents come back?*
> It costs one lookup. No model, no tokens.
>
> **There.** *(point at the RETRIEVE marker crossing θ)* It's searching — and
> the customer is still talking.
>
> A second question opens, and it branches — a parallel search, not a blend.
> Then the fusion panel: a guaranteed slot for each question. If you pool them
> instead, one loud question takes the whole budget and a third of the answer
> disappears without any metric noticing."

**Screen:** the TTFT tile flips negative.

> "Time to first token, measured against the end of the sentence: about
> **minus one second** here. This recording uses our keyless mode, where the
> model answers instantly, so read it as how early Ripple *starts*. On a real
> model — Gemini — when a customer asks two things, the first answer lands a
> median **three seconds before they finish**. A turn-based system cannot
> produce a negative number, because it hasn't started yet."

## 2:00 – 3:05 · Beat 2 — late detail, refined not restarted

**Screen:** continue into `dev_late_01`, or run it next.

> "Now the customer adds something."

*(the "I bought it in Dubai" turn streams in)*

> "Watch the diff. **Two claims preserved — byte-identical text, byte-identical
> citations. One superseded. One added, from a single delta search.**
>
> The citation-drift counter reads zero, and that's structural, not lucky. The
> answer isn't a paragraph — it's a set of claims, each bound to the evidence it
> rests on. Untouched claims are never regenerated, so their citations *can't*
> move. No model ever writes a citation string.
>
> And the scope came from the corpus, not from a rule we wrote. The constraint
> retrieved the cross-border policy, so it invalidated the claims that assumed a
> domestic purchase — and left the device diagnosis standing, because a fault
> doesn't change based on where you bought the phone."

**Screen:** click any citation. The exact passage opens.

> "Every claim is one click from the passage it came from."

## 3:05 – 3:35 · Beat 3 — suppression, then abstention

> "The agent asks for a reformat."

*(the "give me that in two bullets" turn)*

> "`retrieval_required: false`. No search, no tokens, citations retained, none
> invented. It's answered out of session state, which is only possible because
> the answer has structure.
>
> And now the opposite —"

*(run `dev_hole_02`, the insurance question)*

> "— a question our corpus genuinely doesn't answer. It says so, and cites
> nothing. That's the uncertainty indicator the theme asks for, and it's the
> behaviour most demos quietly skip."

## 3:35 – 4:30 · The numbers

**Screen:** cut to the terminal. `python -m evaluation.run_bench --split dev --ablations`
(pre-run; show the output, don't wait for it).

> "Forty scenarios, fifty-one labelled turns, four systems on one harness.
> All six acceptance gates pass. And on a real model — Gemini 3.5 Flash Lite,
> all forty scenarios — 158 of 158 claims were backed by the passage they
> cite, with zero invented citations, for 1.44 times static RAG's tokens.
>
> Against naive streaming — retrieve on every chunk — we use **thirty per cent
> fewer retrievals** and trigger on **zero** of the turns that needed no
> retrieval at all. Naive streaming triggers on **all** of them. That's the half
> of gate G2 that's easy to leave out of a report.
>
> Five ablations, each changing one thing. The sharpest: turn off
> coverage-budgeted fusion and recall stays *identical* at 0.960 — every right
> passage was still retrieved — but coverage drops from 0.772 to 0.718. The
> chunks were found and then not used. That's sub-intent starvation, measured.
>
> Two components we built are gone, because they measured worse than nothing:
> BM25, and our own reranker."

## 4:30 – 5:00 · Reproducibility and honesty

**Screen:** clean terminal. `docker compose up`. Then the replay CLI.

> "One command, clean machine, no API key, no GPU. The replay CLI runs the whole
> suite headless and emits the guide's own §4 JSON — their field names — so a
> private harness written against the spec can read our output unmodified.
>
> Three things we won't overclaim. On a real model, our slowest answers are
> slower than static RAG's, because we make more calls. Static RAG slightly
> beats us at covering every sub-question. And our bottleneck is selection,
> not retrieval. All three are in the report, with numbers."
>
> Ripple. Retrieval that starts before the sentence lands, and an answer that
> updates itself instead of starting over."

---

## Checklist before upload

- [ ] Under 5:00. Overrunning is a guideline breach, and the checklist asks.
- [ ] All six §8 behaviours visible: early retrieval · multi-intent · late-detail
      refinement · suppression · citation traceability · telemetry.
- [ ] The clock is legible whenever you claim something happened early.
- [ ] The citation-drift **0** is on screen during beat 2 — that is gate G5,
      visible.
- [ ] No API keys, `.env` contents, or personal information on screen at any
      point. Check the terminal scrollback before you record.
- [ ] Uploaded unlisted to YouTube or shared on Drive **with link sharing on**.
      A private link is a failed submission.
- [ ] Link pasted into the README and into the deck's checklist slide.
