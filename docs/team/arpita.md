# Arpita — proofread the documents, own the jury Q&A

**Why this matters.** Judges read the README and the evaluation report,
and the top 15 teams face a live Q&A on 15 October. Mistakes in the text
cost credibility; a team that answers well together scores.

**Time:** about 3 hours. **No installation needed** — everything is done
in the browser on GitHub.

## Part 1 — proofread (~1.5 h)

Read these on https://github.com/deathKiller10/ripple :

1. `README.md`
2. `docs/evaluation-report.md` (the long one — sections 0, 2, 2A, 2B and 6
   matter most)
3. the deck: `VITVellore_Spark_Submission.pptx` (download and open)

Look for:

- typos, broken sentences, anything you had to read twice
- **numbers that disagree between files.** Key ones to cross-check:

  | Figure | Should read |
  |---|---|
  | Ripple dev: early retrieval / false triggers / multi-intent | 0.884 / 0.000 / 0.889 |
  | Ripple dev: recall / coverage | 0.960 / 0.772 |
  | Held-out: early / multi-intent / recall / coverage | 0.868 / 0.875 / 0.937 / 0.778 |
  | Real model (gemini-3.5-flash-lite): claims supported | 158 of 158 |
  | Real model: TTFT on two-question calls | −3.17 s |
  | Corpus / scenarios | 60 docs, 151 chunks / 40 dev + 34 held-out |

- **one known item to confirm with Priyanshu:** the README's third line
  says "due 25 Sep 2026". Samsung's PDF says 25 Sep; the team is working
  to 30 Sep. Ask him which is correct and fix the line accordingly.

**How to fix text on GitHub:** open the file → pencil icon (Edit) → change
the words → **Commit changes** → write what you fixed (e.g. "docs: fix
typos in evaluation report §2A") → commit directly to `master`.

Rules:
- **Text only.** Never edit `.py`, `.html`, `.json`, `.jsonl` files or
  anything in `results*` folders.
- **Never change a number yourself.** If two files disagree, tell
  Priyanshu which and where — the right value comes from the results
  files, not from either document.
- Keep a list of what you changed; it goes in the team contributions
  section.

## Part 2 — own the jury Q&A (~1.5 h)

1. Read `docs/judge-qa.md`. It has prepared answers to the questions a
   jury is likely to ask.
2. Practise answering the first ten **out loud, in your own words**,
   without reading.
3. Add at the bottom any question you think a judge would ask that is not
   there, with the best answer you can find in the report. Mark any you
   cannot answer — Priyanshu will fill those in.
4. Add a short table at the top saying which questions each of the four
   of you will take in the final (agree it with the team).

Commit the changes to `docs/judge-qa.md` on GitHub the same way as above.

## In the video

You speak 3:35–4:30: the numbers. The script is in
`docs/demo-script.md`, section "The numbers". Coordinate with Anushka,
who is putting the video together.
