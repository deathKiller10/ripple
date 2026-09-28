# Anushka — dashboard check and the demo video

**Why this matters.** The video is the last missing deliverable, and
Samsung asks for it explicitly (max 5 minutes). You run the dashboard on
screen and narrate the demo — the longest speaking part.

**Time:** about 3 hours (setup, practice, recording).

## Part 1 — get the dashboard running (~1 h)

Either use Priyanshu's laptop, where it already works, or set it up on
yours by following steps 1–4 of [souptik.md](souptik.md). Then, in
PowerShell in the `ripple` folder:

```powershell
$env:RIPPLE_PROVIDER="stub"
uvicorn ripple.server:app --port 8000
```

Open http://localhost:8000. The first line makes it run **without any API
key**, so every run looks the same and costs nothing.

**Only choose scenarios whose name starts with `dev_`.** The list also
shows `heldout_` ones — never open those.

## What the screen shows

- **Left — Live transcript.** What the customer is saying, word by word.
- **Middle — Controller & retrieval timeline.** The system's decisions
  over time. The rising line is its confidence that the request has
  settled enough to search. `RETRIEVE` = it searched, `SUPPRESS` = it
  chose not to, grey codes like `DOC_KB_07 §2` = documents it found.
- **Right — Answer.** Top-left box: how many seconds *before* the
  customer finished the first answer appeared (negative = early). Below:
  each claim with its source. **Click a source code** and the exact
  passage opens at the bottom right.

## Part 2 — practise the four scenarios (~45 min)

Set speed to **1× realtime**. For each, press **Replay** and watch it to
the end:

| Scenario | What to point at |
|---|---|
| `dev_multi_01` | `RETRIEVE` fires while the customer is still talking; the question splits into three (`decompose`); the top-right box shows about **−7 s** |
| `dev_late_01` | "I bought it in Dubai" arrives late: the green `preserved`, red `superseded` labels and **0 citation drift** — the answer is updated, not restarted. Click a citation. |
| `dev_suppress_01` | "Please repeat that in two bullets": `SUPPRESS`, `retrieval_required: false` — no new search |
| `dev_hole_02` | An insurance question the documents do not cover: the orange **Uncertainty** box, and no citation |

Note anything that looks wrong or confusing and tell Priyanshu — that is
part of your check.

## Part 3 — record (~1 h)

The full word-for-word script is `docs/demo-script.md`. Speaking parts:

| Time | Who | Section |
|---|---|---|
| 0:00–0:30 | Priyanshu | The gap — why listening assistants need this |
| 0:30–3:35 | **Anushka** | Live demo: the four scenarios above |
| 3:35–4:30 | Arpita | The numbers |
| 4:30–5:00 | Souptik | Runs anywhere, no API key; what we won't overclaim |

How to record:

1. Use **Clipchamp** (Start menu, built into Windows 11) → Record →
   Screen and camera, or screen only with microphone.
2. Browser full screen, notifications off, nothing personal on screen,
   no `.env` or keys visible anywhere.
3. It is fine to record each person's part separately and join the clips
   in Clipchamp. Keep the total **under 5:00**.
4. Speak slowly. Reading from the script is fine.

## Part 4 — upload

- YouTube → **Unlisted** (not Private), or Google Drive → Share →
  "Anyone with the link can view".
- Open the link in a private/incognito window to check it plays.
- Send the link to Priyanshu; he adds it to the README.

## Your commit

Write `docs/team/dashboard-check-anushka.md`: which laptop you used, each
scenario worked / anything odd, and the final video length. Commit it
under your own name:

```powershell
git config user.name "Anushka Paul"
git config user.email "<the email on your GitHub account>"
git add docs/team/dashboard-check-anushka.md
git commit -m "docs: dashboard walkthrough check for the demo video"
git pull
git push
```

(Or on GitHub: open the `docs/team` folder → Add file → Create new file →
paste → Commit changes.)
