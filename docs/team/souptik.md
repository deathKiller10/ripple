# Souptik — reproduce Ripple on a fresh laptop

**Why this matters.** A judge will download the repo onto their own
computer and follow the README. So far the project has only ever been set
up on Priyanshu's laptop. You are the first stranger. If the README has a
gap, you find it before the judges do.

**Time:** about 2 hours, most of it waiting for installs.

## Before you start

- A GitHub account. Priyanshu will add you as a collaborator — accept the
  email invitation (or open https://github.com/deathKiller10/ripple and
  accept the banner).
- **Git for Windows** (https://git-scm.com). Check in PowerShell:
  `git --version`
- **Python 3.11** — not 3.13 or 3.14; on 3.14 `pip install` fails at
  numpy (found by Anushka on 30 Sep). Install it from python.org if needed,
  and in step 2 use `py -3.11 -m venv .venv` instead of `python -m venv .venv`.
- Keep a notes file open. **Write down every step, how long it took, and
  anything that failed or confused you.** That record is your deliverable.

## Steps (Windows PowerShell)

1. Pick a folder that is **not** inside Priyanshu's files, then download
   the project:
   ```powershell
   git clone https://github.com/deathKiller10/ripple.git
   cd ripple
   ```
2. Make a private Python environment so nothing on your laptop is changed:
   ```powershell
   python -m venv .venv
   Set-ExecutionPolicy -Scope Process Bypass
   .venv\Scripts\Activate.ps1
   ```
   Your prompt should now start with `(.venv)`.
3. Install the dependencies (this can take several minutes):
   ```powershell
   pip install -r requirements.txt
   ```
4. Run the one-command setup from the README:
   ```powershell
   python scripts/setup.py
   ```
   Note what it says at the end. It should report that no LLM key is
   configured — that is correct and expected.
5. Run the property tests:
   ```powershell
   python tests/test_gates.py
   ```
   Expected last line: `11/11 passed`
6. Run the benchmark on the **dev** split (a few minutes):
   ```powershell
   python -m evaluation.run_bench --split dev
   ```
   Expected: six `[PASS]` lines, and the `B3_ripple` row starting
   `0.884  0.000  0.889  0.960  0.772`. Tiny differences in the TTFT
   column's last digit are fine; anything else, write down.
7. Start the dashboard and look at it:
   ```powershell
   uvicorn ripple.server:app --port 8000
   ```
   Open http://localhost:8000, pick `dev_multi_01`, press **Replay**.
   Text should appear on the left and an answer with citations on the
   right. Stop it with **Ctrl + C**.
8. **Optional**, only if you already have Docker Desktop:
   ```powershell
   docker compose up --build
   ```
   then open http://localhost:8000 again. This is exactly the judges' path.

## Your deliverable

Create `docs/team/reproduction-souptik.md` in your clone with:

- your Windows version, Python version and whether you used Docker
- each step above: worked / failed, and how long it took
- the numbers from step 6 (copy the table)
- anything unclear in the README, in your own words

Then commit and push it **under your own name**:

```powershell
git config user.name "Souptik Hazra"
git config user.email "<the email on your GitHub account>"
git add docs/team/reproduction-souptik.md
git commit -m "docs: independent reproduction on a fresh Windows laptop"
git pull
git push
```

If any step fails: **do not fix code.** Put the exact error in your notes
and send it to Priyanshu the same day — there is still time to fix the
README.

**In the video** you speak the last part (4:30–5:00): that the project
runs with one command on a clean machine with no API key — which you
will have just checked yourself.
