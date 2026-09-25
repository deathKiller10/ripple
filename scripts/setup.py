#!/usr/bin/env python3
"""One-command setup and health check.

    python scripts/setup.py

Builds the corpus, the classifier labels, the benchmark scenarios and the
search index, then runs the test suite and tells you what to do next. Safe to
re-run at any time; everything it builds is regenerated from source files in
the repo, so nothing is lost.

Cross-platform on purpose: plain Python rather than a shell script, so the
same command works in Windows PowerShell, macOS Terminal and WSL.
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

OK = "  [ok]  "
BAD = "  [!!]  "
INFO = "         "


def hr(title=""):
    print("\n" + "=" * 68)
    if title:
        print(title)
        print("=" * 68)


def run(desc, args, fatal=True):
    print(f"\n>>> {desc}")
    r = subprocess.run([PY] + args, cwd=ROOT, capture_output=True, text=True)
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    if out:
        print("\n".join("     " + l for l in out.splitlines()[-12:]))
    if r.returncode != 0:
        print(BAD + "failed")
        if err:
            print("\n".join("     " + l for l in err.splitlines()[-15:]))
        if fatal:
            print("\nStopped. Fix the error above, then run this script again.")
            sys.exit(1)
        return False
    return True


def check_imports():
    missing = []
    for mod, pkg in [("numpy", "numpy"), ("sklearn", "scikit-learn"),
                     ("faiss", "faiss-cpu"), ("fastapi", "fastapi"),
                     ("httpx", "httpx"), ("uvicorn", "uvicorn")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    return missing


def main():
    hr("Ripple setup")
    print(f"{INFO}repo     {ROOT}")
    print(f"{INFO}python   {sys.version.split()[0]}  ({PY})")

    if sys.version_info < (3, 10):
        print(BAD + "Python 3.10 or newer is required.")
        sys.exit(1)

    missing = check_imports()
    if missing:
        print(BAD + "missing packages: " + ", ".join(missing))
        print(INFO + "run this first, then run me again:")
        print(f"\n     {PY} -m pip install -r requirements.txt\n")
        sys.exit(1)
    print(OK + "all required packages are installed")

    hr("Building data")
    run("corpus", ["data/corpus/care/build_corpus.py"])
    run("classifier labels", ["data/labels/build_labels.py"])
    run("benchmark scenarios", ["data/scenarios/build_scenarios.py"])
    run("search index", ["-c",
                         "from ripple.retrieval.index import build_and_save;"
                         "i=build_and_save('data/corpus/care','.index',"
                         "__import__('os').environ.get('RIPPLE_EMBEDDER',"
                         "'tfidf-svd'));print(i.stats())"])

    hr("Checking it works")
    passed = run("property tests", ["tests/test_gates.py"], fatal=False)
    if not passed:
        print(BAD + "tests failed — do not push this. Send the output above "
                    "to Claude.")
        sys.exit(1)

    # --- provider status ---------------------------------------------------
    hr("LLM provider")
    env_path = os.path.join(ROOT, ".env")
    has_env = os.path.exists(env_path)
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key and has_env:
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip()

    if not has_env:
        print(BAD + "no .env file found")
        print(INFO + "The system works without one (keyless mode), but the")
        print(INFO + "groundedness and cost numbers stay meaningless.")
        print(INFO + "Create a file called  .env  in this folder containing:")
        print("\n     RIPPLE_PROVIDER=gemini")
        print("     GEMINI_API_KEY=your_key_here")
        print("     RIPPLE_RPM=12\n")
    elif not key:
        print(BAD + ".env exists but GEMINI_API_KEY is empty")
    else:
        print(OK + f"Gemini key found (ends ...{key[-4:]})")
        print(INFO + "Never commit .env, never paste the key into a chat.")

    hr("What to run next")
    print("""
  Keyless benchmark (fast, always works):
      python -m evaluation.run_bench --split dev --ablations

  Gemini benchmark — START WITH THIS ONE. Two systems only, about
  130 calls, roughly 11 minutes at 12 requests/minute:
      python -m evaluation.run_bench --split dev --provider gemini \\
             --systems B1_static_rag,B3_ripple --out results_gemini

  All four systems with Gemini — ONLY if the above finished and you
  have quota left. B2 retrieves on every chunk by design, so it alone
  is ~400 calls and may exhaust a free tier:
      python -m evaluation.run_bench --split dev --provider gemini \\
             --out results_gemini_full

  The dashboard:
      uvicorn ripple.server:app --port 8000     →  http://localhost:8000
""")
    print("=" * 68)
    print("Setup complete.")
    print("=" * 68)


if __name__ == "__main__":
    main()
