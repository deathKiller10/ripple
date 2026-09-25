#!/usr/bin/env python3
"""Two-second check that your LLM key works.

    python scripts/check_key.py

Makes exactly one tiny API call and tells you plainly whether it worked.
Run this before any benchmark: a rejected key inside a long run wastes the
run, and until this existed it looked like a hang rather than an error.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ripple.config import Config                       # noqa: E402
from ripple.synthesis.providers import build_provider   # noqa: E402


def main() -> int:
    cfg = Config()
    wanted = (cfg.synthesis.provider or "stub").lower()
    print(f"provider requested : {wanted}")
    print(f"model              : {cfg.synthesis.model}")

    key_var = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}.get(wanted)
    if key_var:
        key = os.environ.get(key_var, "")
        if key:
            print(f"{key_var:<19}: set, {len(key)} characters, "
                  f"ends ...{key[-4:]}")
            if wanted == "gemini" and not key.startswith("AIza"):
                print()
                print("  NOTE: Google AI Studio API keys normally begin 'AIza'.")
                print("  A key starting with something else is usually an OAuth")
                print("  credential, which this API will reject. Create a real")
                print("  API key at https://aistudio.google.com/app/apikey")
        else:
            print(f"{key_var:<19}: NOT SET")
            print()
            print("  Create a file called .env in the repo root containing:")
            print(f"     RIPPLE_PROVIDER={wanted}")
            print(f"     {key_var}=your_key_here")
            print("     RIPPLE_RPM=12")
            return 1

    provider = build_provider(cfg.synthesis, cfg.cost)
    if provider.name == "stub":
        print()
        print("RESULT: falling back to the keyless stub — no usable key.")
        return 1

    print()
    print("making one test call ...")
    ok, detail = provider.preflight()
    print()
    if ok:
        print(f"RESULT: WORKING. The model replied {detail!r}.")
        print()
        print("You are clear to run:")
        print("  python -m evaluation.run_bench --split dev --provider "
              f"{wanted} \\")
        print("         --systems B1_static_rag,B3_ripple --out results_gemini")
        return 0

    print(f"RESULT: REJECTED — {detail}")
    print()
    print("Common causes, in order of likelihood:")
    print("  1. The key is an OAuth credential, not an API key "
          "(should start 'AIza').")
    print("  2. The key was revoked or belongs to a different project.")
    print("  3. The model name is wrong for your account. Try setting")
    print("     RIPPLE_MODEL=gemini-1.5-flash in .env and run this again.")
    print("  4. No billing/quota on the project.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
