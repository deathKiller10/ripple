#!/usr/bin/env python3
"""Diagnose your LLM key in about five seconds.

    python scripts/check_key.py

Runs two checks, in this order, because they fail for different reasons and
the fix is different:

  1. CAN THE KEY TALK TO THE API AT ALL?  (lists available models)
  2. DOES THE MODEL WE ASK FOR EXIST FOR THIS KEY?  (one tiny generation)

Splitting them matters. A rejected key and a wrong model name produce the
same unhelpful error from a generation call, and they need opposite fixes.

Deliberately makes no judgement about what a key "should look like". Key
formats change, and a prefix heuristic that is out of date sends you looking
in the wrong place -- which is exactly what happened here once already.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ripple.config import Config                       # noqa: E402
from ripple.synthesis.providers import build_provider   # noqa: E402


def line(char="-"):
    print(char * 70)


def main() -> int:
    cfg = Config()
    wanted = (cfg.synthesis.provider or "stub").lower()

    # If the provider says "stub" but a key is sitting in the environment,
    # something ate the RIPPLE_PROVIDER line -- almost always an encoding
    # problem in .env. Say so, then test the key anyway rather than stopping
    # on a technicality.
    if wanted == "stub":
        for guess, var in (("gemini", "GEMINI_API_KEY"),
                           ("openai", "OPENAI_API_KEY")):
            if os.environ.get(var):
                print("  NOTE: RIPPLE_PROVIDER is not set, but " + var)
                print(f"        is. Assuming provider={guess!r} and testing "
                      "it anyway.")
                print("        Fix .env so RIPPLE_PROVIDER=" + guess
                      + " is picked up,")
                print("        or the benchmark will need --provider "
                      + guess + " every time.")
                print()
                wanted = guess
                cfg.synthesis.provider = guess
                break

    line("=")
    print("Ripple key check")
    line("=")
    print(f"  provider requested : {wanted}")
    print(f"  model requested    : {cfg.synthesis.model}")

    key_var = {"gemini": "GEMINI_API_KEY",
               "openai": "OPENAI_API_KEY"}.get(wanted)
    if not key_var:
        print("\n  Provider is 'stub' (keyless). Nothing to check.")
        print("  Set RIPPLE_PROVIDER=gemini in .env to use a real model.")
        return 0

    key = os.environ.get(key_var, "")
    if not key:
        print(f"  {key_var:<18} : NOT SET")
        print()
        print("  Create a file called  .env  in the repo root containing:")
        print(f"     RIPPLE_PROVIDER={wanted}")
        print(f"     {key_var}=your_key_here")
        print("     RIPPLE_RPM=12")
        return 1
    print(f"  {key_var:<18} : set, {len(key)} chars, ends ...{key[-4:]}")

    provider = build_provider(cfg.synthesis, cfg.cost)
    if provider.name == "stub":
        print("\n  RESULT: fell back to the keyless stub. No usable key.")
        return 1

    # --- check 1: authentication ------------------------------------------
    print()
    line()
    print("CHECK 1  Can this key reach the API?")
    line()
    if not hasattr(provider, "list_models"):
        print("  (not supported for this provider; skipping to check 2)")
        models = None
    else:
        ok, result = provider.list_models()
        if not ok:
            print(f"  FAILED: {result}")
            print()
            print("  The key itself is being rejected. Things to check:")
            print("    1. Is the Generative Language API enabled on the")
            print("       project this key belongs to? A brand-new project")
            print("       usually has it switched off. Enable it here:")
            print("       https://console.cloud.google.com/apis/library/"
                  "generativelanguage.googleapis.com")
            print("    2. Does the key have an application restriction (IP,")
            print("       referrer) or an API restriction that excludes the")
            print("       Generative Language API?")
            print("    3. Was the key created in Google AI Studio")
            print("       (https://aistudio.google.com/app/apikey) rather")
            print("       than as a generic Cloud credential?")
            print("    4. Is a proxy, VPN or firewall blocking the request?")
            return 1
        models = result
        print(f"  OK. This key can use {len(models)} model(s).")
        show = [m for m in models if "gemini" in m][:12] or models[:12]
        for m in show:
            print(f"     - {m}")
        if len(models) > len(show):
            print(f"     ... and {len(models) - len(show)} more")

    # --- check 2: the specific model --------------------------------------
    print()
    line()
    print(f"CHECK 2  Does {cfg.synthesis.model!r} work?")
    line()
    if models is not None and cfg.synthesis.model not in models:
        print(f"  {cfg.synthesis.model!r} is NOT in the list above.")
        pref = ("gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash",
                "gemini-flash-latest")
        pick = next((p for p in pref if p in models),
                    next((m for m in models if "flash" in m),
                         models[0] if models else None))
        if pick:
            print()
            print(f"  Use this instead — add to your .env:")
            print(f"     RIPPLE_MODEL={pick}")
            print()
            print("  Then run this script again.")
        return 1

    print("  Making one small generation call ...")
    ok, detail = provider.preflight()
    print()
    if ok:
        line("=")
        print(f"  RESULT: WORKING. The model replied {detail!r}.")
        line("=")
        print()
        print("  You are clear to run the benchmark:")
        print("     python -m evaluation.run_bench --split dev \\")
        print(f"            --provider {wanted} "
              "--systems B1_static_rag,B3_ripple \\")
        print("            --out results_gemini")
        return 0

    print(f"  FAILED: {detail}")
    print()
    print("  Authentication worked, so the key is fine — this is about the")
    print("  model or your quota. Check the model name above, and whether")
    print("  the project has free-tier quota left for today.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
