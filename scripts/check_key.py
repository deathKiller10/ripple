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



# Names that exist but are not text-generation models for our purposes.
_SKIP = ("tts", "image", "embedding", "embed", "aqa", "vision", "imagen",
         "veo", "audio", "live", "native-audio", "dialog", "computer-use",
         "robotics", "guard")


def _rank(name: str) -> tuple:
    """Order candidates: cheap-and-fast first, newest first, aliases last.

    A benchmark wants a flash-class model -- cheap, quick, and the tier a
    support assistant would realistically run on. Pinned names sort ahead of
    "-latest" aliases because an alias that moves under you makes a benchmark
    unreproducible.
    """
    import re

    lowered = name.lower()
    flash = 0 if "flash" in lowered else 1
    lite = 0 if "lite" in lowered else 1
    preview = 1 if ("preview" in lowered or "exp" in lowered) else 0
    alias = 1 if "latest" in lowered else 0
    m = re.search(r"(\d+(?:\.\d+)?)", lowered)
    version = -float(m.group(1)) if m else 0.0
    # Version outranks "lite": a newer flash model beats an older lite one.
    # The first ordering had these swapped and preferred gemini-2.5-flash-lite
    # over gemini-3.8-flash, which is exactly backwards.
    return (flash, preview, alias, version, lite, name)


def find_working_model(provider, models, requested, limit=6, hint=""):
    """Probe real generation calls until one succeeds.

    Costs one tiny call per candidate, capped. Worth it: the alternative is
    discovering the model is dead partway through a ten-minute benchmark.

    `hint` is a model name parsed out of the API's own error text. Providers
    often name the replacement in the message that rejects the old one
    ("Please update your code to use models/x"), and that is a better signal
    than any ordering we invent, so it is tried first.
    """
    candidates = [m for m in models
                  if not any(sk in m.lower() for sk in _SKIP)
                  and m != requested]
    candidates.sort(key=_rank)
    if hint and hint != requested:
        candidates = [hint] + [c for c in candidates if c != hint]
    tried = []
    for name in candidates[:limit]:
        print(f"     trying {name} ...", end="", flush=True)
        ok, detail = provider.try_model(name)
        if ok:
            print(" WORKS")
            return name, detail, tried
        short = detail.split(".")[0][:70]
        print(f" no ({short})")
        tried.append((name, detail))
    return None, "", tried


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
    print(f"  Making one small generation call with "
          f"{cfg.synthesis.model!r} ...")
    ok, detail = provider.preflight()

    if not ok and models:
        # The requested model did not work. Do NOT trust the models list to
        # tell us what will -- it advertises names that 404 on generation.
        # Probe until something actually generates.
        print()
        print(f"  {cfg.synthesis.model!r} failed:")
        print(f"     {detail[:300]}")
        print()
        import re as _re
        m = _re.search(r"models/([A-Za-z0-9.\-]+)", detail.split("no longer")[-1])
        hint = m.group(1) if m else ""
        if hint:
            print(f"  The API suggested {hint!r}; trying that first.")
        print("  Probing for a model that actually generates:")
        found, reply, tried = find_working_model(provider, models,
                                                 cfg.synthesis.model,
                                                 hint=hint)
        if found:
            print()
            line("=")
            print(f"  RESULT: WORKING MODEL FOUND \u2014 {found}")
            line("=")
            print(f"  It replied {reply!r}.")
            print()
            print("  Put this line in your .env (replace any RIPPLE_MODEL "
                  "line):")
            print()
            print(f"     RIPPLE_MODEL={found}")
            print()
            if "latest" in found or "preview" in found:
                # Say this out loud rather than quietly recording a number
                # against a name that may not mean the same thing next week.
                print("  NOTE: that is a moving name, not a pinned version. It")
                print("        can change model under you, so the benchmark")
                print("        number it produces is only reproducible if you")
                print("        also record the date you ran it.")
                print()
            print("  PowerShell:")
            print(f'     Add-Content -Path .env -Value "RIPPLE_MODEL={found}"')
            print()
            print("  Then run this script once more to confirm, and start "
                  "the benchmark.")
            return 0
        print()
        print("  RESULT: no model generated successfully.")
        print()
        print("  Tried:")
        for name, err in tried:
            print(f"     {name}: {err.split('.')[0][:80]}")
        print()
        print("  This usually means the project has no free-tier quota for")
        print("  generation, even though the key authenticates. Check quota")
        print("  at https://aistudio.google.com/ and try again.")
        return 1

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
