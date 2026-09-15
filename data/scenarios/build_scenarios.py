#!/usr/bin/env python3
"""Benchmark scenarios with gold labels.

SPLIT POLICY, declared before any tuning happens:

    dev      thresholds (theta, nu, compoundness, suppression) are calibrated
             on this split and only this split
    heldout  run ONCE, on day 8, after feature freeze. Never inspected while
             tuning. Reported numbers come from here.

Declaring the split in advance is cheap and it is the only credible answer to
"did you overfit to your own demo?". A team that tunes on everything cannot
make that claim afterwards.

G2 ELIGIBILITY TAXONOMY -- `turn_kind` tells the harness what each turn is,
which is what makes gate G2 measurable at all:

    informational   asks for something the corpus must supply.
                    needs_retrieval = True. Counts toward the G2 numerator.
    refinement      supplies a fact that changes a prior answer.
                    needs_retrieval = True, but only a DELTA search is correct;
                    a full re-search counts against delta efficiency.
    presentation    reformats or repeats prior output.
                    needs_retrieval = False. A retrieval here is a FALSE
                    TRIGGER -- the half of G2 that teams who retrieve on every
                    chunk quietly omit.
    social          greetings, acknowledgements, hold messages.
                    needs_retrieval = False.
    uncoverable     asks something the corpus deliberately does not cover.
                    needs_retrieval = True, and the correct answer is an
                    explicit uncertainty indicator, not a confident reply.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SCENARIOS = []


def scn(sid, split, chunks, gold, notes=""):
    SCENARIOS.append({
        "scenario_id": sid, "split": split, "notes": notes,
        "chunks": [{"t": t, "text": x, "speaker": sp, "is_final": f}
                   for (t, x, sp, f) in chunks],
        "gold": gold,
    })


def turn(i, needs, kind, subs=None, docs=None, keep=None, unc=False):
    return {
        "utterance_index": i, "needs_retrieval": needs, "turn_kind": kind,
        "gold_sub_intents": subs or [], "gold_doc_ids": docs or {},
        "claims_that_must_not_change": keep or [], "expect_uncertainty": unc,
    }


C, A = "customer", "agent"

# ===========================================================================
# DEV SPLIT
# ===========================================================================

scn("dev_multi_01", "dev", [
    (0.0, "My Galaxy S24 screen started", C, False),
    (0.7, "flickering after the last update", C, False),
    (1.4, "and the battery is dead by two in the afternoon now", C, False),
    (2.1, "is any of this covered", C, False),
    (2.8, "and how long would a repair take", C, True),
], [turn(0, True, "informational",
         ["display flicker after update", "battery drain after update",
          "warranty coverage", "repair turnaround"],
         {"display flicker after update": ["DOC_KB_01 §1", "DOC_KB_01 §3"],
          "battery drain after update": ["DOC_KB_03 §1", "DOC_KB_03 §3"],
          "warranty coverage": ["DOC_WAR_01 §3", "DOC_WAR_01 §2"],
          "repair turnaround": ["DOC_POL_11 §1", "DOC_POL_11 §2"]})],
    "Four intents in one breath. The guide's Example 1 shape.")

scn("dev_late_01", "dev", [
    (0.0, "Can you tell me whether a display replacement", C, False),
    (0.8, "is covered on a phone that is ten months old", C, True),
    (6.0, "Oh I should mention", C, False),
    (6.6, "I bought it in Dubai while travelling", C, True),
], [turn(0, True, "informational", ["display replacement warranty coverage"],
         {"display replacement warranty coverage": ["DOC_WAR_01 §3",
                                                    "DOC_WAR_01 §2"]}),
    turn(1, True, "refinement", ["cross-border purchase coverage"],
         {"cross-border purchase coverage": ["DOC_WAR_03 §1", "DOC_WAR_03 §2"]},
         keep=["DOC_KB_01", "DOC_POL_11"])],
    "Guide Example 2. Region constraint must supersede the domestic warranty "
    "claim and leave device-level claims standing.")

scn("dev_suppress_01", "dev", [
    (0.0, "What is the turnaround for a display repair", C, True),
    (5.0, "Please repeat that in two bullets", A, True),
], [turn(0, True, "informational", ["display repair turnaround"],
         {"display repair turnaround": ["DOC_POL_11 §2", "DOC_POL_11 §1"]}),
    turn(1, False, "presentation")],
    "Guide Example 3. Zero retrievals and zero tokens on turn 2.")

scn("dev_hole_01", "dev", [
    (0.0, "The customer wants to know", C, False),
    (0.7, "if we reimburse the cost of the screen protector", C, True),
], [turn(0, True, "uncoverable", ["screen protector reimbursement"], {},
         unc=True)],
    "Deliberate coverage hole. Correct behaviour is an explicit uncertainty "
    "indicator, not a plausible improvisation.")

scn("dev_single_01", "dev", [
    (0.0, "What proof of purchase do we accept", C, True),
], [turn(0, True, "informational", ["proof of purchase requirements"],
         {"proof of purchase requirements": ["DOC_WAR_01 §4"]})],
    "Single intent. Must NOT be decomposed -- tests pitfall 5.")

scn("dev_social_01", "dev", [
    (0.0, "Hi there thanks for holding", A, False),
    (0.9, "let me just pull that up for you", A, True),
], [turn(0, False, "social")],
    "No retrieval. Tests the false-trigger half of G2.")

scn("dev_conflict_01", "dev", [
    (0.0, "What is the labour charge", C, False),
    (0.7, "on an out of warranty display replacement", C, True),
], [turn(0, True, "informational", ["out of warranty labour charge"],
         {"out of warranty labour charge": ["DOC_POL_09 §1", "DOC_POL_07 §1"]})],
    "Contradiction pair: DOC_POL_09 supersedes DOC_POL_07 §1. The answer must "
    "surface the current figure and may cite both.")

scn("dev_multi_02", "dev", [
    (0.0, "He says the phone overheats when charging", C, False),
    (0.8, "and he wants to know what a battery costs", C, False),
    (1.6, "and whether he can get a loaner", C, True),
], [turn(0, True, "informational",
         ["overheating while charging", "battery part cost", "loaner device"],
         {"overheating while charging": ["DOC_KB_11 §2", "DOC_KB_11 §3"],
          "battery part cost": ["DOC_PRT_03 §2", "DOC_PRT_03 §1"],
          "loaner device": ["DOC_POL_15 §1", "DOC_POL_15 §2"]})],
    "Three intents, explicit coordination.")

scn("dev_late_02", "dev", [
    (0.0, "How fast can we turn around a display repair", C, True),
    (5.0, "It is an imported unit from Singapore", C, True),
], [turn(0, True, "informational", ["display repair turnaround"],
         {"display repair turnaround": ["DOC_POL_11 §2", "DOC_POL_11 §1"]}),
    turn(1, True, "refinement", ["imported device turnaround"],
         {"imported device turnaround": ["DOC_POL_12 §2", "DOC_POL_12 §1"]})],
    "Near-duplicate pair: DOC_POL_12 restates DOC_POL_11 with different SLAs.")

scn("dev_distract_01", "dev", [
    (0.0, "The customer says the screen keeps flickering", C, False),
    (0.8, "it is a Galaxy S24 not a monitor", C, True),
], [turn(0, True, "informational", ["phone display flicker"],
         {"phone display flicker": ["DOC_KB_01 §1", "DOC_KB_01 §2"]})],
    "Lexical distractor: DOC_KB_02 also matches 'flicker' strongly. Tests "
    "whether early retrieval on the prefix locks onto the wrong document.")

scn("dev_multi_03", "dev", [
    (0.0, "I need the display assembly part number for an S24 Ultra", C, False),
    (0.9, "and what the exclusions are if there is liquid damage", C, True),
], [turn(0, True, "informational",
         ["S24 Ultra display part number", "liquid damage exclusion"],
         {"S24 Ultra display part number": ["DOC_PRT_02 §1", "DOC_PRT_02 §2"],
          "liquid damage exclusion": ["DOC_WAR_05 §1"]})],
    "Exact identifier retrieval: the sparse half of the hybrid must catch "
    "GH82-S24U-DA1.")

scn("dev_suppress_02", "dev", [
    (0.0, "What documentation does an imported device need", C, True),
    (5.0, "Can you make that shorter", A, True),
], [turn(0, True, "informational", ["imported device documentation"],
         {"imported device documentation": ["DOC_WAR_03 §3", "DOC_SVC_01 §2"]}),
    turn(1, False, "presentation")],
    "Suppression with a different phrasing than dev_suppress_01, to check the "
    "classifier generalises rather than pattern-matching.")

# ===========================================================================
# HELD-OUT SPLIT -- do not inspect while tuning
# ===========================================================================

scn("heldout_multi_01", "heldout", [
    (0.0, "So the panel has a green line down the side", C, False),
    (0.8, "he is asking if that is covered", C, False),
    (1.5, "and what the escalation path is if we decline it", C, True),
], [turn(0, True, "informational",
         ["green line panel defect", "warranty coverage", "escalation path"],
         {"green line panel defect": ["DOC_KB_07 §1", "DOC_KB_07 §2"],
          "warranty coverage": ["DOC_WAR_01 §3", "DOC_KB_07 §2"],
          "escalation path": ["DOC_SVC_04 §1", "DOC_SVC_04 §2"]})])

scn("heldout_late_01", "heldout", [
    (0.0, "Is accidental damage covered on this handset", C, True),
    (5.5, "He does have the extended plan by the way", C, True),
], [turn(0, True, "informational", ["accidental damage coverage"],
         {"accidental damage coverage": ["DOC_WAR_05 §1"]}),
    turn(1, True, "refinement", ["extended plan accidental damage"],
         {"extended plan accidental damage": ["DOC_WAR_04 §2", "DOC_WAR_04 §1"]})])

scn("heldout_late_02", "heldout", [
    (0.0, "What is the standard repair turnaround and does doorstep pickup apply",
     C, True),
    (6.0, "Actually the device was purchased overseas", C, True),
], [turn(0, True, "informational",
         ["repair turnaround", "doorstep pickup availability"],
         {"repair turnaround": ["DOC_POL_11 §1"],
          "doorstep pickup availability": ["DOC_SVC_02 §1", "DOC_SVC_02 §2"]}),
    turn(1, True, "refinement", ["imported device service"],
         {"imported device service": ["DOC_POL_12 §2", "DOC_SVC_02 §2"]})])

scn("heldout_suppress_01", "heldout", [
    (0.0, "Tell me the exclusions on the standard warranty", C, True),
    (5.0, "Just give me the first two as a list", A, True),
], [turn(0, True, "informational", ["warranty exclusions"],
         {"warranty exclusions": ["DOC_WAR_05 §1", "DOC_WAR_05 §2"]}),
    turn(1, False, "presentation")])

scn("heldout_hole_01", "heldout", [
    (0.0, "What is the trade in value we can offer against a new handset",
     C, True),
], [turn(0, True, "uncoverable", ["trade-in valuation"], {}, unc=True)])

scn("heldout_social_01", "heldout", [
    (0.0, "Right, bear with me one second", A, True),
], [turn(0, False, "social")])

scn("heldout_single_01", "heldout", [
    (0.0, "When does the new pricing amendment take effect", C, True),
], [turn(0, True, "informational", ["pricing amendment effective date"],
         {"pricing amendment effective date": ["DOC_POL_09 §1", "DOC_POL_09 §2"]})])

scn("heldout_multi_02", "heldout", [
    (0.0, "She wants to know what she needs to bring", C, False),
    (0.9, "and whether her data is safe during the repair", C, True),
], [turn(0, True, "informational",
         ["what to bring to appointment", "data handling during service"],
         {"what to bring to appointment": ["DOC_SVC_01 §2"],
          "data handling during service": ["DOC_POL_14 §1", "DOC_POL_14 §2"]})])

scn("heldout_multi_03", "heldout", [
    (0.0, "The bootloader is unlocked on this one", C, False),
    (0.8, "does that void everything", C, False),
    (1.5, "and what is the diagnostic fee if he declines the quote", C, True),
], [turn(0, True, "informational",
         ["unlocked bootloader exclusion", "diagnostic fee"],
         {"unlocked bootloader exclusion": ["DOC_WAR_05 §3"],
          "diagnostic fee": ["DOC_POL_07 §2", "DOC_POL_09 §2"]})])

scn("heldout_distract_01", "heldout", [
    (0.0, "Which S24 cases are compatible with the Ultra", C, True),
], [turn(0, True, "informational", ["S24 case compatibility"],
         {"S24 case compatibility": ["DOC_PRT_05 §1"]})],
    "Distractor check: 'S24' appears heavily in DOC_PRT_02 (display parts) "
    "which is the wrong document here.")

scn("heldout_late_03", "heldout", [
    (0.0, "Can we book him in for a same day display swap", C, True),
    (5.0, "He mentioned there is water damage as well", C, True),
], [turn(0, True, "informational", ["same day display repair"],
         {"same day display repair": ["DOC_POL_11 §2"]}),
    turn(1, True, "refinement", ["liquid damage exclusion"],
         {"liquid damage exclusion": ["DOC_WAR_05 §1", "DOC_SVC_02 §2"]})])

scn("heldout_conflict_01", "heldout", [
    (0.0, "How long is a repair quotation valid for", C, True),
], [turn(0, True, "informational", ["quotation validity"],
         {"quotation validity": ["DOC_POL_09 §2", "DOC_POL_07 §1"]})],
    "DOC_POL_09 §2 extends validity from fourteen to thirty days. The answer "
    "must not state the superseded figure as current.")


WORDS_PER_SEC = 2.9     # unhurried conversational speech
WORDS_PER_FRAGMENT = 3  # ASR partial-hypothesis granularity


def fragment(scenario: dict) -> dict:
    """Re-emit each authored utterance as realistic streaming fragments.

    Scenarios are AUTHORED as whole clauses because that is readable. They must
    be REPLAYED as the fragments a streaming recogniser actually emits --
    roughly three words every second -- or the benchmark measures nothing.

    This was a real measurement bug, worth recording: with clause-sized chunks
    the early-retrieval rate came out at 8%, far below gate G2's 80% target,
    and the obvious conclusion was that the controller was broken. It was not.
    A single-chunk utterance offers no opportunity to retrieve early, because
    the first chunk to arrive is also the last. The controller needs at least
    three observations before its stability estimate means anything, which is
    exactly what a real transcript supplies and what the authored form did not.
    Measure the wrong input and you will fix the wrong component.
    """
    utterances, current, start = [], [], 0.0
    for ch in scenario["chunks"]:
        current.append(ch)
        if ch["is_final"]:
            utterances.append(current)
            current = []
    if current:
        utterances.append(current)

    out = []
    clock = 0.0
    for u_i, utt in enumerate(utterances):
        text = " ".join(c["text"] for c in utt)
        speaker = utt[0]["speaker"]
        # Authored start times assume clause-sized chunks. Once expanded to
        # fragments an utterance takes much longer, so turn starts are chained
        # rather than taken literally: the next speaker begins after the
        # previous one has finished, plus the authored gap or a default pause.
        authored_gap = (utt[0]["t"] - utterances[u_i - 1][-1]["t"]
                        if u_i else 0.0)
        t0 = 0.0 if u_i == 0 else clock + max(1.2, min(authored_gap, 3.0))
        words = text.split()
        frags = [" ".join(words[i:i + WORDS_PER_FRAGMENT])
                 for i in range(0, len(words), WORDS_PER_FRAGMENT)]
        for j, f in enumerate(frags):
            spoken = sum(len(x.split()) for x in frags[:j])
            t = t0 + spoken / WORDS_PER_SEC
            out.append({
                "t": round(t, 3), "text": f, "speaker": speaker,
                "is_final": j == len(frags) - 1,
            })
        clock = t0 + len(words) / WORDS_PER_SEC
    scenario["chunks"] = out
    return scenario


def main():
    for s in SCENARIOS:
        fragment(s)
    for split in ("dev", "heldout"):
        rows = [s for s in SCENARIOS if s["split"] == split]
        path = os.path.join(HERE, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        kinds: dict[str, int] = {}
        for r in rows:
            for g in r["gold"]:
                kinds[g["turn_kind"]] = kinds.get(g["turn_kind"], 0) + 1
        print(f"{split}: {len(rows)} scenarios, "
              f"{sum(len(r['gold']) for r in rows)} labelled turns  {kinds}")


if __name__ == "__main__":
    main()
