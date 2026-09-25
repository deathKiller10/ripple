#!/usr/bin/env python3
"""Benchmark scenarios, part two. Merged by build_scenarios.py.

Why more: at 12 scenarios and 16 labelled turns per split, a gate percentage
moves by six points when a single turn flips. "84.6% early retrieval" from 13
eligible turns is not a measurement, it is an anecdote with a decimal point.
This part takes each split past 40 scenarios and 50 labelled turns, which is
enough that a gate figure is stable against one scenario going either way.

Same rules as part one. In particular:

  * the SPLIT POLICY still holds -- calibrate on dev, run heldout once after
    feature freeze, never inspect heldout while tuning
  * every entry in `gold_doc_ids` must be a citation that actually exists;
    validate_gold() in build_scenarios.py checks this and fails the build
  * turn kinds carry the G2 eligibility taxonomy; see part one's docstring
"""

C, A = "customer", "agent"

EXTRA = []


def _s(sid, split, chunks, gold, notes=""):
    EXTRA.append({
        "scenario_id": sid, "split": split, "notes": notes,
        "chunks": [{"t": t, "text": x, "speaker": sp, "is_final": f}
                   for (t, x, sp, f) in chunks],
        "gold": gold,
    })


def _t(i, needs, kind, subs=None, docs=None, keep=None, unc=False):
    return {
        "utterance_index": i, "needs_retrieval": needs, "turn_kind": kind,
        "gold_sub_intents": subs or [], "gold_doc_ids": docs or {},
        "claims_that_must_not_change": keep or [], "expect_uncertainty": unc,
    }


# ===========================================================================
# DEV
# ===========================================================================

_s("dev_multi_04", "dev", [
    (0.0, "The port is full of lint and it will not charge", C, False),
    (0.9, "and he wants to know what a new charging board costs", C, True),
], [_t(0, True, "informational",
       ["charging port not charging", "charging assembly price"],
       {"charging port not charging": ["DOC_KB_05 §2", "DOC_KB_05 §1"],
        "charging assembly price": ["DOC_PRT_06 §2", "DOC_PRT_06 §1"]})])

_s("dev_multi_05", "dev", [
    (0.0, "Her camera has a blurry patch in the same spot", C, False),
    (0.9, "and she is asking whether that is covered", C, False),
    (1.8, "and what a module would cost if it is not", C, True),
], [_t(0, True, "informational",
       ["camera blurred patch", "camera warranty coverage",
        "camera module price"],
       {"camera blurred patch": ["DOC_KB_06 §2", "DOC_KB_06 §1"],
        "camera warranty coverage": ["DOC_KB_06 §3", "DOC_WAR_01 §3"],
        "camera module price": ["DOC_PRT_07 §2", "DOC_PRT_07 §1"]})])

_s("dev_late_03", "dev", [
    (0.0, "Can we book him in for a same day display swap", C, True),
    (5.0, "The liquid damage indicator is triggered", C, True),
], [_t(0, True, "informational", ["same day display repair"],
       {"same day display repair": ["DOC_POL_11 §2"]}),
    _t(1, True, "refinement", ["liquid damage exclusion"],
       {"liquid damage exclusion": ["DOC_WAR_05 §1", "DOC_POL_25 §2"]})])

_s("dev_late_04", "dev", [
    (0.0, "What does a battery replacement cost on an S24", C, True),
    (5.5, "The diagnostic says capacity is eighty four percent", C, True),
], [_t(0, True, "informational", ["battery replacement price"],
       {"battery replacement price": ["DOC_PRT_03 §2", "DOC_PRT_03 §1"]}),
    _t(1, True, "refinement", ["battery above degradation threshold"],
       {"battery above degradation threshold": ["DOC_WAR_08 §1",
                                                "DOC_WAR_08 §2"]})],
    "Refinement that makes a covered repair chargeable rather than the "
    "reverse.")

_s("dev_single_02", "dev", [
    (0.0, "What warranty applies to a repair we already did", C, True),
], [_t(0, True, "informational", ["repair warranty period"],
       {"repair warranty period": ["DOC_POL_17 §1"]})])

_s("dev_single_03", "dev", [
    (0.0, "Can we fit a part the customer brings in himself", C, True),
], [_t(0, True, "informational", ["customer supplied parts"],
       {"customer supplied parts": ["DOC_POL_24 §1"]})])

_s("dev_single_04", "dev", [
    (0.0, "How long do we hold a device nobody collects", C, True),
], [_t(0, True, "informational", ["uncollected devices"],
       {"uncollected devices": ["DOC_POL_18 §2", "DOC_POL_18 §1"]})])

_s("dev_suppress_03", "dev", [
    (0.0, "What are the walk in hours", C, True),
    (5.0, "Say that back to me one more time", A, True),
], [_t(0, True, "informational", ["walk in hours"],
       {"walk in hours": ["DOC_SVC_08 §1"]}),
    _t(1, False, "presentation")])

_s("dev_suppress_04", "dev", [
    (0.0, "What do we tell a customer when we decline a claim", C, True),
    (5.0, "Put that into three short bullets for me", A, True),
], [_t(0, True, "informational", ["declining a warranty claim"],
       {"declining a warranty claim": ["DOC_SVC_09 §1", "DOC_SVC_09 §2"]}),
    _t(1, False, "presentation")])

_s("dev_social_02", "dev", [
    (0.0, "Good morning, my name is Priya, how can I help", A, True),
], [_t(0, False, "social")])

_s("dev_social_03", "dev", [
    (0.0, "Sorry about the wait, thanks for holding", A, True),
], [_t(0, False, "social")])

_s("dev_hole_02", "dev", [
    (0.0, "He is asking how to file an insurance claim", C, False),
    (0.8, "with his own insurer for the damage", C, True),
], [_t(0, True, "uncoverable", ["insurance claim procedure"], {}, unc=True)],
    "Out-of-domain hole: DOC_SVC_07 §2 mentions third parties but describes no "
    "procedure. The near-miss is deliberate.")

_s("dev_conflict_02", "dev", [
    (0.0, "Is there still a handling fee on a replacement cable", C, True),
], [_t(0, True, "informational", ["accessory handling fee"],
       {"accessory handling fee": ["DOC_POL_23 §1", "DOC_POL_21 §1"]})],
    "Second contradiction pair: DOC_POL_23 withdraws the fee that DOC_POL_21 "
    "charges. The current answer is 'no fee'.")

_s("dev_distract_02", "dev", [
    (0.0, "The battery is swollen and the back is lifting", C, True),
], [_t(0, True, "informational", ["battery swelling safety"],
       {"battery swelling safety": ["DOC_KB_16 §2", "DOC_KB_16 §1"]})],
    "'Battery' now appears across eight documents; only DOC_KB_16 is right.")

_s("dev_distract_03", "dev", [
    (0.0, "The screen keeps dimming while he records video", C, True),
], [_t(0, True, "informational", ["thermal dimming is expected"],
       {"thermal dimming is expected": ["DOC_KB_14 §2", "DOC_KB_14 §1"]})],
    "Must not be confused with the flicker fault in DOC_KB_01.")

_s("dev_multi_06", "dev", [
    (0.0, "She wants doorstep pickup", C, False),
    (0.8, "and to know if her data is safe", C, True),
], [_t(0, True, "informational",
       ["doorstep pickup availability", "data handling during service"],
       {"doorstep pickup availability": ["DOC_SVC_02 §1", "DOC_SVC_02 §2"],
        "data handling during service": ["DOC_POL_14 §1", "DOC_POL_14 §2"]})])

_s("dev_multi_07", "dev", [
    (0.0, "He bought it second hand and wants to know", C, False),
    (0.9, "if the warranty still applies", C, False),
    (1.8, "and whether the protection plan came with it", C, True),
], [_t(0, True, "informational",
       ["warranty transfer on resale", "extended plan transfer"],
       {"warranty transfer on resale": ["DOC_WAR_09 §1"],
        "extended plan transfer": ["DOC_WAR_09 §2"]})])

_s("dev_late_05", "dev", [
    (0.0, "What is the walk in turnaround today", C, True),
    (5.0, "We are in the declared peak period", C, True),
], [_t(0, True, "informational", ["walk in turnaround"],
       {"walk in turnaround": ["DOC_POL_11 §1"]}),
    _t(1, True, "refinement", ["peak period turnaround"],
       {"peak period turnaround": ["DOC_POL_22 §1", "DOC_POL_22 §2"]})])

_s("dev_single_05", "dev", [
    (0.0, "Do we use refurbished parts and does that change the warranty",
     C, True),
], [_t(0, True, "informational", ["refurbished parts policy"],
       {"refurbished parts policy": ["DOC_POL_16 §1", "DOC_POL_16 §2"]})])

_s("dev_single_06", "dev", [
    (0.0, "Can we reprice the labour after work has started", C, True),
], [_t(0, True, "informational", ["quotation revision"],
       {"quotation revision": ["DOC_POL_20 §2", "DOC_POL_20 §1"]})])

_s("dev_multi_08", "dev", [
    (0.0, "The phone restarts on its own", C, False),
    (0.8, "and it gets very hot when charging", C, True),
], [_t(0, True, "informational",
       ["spontaneous restarts", "overheating while charging"],
       {"spontaneous restarts": ["DOC_KB_13 §2", "DOC_KB_13 §3"],
        "overheating while charging": ["DOC_KB_11 §2", "DOC_KB_11 §3"]})])

_s("dev_single_07", "dev", [
    (0.0, "What priority do customers with a disability get", C, True),
], [_t(0, True, "informational", ["accessibility priority"],
       {"accessibility priority": ["DOC_POL_19 §1", "DOC_POL_19 §2"]})])

_s("dev_single_08", "dev", [
    (0.0, "His fingerprint stopped working after the screen repair", C, True),
], [_t(0, True, "informational", ["fingerprint after screen repair"],
       {"fingerprint after screen repair": ["DOC_KB_10 §2", "DOC_KB_10 §3"]})])

_s("dev_late_06", "dev", [
    (0.0, "Is the display replacement covered", C, True),
    (5.0, "It is a fleet device on the business account", C, True),
], [_t(0, True, "informational", ["display coverage"],
       {"display coverage": ["DOC_WAR_01 §3"]}),
    _t(1, True, "refinement", ["fleet device coverage"],
       {"fleet device coverage": ["DOC_WAR_10 §1", "DOC_WAR_10 §2"]})])

_s("dev_multi_09", "dev", [
    (0.0, "Bluetooth keeps cutting out on his earbuds", C, False),
    (0.9, "and wifi drops at home as well", C, True),
], [_t(0, True, "informational",
       ["bluetooth audio stuttering", "wifi disconnects"],
       {"bluetooth audio stuttering": ["DOC_KB_17 §2", "DOC_KB_17 §3"],
        "wifi disconnects": ["DOC_KB_08 §2", "DOC_KB_08 §3"]})])

_s("dev_single_09", "dev", [
    (0.0, "It failed within four days of purchase, what now", C, True),
], [_t(0, True, "informational", ["dead on arrival handling"],
       {"dead on arrival handling": ["DOC_WAR_06 §2", "DOC_WAR_06 §1"]})])

_s("dev_single_10", "dev", [
    (0.0, "What is in a technical confirmation report", C, True),
], [_t(0, True, "informational", ["technical confirmation report"],
       {"technical confirmation report": ["DOC_SVC_07 §1"]})])

_s("dev_suppress_05", "dev", [
    (0.0, "What are the exclusions on the standard warranty", C, True),
    (5.0, "Translate that into Hindi for me", A, True),
], [_t(0, True, "informational", ["warranty exclusions"],
       {"warranty exclusions": ["DOC_WAR_05 §1", "DOC_WAR_05 §2"]}),
    _t(1, False, "presentation")])

# ===========================================================================
# HELD-OUT  -- do not inspect while tuning
# ===========================================================================

_s("heldout_multi_04", "heldout", [
    (0.0, "The earpiece is really faint on calls", C, False),
    (0.9, "and he wants to know what that part costs", C, True),
], [_t(0, True, "informational",
       ["faint call audio", "receiver part price"],
       {"faint call audio": ["DOC_KB_09 §2", "DOC_KB_09 §3"],
        "receiver part price": ["DOC_PRT_09 §2", "DOC_PRT_09 §1"]})])

_s("heldout_multi_05", "heldout", [
    (0.0, "The keyboard types by itself", C, False),
    (0.8, "and he asks if it is covered", C, False),
    (1.7, "and how quickly we could do it", C, True),
], [_t(0, True, "informational",
       ["phantom touch input", "warranty coverage", "repair turnaround"],
       {"phantom touch input": ["DOC_KB_04 §2", "DOC_KB_04 §3"],
        "warranty coverage": ["DOC_WAR_01 §3"],
        "repair turnaround": ["DOC_POL_11 §1", "DOC_POL_11 §2"]})])

_s("heldout_late_04", "heldout", [
    (0.0, "What does a back glass replacement cost", C, True),
    (5.0, "The frame is bent as well", C, True),
], [_t(0, True, "informational", ["back glass price"],
       {"back glass price": ["DOC_PRT_08 §2", "DOC_PRT_08 §1"]}),
    _t(1, True, "refinement", ["frame damage not serviceable"],
       {"frame damage not serviceable": ["DOC_PRT_08 §1"]})])

_s("heldout_late_05", "heldout", [
    (0.0, "Can we courier the phone back to him when it is done", C, True),
    (5.0, "It was bought in Singapore originally", C, True),
], [_t(0, True, "informational", ["courier return availability"],
       {"courier return availability": ["DOC_SVC_06 §1", "DOC_SVC_06 §2"]}),
    _t(1, True, "refinement", ["imported device exclusions"],
       {"imported device exclusions": ["DOC_SVC_06 §2", "DOC_WAR_03 §1"]})])

_s("heldout_single_02", "heldout", [
    (0.0, "What happens after three missed appointments", C, True),
], [_t(0, True, "informational", ["repeated no shows"],
       {"repeated no shows": ["DOC_SVC_05 §2"]})])

_s("heldout_single_03", "heldout", [
    (0.0, "Does a 45 watt charger damage a phone that only takes 25", C, True),
], [_t(0, True, "informational", ["charger compatibility"],
       {"charger compatibility": ["DOC_PRT_10 §1"]})])

_s("heldout_single_04", "heldout", [
    (0.0, "What coverage does an exchange unit carry", C, True),
], [_t(0, True, "informational", ["refurbished exchange coverage"],
       {"refurbished exchange coverage": ["DOC_WAR_07 §1"]})])

_s("heldout_single_05", "heldout", [
    (0.0, "The update keeps failing to install", C, True),
], [_t(0, True, "informational", ["failed software update"],
       {"failed software update": ["DOC_KB_15 §2", "DOC_KB_15 §3"]})])

_s("heldout_single_06", "heldout", [
    (0.0, "Storage is full but he has hardly any photos", C, True),
], [_t(0, True, "informational", ["storage full with little user data"],
       {"storage full with little user data": ["DOC_KB_12 §2",
                                               "DOC_KB_12 §1"]})])

_s("heldout_suppress_02", "heldout", [
    (0.0, "What is the escalation path if we decline", C, True),
    (5.0, "Shorten that, I have to read it out", A, True),
], [_t(0, True, "informational", ["escalation path"],
       {"escalation path": ["DOC_SVC_04 §1", "DOC_SVC_04 §2"]}),
    _t(1, False, "presentation")])

_s("heldout_suppress_03", "heldout", [
    (0.0, "What must we record when the liquid indicator is triggered",
     C, True),
    (5.0, "Just give me the first point again", A, True),
], [_t(0, True, "informational", ["liquid damage recording"],
       {"liquid damage recording": ["DOC_POL_25 §1", "DOC_POL_25 §2"]}),
    _t(1, False, "presentation")])

_s("heldout_social_02", "heldout", [
    (0.0, "One moment please, I am bringing up the account", A, True),
], [_t(0, False, "social")])

_s("heldout_hole_02", "heldout", [
    (0.0, "Can he pay for the repair in monthly instalments", C, True),
], [_t(0, True, "uncoverable", ["financing options"], {}, unc=True)])

_s("heldout_conflict_02", "heldout", [
    (0.0, "What labour charge applies to a mainboard repair", C, True),
], [_t(0, True, "informational", ["mainboard labour tier"],
       {"mainboard labour tier": ["DOC_POL_09 §1", "DOC_POL_07 §1"]})])

_s("heldout_distract_02", "heldout", [
    (0.0, "Which charging assembly fits an imported S24 Ultra", C, True),
], [_t(0, True, "informational", ["charging assembly variant coding"],
       {"charging assembly variant coding": ["DOC_PRT_06 §2",
                                             "DOC_POL_26 §1"]})],
    "Distractor: DOC_PRT_02 §3 says display assemblies ARE common across "
    "variants. Charging assemblies are not. Retrieving the display doc and "
    "answering from it would be wrong.")

_s("heldout_multi_06", "heldout", [
    (0.0, "What do we do about a device nobody has collected", C, False),
    (0.9, "and do we charge storage for it", C, True),
], [_t(0, True, "informational",
       ["uncollected device handling", "storage charges"],
       {"uncollected device handling": ["DOC_POL_18 §1", "DOC_POL_18 §2"],
        "storage charges": ["DOC_POL_18 §2"]})])

_s("heldout_late_06", "heldout", [
    (0.0, "What does a periscope camera module cost", C, True),
    (5.0, "The device is an imported variant", C, True),
], [_t(0, True, "informational", ["periscope module price"],
       {"periscope module price": ["DOC_PRT_07 §2", "DOC_PRT_07 §1"]}),
    _t(1, True, "refinement", ["variant parts sourcing and freight"],
       {"variant parts sourcing and freight": ["DOC_POL_26 §1",
                                               "DOC_POL_26 §2"]})])

_s("heldout_single_07", "heldout", [
    (0.0, "His import papers are in Arabic, is that acceptable", C, True),
], [_t(0, True, "informational", ["translated import documents"],
       {"translated import documents": ["DOC_WAR_11 §1"]})])

_s("heldout_single_08", "heldout", [
    (0.0, "Who gets served first at a walk in counter", C, True),
], [_t(0, True, "informational", ["queue priority"],
       {"queue priority": ["DOC_SVC_08 §2"]})])

_s("heldout_multi_07", "heldout", [
    (0.0, "The moisture warning is on but the port is dry", C, False),
    (0.9, "and he says he never got it wet", C, True),
], [_t(0, True, "informational",
       ["moisture warning on dry port", "liquid damage disclosure"],
       {"moisture warning on dry port": ["DOC_KB_05 §3"],
        "liquid damage disclosure": ["DOC_POL_25 §2", "DOC_POL_25 §1"]})])

_s("heldout_single_09", "heldout", [
    (0.0, "What quotation validity applies now", C, True),
], [_t(0, True, "informational", ["quotation validity"],
       {"quotation validity": ["DOC_POL_09 §2"]})])

_s("heldout_single_10", "heldout", [
    (0.0, "Do we handle fleet drop offs in the normal queue", C, True),
], [_t(0, True, "informational", ["fleet batch handling"],
       {"fleet batch handling": ["DOC_WAR_10 §2", "DOC_SVC_08 §2"]})])
