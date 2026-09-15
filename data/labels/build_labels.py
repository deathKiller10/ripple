#!/usr/bin/env python3
"""Training data for the presentation-only classifier (controller SUPPRESS).

label 1 = NO-RETRIEVAL turn. Two sub-kinds share this label because they
          share the required behaviour:
            presentation -- restructures, reformats, shortens, translates or
                            repeats prior output; answerable from session state
            social       -- greetings, hold messages, acknowledgements; nothing
                            is being asked at all
          Either way a corpus query is wasted work, so the controller must
          suppress.
label 0 = informational: asks for something the corpus must supply, or adds a
          constraint that changes the answer. Retrieval (or delta retrieval)
          IS required.

These are generic conversational phrasings. They contain no corpus-specific
facts, no benchmark prompts and no answers, so they do not violate the guide's
no-hardcoding rule -- they teach the shape of a reformatting request, not the
content of any reply.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

PRESENTATION = [
    "please repeat your last answer in two bullets",
    "say that again",
    "can you repeat that",
    "give me that in bullet points",
    "bullet that for me",
    "make that shorter",
    "shorten it",
    "can you tighten that up",
    "condense that",
    "summarise what you just said",
    "summarize that",
    "give me the short version",
    "tl;dr that",
    "just the headline",
    "one line please",
    "in one sentence",
    "two lines max",
    "rephrase that",
    "reword it",
    "say that in plain english",
    "simpler please",
    "explain that more simply",
    "put that in simpler terms",
    "read that back to me",
    "what did you just say",
    "repeat the last part",
    "say the second point again",
    "list those out",
    "as a numbered list",
    "format that as a list",
    "break that into steps",
    "can you put that in a table",
    "translate that to hindi",
    "say that in tamil",
    "give me that in marathi",
    "spell that out",
    "slower please",
    "one more time",
    "again, shorter",
    "drop the last point",
    "leave out the pricing part",
    "just the first two points",
    "only the policy bit",
    "reorder those by priority",
    "put the important one first",
    "capitalise the document ids",
    "without the citations",
    "with the citations this time",
    "make it more formal",
    "make it friendlier",
    "rewrite that for the customer",
    "phrase that so i can read it out",
    "how would i say that to the customer",
    "give me a script for that",
    "trim it down",
    "cut it in half",
    "less detail",
    "more concise",
    "restate that",
    "recap",
    # --- social / procedural turns: also require NO retrieval -------------
    "hi there thanks for holding",
    "let me just pull that up for you",
    "bear with me one second",
    "one moment please",
    "give me a second",
    "thanks for waiting",
    "sorry about the wait",
    "are you still there",
    "can you hear me okay",
    "no problem at all",
    "sure thing",
    "of course",
    "let me check that for you",
    "i will look into that now",
    "okay got it",
    "understood",
    "right then",
    "thank you for confirming",
    "is there anything else",
    "have a good day",
    "good morning how can i help",
    "my name is priya i will be assisting you today",
    "let me put you on hold briefly",
    "i am just bringing up your account",
]

INFORMATIONAL = [
    "what is the warranty period",
    "how long does a repair take",
    "is this covered under warranty",
    "what does a display replacement cost",
    "what are the part numbers for the display",
    "can you check the cancellation policy",
    "what is the escalation process",
    "does doorstep pickup cover this",
    "the device was bought in another country",
    "it was purchased abroad",
    "he has the extended protection plan",
    "the phone is fourteen months old",
    "actually it is the ultra not the base model",
    "add that the screen is cracked",
    "there is liquid damage as well",
    "what happens if there is no invoice",
    "is a loaner device available",
    "what is the diagnostic fee",
    "what if the customer declines the quote",
    "how much is the labour charge",
    "what is the turnaround for an imported device",
    "can you look up the battery part number",
    "what does the policy say about data backup",
    "does the plan cover accidental damage",
    "how many claims are allowed per year",
    "what documentation is needed",
    "what proof of purchase is accepted",
    "what is the coverage for accessories",
    "what are the exclusions",
    "is an unlocked bootloader a problem",
    "what diagnostic code should i quote",
    "which build has the fix",
    "what should i check first",
    "does safe mode rule it out",
    "when do i escalate to a service centre",
    "what is the sla for a walk in repair",
    "is the part in stock",
    "can this be repaired locally",
    "what if the variant is not sold here",
    "how do i book an appointment",
    "what does the customer need to bring",
    "is there a wait for walk ins",
    "how do i raise a complaint",
    "who handles regional escalations",
    "what is the response time on an escalation",
    "tell me about the battery degradation threshold",
    "what counts as degraded",
    "is the charger covered",
    "and what about the cable",
    "also check the pricing amendment",
    "which policy is current",
    "has that been superseded",
    "what changed in the amendment",
    "when did the new pricing take effect",
    "i need the customer to know the lead time",
    "find the section on foreign currency",
    "what is the reimbursement process",
    "does the plan transfer between countries",
    "the customer bought it from a retailer in dubai",
    "it is out of warranty now",
]


def main():
    rows = [{"text": t, "label": 1} for t in PRESENTATION]
    rows += [{"text": t, "label": 0} for t in INFORMATIONAL]
    path = os.path.join(HERE, "presentation.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} labelled examples "
          f"({sum(r['label'] for r in rows)} presentation-only)")


if __name__ == "__main__":
    main()
