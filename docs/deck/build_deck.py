#!/usr/bin/env python3
"""Populate the official Samsung PRISM submission template.

Design constraints, in priority order:
  1. It is THEIR template. The palette, fonts and title treatment are taken
     from the file itself (purple 6D28D9 / 704EA6, ink 14142B, muted 63637E,
     lilac D9D3F0, Calibri + Arial) so added content reads as part of the deck
     rather than pasted into it.
  2. Every number on these slides comes from results/benchmark_dev.json
     (keyless run) or results_gemini/run2_2026-09-26/ (real model, named
     beside each figure).
     Nothing is estimated.
  3. No slide is bullets-on-white. Each has a visual structure appropriate to
     its content: stat tiles, comparison columns, a pipeline diagram, a
     results table.
"""

import copy
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "template.pptx")
OUT = os.path.join(HERE, "VITVellore_Spark_Submission.pptx")

# --- palette, lifted from the template -----------------------------------
PURPLE = RGBColor(0x6D, 0x28, 0xD9)
PURPLE_D = RGBColor(0x70, 0x4E, 0xA6)
INK = RGBColor(0x14, 0x14, 0x2B)
MUTED = RGBColor(0x63, 0x63, 0x7E)
LILAC = RGBColor(0xD9, 0xD3, 0xF0)
TINT = RGBColor(0xF4, 0xF1, 0xFB)
GOOD = RGBColor(0x0E, 0x7C, 0x66)
GOOD_T = RGBColor(0xE2, 0xF2, 0xEE)
WARN = RGBColor(0x9B, 0x1B, 0x3C)
WARN_T = RGBColor(0xFA, 0xEC, 0xF0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RULE = RGBColor(0xDD, 0xDA, 0xE8)

BODY = "Calibri"
HEAD = "Arial"

CONTENT_L = 0.92
CONTENT_W = 11.50

# Slides are authored in a 1.70-5.65 band and mapped onto the slide's real
# 2.00-7.05 content area. Doing it as a transform in the helpers, rather than
# by hand per shape, is what stopped the first render leaving two inches of
# dead white space along the bottom of every slide.
# Positions spread more than heights grow: blocks move apart to fill the
# slide, while each card stays close to the size its content needs. Scaling
# both by the same factor left every card with dead space inside it.
_Y0_SRC, _Y0_DST, _POS_SCALE, _H_SCALE = 1.70, 2.00, 1.30, 1.10


def Y(y):
    return _Y0_DST + (y - _Y0_SRC) * _POS_SCALE


def H(h):
    return h * _H_SCALE


# ---------------------------------------------------------------- helpers

def body_placeholder(slide):
    """The empty content placeholder on a template slide."""
    best = None
    for sh in slide.shapes:
        if sh.is_placeholder and sh.has_text_frame and not sh.text_frame.text.strip():
            if best is None or sh.top > best.top:
                best = sh
    return best


def drop(shape):
    shape._element.getparent().remove(shape._element)


def textbox(slide, x, y, w, h, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            raw=False):
    if not raw:
        y, h = Y(y), H(h)
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    tf.paragraphs[0].alignment = align
    return tb, tf


def para(tf, text, size=14, bold=False, color=INK, font=BODY, space_after=4,
         space_before=0, first=False, align=None, italic=False,
         line_spacing=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    # Autoshapes centre their text by default; every card here is read as a
    # left-aligned block, so set it explicitly rather than inheriting.
    p.alignment = align if align is not None else PP_ALIGN.LEFT
    if line_spacing:
        p.line_spacing = line_spacing
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = font
    return p


def rich(tf, parts, size=14, space_after=4, first=False, align=None):
    """parts = [(text, bold, color), ...] in one paragraph."""
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_after = Pt(space_after)
    p.alignment = align if align is not None else PP_ALIGN.LEFT
    for text, bold, color in parts:
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = BODY
    return p


def card(slide, x, y, w, h, fill=TINT, line=None, radius=True):
    y, h = Y(y), H(h)
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1)
    shp.shadow.inherit = False
    if radius:
        try:
            shp.adjustments[0] = 0.06
        except Exception:
            pass
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.16)
    tf.margin_top = tf.margin_bottom = Inches(0.13)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    return shp, tf


def stat(slide, x, y, w, value, label, vcolor=PURPLE, h=1.12, note=None):
    shp, tf = card(slide, x, y, w, h, fill=TINT)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(tf, value, size=30, bold=True, color=vcolor, font=HEAD,
         space_after=1, first=True)
    para(tf, label, size=10.5, color=MUTED, space_after=0)
    if note:
        para(tf, note, size=9, color=MUTED, italic=True, space_after=0)
    return shp


def arrow(slide, x, y, w=0.22, h=0.16, color=LILAC):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(Y(y)),
                               Inches(w), Inches(H(h)))
    a.fill.solid()
    a.fill.fore_color.rgb = color
    a.line.fill.background()
    a.shadow.inherit = False
    return a


def kicker(slide, text, y=1.62):
    _, tf = textbox(slide, CONTENT_L, y, CONTENT_W, 0.30, raw=True)
    para(tf, text, size=11.5, color=MUTED, italic=True, first=True,
         space_after=0)


def table(slide, x, y, w, rows, col_w, header=True, size=11, row_h=0.30):
    n_r, n_c = len(rows), len(rows[0])
    row_h = H(row_h)
    gt = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(Y(y)), Inches(w),
                                Inches(row_h * n_r)).table
    for j, cw in enumerate(col_w):
        gt.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        gt.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            txt, bold, color = (val if isinstance(val, tuple)
                                else (val, False, INK))
            cell = gt.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = (LILAC if (header and i == 0)
                                        else (WHITE if i % 2 else TINT))
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if j else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = txt
            r.font.size = Pt(size)
            r.font.bold = bold or (header and i == 0)
            r.font.color.rgb = color if not (header and i == 0) else PURPLE_D
            r.font.name = BODY
    return gt


def _numeric(s):
    return bool(s) and s[0].isdigit() or s.startswith(("−", "-", "+", "0", "."))


# ---------------------------------------------------------------- slides

prs = Presentation(SRC)
S = prs.slides


def clear_body(i):
    ph = body_placeholder(S[i])
    if ph is not None:
        drop(ph)


# --- 1. title -------------------------------------------------------------
s = S[0]
for sh in s.shapes:
    if sh.has_text_frame and sh.text_frame.text.strip().startswith("Theme ID"):
        lines = [
            "Theme ID - 04  Streaming Live RAG",
            "Team Name - Spark  ·  Project - Ripple",
            "College Name - Vellore Institute of Technology, Vellore",
            "Priyanshu Kundu - priyanshuwork.10@gmail.com",
            "Souptik Hazra",
            "Anushka Paul",
            "Arpita Bhaumik",
            "GitHub - github.com/deathKiller10/ripple",
        ]
        ps = sh.text_frame.paragraphs
        for k, line in enumerate(lines):
            if k < len(ps):
                p = ps[k]
            else:
                p = copy.deepcopy(ps[-1]._p)
                ps[-1]._p.addnext(p)
                p = sh.text_frame.paragraphs[-1]
            runs = p.runs
            if runs:
                runs[0].text = line
                for extra in runs[1:]:
                    extra.text = ""
            else:
                r = p.add_run()
                r.text = line
                r.font.size = Pt(10)
                r.font.name = BODY
        break

# --- 2. Theme -------------------------------------------------------------
clear_body(1)
s = S[1]
kicker(s, "The customer is not talking to the assistant. That is the whole problem.")
shp, tf = card(s, CONTENT_L, 1.66, CONTENT_W, 1.02, fill=TINT)
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
para(tf, "“My S24’s screen is flickering after the update, the battery "
         "dies by 2pm, is any of this covered, and how long’s a repair?”",
     size=17, italic=True, color=INK, font=HEAD, first=True, space_after=2)
para(tf, "One breath. Four questions. The agent has ~1.5 seconds.",
     size=11, color=MUTED, space_after=0)

cards = [
    ("No turn to wait for",
     "There is no submit action and no endpoint. A turn-based system waits for "
     "a query that is never addressed to it."),
    ("Several questions, one utterance",
     "Answer them as a blend and a third of the answer silently disappears — "
     "and no standard retrieval metric can see it."),
    ("Details arrive late",
     "“Oh — I bought it in Dubai.” The answer must get better, "
     "not start over, and the citations already given must survive."),
]
w = (CONTENT_W - 0.44) / 3
for i, (h1, b) in enumerate(cards):
    x = CONTENT_L + i * (w + 0.22)
    shp, tf = card(s, x, 2.96, w, 1.62, fill=WHITE, line=RULE)
    para(tf, h1, size=13.5, bold=True, color=PURPLE_D, font=HEAD, first=True,
         space_after=4)
    para(tf, b, size=11.5, color=INK, space_after=0)

_, tf = textbox(s, CONTENT_L, 4.78, CONTENT_W, 0.9)
rich(tf, [("Ripple ", True, PURPLE),
          ("starts retrieving before the sentence lands, fans out across the "
           "questions hidden inside it, and holds the answer as individually "
           "cited claims — so a late detail patches two sentences instead "
           "of restarting the search.", False, INK)], size=13.5, first=True)

# --- 3. Existing solutions & gaps ----------------------------------------
clear_body(2)
s = S[2]
kicker(s, "The gap is not retrieval quality. It is the absence of state between "
          "the first word and the last.")
cols = [
    ("Contact-centre assist", "Keyword triggers after the utterance ends. "
     "Surfaces a list of articles, not a grounded answer.",
     "Fires too late; no answer, just links"),
    ("Conversational RAG", "Appends to history and re-retrieves on every "
     "follow-up. Rewrites the whole answer each time.",
     "This is the “restart” the theme rules out"),
    ("Streaming ASR + RAG", "Retrieves on every transcript chunk. Fast, and "
     "indiscriminate.",
     "Measured: 100% false triggers, 49% more retrievals"),
]
w = (CONTENT_W - 0.44) / 3
for i, (h1, b, gap) in enumerate(cols):
    x = CONTENT_L + i * (w + 0.22)
    shp, tf = card(s, x, 1.70, w, 1.70, fill=WHITE, line=RULE)
    para(tf, h1, size=14, bold=True, color=INK, font=HEAD, first=True,
         space_after=5)
    para(tf, b, size=11.5, color=MUTED, space_after=0)
    shp2, tf2 = card(s, x, 3.48, w, 0.78, fill=WARN_T)
    tf2.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(tf2, "GAP", size=8.5, bold=True, color=WARN, font=HEAD, first=True,
         space_after=2)
    para(tf2, gap, size=11, color=INK, space_after=0)

shp, tf = card(s, CONTENT_L, 4.50, CONTENT_W, 1.14, fill=TINT)
para(tf, "What none of them do", size=13.5, bold=True, color=PURPLE_D,
     font=HEAD, first=True, space_after=5)
para(tf, "Maintain an evidence pool across a single utterance ·  allocate "
         "a context budget per sub-question ·  represent the answer with "
         "enough structure to patch it without breaking its citations",
     size=12.5, color=INK, space_after=0)

# --- 4. Solution & architecture ------------------------------------------
clear_body(3)
s = S[3]
kicker(s, "Nine stages. Two of them cost a token. The diagram doubles as a cost "
          "ledger, because the guide grades cost-to-performance.")
stages = [
    ("1", "Retrieval\ncontroller", "0 tokens", True),
    ("2", "Compound\u00adness gate", "0 tokens", True),
    ("3", "Sub-query\nextractor", "1 call", False),
    ("4", "Parallel\nretrieval", "0 tokens", True),
    ("5", "Coverage\nfusion", "0 tokens", True),
    ("6", "Abstention\ngate", "0 tokens", True),
    ("7", "Claim\nsynthesiser", "1 call", False),
    ("8", "Grounding\nverifier", "0 tokens", True),
]
bw, gap = 1.24, 0.185
x0 = CONTENT_L
for i, (n, name, cost, free) in enumerate(stages):
    x = x0 + i * (bw + gap)
    shp, tf = card(s, x, 1.70, bw, 1.28,
                   fill=WHITE if free else TINT, line=RULE)
    para(tf, n, size=9.5, bold=True, color=PURPLE, font=HEAD, first=True,
         space_after=2)
    para(tf, name, size=10.5, bold=True, color=INK, space_after=3)
    para(tf, cost, size=9, bold=True,
         color=GOOD if free else PURPLE_D, font=HEAD, space_after=0)
    if i < len(stages) - 1:
        arrow(s, x + bw + 0.02, 2.31, w=gap - 0.04, h=0.14)

mechs = [
    ("M1  Retrieval-space stability",
     "Everyone measures whether the sentence is finished. We measure whether "
     "more words would change which documents come back — by comparing "
     "successive result sets. One embedding, one lookup, zero tokens. "
     "θ is derived from a cost curve, not chosen."),
    ("M2  The claim graph",
     "The answer is a set of claims, each bound to its evidence IDs. A late "
     "constraint invalidates, re-scores, delta-retrieves, patches. Unaffected "
     "claims are never regenerated, so their citations cannot drift — and "
     "no model ever writes a citation string."),
    ("M3  Coverage-budgeted fusion",
     "RRF pools; we allocate. A floor per sub-question, then marginal gain, "
     "then de-duplication. Fixes sub-intent starvation, which recall@k cannot "
     "see: identical 0.960 recall either way, 0.772 vs 0.718 coverage."),
]
w = (CONTENT_W - 0.44) / 3
for i, (h1, b) in enumerate(mechs):
    x = CONTENT_L + i * (w + 0.22)
    shp, tf = card(s, x, 3.20, w, 2.10, fill=TINT)
    para(tf, h1, size=12.5, bold=True, color=PURPLE_D, font=HEAD, first=True,
         space_after=4)
    para(tf, b, size=10.5, color=INK, space_after=0)

_, tf = textbox(s, CONTENT_L, 5.42, CONTENT_W, 0.32)
para(tf, "Engine is transport-free: the WebSocket server and the headless "
         "replay CLI are both clients of it. No agent framework, no vector "
         "database server, no Redis — session-scoped memory makes "
         "persistence a spec violation, not a missing feature.",
     size=10.5, color=MUTED, italic=True, first=True, space_after=0)

# --- 5. Demo & walkthrough ------------------------------------------------
clear_body(4)
s = S[4]
kicker(s, "One customer call, four beats, no cuts. The dashboard shows the "
          "controller deciding on a clock.")
beats = [
    ("0:30", "Early retrieval + multi-intent",
     "Stability climbs; RETRIEVE fires at 0.8 s while the customer is still "
     "speaking. Second and third questions open parallel branches.",
     "Answer starts before they finish"),
    ("2:00", "Late detail, refined not restarted",
     "“I bought it in Dubai.” Version diff animates: claims "
     "preserved byte-identical, one superseded, delta retrieval only.",
     "0 citation drift"),
    ("3:05", "Suppression + abstention",
     "“Give me that in two lines” → no search, no tokens, "
     "citations retained. Then a question the corpus cannot answer.",
     "0 retrievals, 0 tokens"),
    ("3:35", "The numbers, then reproducibility",
     "Six gates against measured values, five ablations, then "
     "docker compose up on a clean machine and the headless replay CLI.",
     "No API key required"),
]
w = (CONTENT_W - 0.66) / 4
for i, (t, h1, b, tag) in enumerate(beats):
    x = CONTENT_L + i * (w + 0.22)
    shp, tf = card(s, x, 1.70, w, 2.30, fill=WHITE, line=RULE)
    para(tf, t, size=10, bold=True, color=PURPLE, font=HEAD, first=True,
         space_after=3)
    para(tf, h1, size=13, bold=True, color=INK, font=HEAD, space_after=5)
    para(tf, b, size=11, color=MUTED, space_after=0)
    shp2, tf2 = card(s, x, 4.10, w, 0.46, fill=GOOD_T)
    tf2.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(tf2, tag, size=11, bold=True, color=GOOD, font=HEAD, first=True,
         space_after=0, align=PP_ALIGN.CENTER)

shp, tf = card(s, CONTENT_L, 4.76, CONTENT_W, 0.88, fill=TINT)
para(tf, "Why a judge can trust the screen", size=12.5, bold=True,
     color=PURPLE_D, font=HEAD, first=True, space_after=4)
para(tf, "The dashboard computes nothing. Every number on it comes from a "
         "telemetry event, so the screen and traces/<session>.jsonl cannot "
         "disagree.", size=12, color=INK, space_after=0)

# --- 6. Tools and tech stack ---------------------------------------------
clear_body(5)
s = S[5]
kicker(s, "Every dependency justifies its latency, or it was removed. Two were "
          "removed.")
groups = [
    ("Engine", "Python 3.11  ·  FastAPI + WebSockets  ·  asyncio\n"
               "Single process, one port, one container"),
    ("Retrieval", "FAISS flat inner-product (exact)\n"
                  "scikit-learn TF-IDF + SVD embeddings\n"
                  "No GPU, no model download at run time"),
    ("Generation", "Pluggable provider: Gemini · OpenAI · keyless stub\n"
                   "Client-side rate limiting for free tiers"),
    ("Frontend", "Vanilla HTML dashboard (ships)\nVite + React + Tailwind (migration)"),
    ("Evaluation", "Custom harness: 4 systems, 6 gates, 5 ablations\n"
                   "Virtual-clock replay — machine-independent timings"),
    ("Packaging", "Docker multi-stage; corpus, labels and index baked in at\n"
                  "build time so the container needs no network"),
]
w = (CONTENT_W - 0.44) / 3
for i, (h1, b) in enumerate(groups):
    x = CONTENT_L + (i % 3) * (w + 0.22)
    y = 1.70 + (i // 3) * 1.16
    shp, tf = card(s, x, y, w, 1.02, fill=WHITE, line=RULE)
    para(tf, h1, size=12.5, bold=True, color=PURPLE_D, font=HEAD, first=True,
         space_after=4)
    para(tf, b, size=10.5, color=INK, space_after=0)

shp, tf = card(s, CONTENT_L, 4.08, CONTENT_W, 1.52, fill=WARN_T)
para(tf, "Deliberately not used — and two things we built and then deleted",
     size=12.5, bold=True, color=WARN, font=HEAD, first=True, space_after=5)
para(tf, "LangChain · LlamaIndex · LangGraph · any agent framework "
         "· Kafka · Celery · Kubernetes · Redis · a "
         "vector-database server", size=11.5, color=INK, space_after=6)
rich(tf, [("BM25 removed. ", True, INK),
          ("Recall improved 0.893 → 0.960 without it, and it won nothing on "
           "a 16-query part-number probe.  ", False, INK),
          ("Reranker removed. ", True, INK),
          ("Coverage improved 0.726 → 0.772, and it cost 90 ms per "
           "sub-query. Deleting your own work on evidence is the parsimony rule "
           "applied to yourself.", False, INK)], size=11.5)

# --- 7. Impact & use case -------------------------------------------------
clear_body(6)
s = S[6]
kicker(s, "Samsung runs support centres. Handle time and first-call resolution "
          "are budget lines, not metrics.")
tiles = [("−3.17 s", "first answer, two-question calls", PURPLE,
          "before the customer finishes · real model"),
         ("0.884", "of eligible turns retrieve early", PURPLE, "gate G2 ≥ 0.80"),
         ("0.000", "false triggers", GOOD, "naive streaming: 1.000"),
         ("158/158", "claims backed by the cited passage", GOOD,
          "gemini-3.5-flash-lite · 0 fabricated")]
w = (CONTENT_W - 0.66) / 4
for i, (v, lab, c, note) in enumerate(tiles):
    stat(s, CONTENT_L + i * (w + 0.22), 1.70, w, v, lab, vcolor=c, h=1.20,
         note=note)

uses = [
    ("Live agent assist", "The reference application. The agent reads a grounded "
     "answer while the customer is still speaking, with every claim traceable "
     "to a policy clause."),
    ("Any listening assistant", "The engine is corpus-agnostic and domain-free: "
     "scope comes from corpus metadata, never from rules in code. Point it at "
     "field service, insurance adjudication or regulatory circulars unchanged."),
    ("A worklet, not a demo", "Engine and application are separate. The "
     "dashboard and the headless replay CLI are both clients of the same "
     "object, which is why it can be evaluated without a browser."),
]
w = (CONTENT_W - 0.44) / 3
for i, (h1, b) in enumerate(uses):
    x = CONTENT_L + i * (w + 0.22)
    shp, tf = card(s, x, 3.14, w, 1.62, fill=WHITE, line=RULE)
    para(tf, h1, size=13, bold=True, color=PURPLE_D, font=HEAD, first=True,
         space_after=4)
    para(tf, b, size=11.5, color=INK, space_after=0)

shp, tf = card(s, CONTENT_L, 4.96, CONTENT_W, 0.68, fill=TINT)
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
rich(tf, [("Session-scoped memory only. ", True, PURPLE_D),
          ("No cross-session profile, no persistent user tracking. A test "
           "asserts two identical sessions produce identical state — "
           "non-profiling demonstrated, not promised.", False, INK)],
     size=12, first=True)

# --- 8. Innovation, results, limitations ---------------------------------
clear_body(7)
s = S[7]
kicker(s, "Table: dev split, 40 scenarios, keyless provider. Right card: the "
          "same split on a real model. Held-out (34 unseen, run once): all six "
          "gates pass.")
rows = [
    ["System", "early retr", "false trig", "multi-intent", "recall@k",
     "intent cov", "TTFT med", "retr/turn"],
    ["B0  LLM only", "0.000", "0.000", "0.000", "0.000", "0.000", "0.000", "0.00"],
    ["B1  static RAG", "0.000", "1.000", "0.000", "0.835", "0.768", "+0.010", "1.00"],
    ["B2  naive streaming", "1.000", "1.000", "0.000", "0.835", "0.768",
     "−2.069", "4.04"],
    [("B3  Ripple", True, PURPLE_D), ("0.884", True, INK), ("0.000", True, GOOD),
     ("0.889", True, INK), ("0.960", True, GOOD), ("0.772", True, INK),
     ("−1.036", True, GOOD), ("2.75", True, INK)],
]
table(s, CONTENT_L, 1.68, CONTENT_W, rows,
      [2.50, 1.30, 1.30, 1.50, 1.24, 1.32, 1.20, 1.14], size=10.5, row_h=0.28)

shp, tf = card(s, CONTENT_L, 3.30, 5.58, 1.16, fill=GOOD_T)
para(tf, "All six acceptance gates PASS", size=12.5, bold=True, color=GOOD,
     font=HEAD, first=True, space_after=4)
para(tf, "G2 early retrieval 0.884 · G3 multi-intent 0.889 · "
         "G4 zero fabricated IDs · G5 state continuity 1.000 · "
         "G6 trace coverage 1.000 · G1 one-command container",
     size=10.5, color=INK, space_after=0)

shp, tf = card(s, CONTENT_L + 5.80, 3.30, 5.70, 1.16, fill=TINT)
para(tf, "Real model — gemini-3.5-flash-lite, 26 Sep 2026", size=12.5,
     bold=True, color=PURPLE_D, font=HEAD, first=True, space_after=4)
para(tf, "Ripple vs static RAG, all 40 dev scenarios: 158/158 claims backed by "
         "their cited passage · 0 fabricated · recall 0.942 vs 0.835 · "
         "1.44× tokens per turn · first answer −3.17 s on two-question calls.",
     size=10.5, color=INK, space_after=0)

shp, tf = card(s, CONTENT_L, 4.60, CONTENT_W, 1.04, fill=WARN_T)
para(tf, "Limitations, stated plainly", size=12.5, bold=True, color=WARN,
     font=HEAD, first=True, space_after=4)
para(tf, "The table's −1.036 s assumes an instant model; with ~2 s real replies "
         "the median is +1.80 s (static RAG +2.07 s) and the p90 tail is worse "
         "(10.1 s vs 2.5 s).  ·  With a real model static RAG edges coverage "
         "(0.854 vs 0.833).  ·  Grounding is an automated check, not human "
         "review.  ·  Keyless mode declined 0 of 2 unanswerable held-out "
         "questions (near-misses).  ·  No currency cost: no published price "
         "configured.",
     size=10.5, color=INK, space_after=0)

# --- 9. What's next -------------------------------------------------------
clear_body(8)
s = S[8]
kicker(s, "Ordered by measured expected value, not by novelty.")
nexts = [
    ("1", "Cross-encoder reranking",
     "The measured bottleneck is selection: recall 0.960, coverage 0.772 (keyless). Our "
     "lexical reranker shared the retriever's bias and made it worse. A "
     "cross-encoder scores relevance rather than term overlap — the "
     "highest-value next experiment, and currently unmeasured."),
    ("2", "Semantic embeddings",
     "bge-small behind a switch. Ablation A4 must be re-run with it: BM25 "
     "would likely earn its place back against an embedder with no character "
     "n-grams."),
    ("3", "Learned controller",
     "The stability signal is unsupervised today. With labelled retrieval-"
     "timing data, a small classifier over stability, novelty, prefix length "
     "and jump similarity should beat a single threshold."),
    ("4", "Scale and a real corpus",
     "151 chunks is enough to measure; production knowledge bases are "
     "10⁴–10⁵ sections, where the flat-index decision is "
     "revisited. Swapping corpora is a config change by design."),
]
w = (CONTENT_W - 0.22) / 2
for i, (n, h1, b) in enumerate(nexts):
    x = CONTENT_L + (i % 2) * (w + 0.22)
    y = 1.70 + (i // 2) * 1.96
    shp, tf = card(s, x, y, w, 1.76, fill=WHITE, line=RULE)
    rich(tf, [(n + "   ", True, PURPLE), (h1, True, INK)], size=13.5, first=True,
         space_after=5)
    para(tf, b, size=11.5, color=MUTED, space_after=0)

# --- 10. Brownie points / differentiation --------------------------------
clear_body(9)
s = S[9]
kicker(s, "What another team cannot retrofit in the time remaining.")
diffs = [
    ("The claim graph is a one-way door",
     "Build refinement as string rewriting and converting to claim-level state "
     "is a rewrite of synthesis, state and the UI. Nobody switches on day six. "
     "It is why our citation drift is structurally 0, not prompted."),
    ("θ derived from a cost curve",
     "“Why 0.40?” has a plot, not a shrug: E[cost] = P(false "
     "trigger)·c_waste + P(late)·c_late, from the ~1000× "
     "asymmetry between a 4 ms lookup and 700 ms of silence."),
    ("We report the half of G2 others omit",
     "Early-retrieval rate AND false-trigger rate. A system that retrieves on "
     "every chunk scores 1.000 on the first and 1.000 on the second. We report "
     "0.884 and 0.000."),
    ("Held-out split, run once after freeze",
     "34 unseen scenarios: all six gates pass, one to two points below dev "
     "(recall 0.937). The honest answer to “did you overfit to your demo?”"),
    ("Output matches the guide's §4 JSON exactly",
     "retrieval_events, sub_queries, answer, citations, uncertainty — "
     "their field names, so a private harness written against the guide "
     "consumes our replay output unmodified."),
    ("Negative results published",
     "Two abstention hypotheses measured and discarded, two components deleted "
     "on evidence, and five bugs found by a number disagreeing with the "
     "mechanism. All in the repo, not hidden."),
]
w = (CONTENT_W - 0.44) / 3
for i, (h1, b) in enumerate(diffs):
    x = CONTENT_L + (i % 3) * (w + 0.22)
    y = 1.70 + (i // 3) * 1.86
    shp, tf = card(s, x, y, w, 1.70, fill=TINT)
    para(tf, h1, size=12.5, bold=True, color=PURPLE_D, font=HEAD, first=True,
         space_after=4)
    para(tf, b, size=10.5, color=INK, space_after=0)

_, tf = textbox(s, CONTENT_L, 5.48, CONTENT_W, 0.30)
para(tf, "Plus the three §8 deliverables most teams will not notice are "
         "required: a 6-page architecture brief, an evaluation report with "
         "analysed failures and five ablations, and a documented telemetry "
         "schema.", size=10.5, color=MUTED, italic=True, first=True,
     space_after=0)

# --- 11. Checklist --------------------------------------------------------
s = S[10]
ph = body_placeholder(s)
if ph is not None:
    drop(ph)
for sh in list(s.shapes):
    if sh.has_text_frame and sh.text_frame.text.strip().startswith(
            "Working prototype code"):
        drop(sh)
items = [
    ("Working prototype code — public GitHub repo", "Y",
     "github.com/deathKiller10/ripple"),
    ("README with reproducible setup instructions", "Y",
     "docker compose up — no API key, no GPU"),
    ("Docker files and other requirements", "Y",
     "Multi-stage Dockerfile; index baked in at build time"),
    ("Demo video, max 5 minutes", "Y", "Linked in the repo README"),
    ("Presentation file (PPT or PDF)", "Y", "This deck"),
    ("Release tag PRISM_GENAI_HACKATHON_Y2026", "Y",
     "On the final commit; everything referenced is inside it"),
    ("Architecture brief, evaluation report, telemetry schema", "Y",
     "docs/ — Theme 4 Guide §8 deliverables"),
    ("AI usage disclosure", "Y", "docs/ai-log.md, maintained from day 0"),
]
for i, (label, yn, note) in enumerate(items):
    y = 1.68 + i * 0.47
    shp, tf = card(s, CONTENT_L, y, CONTENT_W, 0.40,
                   fill=WHITE if i % 2 else TINT, radius=False)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    rich(tf, [(yn + "   ", True, GOOD), (label + "   ", True, INK),
              (note, False, MUTED)], size=12, first=True, space_after=0)

# --- 12. Thank you --------------------------------------------------------
s = S[11]
_, tf = textbox(s, 1.14, 5.06, 8.0, 0.60, raw=True)
para(tf, "Ripple  ·  Theme 04 Streaming Live RAG", size=14, bold=True,
     color=PURPLE_D, font=HEAD, first=True, space_after=3)
para(tf, "github.com/deathKiller10/ripple", size=12, color=MUTED, space_after=0)

prs.save(OUT)
print("wrote", OUT)
