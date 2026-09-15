"""FastAPI + WebSocket transport.

A thin client of the engine. Nothing in ripple/engine.py imports this module,
which is what keeps the replay CLI (and therefore gate G1) independent of the
web stack.

Endpoints
---------
GET  /                      the dashboard
GET  /api/scenarios         list replayable scenarios
GET  /api/corpus/{cite}     fetch a chunk by citation, for the evidence panel
WS   /ws                    live session

WebSocket protocol, client -> server:
    {"type": "chunk",   "t": 1.2, "text": "...", "speaker": "customer"}
    {"type": "end",     "t": 3.1}
    {"type": "replay",  "scenario_id": "dev_multi_01", "speed": 1.0}
    {"type": "reset"}

server -> client: every telemetry event verbatim, plus {"type": "turn", ...}
when an utterance resolves. The dashboard renders the trace; it does not
compute anything the engine did not already emit, so what a judge sees on
screen is exactly what is in the JSONL.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Config
from .engine import RippleEngine
from .schemas import Scenario

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(os.path.dirname(HERE), "frontend", "dist")
FALLBACK = os.path.join(os.path.dirname(HERE), "frontend", "dashboard.html")

app = FastAPI(title="Ripple", version="0.1.0")
_cfg = Config()
_engine: RippleEngine | None = None


def engine() -> RippleEngine:
    global _engine
    if _engine is None:
        if not os.path.exists(os.path.join(_cfg.index_path, "chunks.jsonl")):
            from .retrieval.index import build_and_save
            build_and_save(_cfg.corpus_path, _cfg.index_path, _cfg.embedder)
        _engine = RippleEngine(_cfg)
    return _engine


@app.get("/api/health")
def health():
    e = engine()
    return {
        "ok": True,
        "provider": e.provider.name,
        "embedder": _cfg.embedder,
        "reranker": getattr(e.reranker, "name", "?"),
        "index": e.index.stats(),
        "theta": _cfg.controller.theta,
        "nu": _cfg.controller.nu,
    }


@app.get("/api/scenarios")
def scenarios():
    out = []
    for split in ("dev", "heldout"):
        path = f"data/scenarios/{split}.jsonl"
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                out.append({
                    "scenario_id": d["scenario_id"], "split": d["split"],
                    "notes": d.get("notes", ""),
                    "turns": len([g for g in d["gold"]]),
                    "text": " ".join(c["text"] for c in d["chunks"])[:220],
                })
    return out


@app.get("/api/corpus")
def corpus_chunk(cite: str):
    e = engine()
    c = e.index.by_cite.get(cite.strip())
    if not c:
        return JSONResponse({"error": "not found"}, status_code=404)
    return c.to_dict()


def _load_scenario(scenario_id: str) -> Scenario | None:
    for split in ("dev", "heldout"):
        path = f"data/scenarios/{split}.jsonl"
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip() and json.loads(line)["scenario_id"] == scenario_id:
                    return Scenario.from_dict(json.loads(line))
    return None


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    e = engine()
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue = asyncio.Queue()

    def on_event(ev):
        # Called from the engine's thread of control; hop back to the loop.
        loop.call_soon_threadsafe(outbox.put_nowait,
                                  json.loads(ev.to_json()))

    session = e.session(on_event=on_event, sink_dir=_cfg.telemetry_path)

    async def pump():
        while True:
            item = await outbox.get()
            await sock.send_text(json.dumps(item))

    pump_task = asyncio.create_task(pump())

    async def emit_turn(result):
        await outbox.put({"type": "turn", **result.to_dict()})
        await outbox.put({"type": "state",
                          "claims": [c.to_dict() for c in
                                     session.graph.active_claims()],
                          "versions": [v.to_dict() for v in
                                       session.graph.versions],
                          "pool": session.pool.snapshot(),
                          "pool_counts": session.pool.counts(),
                          "intents": [i.to_dict() for i in
                                      session.graph.intents.values()],
                          "cost": session.bus.cost.to_dict()})

    try:
        while True:
            raw = await sock.receive_text()
            msg = json.loads(raw)
            kind = msg.get("type")

            if kind == "chunk":
                session.on_chunk(float(msg.get("t", 0.0)), msg.get("text", ""),
                                 msg.get("speaker", "customer"),
                                 bool(msg.get("is_final", False)))
            elif kind == "end":
                result = session.end_utterance(float(msg.get("t", 0.0)))
                await emit_turn(result)
            elif kind == "reset":
                session.close()
                session = e.session(on_event=on_event,
                                    sink_dir=_cfg.telemetry_path)
                await outbox.put({"type": "reset_ok"})
            elif kind == "replay":
                sc = _load_scenario(msg.get("scenario_id", ""))
                if not sc:
                    await outbox.put({"type": "error",
                                      "message": "unknown scenario"})
                    continue
                speed = float(msg.get("speed", 1.0)) or 1.0
                session.close()
                session = e.session(session_id=sc.scenario_id,
                                    on_event=on_event,
                                    sink_dir=_cfg.telemetry_path)
                prev = 0.0
                for ch in sc.chunks:
                    await asyncio.sleep(max(0.0, (ch.t - prev)) / speed)
                    prev = ch.t
                    await outbox.put({"type": "transcript", "t": ch.t,
                                      "text": ch.text, "speaker": ch.speaker,
                                      "is_final": ch.is_final})
                    session.on_chunk(ch.t, ch.text, ch.speaker, ch.is_final)
                    if ch.is_final:
                        result = session.end_utterance(ch.t + 0.001)
                        await emit_turn(result)
                await outbox.put({"type": "replay_done",
                                  "scenario_id": sc.scenario_id})
    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()
        # Session-scoped memory: state dies with the connection. There is no
        # persistence layer to clean up because the spec forbids one.
        session.close()


# The built React bundle when present, otherwise the zero-build dashboard.
if os.path.isdir(STATIC):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC, "assets")),
              name="assets")


@app.get("/")
def index():
    built = os.path.join(STATIC, "index.html")
    if os.path.exists(built):
        return FileResponse(built)
    return FileResponse(FALLBACK)
