"""FastAPI chat server.

One process loads RC1. Each reply is a DecisionPlan rendered locally. Gemini,
when configured, may rewrite only the reason.
"""
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import secrets
import sys
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "decision-corpus-ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


def _load_local_env():
    path = BASE_DIR / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_local_env()

from app.conversation import detect_query_intent, formulate_retrieval_query
from app.editor import accept_edit, edit_reason, editor_enabled
from app.planner import build_plan
from app.render import render_plan

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSIONS: Dict[str, List[Dict[str, Any]]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("DHARM_SKIP_ENGINE") == "1":
        app.state.engine = getattr(app.state, "engine", None)
        app.state.records = getattr(app.state, "records", {})
        app.state.model_lock = getattr(app.state, "model_lock", None) or asyncio.Lock()
        yield
        return
    print("Initializing Dharm RC1 Engine...", flush=True)
    from rc1_inference import DharmRC1Engine
    app.state.engine = DharmRC1Engine(device="cpu")
    app.state.model_lock = asyncio.Lock()
    app.state.records = app.state.engine.records
    print(f"Dharm RC1 Engine Ready. Loaded {len(app.state.records)} records.", flush=True)
    yield


app = FastAPI(title="Dharm", lifespan=lifespan)


@app.middleware("http")
async def backend_token(request: Request, call_next):
    token = os.getenv("DHARM_BACKEND_TOKEN")
    if token and not secrets.compare_digest(
        request.headers.get("authorization", "").encode(), ("Bearer " + token).encode()
    ):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


class ChatRequest(BaseModel):
    message: str = Field(max_length=4000)
    session_id: Optional[str] = Field(default=None, max_length=128)


@app.get("/")
async def root():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_file)


@app.get("/api/health")
async def health():
    records = getattr(app.state, "records", {}) or {}
    return {
        "status": "online" if records else "starting",
        "ready": bool(records),
        "records_loaded": len(records),
        "editor": editor_enabled(),
    }


@app.get("/api/citations/{passage_id}")
async def get_citation(passage_id: str):
    pid = passage_id.strip().upper()
    records = app.state.records
    if pid not in records:
        raise HTTPException(status_code=404, detail=f"Passage ID '{pid}' not found in corpus.")
    row = records[pid]
    return _citation(pid, row)


def _citation(pid, row):
    return {
        "id": pid,
        "work": row.get("work", ""),
        "locator": row.get("locator", ""),
        "translation": row.get("english_translation", ""),
        "principle": row.get("derived_principle", ""),
        "caveat": _caveat(row),
        "source_url": row.get("source_url", ""),
    }


def _caveat(row):
    text = (row.get("application_limit") or "").strip()
    sentence = text.split(".")[0].strip()
    if not sentence or "sattvic" in sentence.casefold() or "editorial" in sentence.casefold():
        return ""
    return sentence


def _empty_retrieval(abstained=False):
    return {
        "abstained": abstained,
        "abstention_prob": 1.0 if abstained else 0.0,
        "primary": "Abstain" if abstained else None,
        "top3": [],
        "pair_promoted": False,
        "promoted_pair": None,
        "acceptance_score": None,
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    user_msg = request.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = request.session_id or str(uuid.uuid4())
    history = SESSIONS.setdefault(session_id, [])
    records = app.state.records or {}
    intent = detect_query_intent(user_msg, history)
    query_text = formulate_retrieval_query(user_msg, history, intent)

    if intent in {"greeting", "clarify"}:
        retrieval = _empty_retrieval(False)
    elif intent == "out_of_scope":
        retrieval = _empty_retrieval(True)
    else:
        async with app.state.model_lock:
            retrieval = await run_in_threadpool(app.state.engine.predict, query_text)

    plan = build_plan(user_msg, history, intent, retrieval, records)
    edited = False
    if plan.disposition == "answer" and editor_enabled():
        evidence = [
            {"id": item.id, "principle": item.principle}
            for item in plan.supports
        ]
        try:
            rewritten = await asyncio.wait_for(
                run_in_threadpool(edit_reason, plan, evidence),
                timeout=12,
            )
        except Exception:
            rewritten = None
        if rewritten and accept_edit(plan, rewritten):
            plan.reason = rewritten
            edited = True
    text = render_plan(plan)
    cited_ids = [item.id for item in plan.supports]
    citations = [_citation(pid, records[pid]) for pid in (retrieval.get("top3") or []) if pid in records]

    metadata = {
        "session_id": session_id,
        "intent": intent,
        "disposition": plan.disposition,
        "plan": plan.as_dict(),
        "cited_ids": cited_ids,
        "citations": citations,
        "edited": edited,
        "provider_mode": "Edited" if edited else "Dharm",
    }
    if os.getenv("DHARM_TRACE"):
        print(json.dumps({
            "user": user_msg,
            "intent": intent,
            "retrieval_query": query_text,
            "top3": retrieval.get("top3"),
            "abstained": retrieval.get("abstained"),
            "pair_promoted": retrieval.get("pair_promoted"),
            "disposition": plan.disposition,
            "cited_ids": cited_ids,
            "edited": edited,
        }, ensure_ascii=False), flush=True)

    async def event_generator():
        yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"
        yield f"event: chunk\ndata: {json.dumps({'text': text})}\n\n"
        history.append({"role": "user", "content": user_msg})
        stored_intent = "clarify" if plan.disposition == "clarify" and intent == "new_query" else intent
        history.append({
            "role": "assistant",
            "content": text,
            "metadata": {
                "intent": stored_intent,
                "clarify_kind": plan.clarify_kind,
                "plan": plan.as_dict(),
            },
        })
        del history[:-12]
        yield 'event: done\ndata: {"status":"complete"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=False)
