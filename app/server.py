"""FastAPI Application Server for Dharm Decision Corpus Chat.

Serves:
- Single-process model initialization via lifespan (DharmRC1Engine loaded ONCE).
- SSE streaming chat endpoint with real-time token delivery and metadata events.
- Server-side validated citation lookup endpoint (/api/citations/{id}).
- In-memory isolated multi-turn session management.
- Static web chat UI (/ and static assets).
"""
import secrets
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Add decision-corpus-ml to python path so rc1_inference can be imported cleanly
BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "decision-corpus-ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


from app.generator import (
    build_system_prompt,
    detect_query_intent,
    formulate_retrieval_query,
    MODEL, WELCOME, local_model_available, stream_answer,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# In-memory session store: session_id -> list of {"role": "user"|"assistant", "content": str, "metadata": dict}
SESSIONS: Dict[str, List[Dict[str, Any]]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load retrieval models and indices once at server startup."""
    print("=" * 80)
    print("Initializing Dharm RC1 Engine (Loading models once per process)...")
    from rc1_inference import DharmRC1Engine
    app.state.engine = DharmRC1Engine(device="cpu")
    app.state.model_lock = asyncio.Lock()
    app.state.records = app.state.engine.records
    print(f"Dharm RC1 Engine Ready! Loaded {len(app.state.records)} corpus records.")
    print("=" * 80)
    yield
    print("Shutting down Dharm server...")


app = FastAPI(title="Dharm Local Chat Application", lifespan=lifespan)

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
    """Serve the chat frontend."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_file)


@app.get("/api/health")
async def health():
    """Health check endpoint showing model, provider, and active session count."""
    ready = await local_model_available()
    return {
        "status": "online",
        "engine": "DharmRC1Engine",
        "release_candidate": "pair-verifier-rc1",
        "provider_mode": MODEL if ready else "Sources only",
        "generation_ready": ready,
        "total_sessions": len(SESSIONS),
        "records_loaded": len(app.state.records),
    }


@app.get("/api/citations/{passage_id}")
async def get_citation(passage_id: str):
    """Return validated record details for a specific citation ID."""
    pid = passage_id.strip().upper()
    records = app.state.records
    if pid not in records:
        raise HTTPException(status_code=404, detail=f"Passage ID '{pid}' not found in corpus.")

    r = records[pid]
    return {
        "id": pid,
        "work": r["work"],
        "locator": r["locator"],
        "decision_domain": r["decision_domain"],
        "urgency_level": r["urgency_level"],
        "reversibility": r["reversibility"],
        "derived_principle": r["derived_principle"],
        "context_summary": r["context_summary"],
        "english_translation": r["english_translation"],
        "modern_application": r["scenarios_json"][0]["scenario"] if r.get("scenarios_json") else "",
        "sequence_role": r.get("complementary_passage_id", {}).get("sequence_role", ""),
        "contrast_condition": r.get("competing_passage_id", {}).get("condition", ""),
        "contrast_target_id": r.get("competing_passage_id", {}).get("id", ""),
        "source_url": r.get("source_url", ""),
    }


@app.post("/api/sessions/clear")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"status": "cleared", "session_id": session_id}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events (SSE)."""
    user_msg = request.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in SESSIONS:
        SESSIONS[session_id] = []

    history = SESSIONS[session_id]

    engine = app.state.engine
    records = app.state.records

    # 1. Detect conversational intent & formulate retrieval query
    intent = detect_query_intent(user_msg, history)
    query_text = formulate_retrieval_query(user_msg, history, intent)

    # 2. Run Dharm retrieval pipeline (unless it is an ordinary greeting or out-of-scope)
    if intent == "greeting":
        retrieval_res = {
            "abstained": False,
            "abstention_prob": 0.0,
            "primary": None,
            "top3": [],
            "pair_promoted": False,
            "promoted_pair": None,
            "acceptance_score": None,
            "fallback_reason": None,
        }
    elif intent == "out_of_scope":
        retrieval_res = {
            "abstained": True,
            "abstention_prob": 1.0,
            "primary": "Abstain",
            "top3": [],
            "pair_promoted": False,
            "promoted_pair": None,
            "acceptance_score": None,
            "fallback_reason": "out_of_scope_domain",
        }
    else:
        # Run local RC1 inference
        async with app.state.model_lock:
            retrieval_res = await run_in_threadpool(engine.predict, query_text)

    # 3. Build verified citation metadata
    validated_citations = []
    top3_ids = retrieval_res.get("top3", [])
    for pid in top3_ids:
        if pid in records:
            r = records[pid]
            validated_citations.append({
                "id": pid,
                "work": r["work"],
                "locator": r["locator"],
                "principle": r["derived_principle"],
                "translation": r["english_translation"],
                "context": r["context_summary"],
                "modern_application": r["scenarios_json"][0]["scenario"] if r.get("scenarios_json") else "",
                "source_url": r.get("source_url", ""),
            })

    ready = await local_model_available() if intent != "greeting" and not retrieval_res["abstained"] else False
    provider_mode = MODEL if ready else "Sources only"

    metadata_payload = {
        "session_id": session_id,
        "intent": intent,
        "abstained": retrieval_res.get("abstained", False),
        "abstention_prob": retrieval_res.get("abstention_prob", 0.0),
        "primary": retrieval_res.get("primary"),
        "top3": retrieval_res.get("top3", []),
        "pair_promoted": retrieval_res.get("pair_promoted", False),
        "promoted_pair": retrieval_res.get("promoted_pair"),
        "acceptance_score": retrieval_res.get("acceptance_score"),
        "fallback_reason": retrieval_res.get("fallback_reason"),
        "citations": validated_citations,
        "provider_mode": provider_mode,
        "reformulated_query": query_text if intent == "followup" else None,
    }

    system_prompt = build_system_prompt(records, retrieval_res, intent) if ready else ""

    async def event_generator():
        # Emit metadata event first
        yield f"event: metadata\ndata: {json.dumps(metadata_payload)}\n\n"

        full_assistant_reply = []

        try:
            if ready:
                async for chunk in stream_answer(system_prompt, query_text):
                    full_assistant_reply.append(chunk)
                    yield f"event: chunk\ndata: {json.dumps({'text': chunk})}\n\n"
            else:
                if intent == "greeting":
                    text = WELCOME
                elif retrieval_res["abstained"]:
                    text = "I couldn't ground this question in a passage. What decision are you trying to make?"
                else:
                    text = "Local generation is unavailable. You can explore the retrieved sources below."
                full_assistant_reply.append(text)
                yield f"event: chunk\ndata: {json.dumps({'text': text})}\n\n"

            # Save to conversation history
            complete_text = "".join(full_assistant_reply)
            history.append({"role": "user", "content": user_msg})
            history.append({
                "role": "assistant",
                "content": complete_text,
                "metadata": metadata_payload,
            })

            del history[:-12]
            yield 'event: done\ndata: {"status":"complete"}\n\n'

        except Exception:
            yield f"event: error\ndata: {json.dumps({'error': 'Could not complete the reply. Please retry.'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=False)
