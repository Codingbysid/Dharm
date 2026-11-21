"""FastAPI Application Server for Dharm Decision Corpus Chat.

Serves:
- Single-process model initialization via lifespan (DharmRC1Engine loaded ONCE).
- SSE streaming chat endpoint with real-time token delivery and metadata events.
- Server-side validated citation lookup endpoint (/api/citations/{id}).
- In-memory isolated multi-turn session management.
- Static web chat UI (/ and static assets).
"""
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# Add decision-corpus-ml to python path so rc1_inference can be imported cleanly
BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "decision-corpus-ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from rc1_inference import DharmRC1Engine
from app.generator import (
    build_system_prompt,
    detect_query_intent,
    formulate_retrieval_query,
    get_api_key,
    stream_gemini_response,
    stream_retrieval_preview_response,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# In-memory session store: session_id -> list of {"role": "user"|"assistant", "content": str, "metadata": dict}
SESSIONS: Dict[str, List[Dict[str, Any]]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load retrieval models and indices once at server startup."""
    print("=" * 80)
    print("Initializing Dharm RC1 Engine (Loading models once per process)...")
    app.state.engine = DharmRC1Engine(device="cpu")
    app.state.records = app.state.engine.records
    print(f"Dharm RC1 Engine Ready! Loaded {len(app.state.records)} corpus records.")
    print("=" * 80)
    yield
    print("Shutting down Dharm server...")


app = FastAPI(title="Dharm Local Chat Application", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


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
    api_key = get_api_key()
    mode = "Live Generation (Gemini 2.5 Flash)" if api_key else "Retrieval Preview Mode (Set GEMINI_API_KEY)"
    return {
        "status": "online",
        "engine": "DharmRC1Engine",
        "release_candidate": "pair-verifier-rc1",
        "provider_mode": mode,
        "api_key_configured": bool(api_key),
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
        "source_url": r.get("source_url") or r.get("retrieval_cues", ""),
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
        retrieval_res = engine.predict(query_text)

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
                "source_url": r.get("source_url") or r.get("retrieval_cues", ""),
            })

    api_key = get_api_key()
    provider_mode = "Live Generation (Gemini 2.5 Flash)" if api_key else "Retrieval Preview Mode (Set GEMINI_API_KEY)"

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

    system_prompt = build_system_prompt(records, retrieval_res, intent)

    async def event_generator():
        # Emit metadata event first
        yield f"event: metadata\ndata: {json.dumps(metadata_payload)}\n\n"

        full_assistant_reply = []

        try:
            if api_key and intent != "greeting":
                async for chunk in stream_gemini_response(api_key, system_prompt, user_msg, history):
                    full_assistant_reply.append(chunk)
                    yield f"event: chunk\ndata: {json.dumps({'text': chunk})}\n\n"
            else:
                # Use structured preview mode
                async for chunk in stream_retrieval_preview_response(records, retrieval_res, user_msg, intent):
                    full_assistant_reply.append(chunk)
                    yield f"event: chunk\ndata: {json.dumps({'text': chunk})}\n\n"
                    # Gentle artificial pacing for preview mode
                    await asyncio.sleep(0.02)

            yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

            # Save to conversation history
            complete_text = "".join(full_assistant_reply)
            history.append({"role": "user", "content": user_msg})
            history.append({
                "role": "assistant",
                "content": complete_text,
                "metadata": metadata_payload,
            })

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

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
