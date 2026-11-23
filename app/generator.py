"""Conversational grounding and local Ollama generation for the frozen RC1 index."""
import json
import os
import re

import httpx

MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
WELCOME = "What decision are you weighing? I can help you explore it using the Vedas, Upanishads, Bhagavad Gita, and Chanakya tradition."


def detect_query_intent(text, history):
    words = text.strip().casefold().strip(" .!?")
    if re.fullmatch(r"(hi|hello|hey|namaste|good (morning|afternoon|evening)|help|who are you|what can you do|what is dharm)", words):
        return "greeting"
    if re.match(r"^(new topic|different question|unrelated)\b", words):
        return "new_query"
    if re.match(r"^(?:how (?:do|can) i (?:fix a (?:leaking )?pipe|bake a cake)|write a python script to|give me a recipe for)\b", words):
        return "out_of_scope"
    if history and re.match(r"^(what if|suppose|in that case|instead|what about|can we still|what should we do then|why is that|why$|what changes|and if|does that|how soon|the deadline (?:moved|changed))\b", words):
        return "followup"
    return "new_query"


def retrieval_wording(text):
    # Disambiguate an explicitly named software object while preserving financial uses of release.
    return re.sub(r"\b(?:releasing|shipping|launching)\s+(?:a |the |our )?(software )?feature\b",
                  "rolling out a software feature", text, flags=re.I)


def formulate_retrieval_query(message, history, intent):
    if intent != "followup":
        return retrieval_wording(message)
    # Each assistant turn stores the active topic and its correlated user turn.
    # Stop at a topic boundary even when the older question was longer.
    turns = []
    for user, answer in reversed(list(zip(history[::2], history[1::2]))):
        previous = answer.get("metadata", {}).get("intent", "new_query")
        if previous in {"greeting", "out_of_scope"}:
            break
        turns.append(user["content"])
        if previous != "followup":
            break
    if not turns:
        return retrieval_wording(message)
    context = list(reversed(turns))
    return retrieval_wording("\n".join([context[0], *context[1:][-2:], "Latest update (supersedes earlier constraints): " + message]))


def build_system_prompt(records, result, intent):
    evidence = [
        {key: records[pid][key] for key in ("id", "work", "locator", "derived_principle", "english_translation", "application_limit")}
        for pid in result["top3"] if pid in records
    ]
    return (
        "You are Dharm. Answer the user's actual decision in plain English. Treat the latest "
        "update as current. Begin with a concrete next step or one focused clarifying question. "
        "The following retrieved records are possible evidence. Check that their meaning fits "
        "the user's situation. A shared word alone is insufficient. Omit weak analogies. "
        "Distinguish practical suggestions from the text's teaching. Cite a supporting record "
        "as [Dnnn] only when it supports the specific claim. Use only the supplied record IDs. "
        "If no record fits, say so briefly. Do not invent quotations or facts about the user.\n"
        + json.dumps(evidence, ensure_ascii=False)
    )


async def local_model_available():
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(OLLAMA + "/api/tags")
            response.raise_for_status()
            return any(row.get("name") == MODEL for row in response.json().get("models", []))
    except (httpx.HTTPError, ValueError):
        return False


async def stream_answer(system, query):
    payload = {"model": MODEL, "stream": True, "think": False,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": query}],
               "options": {"temperature": 0.2, "num_predict": 350}}
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=3)) as client:
        async with client.stream("POST", OLLAMA + "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise RuntimeError("Local generation failed.")
                chunk = data.get("message", {}).get("content")
                if chunk:
                    yield chunk
                if data.get("done"):
                    return
    raise RuntimeError("Local model stream ended early.")
