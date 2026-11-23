## Run locally

Starting point for this branch: `4e33b4e`.

One command, from the repository root:

```sh
./scripts/run_local.sh
```

That starts the chat at `http://127.0.0.1:8000`. The process loads the RC1 retriever once, from `decision-corpus-ml/data`, `artifacts-v1.1/abstention_clf.joblib`, and the router, verifier, and configs in `artifacts-v1.6`. Hugging Face weights stay in the shared cache. No Ollama model is used.

With `GEMINI_EDITOR=0` or no `GEMINI_API_KEY`, every reply is the local DecisionPlan renderer: an answer, one clarifying question, or a short abstention. A list of passages is not the reply. Copy `.env.example` to `.env` and set `GEMINI_EDITOR=1` only when Gemini should rewrite the reason. The action, order, limit, and passage IDs stay with Dharm. The key is read on the server and is not committed.

## Vercel

Import this branch. Framework **Other**, root `app/static`. The page calls `/api/health` and `/api/chat`, which `app/static/vercel.json` sends through `app/static/api/relay.mjs`. Set `DHARM_BACKEND_URL` to an HTTPS origin and `DHARM_BACKEND_TOKEN` to the same secret as the backend. Until those exist, the page shows that the decision service is not connected. Do not treat that preview as a working public chatbot, and do not start a paid backend until retrieval-only memory and latency are accepted.

`compose.beta.yaml` is the later backend shape. It does not include Ollama.
