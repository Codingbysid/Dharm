## Run locally

Starting point for this branch: `4e33b4e`.

One command, from the repository root:

```sh
./scripts/run_local.sh
```

That starts the chat at `http://127.0.0.1:8000`. The process loads the RC1 retriever once, from `decision-corpus-ml/data`, `artifacts-v1.1/abstention_clf.joblib`, and the router, verifier, and configs in `artifacts-v1.6`. Hugging Face weights stay in the shared cache. No Ollama model is used.

Gemini rewrites the explanation when `GEMINI_API_KEY` is set. Set `GEMINI_EDITOR=0` to keep Dharm's own wording. The action, order, limit, and passage IDs stay with Dharm. A rejected edit is discarded. Put the key on the backend only, and do not commit it.

## Vercel

Import this branch. Framework **Other**, root `app/static`. The page calls same-origin `/api/health` and `/api/chat` (`API_BASE` is empty). `app/static/vercel.json` sends those through `app/static/api/relay.mjs`. Set server-side `DHARM_BACKEND_URL` (HTTPS) and `DHARM_BACKEND_TOKEN`. Until both exist, the page says the decision service is offline. Do not put the API key or model files in the frontend. Do not treat that page as a working public chatbot, and do not start a paid backend until retrieval-only memory and latency are accepted.

Check the static output locally:

```sh
cd app/static && mkdir -p public && cp index.html public/index.html
```

`compose.beta.yaml` is the later backend shape. It does not include Ollama.
