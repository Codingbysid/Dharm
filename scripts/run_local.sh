#!/bin/sh
# One local command. Retrieval runs in this process. Gemini stays off unless
# GEMINI_API_KEY is set and GEMINI_EDITOR is not 0. The editor defaults open.
cd "$(dirname "$0")/.." || exit 1
PYTHON="decision-corpus-ml/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Missing decision-corpus-ml/.venv. Create it with Python 3.11 and install app/requirements.txt." >&2
  exit 1
fi
exec "$PYTHON" -m uvicorn app.server:app --host 127.0.0.1 --port 8000
