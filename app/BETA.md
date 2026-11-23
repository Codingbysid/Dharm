## Friends preview

Run the inference backend on one persistent host with Python 3.11, the repository, the RC1 model files, and Ollama. Use one worker because conversation history is process-local. Sessions reset at restart. Run these commands from the repository root in a virtual environment.

```sh
python -m pip install -r app/requirements.txt
ollama pull qwen3:4b
python -m uvicorn app.server:app --port 8000
```

Start Ollama before launching the app. The default Ollama origin is local port 11434. Set OLLAMA_BASE_URL when the model service is on a private network. The first RC1 startup also needs access to the pinned pipeline's pretrained Hugging Face weights.

For the hosted preview, place the backend behind HTTPS, bind uvicorn to the interface required by that host, and set DHARM_BACKEND_TOKEN to a generated secret before exposing it. Keep the Ollama port private. Set a hosting budget or request limit for the trial.

Import the beta Git branch in Vercel. Select **Other** as the framework and **app/static** as the project root. Configure DHARM_BACKEND_URL (HTTPS origin) and DHARM_BACKEND_TOKEN (matching secret) as server-side preview environment variables. Deploy the preview. The frontend sends same-origin requests to the Vercel relay. No Torch or Ollama weights belong in the Vercel deployment. Confirm preview access settings allow the invited testers.

Open the preview on another device. Confirm the status displays the configured model and send a new dilemma followed by a changed deadline. Also test a short unrelated question, a new chat, and an unavailable inference backend. With Ollama offline the app exposes retrieved sources only. Check a cited passage against the answer before inviting testers.

### Persistent Linux backend

On a host with Docker Compose and sufficient RAM for RC1 and Qwen together, export DHARM_BACKEND_TOKEN in your shell. From the repository root run:

```sh
docker compose -f compose.beta.yaml up -d --build
docker compose -f compose.beta.yaml exec ollama ollama pull qwen3:4b
```

Configure the host's HTTPS reverse proxy to forward to 127.0.0.1:8000 and allow streaming. Use its HTTPS origin as DHARM_BACKEND_URL. Verify health through Vercel. Named volumes retain model downloads across restarts. Do not commit secrets.
