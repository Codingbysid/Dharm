FROM python:3.11-slim
WORKDIR /srv/dharm
ENV PYTHONUNBUFFERED=1 HF_HOME=/tmp/huggingface
COPY decision-corpus-ml/requirements.txt decision-corpus-ml/requirements.txt
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 && pip install --no-cache-dir -r app/requirements.txt
COPY app/ app/
COPY decision-corpus-ml/rc1_inference.py decision-corpus-ml/
COPY decision-corpus-ml/data/decisions_v0.2.jsonl decision-corpus-ml/data/chanakya_daily_supplement.jsonl decision-corpus-ml/data/
COPY decision-corpus-ml/artifacts-v1.1/abstention_clf.joblib decision-corpus-ml/artifacts-v1.1/
COPY decision-corpus-ml/artifacts-v1.6/workflow_router.joblib decision-corpus-ml/artifacts-v1.6/pair_verifier_model.joblib decision-corpus-ml/artifacts-v1.6/pair_scorer_config.json decision-corpus-ml/artifacts-v1.6/verifier_config.json decision-corpus-ml/artifacts-v1.6/
EXPOSE 8000
ENV PORT=8000
CMD ["sh", "-c", "python -m uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
