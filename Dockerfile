# syntax=docker/dockerfile:1
# FastAPI service + browser demo. Used by Render (render.yaml) and docker compose.
# Models, the evaluation report and a streaming run are produced at build time from seeded synthetic
# data, so the image is self-contained and no data or model binaries are committed to Git.
# The optional PySpark demo image is in Dockerfile.spark.

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    VTAD_CONFIG=/app/configs/config.yaml
WORKDIR /app

# libgomp1: OpenMP runtime required by the XGBoost and scikit-learn wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The Linux "xgboost" wheel depends on the CUDA NCCL library; "xgboost-cpu" is the same Python
# package (import name "xgboost") without GPU dependencies, which keeps the image small.
COPY requirements.txt ./
RUN sed -E 's/^xgboost([<>=! ]|$)/xgboost-cpu\1/' requirements.txt > /tmp/requirements.txt \
    && grep -q '^xgboost-cpu' /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . && rm -rf build src/*.egg-info

COPY configs ./configs
COPY scripts ./scripts
# Only the benchmark report passes .dockerignore: it was measured on a developer machine (see README)
# and is shown in the demo with that environment. Evaluation and streaming reports are rebuilt below.
COPY reports ./reports

RUN python scripts/generate_data.py \
    && python scripts/train_model.py \
    && python scripts/run_streaming.py --engine local --reset \
    && rm -rf data

RUN useradd --create-home --uid 10001 appuser
USER appuser

# Render injects PORT (default 10000); 8000 is used when it is not set.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"
CMD ["python", "-m", "vehicle_anomaly.api"]
