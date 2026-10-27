# syntax=docker/dockerfile:1
# Targets:
#   api   (default) - FastAPI service; models are trained at build time from seeded synthetic data
#   spark           - adds a Java 17 runtime + PySpark and runs the Structured Streaming demo
# Debian bookworm is pinned so the openjdk-17 package used by the spark target is available.

FROM python:3.11-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VTAD_CONFIG=/app/configs/config.yaml
WORKDIR /app

# libgomp1: OpenMP runtime required by the XGBoost and scikit-learn wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN pip install --no-deps .

# Deterministic synthetic data + training so the image is self-contained without committing binaries.
RUN python scripts/generate_data.py --target training \
    && python scripts/train_model.py \
    && rm -rf data/raw

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app


FROM base AS spark
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-spark.txt ./
RUN pip install -r requirements-spark.txt
USER appuser
CMD ["sh", "-c", "python scripts/generate_data.py --target streaming && python scripts/run_streaming.py --engine spark --reset && cat reports/streaming_summary.md"]


FROM base AS api
USER appuser
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"
CMD ["python", "-m", "vehicle_anomaly.api"]
