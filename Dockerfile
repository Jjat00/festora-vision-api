# ==============================================================
# festora-vision-api — Production Dockerfile
#
# Build strategy:
#   Stage 1 (builder): Install Python deps + download all ML models.
#   Stage 2 (runtime): Copy only the installed packages and model cache.
#
# This means `docker run` is fast — models are already on disk.
# The image is large (~5 GB) by necessity (PyTorch + model weights).
# CPU-only torch keeps it ~2 GB smaller than the GPU variant.
# ==============================================================

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

# System dependencies for OpenCV + DeepFace
# libgl1-mesa-glx was renamed to libgl1 in Debian Trixie (Debian 13).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only requirements first to maximise Docker layer caching.
COPY requirements.txt .

# Install CPU-only PyTorch first (smaller wheel, no CUDA).
# The --index-url overrides PyPI for torch/torchvision only.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.10.0 \
        torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cpu

# Install pyiqa WITHOUT its declared dependencies.
# pyiqa 0.1.14.1 hard-pins transformers==4.37.2 but we need 5.x for CLIP.
# BRISQUE and NIMA don't call the transformers library internally — they
# only need torch, torchvision, timm, scipy, and addict, all of which
# are provided by requirements.txt.
RUN pip install --no-cache-dir --no-deps pyiqa==0.1.14.1

# Install remaining dependencies from PyPI (includes transformers 5.2.0).
# torch/torchvision are already satisfied — pip will skip reinstalling.
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download all ML model weights at build time.
# This runs a helper script that triggers each library's auto-download
# mechanism, placing files in /app/models_cache.
ENV TRANSFORMERS_CACHE=/app/models_cache
ENV HF_HOME=/app/models_cache
ENV TORCH_HOME=/app/models_cache

COPY scripts/download_models.py /build/download_models.py
RUN python /build/download_models.py


# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="festora-vision-api"
LABEL org.opencontainers.image.description="Reusable image analysis microservice"
LABEL org.opencontainers.image.source="https://github.com/Jjat00/festora-vision-api"
LABEL org.opencontainers.image.licenses="MIT"

# Runtime system libraries (same as builder, minus build tools).
# libgl1-mesa-glx was renamed to libgl1 in Debian Trixie (Debian 13).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security.
RUN groupadd --gid 1001 visionapi && \
    useradd --uid 1001 --gid visionapi --shell /bin/bash --create-home visionapi

WORKDIR /app

# Copy installed Python packages from builder.
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-downloaded model weights and fix ownership so the non-root
# visionapi user can write HuggingFace cache metadata at runtime.
COPY --from=builder /app/models_cache /app/models_cache
RUN chown -R visionapi:visionapi /app/models_cache

# Copy application source.
COPY --chown=visionapi:visionapi app/ /app/app/

# Point all model libraries at the pre-downloaded cache.
ENV TRANSFORMERS_CACHE=/app/models_cache
ENV HF_HOME=/app/models_cache
ENV TORCH_HOME=/app/models_cache
ENV DEEPFACE_HOME=/app/models_cache

# Default service configuration — all overridable at runtime via env.
ENV ENVIRONMENT=production
ENV LOG_JSON=true
ENV LOG_LEVEL=INFO
ENV RATE_LIMIT_ENABLED=true
ENV RATE_LIMIT_PER_MINUTE=60

USER visionapi

# Reduce TensorFlow/Keras memory and startup overhead.
# TF tries to allocate aggressively on CPU; these caps prevent OOM in
# constrained environments (Railway, Fly.io, etc.).
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=2
ENV OMP_NUM_THREADS=2

EXPOSE 8000

# Allow up to 90s for model loading before health checks start.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Use 2 workers by default; scale via UVICORN_WORKERS env var.
# Set UVICORN_LOG_LEVEL=info (docker-compose) to see HTTP access logs in dev.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS:-1} --log-level ${UVICORN_LOG_LEVEL:-warning}"]
