# festora-vision-api

An independent, reusable image analysis microservice built with FastAPI.

Provides blur detection, technical quality scoring, aesthetic scoring, emotion
analysis, CLIP embeddings, and DBSCAN clustering — all over a clean REST API.

Designed to be used by any project (e-commerce, social media, stock photo,
etc.), not just Festora. Has no knowledge of any external database schema.

---

## Quick Start

```bash
# Clone
git clone https://github.com/Jjat00/festora-vision-api.git
cd festora-vision-api

# Copy config (all defaults work out of the box)
cp .env.example .env

# Build and start (downloads ~4 GB of model weights on first build)
docker compose up --build
```

The service is ready when you see:

```
INFO | Service ready.
```

Open the interactive API docs: http://localhost:8000/docs

---

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Service + model status |
| GET | `/v1/health` | No | Versioned alias |
| POST | `/v1/analyze` | Yes* | Analyse a single image (blur, quality, emotion, embedding) |
| POST | `/v1/analyze/batch` | Yes* | Analyse up to 20 images in parallel |
| POST | `/v1/analyze/llm` | Yes* | Analyse a single image with a vision LLM (LiteLLM) |
| POST | `/v1/cluster` | Yes* | DBSCAN on CLIP embeddings |

*Auth is only enforced when `API_KEY` env var is set.

---

## Example: Analyse a single image

```bash
curl -X POST http://localhost:8000/v1/analyze \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-secret-key" \
  -d '{
    "image": {
      "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png",
      "image_id": "demo_001"
    },
    "run_blur": true,
    "run_quality": true,
    "run_emotion": false,
    "run_embedding": true
  }'
```

Response:

```json
{
  "api_version": "1.0",
  "result": {
    "image_id": "demo_001",
    "source_url": "https://...",
    "width_px": 240,
    "height_px": 240,
    "blur": {
      "laplacian_variance": 312.5,
      "is_blurry": false,
      "blur_threshold": 100.0
    },
    "quality": {
      "brisque_score": 14.3,
      "nima_technical_score": 6.8,
      "overall_quality": "high"
    },
    "aesthetic": {
      "nima_aesthetic_score": 7.2,
      "aesthetic_label": "high"
    },
    "emotion": null,
    "embedding": {
      "model": "openai/clip-vit-base-patch32",
      "dimensions": 512,
      "vector": [0.021, -0.014, "...512 floats..."]
    },
    "processing_time_ms": 843.2,
    "error": null
  }
}
```

---

## Example: Batch analyse

```bash
curl -X POST http://localhost:8000/v1/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      { "url": "https://example.com/photo1.jpg", "image_id": "p1" },
      { "url": "https://example.com/photo2.jpg", "image_id": "p2" }
    ],
    "run_blur": true,
    "run_quality": true,
    "run_emotion": false,
    "run_embedding": true
  }'
```

---

## Example: Cluster embeddings

```bash
curl -X POST http://localhost:8000/v1/cluster \
  -H "Content-Type: application/json" \
  -d '{
    "embeddings": [[...512 floats...], [...512 floats...]],
    "image_ids": ["p1", "p2"],
    "eps": 0.35,
    "min_samples": 2
  }'
```

---

## Authentication

Set `API_KEY=your-secret` in your `.env` file to enable authentication.
All protected endpoints require the `X-Api-Key: your-secret` header.

Leave `API_KEY` blank to run in open mode (useful for local development).

---

## Configuration

All configuration is via environment variables. See `.env.example` for the
full reference with descriptions and defaults.

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `""` | API key. Blank = auth disabled. |
| `ENVIRONMENT` | `development` | `development`, `staging`, `production` |
| `RATE_LIMIT_ENABLED` | `true` | Enable per-IP rate limiting |
| `RATE_LIMIT_PER_MINUTE` | `60` | Max requests per IP per minute |
| `MAX_BATCH_SIZE` | `20` | Max images per batch request |
| `CLIP_MODEL_NAME` | `openai/clip-vit-base-patch32` | HuggingFace model ID |
| `DEEPFACE_DETECTOR` | `opencv` | Face detector backend |

---

## Running locally without Docker

```bash
python -m venv .venv
source .venv/bin/activate

# Install CPU-only torch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
pip install -r requirements-dev.txt

# Download models (one-time, ~4 GB)
python scripts/download_models.py

# Run
uvicorn app.main:app --reload --port 8000
```

---

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
pytest --cov=app --cov-report=html   # with coverage
```

---

## Architecture

```
app/
  main.py                   FastAPI app + lifespan (model loading)
  core/
    config.py               Pydantic Settings (all env vars)
    security.py             API key auth dependency
    errors.py               Structured error responses + handlers
    logging.py              JSON / human-readable log setup
  models/
    registry.py             Model loading + singleton registry
  schemas/
    common.py               ImageInput, AnalysisFlags
    analysis.py             Request/response schemas for /analyze
    cluster.py              Request/response schemas for /cluster
    health.py               Health response schema
  services/
    blur_service.py         OpenCV Laplacian variance
    quality_service.py      PyIQA BRISQUE + NIMA
    emotion_service.py      DeepFace (thread-pooled)
    embedding_service.py    CLIP 512D vectors
    cluster_service.py      DBSCAN
    analysis_orchestrator.py  Coordinates all services per image
  api/v1/
    router.py               Aggregate v1 router
    endpoints/
      analyze.py            POST /v1/analyze, /v1/analyze/batch
      cluster.py            POST /v1/cluster
      health.py             GET /health, /v1/health
```

---

## Example: LLM visual analysis

```bash
# Requires ANTHROPIC_API_KEY (or OPENAI_API_KEY / GEMINI_API_KEY) in .env
curl -X POST http://localhost:8000/v1/analyze/llm \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-secret-key" \
  -d '{
    "image_url": "https://example.com/photo.jpg",
    "ref_id": "photo_abc123",
    "model": "anthropic/claude-opus-4-6"
  }'
```

Response:

```json
{
  "ref_id": "photo_abc123",
  "analysis": {
    "overall_score": 8,
    "discard_reason": null,
    "best_in_group": true,
    "composition": "rule of thirds",
    "pose_quality": "natural and relaxed",
    "background_quality": "soft bokeh, non-distracting",
    "highlights": ["genuine smile", "sharp focus on subject", "good lighting"],
    "issues": [],
    "summary": "Natural portrait with excellent composition and a genuine expression."
  },
  "model_used": "claude-opus-4-6-20250514",
  "tokens_used": 387,
  "processing_ms": 2140
}
```

Switch providers by changing the `model` field — no code changes needed:

```json
{ "model": "openai/gpt-4o" }
{ "model": "gemini/gemini-1.5-pro" }
```

---

## Tech stack

| Component | Library | Why |
|-----------|---------|-----|
| Web framework | FastAPI 0.115 | Async, auto OpenAPI |
| Validation | Pydantic v2 | Fast, type-safe |
| Image I/O | Pillow + httpx | Async URL fetch |
| Blur detection | OpenCV | Laplacian — fast, no GPU |
| Quality / aesthetic | PyIQA (BRISQUE, NIMA) | State-of-the-art NR-IQA |
| Emotion | DeepFace | Best OSS face analysis |
| Embeddings | HuggingFace CLIP | 512D semantic vectors |
| Clustering | scikit-learn DBSCAN | Cosine metric, no k needed |
| LLM analysis | LiteLLM | Unified interface for 100+ LLMs |
| Rate limiting | slowapi | Plugs into FastAPI |

---

## License

MIT
