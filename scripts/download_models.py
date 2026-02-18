"""
Model pre-download script — runs once during `docker build`.

Each library is triggered to auto-download its weights into
/app/models_cache (set via env vars before this script runs).

This script must be runnable standalone (no app imports).
"""

from __future__ import annotations

import os
import sys

MODELS_DIR = os.environ.get("HF_HOME", "/app/models_cache")
CLIP_MODEL = os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

print(f"[download_models] Model cache dir: {MODELS_DIR}", flush=True)


def download_clip() -> None:
    print(f"[download_models] Downloading CLIP: {CLIP_MODEL}", flush=True)
    from transformers import CLIPModel, CLIPProcessor  # type: ignore

    CLIPProcessor.from_pretrained(CLIP_MODEL, cache_dir=MODELS_DIR)
    CLIPModel.from_pretrained(CLIP_MODEL, cache_dir=MODELS_DIR)
    print("[download_models] CLIP done.", flush=True)


def download_pyiqa() -> None:
    print("[download_models] Downloading PyIQA models (brisque, nima, nima-koniq)", flush=True)
    import pyiqa  # type: ignore

    for model_name in ("brisque", "nima", "nima-koniq"):
        print(f"  -> {model_name}", flush=True)
        pyiqa.create_metric(model_name)
    print("[download_models] PyIQA done.", flush=True)


def download_deepface() -> None:
    print("[download_models] Pre-loading DeepFace (opencv detector)", flush=True)
    # DeepFace downloads weights lazily on first call. We trigger it here.
    import numpy as np
    from deepface import DeepFace  # type: ignore

    # Create a tiny black image to trigger detector initialisation.
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    try:
        DeepFace.analyze(
            img_path=dummy,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            silent=True,
        )
    except Exception:
        pass  # Expected — dummy image has no face.
    print("[download_models] DeepFace done.", flush=True)


if __name__ == "__main__":
    errors: list[str] = []

    for fn in (download_clip, download_pyiqa, download_deepface):
        try:
            fn()
        except Exception as exc:
            print(f"[download_models] ERROR in {fn.__name__}: {exc}", file=sys.stderr, flush=True)
            errors.append(str(exc))

    if errors:
        print(f"[download_models] {len(errors)} model(s) failed to download.", file=sys.stderr)
        sys.exit(1)

    print("[download_models] All models downloaded successfully.", flush=True)
