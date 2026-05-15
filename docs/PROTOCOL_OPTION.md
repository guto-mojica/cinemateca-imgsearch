# Protocol Option — Pluggable Model Backends

**Status:** Design option. Not yet implemented.
**Context:** Decided during v0.3.0 stabilisation, before any model swaps begin.

---

## Problem

Every model in `src/cinemateca/` is hardwired to one backend:

- `CLIPEmbedder` → `open_clip` (PyTorch)
- `FaceDetector` → `facenet_pytorch` (PyTorch)
- `ObjectDetector` → `ultralytics` (PyTorch)
- `LLMDescriber` → `transformers` / Moondream 2 (PyTorch)

Swapping a model means editing the class. Migrating to ONNX means rewriting the class.
Each model change is surgery on production code.

With fine-tuned models and an ONNX migration both on the horizon, every change
will touch the same files more than once. That is the damage this design prevents.

---

## Design

Define a **typed interface (Protocol) per model role**. Concrete backends implement
the interface. The pipeline never imports a backend directly — it asks a registry,
which reads `config.yaml` and returns the right implementation.

This is the Strategy pattern applied per model role. It is inspired by ComfyUI's
pluggable node philosophy, adapted to a fixed pipeline (scenes → visual → embeddings
→ llm) rather than a visual graph.

**Adding a new model = one new file + one config line. Zero changes to the pipeline.**

---

## `src/cinemateca/models/base.py` — The Interfaces

```python
"""
cinemateca.models.base
~~~~~~~~~~~~~~~~~~~~~~
Protocol definitions for every model role in the pipeline.

Concrete backends (openclip, onnx, gguf, …) implement these interfaces.
The pipeline imports only from here — never from a concrete backend.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Protocol, runtime_checkable

import numpy as np
import pandas as pd


# ── Embedding ────────────────────────────────────────────────────────────────

@runtime_checkable
class ImageEmbedder(Protocol):
    """Encodes images into L2-normalised float32 vectors."""

    def encode_images(self, image_paths: List[Path]) -> np.ndarray:
        """Returns (N, D) float32 array, L2-normalised."""
        ...

    def encode_text(self, text: str) -> np.ndarray:
        """Returns (D,) float32 vector, L2-normalised.

        Text and image embeddings share the same space (CLIP contract).
        """
        ...

    def encode_image_single(self, image_path: Path) -> np.ndarray:
        """Returns (D,) float32 vector for one image (used in image search)."""
        ...


# ── Face detection ───────────────────────────────────────────────────────────

@runtime_checkable
class FaceDetector(Protocol):
    """Detects human faces in keyframe images."""

    def detect(self, image_path: Path) -> dict:
        """Returns {"num_faces": int, "faces": [{"bbox", "confidence", ...}]}."""
        ...

    def detect_batch(self, image_paths: List[Path]) -> List[dict]:
        """Returns one result dict per path, same order as input."""
        ...


# ── Object detection ─────────────────────────────────────────────────────────

@runtime_checkable
class ObjectDetector(Protocol):
    """Detects objects in keyframe images."""

    def detect(self, image_path: Path) -> dict:
        """Returns {"num_objects": int, "objects": [...], "class_counts": {...}}."""
        ...

    def detect_batch(self, image_paths: List[Path]) -> List[dict]:
        ...


# ── Scene description (VLM) ──────────────────────────────────────────────────

@runtime_checkable
class SceneDescriber(Protocol):
    """Generates natural-language metadata for a keyframe using a VLM."""

    def describe(self, image_path: Path) -> dict:
        """
        Returns a metadata dict with at minimum:
            description   : str
            location      : "interior" | "exterior" | "desconhecido"
            time_of_day   : "dia" | "noite" | "desconhecido"
            num_people    : int  (-1 = vague/many)
            objects       : List[str]
            tags          : List[str]  (kebab-case, auto-generated)
        """
        ...

    def describe_batch(
        self,
        keyframes_df: pd.DataFrame,
        existing_results: list | None = None,
        checkpoint_path: Path | None = None,
    ) -> list[dict]:
        """Processes all rows in keyframes_df, supports resume via existing_results."""
        ...


# ── Environment classification ───────────────────────────────────────────────

@runtime_checkable
class EnvironmentClassifier(Protocol):
    """Heuristic or model-based environment classification."""

    def classify(self, image_path: Path) -> dict:
        """Returns {"time_of_day", "brightness_score", "location", "edge_density"}."""
        ...

    def classify_batch(self, image_paths: List[Path]) -> List[dict]:
        ...
```

---

## `src/cinemateca/models/registry.py` — The Factory

```python
"""
cinemateca.models.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~
Reads the [models] section of config.yaml and returns concrete
implementations. The pipeline imports from here, not from backends.
"""
from __future__ import annotations

from functools import lru_cache
from cinemateca.models.base import (
    ImageEmbedder, FaceDetector, ObjectDetector,
    SceneDescriber, EnvironmentClassifier,
)


def get_image_embedder(cfg) -> ImageEmbedder:
    name = cfg.models.image_embedder  # e.g. "clip_openclip"
    if name == "clip_openclip":
        from cinemateca.models.clip.openclip import OpenClipEmbedder
        return OpenClipEmbedder(cfg)
    if name == "clip_onnx":
        from cinemateca.models.clip.onnx import OnnxClipEmbedder
        return OnnxClipEmbedder(cfg)
    raise ValueError(f"Unknown image_embedder: {name!r}")


def get_face_detector(cfg) -> FaceDetector:
    name = cfg.models.face_detector
    if name == "mtcnn_pytorch":
        from cinemateca.models.face.mtcnn import MTCNNDetector
        return MTCNNDetector(cfg)
    if name == "mtcnn_onnx":
        from cinemateca.models.face.onnx import OnnxMTCNNDetector
        return OnnxMTCNNDetector(cfg)
    if name == "opencv_dnn":
        from cinemateca.models.face.opencv_dnn import OpenCVDNNDetector
        return OpenCVDNNDetector(cfg)
    raise ValueError(f"Unknown face_detector: {name!r}")


def get_object_detector(cfg) -> ObjectDetector:
    name = cfg.models.object_detector
    if name == "yolov8":
        from cinemateca.models.objects.yolov8 import YOLOv8Detector
        return YOLOv8Detector(cfg)
    if name == "yolov8_onnx":
        from cinemateca.models.objects.onnx import OnnxYOLODetector
        return OnnxYOLODetector(cfg)
    raise ValueError(f"Unknown object_detector: {name!r}")


def get_scene_describer(cfg) -> SceneDescriber:
    name = cfg.models.scene_describer
    if name == "moondream_pytorch":
        from cinemateca.models.describer.moondream import MoondreamDescriber
        return MoondreamDescriber(cfg)
    if name == "moondream_gguf":
        from cinemateca.models.describer.gguf import GGUFDescriber
        return GGUFDescriber(cfg)
    raise ValueError(f"Unknown scene_describer: {name!r}")
```

---

## `config/default.yaml` addition

```yaml
models:
  image_embedder: clip_openclip      # clip_openclip | clip_onnx
  face_detector: mtcnn_pytorch       # mtcnn_pytorch | mtcnn_onnx | opencv_dnn
  object_detector: yolov8            # yolov8 | yolov8_onnx
  scene_describer: moondream_pytorch # moondream_pytorch | moondream_gguf
  environment_classifier: opencv_heuristic  # only one impl today
```

Override in `config/local.yaml` to test a new backend without touching code.

---

## Repository layout after restructure

```
src/cinemateca/
  models/
    __init__.py
    base.py              ← Protocols (this document)
    registry.py          ← factory functions
    clip/
      openclip.py        ← current CLIPEmbedder, renamed + interface-compliant
      onnx.py            ← future ONNX Runtime backend
    face/
      mtcnn.py           ← current FaceDetector, renamed
      onnx.py            ← future
      opencv_dnn.py      ← free (OpenCV already a dep)
    objects/
      yolov8.py          ← current ObjectDetector, renamed
      onnx.py            ← future
    describer/
      moondream.py       ← current LLMDescriber, renamed
      gguf.py            ← future (llama-cpp-python)
  embeddings.py          ← SemanticSearch stays here (pure numpy, no model)
  scene_detector.py      ← unchanged (PySceneDetect, no PyTorch)
  annotator.py           ← unchanged
  pipeline.py            ← imports from registry, not from backends
  device.py              ← becomes provider selector for ONNX, or removed
```

**What does NOT move:** `SemanticSearch` (pure numpy dot-product, not a model),
`scene_detector.py` (PySceneDetect/OpenCV), `annotator.py`, `library.py`.

---

## Migration path

This restructure is **not** a rewrite. Each concrete file is the current class,
moved and renamed to implement the Protocol. Logic changes only when a new backend
is added.

```
Step 1  Define Protocols in base.py (this doc).
        No logic changes anywhere. Existing classes implicitly satisfy the Protocol
        via structural subtyping — Python checks duck-typing, no explicit inheritance.

Step 2  Add registry.py pointing to current implementations under models/.
        Update pipeline.py to call registry.get_*() instead of importing directly.
        Tests still pass; behaviour unchanged.

Step 3  (When a model swap is needed) Add one new file under the right subfolder.
        Update config.yaml. Done.

Step 4  (Pre-release) Add ONNX backends as new files. Switch config to clip_onnx,
        yolov8_onnx, etc. Drop PyTorch from dependencies. Build binary.
```

---

## Timing

Do Step 1 + Step 2 **before** any model changes. The restructure is low-risk
(logic-preserving), but it makes every subsequent model change additive instead
of surgical. If model changes come before the Protocol layer, each swap will
touch shared code under pressure.

Do Step 4 (ONNX) only when a release build is the explicit goal — not before,
because ONNX requires one-time model exports and changes the dependency tree.

---

## Relation to ONNX migration

See `ONNX_MIGRATION_OPTION` in project memory. The Protocol layer is the
prerequisite: once registry.py exists, adding an ONNX backend is one new
file per role, not a migration of existing classes. The ONNX migration cost
drops significantly once this structure is in place.
