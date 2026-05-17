# Pluggable Model Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a typed Protocol + registry per model role, hard-cutover all consumers off the old module paths, and replace the scene describer with a single keyless/offline Moondream-2 GGUF backend (no `transformers`).

**Architecture:** Implements `docs/superpowers/specs/2026-05-17-pluggable-model-backends-design.md` (read it first). New `src/cinemateca/models/` package: `base.py` (5 `runtime_checkable` Protocols), `registry.py` (config-driven factory), and per-role backend modules holding the *moved* current classes. `pipeline.py` and `api/services/search.py` consume the registry / new module paths only — no back-compat shims. `VisualAnalyzer` stays in `visual_analyzer.py` as an injection facade; `llm_describer.py` is deleted.

**Tech Stack:** Python 3.10+, `llama-cpp-python` (Moondream GGUF, multimodal chat handler), `huggingface_hub` (weight download), pytest, uv.

---

## File Structure

**Create:**
- `src/cinemateca/models/__init__.py` — empty package marker.
- `src/cinemateca/models/base.py` — 5 Protocols.
- `src/cinemateca/models/registry.py` — `get_*` factories reading `cfg.models.*`.
- `src/cinemateca/models/clip/__init__.py`, `face/__init__.py`, `objects/__init__.py`, `environment/__init__.py`, `describer/__init__.py` — package markers.
- `src/cinemateca/models/clip/openclip.py` — `OpenClipEmbedder` (moved `CLIPEmbedder` + `encode_image_single`).
- `src/cinemateca/models/face/mtcnn.py` — `MTCNNFaceDetector` (moved `FaceDetector`).
- `src/cinemateca/models/objects/yolov8.py` — `YOLOv8ObjectDetector` (moved `ObjectDetector`).
- `src/cinemateca/models/environment/opencv_heuristic.py` — `OpenCVEnvironmentClassifier` (moved `EnvironmentClassifier`).
- `src/cinemateca/models/describer/_common.py` — model-agnostic parsing/tagging extracted from `llm_describer.py`.
- `src/cinemateca/models/describer/gguf.py` — `MoondreamGGUFDescriber` (new, the only describer).
- `tests/test_models_protocols.py` — Protocol conformance + registry tests.
- `tests/test_describer_gguf.py` — describer unit tests (llama mocked) incl. resume-bug regression.

**Modify:**
- `src/cinemateca/embeddings.py` — keep only `SemanticSearch`; `by_image` uses `embedder.encode_image_single`.
- `src/cinemateca/visual_analyzer.py` — keep only `VisualAnalyzer`, injection constructor.
- `src/cinemateca/pipeline.py` — registry calls at the 3 model sites + `describe_batch`.
- `src/cinemateca/__main__.py` — `--steps` alias mapping.
- `api/services/search.py` — `OpenClipEmbedder` import/usage.
- `config/default.yaml` — `models:` block.
- `tests/test_smoke.py` — new import paths.
- `tests/test_search_service.py` — `clipfree` monkeypatch target.
- `pyproject.toml` — deps.
- `docs/PROTOCOL_OPTION.md`, `CLAUDE.md`, `CHANGELOG.md` — status.

**Delete:**
- `src/cinemateca/llm_describer.py`.

---

## Task 1: Dependencies

**Files:**
- Modify: `pyproject.toml:34` (the `full` extra)

- [ ] **Step 1: Verify Moondream was the only `transformers` consumer**

Run:
```bash
cd /mnt/a/1_projects/samaritan/cinemateca-imgsearch
grep -rn -E "import transformers|from transformers" --include="*.py" src api | grep -v __pycache__
```
Expected: matches **only** in `src/cinemateca/llm_describer.py`. If any other
file appears, STOP and report — do not remove `transformers` from deps; keep it
and note the extra consumer. (Confirmed Moondream-only by inspection 2026-05-17;
this step re-verifies before deletion.)

- [ ] **Step 2: Edit `pyproject.toml` `full` extra**

Replace the line `    "transformers>=4.35",` with `    "llama-cpp-python>=0.3,<0.4",`.
The `full` extra block becomes (lines 27–45):

```toml
full = [
    "torch>=2.0",
    "torchvision>=0.15",
    "open-clip-torch>=2.20",
    "scenedetect>=0.6",
    "ultralytics>=8.0",
    "facenet-pytorch>=2.5",
    "llama-cpp-python>=0.3,<0.4",
    "rich>=13.0",
    # v0.3.0 web stack
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "jinja2>=3.1",
    "python-multipart>=0.0.7",
    "aiofiles>=23.0",
    "babel>=2.14",
    # legacy Streamlit (removed after parity confirmed)
    "streamlit>=1.28",
]
```

(There is no `transformers`/`pyvips`/`einops` line to remove elsewhere — the
session edits that added `pyvips`/`einops` were already reverted; `pyproject`
is at HEAD which only has `transformers>=4.35`. `huggingface_hub` arrives
transitively via `llama-cpp-python`/`open-clip-torch`; if a later task's
`from huggingface_hub import ...` fails, add `"huggingface_hub>=0.20",` to the
`full` extra and re-sync.)

- [ ] **Step 3: Sync the environment**

Run: `uv sync --extra full --group dev`
Expected: completes; `llama-cpp-python` installed. Then:
`uv run python -c "from llama_cpp import Llama; print('llama-cpp OK')"`
Expected: `llama-cpp OK`.

- [ ] **Step 4: Run the full suite to capture the green baseline**

Run: `uv run pytest tests/ -q`
Expected: all pass (record the count, e.g. `208 passed`). This is the
regression baseline every later task must hold.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build(deps): swap transformers->llama-cpp-python for GGUF describer

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Protocol definitions (`models/base.py`)

**Files:**
- Create: `src/cinemateca/models/__init__.py`
- Create: `src/cinemateca/models/base.py`
- Create: `tests/test_models_protocols.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_models_protocols.py`:

```python
"""Protocol + registry conformance (no model load, no GPU, hermetic)."""
from __future__ import annotations


def test_base_protocols_exist_and_are_runtime_checkable():
    from typing import get_type_hints  # noqa: F401

    from cinemateca.models.base import (
        EnvironmentClassifier,
        FaceDetector,
        ImageEmbedder,
        ObjectDetector,
        SceneDescriber,
    )

    for proto in (
        ImageEmbedder, FaceDetector, ObjectDetector,
        SceneDescriber, EnvironmentClassifier,
    ):
        # @runtime_checkable Protocols expose _is_runtime_protocol = True
        assert getattr(proto, "_is_runtime_protocol", False), proto
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models_protocols.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'cinemateca.models'`.

- [ ] **Step 3: Create the package marker**

Create `src/cinemateca/models/__init__.py` with a single line:

```python
"""Pluggable model backends (Protocols + registry). See docs/PROTOCOL_OPTION.md."""
```

- [ ] **Step 4: Create `src/cinemateca/models/base.py`**

```python
"""
cinemateca.models.base
~~~~~~~~~~~~~~~~~~~~~~
Protocol definitions for every model role in the pipeline.

Concrete backends (openclip, mtcnn, yolov8, gguf, …) implement these
interfaces. The pipeline imports only from here / the registry — never
from a concrete backend.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class ImageEmbedder(Protocol):
    """Encodes images into L2-normalised float32 vectors."""

    def encode_images(self, image_paths: List[Path]) -> np.ndarray:
        """Returns (N, D) float32 array, L2-normalised."""
        ...

    def encode_text(self, text: str) -> np.ndarray:
        """Returns (D,) float32 vector, L2-normalised (shared CLIP space)."""
        ...

    def encode_image_single(self, image_path: Path) -> np.ndarray:
        """Returns (D,) float32 vector for one image (image search)."""
        ...


@runtime_checkable
class FaceDetector(Protocol):
    """Detects human faces in keyframe images."""

    def detect(self, image_path: Path) -> dict:
        """Returns {"num_faces": int, "faces": [...]}."""
        ...

    def detect_batch(self, image_paths: List[Path]) -> List[dict]:
        """One result dict per path, same order as input."""
        ...


@runtime_checkable
class ObjectDetector(Protocol):
    """Detects objects in keyframe images."""

    def detect(self, image_path: Path) -> dict:
        """Returns {"num_objects": int, "objects": [...], "class_counts": {...}}."""
        ...

    def detect_batch(self, image_paths: List[Path]) -> List[dict]:
        ...


@runtime_checkable
class SceneDescriber(Protocol):
    """Generates natural-language metadata for a keyframe using a VLM."""

    def describe(self, image_path: Path) -> dict:
        """description/location/setting/time_of_day/num_people/objects/tags."""
        ...

    def describe_batch(
        self,
        keyframes_df: pd.DataFrame,
        existing_results: list | None = None,
        checkpoint_path: Path | None = None,
    ) -> list[dict]:
        """Process all rows; resume via existing_results."""
        ...


@runtime_checkable
class EnvironmentClassifier(Protocol):
    """Heuristic or model-based environment classification."""

    def classify(self, image_path: Path) -> dict:
        """Returns {"time_of_day","brightness_score","location","edge_density"}."""
        ...

    def classify_batch(self, image_paths: List[Path]) -> List[dict]:
        ...
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_models_protocols.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cinemateca/models/__init__.py src/cinemateca/models/base.py tests/test_models_protocols.py
git commit -m "feat(models): add base.py with 5 runtime_checkable Protocols

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Move Face / Object / Environment detectors

**Files:**
- Create: `src/cinemateca/models/face/__init__.py`, `src/cinemateca/models/face/mtcnn.py`
- Create: `src/cinemateca/models/objects/__init__.py`, `src/cinemateca/models/objects/yolov8.py`
- Create: `src/cinemateca/models/environment/__init__.py`, `src/cinemateca/models/environment/opencv_heuristic.py`
- Modify: `src/cinemateca/visual_analyzer.py` (remove 3 classes, refactor `VisualAnalyzer`)
- Test: `tests/test_models_protocols.py`

- [ ] **Step 1: Add the failing conformance test**

Append to `tests/test_models_protocols.py`:

```python
def test_detector_backends_conform():
    from cinemateca.models.base import (
        EnvironmentClassifier,
        FaceDetector,
        ObjectDetector,
    )
    from cinemateca.models.environment.opencv_heuristic import (
        OpenCVEnvironmentClassifier,
    )
    from cinemateca.models.face.mtcnn import MTCNNFaceDetector
    from cinemateca.models.objects.yolov8 import YOLOv8ObjectDetector

    assert isinstance(MTCNNFaceDetector(), FaceDetector)
    assert isinstance(YOLOv8ObjectDetector(), ObjectDetector)
    assert isinstance(OpenCVEnvironmentClassifier(), EnvironmentClassifier)


def test_visual_analyzer_injection():
    """VisualAnalyzer accepts injected backends and exposes them."""
    from cinemateca.models.environment.opencv_heuristic import (
        OpenCVEnvironmentClassifier,
    )
    from cinemateca.models.face.mtcnn import MTCNNFaceDetector
    from cinemateca.models.objects.yolov8 import YOLOv8ObjectDetector
    from cinemateca.visual_analyzer import VisualAnalyzer

    fd, od, ec = (
        MTCNNFaceDetector(),
        YOLOv8ObjectDetector(),
        OpenCVEnvironmentClassifier(),
    )
    va = VisualAnalyzer(face_detector=fd, object_detector=od, env_classifier=ec)
    assert va.face_detector is fd
    assert va.object_detector is od
    assert va.env_classifier is ec
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_models_protocols.py -q`
Expected: FAIL — `ModuleNotFoundError: cinemateca.models.face.mtcnn`.

- [ ] **Step 3: Create the package markers**

Create each of `src/cinemateca/models/face/__init__.py`,
`src/cinemateca/models/objects/__init__.py`,
`src/cinemateca/models/environment/__init__.py` with one line:

```python
"""Backend package."""
```

- [ ] **Step 4: Create `models/face/mtcnn.py`**

Move the current `FaceDetector` class verbatim from `visual_analyzer.py`
(lines 24–105), renamed to `MTCNNFaceDetector`. Full file:

```python
"""MTCNN face-detection backend (moved from visual_analyzer.py, unchanged)."""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


class MTCNNFaceDetector:
    """Detects faces in frames using MTCNN (facenet-pytorch)."""

    def __init__(self, cfg=None, device=None):
        self._model = None
        self._device = device

        if cfg is not None:
            fd_cfg = cfg.visual_analysis.face_detection
            self.enabled = fd_cfg.enabled
            self.min_face_size = fd_cfg.min_face_size
            self.thresholds = list(fd_cfg.thresholds)
        else:
            self.enabled = True
            self.min_face_size = 20
            self.thresholds = [0.6, 0.7, 0.7]

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from facenet_pytorch import MTCNN
        except ImportError:
            raise RuntimeError(
                "facenet-pytorch não instalado. Execute: pip install facenet-pytorch"
            )

        # MTCNN has MPS incompatibilities with adaptive pooling — always use CPU
        self._model = MTCNN(
            keep_all=True,
            device="cpu",
            thresholds=self.thresholds,
            min_face_size=self.min_face_size,
        )
        logger.info("MTCNN carregado no device: cpu")

    def detect(self, image_path: str | Path) -> dict:
        if not self.enabled:
            return {"num_faces": 0, "faces": []}

        self._load_model()

        img = Image.open(image_path)
        boxes, probs, landmarks = self._model.detect(img, landmarks=True)

        if boxes is None:
            return {"num_faces": 0, "faces": []}

        faces = []
        for box, prob, lm in zip(boxes, probs, landmarks):
            faces.append({
                "bbox": box.tolist(),
                "confidence": float(prob),
                "landmarks": lm.tolist() if lm is not None else None,
                "area": float((box[2] - box[0]) * (box[3] - box[1])),
            })

        return {"num_faces": len(faces), "faces": faces}

    def detect_batch(self, image_paths: list[Path]) -> list[dict]:
        results = []
        for p in image_paths:
            r = self.detect(p)
            r["frame_path"] = str(p.name)
            results.append(r)
        logger.info("Detecção facial: %d frames processados", len(results))
        return results
```

- [ ] **Step 5: Create `models/objects/yolov8.py`**

Move current `ObjectDetector` (visual_analyzer.py lines 110–188) verbatim,
renamed `YOLOv8ObjectDetector`. Full file:

```python
"""YOLOv8 object-detection backend (moved from visual_analyzer.py, unchanged)."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class YOLOv8ObjectDetector:
    """Detects objects using YOLOv8 (Ultralytics)."""

    def __init__(self, cfg=None, device=None):
        self._model = None
        self._device = device

        if cfg is not None:
            od_cfg = cfg.visual_analysis.object_detection
            self.enabled = od_cfg.enabled
            self.model_name = od_cfg.model
            self.confidence = od_cfg.confidence
        else:
            self.enabled = True
            self.model_name = "yolov8n.pt"
            self.confidence = 0.30

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "ultralytics não instalado. Execute: pip install ultralytics"
            )
        self._model = YOLO(self.model_name)
        logger.info("YOLOv8 carregado: %s", self.model_name)

    def detect(self, image_path: str | Path) -> dict:
        if not self.enabled:
            return {"num_objects": 0, "objects": [], "class_counts": {}}

        self._load_model()
        results = self._model(str(image_path), conf=self.confidence, verbose=False)

        objects = []
        class_counts: dict[str, int] = {}

        for result in results:
            for box in result.boxes:
                cls_name = self._model.names[int(box.cls[0])]
                obj = {
                    "class": cls_name,
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                }
                objects.append(obj)
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        return {
            "num_objects": len(objects),
            "objects": objects,
            "class_counts": class_counts,
        }

    def detect_batch(self, image_paths: list[Path]) -> list[dict]:
        results = []
        for p in image_paths:
            r = self.detect(p)
            r["frame_path"] = str(p.name)
            results.append(r)
        logger.info("Detecção de objetos: %d frames processados", len(results))
        return results
```

- [ ] **Step 6: Create `models/environment/opencv_heuristic.py`**

Move current `EnvironmentClassifier` (visual_analyzer.py lines 193–265)
verbatim, renamed `OpenCVEnvironmentClassifier`. Full file:

```python
"""OpenCV-heuristic environment classifier (moved, unchanged)."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class OpenCVEnvironmentClassifier:
    """Classifies scene environment via brightness + edge-density heuristics."""

    def __init__(self, cfg=None):
        if cfg is not None:
            env_cfg = cfg.visual_analysis.environment
            self.enabled = env_cfg.enabled
            self.brightness_threshold = env_cfg.brightness_threshold
            self.edge_density_threshold = env_cfg.edge_density_threshold
        else:
            self.enabled = True
            self.brightness_threshold = 100
            self.edge_density_threshold = 0.05

    def classify(self, image_path: str | Path) -> dict:
        if not self.enabled:
            return {
                "time_of_day": "desconhecido",
                "brightness_score": 0.0,
                "location": "desconhecido",
                "edge_density": 0.0,
            }

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning("Não foi possível ler frame: %s", image_path)
            return {
                "time_of_day": "desconhecido",
                "brightness_score": 0.0,
                "location": "desconhecido",
                "edge_density": 0.0,
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges.sum() / (edges.shape[0] * edges.shape[1]))

        return {
            "time_of_day": "dia" if brightness > self.brightness_threshold else "noite",
            "brightness_score": brightness,
            "location": "exterior" if edge_density > self.edge_density_threshold else "interior",
            "edge_density": edge_density,
        }

    def classify_batch(self, image_paths: list[Path]) -> list[dict]:
        results = []
        for p in image_paths:
            r = self.classify(p)
            r["frame_path"] = str(p.name)
            results.append(r)
        logger.info("Classificação de ambiente: %d frames processados", len(results))
        return results
```

- [ ] **Step 7: Rewrite `src/cinemateca/visual_analyzer.py` to the injection facade only**

Replace the entire file with:

```python
"""
cinemateca.visual_analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~
VisualAnalyzer facade. Composes injected Face / Object / Environment
backends (provided by cinemateca.models.registry). The detector classes
themselves live under cinemateca.models.{face,objects,environment}.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VisualAnalyzer:
    """Orchestrates injected face / object / environment backends.

    Backends are supplied by the caller (the pipeline builds them via
    cinemateca.models.registry). Saves one consolidated visual-metadata
    JSON, format-compatible with the embeddings and LLM modules.
    """

    def __init__(self, face_detector, object_detector, env_classifier):
        self.face_detector = face_detector
        self.object_detector = object_detector
        self.env_classifier = env_classifier

    def analyze_frame(self, image_path: str | Path) -> dict:
        path = Path(image_path)
        return {
            "frame_path": path.name,
            "face_detection": self.face_detector.detect(path),
            "object_detection": self.object_detector.detect(path),
            "environment": self.env_classifier.classify(path),
        }

    def analyze_keyframes(
        self,
        keyframe_paths: list[Path],
        max_frames: int | None = None,
    ) -> list[dict]:
        paths = keyframe_paths[:max_frames] if max_frames else keyframe_paths
        results = []
        for i, p in enumerate(paths):
            try:
                r = self.analyze_frame(p)
                results.append(r)
                if (i + 1) % 25 == 0:
                    logger.info("Análise visual: %d/%d frames", i + 1, len(paths))
            except Exception as e:
                logger.error("Erro ao analisar %s: %s", Path(p).name, e)
                results.append({"frame_path": Path(p).name, "error": str(e)})

        logger.info("✓ Análise visual concluída: %d frames", len(results))
        return results

    def save_metadata(self, results: list[dict], output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("✓ Metadados visuais salvos: %s (%d frames)", out, len(results))
        return out
```

> Note: `pipeline.py` still constructs `VisualAnalyzer(cfg, device)` (old
> signature) until Task 6. To keep the suite green between Task 3 and Task 6,
> Step 8 below patches the single pipeline call site to build the injected
> backends inline; Task 6 replaces that with the registry.

- [ ] **Step 8: Patch the pipeline VisualAnalyzer construction (interim)**

In `src/cinemateca/pipeline.py`, the visual-analysis step currently has
(around lines 243 + 258):

```python
        from cinemateca.visual_analyzer import VisualAnalyzer
        ...
            analyzer = VisualAnalyzer(self.cfg, self.device)
```

Change the import line to:

```python
        from cinemateca.models.environment.opencv_heuristic import OpenCVEnvironmentClassifier
        from cinemateca.models.face.mtcnn import MTCNNFaceDetector
        from cinemateca.models.objects.yolov8 import YOLOv8ObjectDetector
        from cinemateca.visual_analyzer import VisualAnalyzer
```

and the construction line to:

```python
            analyzer = VisualAnalyzer(
                face_detector=MTCNNFaceDetector(self.cfg, self.device),
                object_detector=YOLOv8ObjectDetector(self.cfg, self.device),
                env_classifier=OpenCVEnvironmentClassifier(self.cfg),
            )
```

- [ ] **Step 9: Run conformance + full suite**

Run: `uv run pytest tests/test_models_protocols.py -q && uv run pytest tests/ -q`
Expected: new tests PASS; full suite still at the Task 1 baseline count.

- [ ] **Step 10: Commit**

```bash
git add src/cinemateca/models/face src/cinemateca/models/objects src/cinemateca/models/environment src/cinemateca/visual_analyzer.py src/cinemateca/pipeline.py tests/test_models_protocols.py
git commit -m "refactor(models): move face/object/environment detectors; VisualAnalyzer becomes injection facade

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Move CLIP + `encode_image_single` + `by_image` decoupling

**Files:**
- Create: `src/cinemateca/models/clip/__init__.py`, `src/cinemateca/models/clip/openclip.py`
- Modify: `src/cinemateca/embeddings.py` (keep only `SemanticSearch`; refactor `by_image`)
- Modify: `api/services/search.py` (lines 151, 154, 189)
- Modify: `src/cinemateca/pipeline.py` (embeddings step import/construct)
- Modify: `tests/test_smoke.py` (line 102), `tests/test_search_service.py` (`clipfree`)
- Test: `tests/test_models_protocols.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_models_protocols.py`:

```python
def test_openclip_embedder_conforms():
    from cinemateca.models.base import ImageEmbedder
    from cinemateca.models.clip.openclip import OpenClipEmbedder

    assert isinstance(OpenClipEmbedder(), ImageEmbedder)


def test_by_image_uses_encode_image_single(monkeypatch, tmp_path):
    """by_image must call embedder.encode_image_single, not embedder privates."""
    import numpy as np
    import pandas as pd

    from cinemateca.embeddings import SemanticSearch

    calls = {}

    class FakeEmbedder:
        def encode_image_single(self, image_path):
            calls["path"] = str(image_path)
            return np.array([1.0, 0.0], dtype="float32")

    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    df = pd.DataFrame({"filepath": ["a.jpg", "b.jpg"], "scene_id": [1, 2]})
    s = SemanticSearch(emb, df, FakeEmbedder())
    out = s.by_image("query.jpg", top_k=2, exclude_self=False)
    assert calls["path"] == "query.jpg"
    assert list(out["scene_id"]) == [1, 2]  # ranked by dot product
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_models_protocols.py -k "openclip or by_image" -q`
Expected: FAIL — `ModuleNotFoundError: cinemateca.models.clip.openclip`.

- [ ] **Step 3: Create `models/clip/__init__.py`**

One line: `"""CLIP embedder backends."""`

- [ ] **Step 4: Create `models/clip/openclip.py`**

Move `CLIPEmbedder` from `embeddings.py` (lines 25–231) verbatim, renamed
`OpenClipEmbedder`, **adding** `encode_image_single`. Full file:

```python
"""open_clip image/text embedder backend (moved from embeddings.py)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


class OpenClipEmbedder:
    """CLIP (ViT-B/32 via open_clip) image/text embedder, L2-normalised."""

    def __init__(self, cfg=None, device=None):
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = device

        if cfg is not None:
            emb = cfg.embeddings
            self.model_name = emb.model
            self.pretrained = emb.pretrained
            self.batch_size = emb.batch_size
        else:
            self.model_name = "ViT-B-32"
            self.pretrained = "openai"
            self.batch_size = 16

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import open_clip
        except ImportError:
            raise RuntimeError(
                "open_clip não instalado. Execute: pip install open-clip-torch"
            )

        logger.info("Carregando CLIP %s (%s)...", self.model_name, self.pretrained)
        t0 = time.time()

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._model = self._model.to(self._device)
        self._model.eval()

        logger.info("✓ CLIP carregado em %.1fs | device=%s", time.time() - t0, self._device)

    def encode_images(self, image_paths: list[Path]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self._load_model()

        all_embeddings = []
        error_count = 0
        t0 = time.time()

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i: i + self.batch_size]
            tensors = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(self._preprocess(img))
                except Exception as e:
                    logger.warning("Erro ao carregar %s: %s", path, e)
                    tensors.append(torch.zeros(3, 224, 224))
                    error_count += 1

            batch = torch.stack(tensors).to(self._device)
            with torch.no_grad():
                feats = self._model.encode_image(batch)
                feats = F.normalize(feats, dim=-1)
            all_embeddings.append(feats.cpu().numpy())

            if (i // self.batch_size + 1) % 10 == 0:
                logger.info(
                    "Embeddings: %d/%d imagens processadas",
                    min(i + self.batch_size, len(image_paths)),
                    len(image_paths),
                )

        embeddings = np.vstack(all_embeddings).astype("float32")
        logger.info(
            "✓ %d embeddings gerados em %.1fs (erros: %d) | shape=%s",
            len(image_paths), time.time() - t0, error_count, embeddings.shape,
        )
        return embeddings

    def encode_text(self, text: str) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self._load_model()

        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            feat = self._model.encode_text(tokens)
            feat = F.normalize(feat, dim=-1)
        return feat.cpu().numpy().astype("float32")[0]

    def encode_image_single(self, image_path: Path) -> np.ndarray:
        """One image -> (D,) float32 L2-normalised vector (image search).

        Extracted from the old SemanticSearch.by_image inline logic;
        math is identical (same preprocess + encode_image + L2 normalise).
        """
        import torch
        import torch.nn.functional as F

        self._load_model()

        img = Image.open(image_path).convert("RGB")
        tensor = self._preprocess(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._model.encode_image(tensor)
            feat = F.normalize(feat, dim=-1)
        return feat.cpu().numpy().astype("float32")[0]

    def save(
        self,
        embeddings: np.ndarray,
        keyframes_df: pd.DataFrame,
        output_dir: str | Path,
        embeddings_filename: str = "keyframe_embeddings.npy",
        mapping_filename: str = "index_mapping.json",
    ) -> tuple[Path, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        emb_path = out / embeddings_filename
        np.save(emb_path, embeddings)
        logger.info("✓ Embeddings salvos: %s | %.1f MB", emb_path, emb_path.stat().st_size / 1e6)

        mapping = {
            "model": f"CLIP {self.model_name} ({self.pretrained})",
            "dimension": embeddings.shape[1],
            "total_vectors": len(embeddings),
            "normalized": True,
            "keyframe_paths": keyframes_df["filepath"].tolist(),
            "scene_ids": keyframes_df["scene_id"].tolist(),
        }
        if "keyframe_id" in keyframes_df.columns:
            mapping["keyframe_ids"] = keyframes_df["keyframe_id"].tolist()

        map_path = out / mapping_filename
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        logger.info("✓ Mapeamento salvo: %s", map_path)

        return emb_path, map_path

    @staticmethod
    def load(
        embeddings_path: str | Path,
        mapping_path: str | Path,
    ) -> tuple[np.ndarray, dict, pd.DataFrame]:
        emb = np.load(embeddings_path)
        with open(mapping_path, encoding="utf-8") as f:
            mapping = json.load(f)

        kf_df = pd.DataFrame({
            "filepath": mapping["keyframe_paths"],
            "scene_id": mapping["scene_ids"],
        })
        if "keyframe_ids" in mapping:
            kf_df["keyframe_id"] = mapping["keyframe_ids"]

        logger.info(
            "✓ Embeddings carregados: shape=%s | %d keyframes mapeados",
            emb.shape, len(kf_df),
        )
        return emb, mapping, kf_df
```

- [ ] **Step 5: Rewrite `src/cinemateca/embeddings.py` to keep only `SemanticSearch`**

Replace the whole file with the block below. `SemanticSearch` is unchanged
**except** `by_image` now calls `self.embedder.encode_image_single(...)`. The
`combined()` method (including its scene-id normalisation comment block) is
copied verbatim from the current file (lines 337–412) — do not alter it.

```python
"""
cinemateca.embeddings
~~~~~~~~~~~~~~~~~~~~~
Busca semântica sobre embeddings CLIP. Pure-numpy dot product
(equivalente a cosseno para vetores normalizados). The CLIP embedder
itself lives in cinemateca.models.clip.openclip.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SemanticSearch:
    """Semantic search over CLIP embeddings (text / image / combined)."""

    def __init__(self, embeddings: np.ndarray, keyframes_df: pd.DataFrame, embedder):
        self.embeddings = embeddings
        self.keyframes_df = keyframes_df
        self.embedder = embedder

    def by_text(self, query: str, top_k: int = 8) -> pd.DataFrame:
        query_emb = self.embedder.encode_text(query)
        similarities = (self.embeddings @ query_emb).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        rows = []
        for rank, idx in enumerate(top_indices):
            row = self.keyframes_df.iloc[idx]
            rows.append({
                "rank": rank + 1,
                "scene_id": row["scene_id"],
                "filepath": row["filepath"],
                "similarity": float(similarities[idx]),
            })
        return pd.DataFrame(rows)

    def by_image(
        self,
        image_path,
        top_k: int = 8,
        exclude_self: bool = True,
    ) -> pd.DataFrame:
        img_emb = self.embedder.encode_image_single(image_path)

        similarities = (self.embeddings @ img_emb).flatten()
        top_indices = np.argsort(similarities)[::-1]

        if exclude_self:
            query_str = str(image_path)
            top_indices = [
                i for i in top_indices
                if str(self.keyframes_df.iloc[i]["filepath"]) != query_str
            ]

        top_indices = top_indices[:top_k]

        rows = []
        for rank, idx in enumerate(top_indices):
            row = self.keyframes_df.iloc[idx]
            rows.append({
                "rank": rank + 1,
                "scene_id": row["scene_id"],
                "filepath": row["filepath"],
                "similarity": float(similarities[idx]),
            })
        return pd.DataFrame(rows)

    def combined(
        self,
        query: str,
        filter_tags: list[str] | None = None,
        tag_index: dict | None = None,
        top_k: int = 8,
    ) -> pd.DataFrame:
        # >>> COPY lines 337-412 of the pre-change embeddings.py VERBATIM
        # >>> (the entire body of combined(), including the SOLE/REQUIRED
        # >>> normalization comment block and the scene_id_key .map() logic).
        # >>> Do not paraphrase — git show HEAD:src/cinemateca/embeddings.py
        # >>> and paste the method body exactly.
        ...
```

> Implementation note for the engineer: run
> `git show HEAD:src/cinemateca/embeddings.py` and copy the `combined`
> method body (def line 337 through line 412) exactly in place of the
> `# >>> COPY …` / `...` placeholder above. It is reproduced out-of-band
> only because it contains a long load-bearing comment that must not drift.

- [ ] **Step 6: Update `api/services/search.py`**

Three edits (lazy imports preserved so the test monkeypatch lands):

`search.py:151` — change
`    from cinemateca.embeddings import CLIPEmbedder`
to
`    from cinemateca.models.clip.openclip import OpenClipEmbedder`

`search.py:154` — change `CLIPEmbedder.load(emb_path, map_path)` to
`OpenClipEmbedder.load(emb_path, map_path)`

`search.py:189` — change `embedder = CLIPEmbedder()` to
`embedder = OpenClipEmbedder()`

(`SemanticSearch` imports at 304/314 are unchanged — it stays in
`embeddings.py`.)

- [ ] **Step 7: Update the pipeline embeddings step**

In `src/cinemateca/pipeline.py` the embeddings step has (around 270 + 292):

```python
        from cinemateca.embeddings import CLIPEmbedder
        ...
            embedder = CLIPEmbedder(self.cfg, self.device)
```

Change to (interim — Task 6 swaps to registry):

```python
        from cinemateca.models.clip.openclip import OpenClipEmbedder
        ...
            embedder = OpenClipEmbedder(self.cfg, self.device)
```

- [ ] **Step 8: Update `tests/test_smoke.py:102`**

Replace:

```python
def test_import_embeddings():
    from cinemateca.embeddings import CLIPEmbedder
    assert CLIPEmbedder is not None
```

with:

```python
def test_import_embeddings():
    from cinemateca.embeddings import SemanticSearch
    from cinemateca.models.clip.openclip import OpenClipEmbedder
    assert SemanticSearch is not None
    assert OpenClipEmbedder is not None
```

- [ ] **Step 9: Update the `clipfree` fixture in `tests/test_search_service.py`**

Replace the fixture body (lines ~55–73) with:

```python
@pytest.fixture()
def clipfree(monkeypatch):
    """Patch OpenClipEmbedder so construction is CLIP-free; .load stays real."""
    import cinemateca.models.clip.openclip as oc

    real_load = oc.OpenClipEmbedder.load

    class _PatchedEmbedder:
        def __init__(self, *a, **k):
            pass

        load = staticmethod(real_load)

        def encode_text(self, query):
            return np.ones(4, dtype="float32")

    monkeypatch.setattr(oc, "OpenClipEmbedder", _PatchedEmbedder)
    return _PatchedEmbedder
```

(`search.py`'s lazy `from cinemateca.models.clip.openclip import
OpenClipEmbedder` inside `_load_and_validate` runs at call time, after the
fixture patches the module attribute, so the patched class is the one used.)

- [ ] **Step 10: Run targeted + full suite**

Run:
```bash
uv run pytest tests/test_models_protocols.py tests/test_search_service.py tests/test_smoke.py tests/test_scene_id_filtering.py -q && uv run pytest tests/ -q
```
Expected: all PASS; full suite at baseline count.

- [ ] **Step 11: Commit**

```bash
git add src/cinemateca/models/clip src/cinemateca/embeddings.py api/services/search.py src/cinemateca/pipeline.py tests/test_models_protocols.py tests/test_smoke.py tests/test_search_service.py
git commit -m "refactor(models): move CLIP to OpenClipEmbedder; add encode_image_single; decouple by_image

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Describer — `_common.py` extract + GGUF backend + delete `llm_describer.py`

**Files:**
- Create: `src/cinemateca/models/describer/__init__.py`, `_common.py`, `gguf.py`
- Delete: `src/cinemateca/llm_describer.py`
- Modify: `tests/test_smoke.py` (lines 107, 119, 128, 136)
- Test: `tests/test_describer_gguf.py`

- [ ] **Step 1: Create `models/describer/__init__.py`**

One line: `"""Scene-describer (VLM) backends."""`

- [ ] **Step 2: Create `models/describer/_common.py` (verbatim extract)**

Copy these symbols **verbatim** from the current `src/cinemateca/llm_describer.py`
(use `git show HEAD:src/cinemateca/llm_describer.py`): module constants
`PROMPTS` (lines 46–60), `LOCATION_MAP` (63–68), `TIME_MAP` (70–74); functions
`_parse_num_people` (79–107), `_parse_objects` (110–122), `_generate_tags`
(125–152); and a `build_metadata` function whose body is the current
`LLMDescriber._build_metadata` (249–291) with `self` removed (it never uses
`self`). Header + structure:

```python
"""
cinemateca.models.describer._common
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model-agnostic prompt set + answer parsing + tag/metadata assembly,
extracted unchanged from the former llm_describer.py. Shared by every
SceneDescriber backend so parsing/tagging behaviour is identical
regardless of the VLM engine.
"""
from __future__ import annotations

import re

import pandas as pd

PROMPTS: dict[str, tuple[str, int]] = {
    # >>> paste lines 46-60 of HEAD:llm_describer.py verbatim
}

LOCATION_MAP = {
    # >>> paste lines 63-68 verbatim
}

TIME_MAP = {
    # >>> paste lines 70-74 verbatim
}


def _parse_num_people(text: str) -> int:
    # >>> paste body of lines 79-107 verbatim
    ...


def _parse_objects(text: str) -> list[str]:
    # >>> paste body of lines 110-122 verbatim
    ...


def _generate_tags(parsed: dict) -> list[str]:
    # >>> paste body of lines 125-152 verbatim
    ...


def build_metadata(row: pd.Series, raw: dict) -> dict:
    """Former LLMDescriber._build_metadata (self removed; logic identical)."""
    # >>> paste body of lines 251-291 verbatim (the method body after the
    # >>> docstring), de-indented one level, `self.`-free (it has no self use)
    ...
```

> The `# >>>` markers mean: open `git show HEAD:src/cinemateca/llm_describer.py`,
> copy the exact line ranges, paste in place of the marker + `...`. No
> behavioural edits.

- [ ] **Step 3: Write the failing describer tests**

Create `tests/test_describer_gguf.py`:

```python
"""GGUF describer unit tests — llama-cpp fully mocked, hermetic."""
from __future__ import annotations

import pandas as pd

from cinemateca.models.base import SceneDescriber


class _FakeLlama:
    """Stand-in for the llama_cpp.Llama wrapped by the backend."""

    def __init__(self):
        self.calls = 0

    def answer(self, prompt: str) -> str:
        self.calls += 1
        # deterministic canned answers keyed by prompt content
        if "indoors or outdoors" in prompt:
            return "outdoor"
        if "time of day" in prompt:
            return "day"
        if "How many people" in prompt:
            return "2 people talking"
        if "notable objects" in prompt:
            return "tree, fence"
        if "setting in" in prompt:
            return "rural field"
        return "A man stands in a field."


def _backend_with_fake(monkeypatch):
    from cinemateca.models.describer import gguf

    fake = _FakeLlama()
    monkeypatch.setattr(
        gguf.MoondreamGGUFDescriber, "_answer",
        lambda self, image_path, prompt, max_tokens: fake.answer(prompt),
    )
    monkeypatch.setattr(
        gguf.MoondreamGGUFDescriber, "_load_model", lambda self: None,
    )
    return gguf.MoondreamGGUFDescriber(), fake


def test_gguf_describer_conforms(monkeypatch):
    backend, _ = _backend_with_fake(monkeypatch)
    assert isinstance(backend, SceneDescriber)


def test_describe_single_builds_metadata(monkeypatch):
    backend, _ = _backend_with_fake(monkeypatch)
    meta = backend.describe("frame.jpg")
    assert meta["location"] == "exterior"
    assert meta["time_of_day"] == "dia"
    assert meta["num_people"] == 2
    assert "tree" in meta["objects"]
    assert isinstance(meta["tags"], list) and meta["tags"]


def test_describe_batch_resume_excludes_error_rows(monkeypatch):
    """Regression: error rows must NOT count as processed (the resume bug)."""
    backend, fake = _backend_with_fake(monkeypatch)
    df = pd.DataFrame([
        {"filepath": "a.jpg", "scene_id": 1},
        {"filepath": "b.jpg", "scene_id": 2},
    ])
    existing = [{"scene_id": 1, "error": "boom", "tags": [], "objects": []}]
    out = backend.describe_batch(df, existing_results=existing)
    ids = sorted(r["scene_id"] for r in out)
    assert ids == [1, 2]               # scene 1 reprocessed, not skipped
    assert any("error" not in r and r["scene_id"] == 1 for r in out)
```

- [ ] **Step 4: Run to verify failure**

Run: `uv run pytest tests/test_describer_gguf.py -q`
Expected: FAIL — `ModuleNotFoundError: cinemateca.models.describer.gguf`.

- [ ] **Step 5: Verify the installed llama-cpp multimodal handler API**

Run:
```bash
uv run python -c "import llama_cpp, inspect; from llama_cpp.llama_chat_format import MoondreamChatHandler; print(llama_cpp.__version__); print(inspect.signature(MoondreamChatHandler.__init__)); print('from_pretrained' , hasattr(MoondreamChatHandler,'from_pretrained'))"
```
Expected: prints a version `0.3.x`, an `__init__` signature taking
`clip_model_path`, and `from_pretrained True`. **If the symbol does not exist**
(older/newer build), fall back to `Llava15ChatHandler` with the Moondream
`mmproj` and record the substitution in the commit message — the rest of the
backend is unchanged.

- [ ] **Step 6: Create `models/describer/gguf.py`**

```python
"""
cinemateca.models.describer.gguf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Keyless, offline Moondream 2 scene describer via llama-cpp-python.
Loads the 2025-01-09 GGUF pair (text model + mmproj) from the
vikhyatk/moondream2 repo at that revision. No transformers, no API key.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cinemateca.models.describer._common import (
    PROMPTS,
    build_metadata,
)

logger = logging.getLogger(__name__)

_REPO = "vikhyatk/moondream2"
_REV = "2025-01-09"
_TEXT_GGUF = "moondream2-text-model-f16.gguf"
_MMPROJ_GGUF = "moondream2-mmproj-f16.gguf"


class MoondreamGGUFDescriber:
    """SceneDescriber backed by Moondream 2 GGUF + llama-cpp-python."""

    def __init__(self, cfg=None, device=None):
        self._llm = None
        if cfg is not None and hasattr(cfg, "llm"):
            self.checkpoint_interval = cfg.llm.checkpoint_interval
            self.process_limit = cfg.llm.process_limit
            self.descriptions_filename = cfg.llm.descriptions_filename
            self.tags_filename = cfg.llm.tags_filename
        else:
            self.checkpoint_interval = 25
            self.process_limit = None
            self.descriptions_filename = "scene_descriptions.json"
            self.tags_filename = "scene_tags.json"

    def _load_model(self):
        if self._llm is not None:
            return
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MoondreamChatHandler

        t0 = time.time()
        text_path = hf_hub_download(_REPO, _TEXT_GGUF, revision=_REV)
        mmproj_path = hf_hub_download(_REPO, _MMPROJ_GGUF, revision=_REV)
        handler = MoondreamChatHandler(clip_model_path=mmproj_path)
        self._llm = Llama(
            model_path=text_path,
            chat_handler=handler,
            n_ctx=2048,
            logits_all=True,
            verbose=False,
        )
        logger.info("✓ Moondream GGUF carregado em %.1fs", time.time() - t0)

    def _answer(self, image_path, prompt: str, max_tokens: int) -> str:
        """One image+prompt -> stripped model text. Re-embeds per call."""
        uri = Path(image_path).resolve().as_uri()
        resp = self._llm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": uri}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"].strip()

    def describe(self, image_path) -> dict:
        self._load_model()
        raw = {}
        for field, (prompt, max_tokens) in PROMPTS.items():
            try:
                raw[field] = self._answer(image_path, prompt, max_tokens)
            except Exception as e:  # noqa: BLE001 - per-field resilience
                raw[field] = f"ERROR: {e}"
        # build_metadata expects a pandas row; synthesize a minimal one.
        row = pd.Series({"filepath": str(image_path)})
        return build_metadata(row, raw)

    def describe_batch(
        self,
        keyframes_df: pd.DataFrame,
        existing_results: list | None = None,
        checkpoint_path: Path | None = None,
    ) -> list[dict]:
        all_results = list(existing_results or [])
        # RESUME-BUG FIX: error rows are NOT processed (mirror pipeline.py:332).
        processed_ids = {
            r["scene_id"] for r in all_results if "error" not in r
        }
        to_process = keyframes_df[
            ~keyframes_df["scene_id"].isin(processed_ids)
        ].reset_index(drop=True)
        if self.process_limit:
            to_process = to_process.head(self.process_limit)

        logger.info(
            "LLM(GGUF): %d a processar (%d já ok, %d total)",
            len(to_process), len(processed_ids), len(keyframes_df),
        )

        for i, row in to_process.iterrows():
            try:
                raw = {}
                self._load_model()
                for field, (prompt, mx) in PROMPTS.items():
                    try:
                        raw[field] = self._answer(row["filepath"], prompt, mx)
                    except Exception as e:  # noqa: BLE001
                        raw[field] = f"ERROR: {e}"
                meta = build_metadata(row, raw)
                meta["scene_id"] = int(row.get("scene_id", -1))
                all_results.append(meta)
            except Exception as e:  # noqa: BLE001 - whole-frame failure
                all_results.append({
                    "scene_id": int(row.get("scene_id", -1)),
                    "keyframe_path": str(row["filepath"]),
                    "error": str(e),
                    "tags": [],
                    "objects": [],
                })
                logger.error("Erro cena %s: %s", row.get("scene_id"), e)

            count = i + 1
            if checkpoint_path and count % self.checkpoint_interval == 0:
                self._save_json(all_results, checkpoint_path)
                logger.info("Checkpoint: %d/%d", count, len(to_process))

        return all_results

    @staticmethod
    def build_tag_index(results: list[dict]) -> dict:
        from collections import defaultdict

        idx: dict[str, list] = defaultdict(list)
        for rec in results:
            for tag in rec.get("tags", []):
                idx[tag].append(rec.get("scene_id"))
        return dict(sorted(idx.items(), key=lambda x: len(x[1]), reverse=True))

    def save(self, results, tag_index, output_dir):
        import json

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        desc_path = out / self.descriptions_filename
        tags_path = out / self.tags_filename
        self._save_json(results, desc_path)
        with open(tags_path, "w", encoding="utf-8") as f:
            json.dump(tag_index, f, indent=2, ensure_ascii=False)
        return desc_path, tags_path

    @staticmethod
    def _save_json(data, path: Path):
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
```

> `build_tag_index`/`save`/`_save_json` mirror the old `LLMDescriber`
> surface so `pipeline.py`'s LLM step (Task 6) calls them unchanged except
> the method rename `describe_keyframes` → `describe_batch`.

- [ ] **Step 7: Run describer tests**

Run: `uv run pytest tests/test_describer_gguf.py -q`
Expected: PASS (all 3).

- [ ] **Step 8: Delete `llm_describer.py` and repoint smoke tests**

```bash
git rm src/cinemateca/llm_describer.py
```

In `tests/test_smoke.py` replace:

```python
def test_import_llm_describer():
    from cinemateca.llm_describer import LLMDescriber
    assert LLMDescriber is not None
```
with
```python
def test_import_describer():
    from cinemateca.models.describer.gguf import MoondreamGGUFDescriber
    assert MoondreamGGUFDescriber is not None
```

and replace the three parsing-import lines:
`from cinemateca.llm_describer import _parse_num_people` →
`from cinemateca.models.describer._common import _parse_num_people`;
`from cinemateca.llm_describer import _parse_objects` →
`from cinemateca.models.describer._common import _parse_objects`;
`from cinemateca.llm_describer import _generate_tags` →
`from cinemateca.models.describer._common import _generate_tags`.

- [ ] **Step 9: Confirm nothing else imports `llm_describer`**

Run:
```bash
grep -rn "llm_describer" --include="*.py" src api tests | grep -v __pycache__
```
Expected: only `src/cinemateca/__init__.py` docstring mention (cosmetic — edit
that line to say `models.describer` instead) and no functional import. Fix the
`__init__.py` docstring line, then re-run; expected: no functional matches.

- [ ] **Step 10: Run full suite**

Run: `uv run pytest tests/ -q`
Expected: all PASS at baseline count (note: one test name changed
`test_import_llm_describer`→`test_import_describer`; total count unchanged).

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "feat(describer): add keyless Moondream GGUF backend + _common extract; delete llm_describer.py

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Registry + pipeline repoint + config block

**Files:**
- Create: `src/cinemateca/models/registry.py`
- Modify: `config/default.yaml` (add `models:` block)
- Modify: `src/cinemateca/pipeline.py` (3 model sites → registry)
- Test: `tests/test_models_protocols.py`

- [ ] **Step 1: Add failing registry tests**

Append to `tests/test_models_protocols.py`:

```python
def test_registry_returns_correct_types():
    from cinemateca.models import registry
    from cinemateca.models.base import (
        EnvironmentClassifier,
        FaceDetector,
        ImageEmbedder,
        ObjectDetector,
        SceneDescriber,
    )

    class _M:
        image_embedder = "clip_openclip"
        face_detector = "mtcnn_pytorch"
        object_detector = "yolov8"
        scene_describer = "moondream_gguf"
        environment_classifier = "opencv_heuristic"

    class _Cfg:
        models = _M()

    cfg = _Cfg()
    assert isinstance(registry.get_image_embedder(cfg), ImageEmbedder)
    assert isinstance(registry.get_face_detector(cfg), FaceDetector)
    assert isinstance(registry.get_object_detector(cfg), ObjectDetector)
    assert isinstance(registry.get_scene_describer(cfg), SceneDescriber)
    assert isinstance(
        registry.get_environment_classifier(cfg), EnvironmentClassifier
    )


def test_registry_unknown_name_raises():
    import pytest

    from cinemateca.models import registry

    class _Cfg:
        class models:
            image_embedder = "nope"

    with pytest.raises(ValueError):
        registry.get_image_embedder(_Cfg())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_models_protocols.py -k registry -q`
Expected: FAIL — `ModuleNotFoundError: cinemateca.models.registry`.

- [ ] **Step 3: Create `src/cinemateca/models/registry.py`**

```python
"""
cinemateca.models.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~
Reads cfg.models.* and returns concrete backends. The pipeline imports
from here, never from a concrete backend module.
"""
from __future__ import annotations


def _name(cfg, attr: str) -> str:
    models = getattr(cfg, "models", None)
    if models is None:
        raise ValueError("config has no [models] section")
    val = getattr(models, attr, None)
    if not val:
        raise ValueError(f"models.{attr} is unset")
    return val


def get_image_embedder(cfg):
    name = _name(cfg, "image_embedder")
    if name == "clip_openclip":
        from cinemateca.models.clip.openclip import OpenClipEmbedder
        return OpenClipEmbedder(cfg, getattr(cfg, "_device", None))
    raise ValueError(f"Unknown image_embedder: {name!r}")


def get_face_detector(cfg):
    name = _name(cfg, "face_detector")
    if name == "mtcnn_pytorch":
        from cinemateca.models.face.mtcnn import MTCNNFaceDetector
        return MTCNNFaceDetector(cfg, getattr(cfg, "_device", None))
    raise ValueError(f"Unknown face_detector: {name!r}")


def get_object_detector(cfg):
    name = _name(cfg, "object_detector")
    if name == "yolov8":
        from cinemateca.models.objects.yolov8 import YOLOv8ObjectDetector
        return YOLOv8ObjectDetector(cfg, getattr(cfg, "_device", None))
    raise ValueError(f"Unknown object_detector: {name!r}")


def get_scene_describer(cfg):
    name = _name(cfg, "scene_describer")
    if name == "moondream_gguf":
        from cinemateca.models.describer.gguf import MoondreamGGUFDescriber
        return MoondreamGGUFDescriber(cfg, getattr(cfg, "_device", None))
    raise ValueError(f"Unknown scene_describer: {name!r}")


def get_environment_classifier(cfg):
    name = _name(cfg, "environment_classifier")
    if name == "opencv_heuristic":
        from cinemateca.models.environment.opencv_heuristic import (
            OpenCVEnvironmentClassifier,
        )
        return OpenCVEnvironmentClassifier(cfg)
    raise ValueError(f"Unknown environment_classifier: {name!r}")
```

- [ ] **Step 4: Run registry tests**

Run: `uv run pytest tests/test_models_protocols.py -k registry -q`
Expected: PASS.

- [ ] **Step 5: Add the `models:` block to `config/default.yaml`**

Insert immediately **after** the `llm:` block (after line 84, before the
`# ── Pipeline` banner at line 86):

```yaml

# ── Model backends (pluggable — see docs/PROTOCOL_OPTION.md) ──────────────────
models:
  image_embedder: clip_openclip            # clip_openclip
  face_detector: mtcnn_pytorch             # mtcnn_pytorch
  object_detector: yolov8                  # yolov8
  scene_describer: moondream_gguf          # moondream_gguf (keyless, offline)
  environment_classifier: opencv_heuristic # opencv_heuristic
```

- [ ] **Step 6: Repoint `pipeline.py` visual-analysis step to the registry**

Replace the interim Task-3 Step-8 import + construction with:

```python
        from cinemateca.models.registry import get_environment_classifier, get_face_detector, get_object_detector
        from cinemateca.visual_analyzer import VisualAnalyzer
        ...
            analyzer = VisualAnalyzer(
                face_detector=get_face_detector(self.cfg),
                object_detector=get_object_detector(self.cfg),
                env_classifier=get_environment_classifier(self.cfg),
            )
```

If the registry backends need the device, set `self.cfg._device = self.device`
once at the top of `pipeline.run` (the registry reads `cfg._device`); add that
single line right after `video_path = Path(video_path)` in `run`:
`self.cfg._device = self.device`.

- [ ] **Step 7: Repoint the embeddings step**

Replace the Task-4 interim:

```python
        from cinemateca.models.clip.openclip import OpenClipEmbedder
        ...
            embedder = OpenClipEmbedder(self.cfg, self.device)
```
with:
```python
        from cinemateca.models.registry import get_image_embedder
        ...
            embedder = get_image_embedder(self.cfg)
```

- [ ] **Step 8: Repoint the LLM step + rename to `describe_batch`**

In `_step_llm_description`, replace:

```python
        from cinemateca.llm_describer import LLMDescriber
        ...
            describer = LLMDescriber(self.cfg, self.device)
            results = describer.describe_keyframes(
                valid_kf,
                existing_results=existing,
                checkpoint_path=desc_path,
            )
```
with:
```python
        from cinemateca.models.registry import get_scene_describer
        ...
            describer = get_scene_describer(self.cfg)
            results = describer.describe_batch(
                valid_kf,
                existing_results=existing,
                checkpoint_path=desc_path,
            )
```

(`describer.build_tag_index(results)` and `describer.save(...)` calls below
remain — the GGUF backend provides both, Task 5 Step 6.)

- [ ] **Step 9: Full suite**

Run: `uv run pytest tests/ -q`
Expected: all PASS at baseline count.

- [ ] **Step 10: Smoke the pipeline wiring (no heavy models)**

Run:
```bash
uv run python -c "
from cinemateca.config import load_config
from cinemateca.models import registry
c = load_config(None)
print('describer:', type(registry.get_scene_describer(c)).__name__)
print('embedder :', type(registry.get_image_embedder(c)).__name__)
"
```
Expected: `describer: MoondreamGGUFDescriber` / `embedder: OpenClipEmbedder`
(no model download — construction is lazy).

- [ ] **Step 11: Commit**

```bash
git add src/cinemateca/models/registry.py config/default.yaml src/cinemateca/pipeline.py tests/test_models_protocols.py
git commit -m "feat(models): registry + config block; pipeline uses registry (no direct backend imports)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Fix the `--steps` alias bug

**Files:**
- Modify: `src/cinemateca/__main__.py:36-44`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smoke.py`:

```python
def test_steps_alias_maps_to_full_names():
    from cinemateca.__main__ import _resolve_steps

    assert _resolve_steps("llm") == {"llm_description"}
    assert _resolve_steps("frames,scenes") == {
        "frame_extraction", "scene_detection",
    }
    assert _resolve_steps("llm_description") == {"llm_description"}  # full name OK
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_smoke.py -k steps_alias -q`
Expected: FAIL — `ImportError: cannot import name '_resolve_steps'`.

- [ ] **Step 3: Add `_resolve_steps` and use it in `cmd_process`**

In `src/cinemateca/__main__.py`, add this module-level function above
`cmd_process`:

```python
_STEP_ALIASES = {
    "frames": "frame_extraction",
    "scenes": "scene_detection",
    "visual": "visual_analysis",
    "embeddings": "embeddings",
    "llm": "llm_description",
}


def _resolve_steps(steps_arg: str) -> set[str]:
    """Map documented short aliases (and full names) to canonical step names."""
    full = set(_STEP_ALIASES.values())
    out = set()
    for tok in steps_arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in _STEP_ALIASES:
            out.add(_STEP_ALIASES[tok])
        elif tok in full:
            out.add(tok)
        else:
            raise SystemExit(
                f"--steps: valor desconhecido {tok!r}. "
                f"Use: {','.join(_STEP_ALIASES)}"
            )
    return out
```

Then in `cmd_process` replace:

```python
    if args.steps:
        enabled = set(args.steps.split(","))
        all_steps = [
            "frame_extraction", "scene_detection",
            "visual_analysis", "embeddings", "llm_description"
        ]
        for step in all_steps:
            setattr(cfg.pipeline.steps, step, step in enabled)
```
with:
```python
    if args.steps:
        enabled = _resolve_steps(args.steps)
        for step in _STEP_ALIASES.values():
            setattr(cfg.pipeline.steps, step, step in enabled)
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_smoke.py -k steps_alias -q`
Expected: PASS.

- [ ] **Step 5: Full suite + commit**

Run: `uv run pytest tests/ -q` → all PASS.

```bash
git add src/cinemateca/__main__.py tests/test_smoke.py
git commit -m "fix(cli): --steps short aliases now map to canonical step names

Previously '--steps llm' silently disabled every step and exited 0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Acceptance — regenerate the Jeca Tatu LLM artefacts (gated, expensive)

**Files:** none (data regeneration). `data/` is the user's archive — this task
is the only one that downloads weights / writes `data/`. Per CLAUDE.md it
requires explicit user confirmation before running.

- [ ] **Step 1: Get explicit go-ahead**

Confirm with the maintainer: this downloads the Moondream GGUF (~3.7 GB f16)
and runs the LLM step over 412 scenes (minutes–low-hours). Do not proceed
without an explicit yes.

- [ ] **Step 2: Clear the corrupt/partial artefacts so resume starts empty**

```bash
cd /mnt/a/1_projects/samaritan/cinemateca-imgsearch
ls -la data/metadata/scene_descriptions.json data/metadata/scene_tags.json 2>/dev/null
rm -f data/metadata/scene_descriptions.json data/metadata/scene_tags.json
```
(The `*.corrupt-bak` files from the investigation stay as forensic backup.)

- [ ] **Step 3: Run the LLM step through the registry path (background)**

```bash
uv run python -m cinemateca process --video data/raw/_unused_llm_only --steps llm_description > /tmp/cinemateca_gguf_regen.log 2>&1 &
```
(Now `--steps llm_description` works regardless of Task 7; using the full name
is intentional and unambiguous. The `--video` arg is a required-but-unused
placeholder for the llm-only run.)

- [ ] **Step 4: Verify acceptance after completion**

```bash
uv run python -c "
import json
d=json.load(open('data/metadata/scene_descriptions.json'))
errs=sum(1 for r in d if 'error' in r)
t=json.load(open('data/metadata/scene_tags.json'))
print(f'rows={len(d)} errors={errs} tags={len(t)}')
assert errs == 0, f'{errs} error rows remain'
assert len(t) > 0, 'empty tag index'
print('ACCEPTANCE PASS')
"
```
Expected: `errors=0`, non-empty tags, `ACCEPTANCE PASS`.

- [ ] **Step 5: Commit the recovery note (not the data)**

`data/` is gitignored — do not `git add` it. Record the outcome in Task 9's
CHANGELOG entry instead.

---

## Task 9: Docs

**Files:**
- Modify: `docs/PROTOCOL_OPTION.md`, `CLAUDE.md`, `CHANGELOG.md`

- [ ] **Step 1: `docs/PROTOCOL_OPTION.md`** — change the top `**Status:**` line
from `Design option. Not yet implemented.` to
`Steps 1–2 implemented 2026-05-17 (pluggable Protocol + registry; describer on keyless Moondream GGUF). Step 4 (ONNX) still deferred. See docs/superpowers/specs/2026-05-17-pluggable-model-backends-design.md.`

- [ ] **Step 2: `CLAUDE.md` migration tracker** — append under the FastAPI
regression-recovery bullet:

```
- [x] Pluggable model backends (PROTOCOL_OPTION Steps 1–2): 5 Protocols +
  registry, all roles moved, hard cutover (no shims), describer replaced
  with keyless offline Moondream-2 GGUF (transformers removed) — see
  docs/superpowers/specs/2026-05-17-pluggable-model-backends-design.md
```

- [ ] **Step 3: `CHANGELOG.md`** — add an entry summarising: Protocol/registry
introduced; `transformers`/Moondream-PyTorch removed; describer now
keyless/offline GGUF; `--steps` alias bug fixed; `by_image` decoupled;
Jeca Tatu artefacts regenerated (state the final `errors=0` count from Task 8).

- [ ] **Step 4: Commit**

```bash
git add docs/PROTOCOL_OPTION.md CLAUDE.md CHANGELOG.md
git commit -m "docs: mark PROTOCOL_OPTION Steps 1-2 done; update tracker + changelog

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** Architecture (Tasks 2,3,4,6) ✓; per-role deltas — ImageEmbedder
`encode_image_single`/`by_image` (Task 4) ✓, SceneDescriber replace+`_common`+
resume-fix (Task 5) ✓, detectors moved (Task 3) ✓; dependency repackaging incl.
grep-gate (Task 1) ✓; `--steps` bug (Task 7) ✓; testing — conformance/registry/
resume/`by_image`/smoke (Tasks 2–7) ✓; acceptance regeneration (Task 8) ✓;
sequencing matches spec (green tree per task) ✓; docs (Task 9) ✓. No spec
requirement is unassigned.

**Placeholder scan:** The only deliberate "copy verbatim" markers are in Task 4
Step 5 (`combined()` body) and Task 5 Step 2 (`_common.py` extract) — these
quote exact `git show HEAD:` line ranges with explicit paste instructions
because the source is load-bearing and must not be paraphrased; they are not
open-ended TODOs. No "add error handling"/"similar to"/"TBD" placeholders.

**Type consistency:** New class names are used identically across tasks —
`OpenClipEmbedder`, `MTCNNFaceDetector`, `YOLOv8ObjectDetector`,
`OpenCVEnvironmentClassifier`, `MoondreamGGUFDescriber`; method names
`encode_image_single`, `describe`/`describe_batch`, registry `get_*`; config
keys `clip_openclip`/`mtcnn_pytorch`/`yolov8`/`moondream_gguf`/`opencv_heuristic`
match `config/default.yaml` (Task 6 Step 5) and `registry.py` (Task 6 Step 3).
`VisualAnalyzer(face_detector=, object_detector=, env_classifier=)` keyword
constructor is consistent between Task 3 Step 7 (def), Task 3 Step 8 (interim
call) and Task 6 Step 6 (registry call).

---

## Risks carried from the spec

- Single describer, no fallback (maintainer-accepted) — Task 8 is the sole gate.
- `MoondreamChatHandler` signature pinned by Task 5 Step 5 verification; a
  build mismatch falls back to `Llava15ChatHandler` + Moondream mmproj.
- 6× per-frame image re-embed: acceptable for the CPU/GPU baseline; if Task 8
  runtime is unacceptable, the documented mitigation (collapse prompts / use
  lower-level `llama_cpp` embed-once API) is a follow-up, not a blocker.
