# Pluggable Model Backends — Design Spec

**Date:** 2026-05-17
**Status:** Approved design, ready for implementation planning
**Supersedes/implements:** Steps 1–2 of `docs/PROTOCOL_OPTION.md`
**Trigger:** A footage-processing run produced a 100%-error
`data/metadata/scene_descriptions.json` (`No module named 'pyvips'`).
Root-cause investigation found a cascade: undeclared Moondream remote-code deps
(`pyvips`, `einops`), and — decisively — `transformers` 5.x is incompatible with
**every** Moondream 2 revision (`'HfMoondream' object has no attribute
'all_tied_weights_keys'`, verified against `2025-01-09` and `2025-06-21`/`main`).
The maintainer chose to use this failure as the trigger to implement the
pluggable-backend refactor `PROTOCOL_OPTION.md` already designed, and to move the
scene describer off the `transformers`/PyTorch Moondream entirely onto a keyless,
offline GGUF backend.

---

## Goals

1. Introduce a typed `Protocol` per model role; backends are selected by config,
   never imported directly by the pipeline. (`PROTOCOL_OPTION.md` Steps 1–2.)
2. Replace the scene describer with a single **keyless, fully-offline,
   transformers-free** backend: Moondream 2 GGUF via `llama-cpp-python`.
3. Recover the corrupt single-film artefacts: regenerate the 412-scene
   *Jeca Tatu* `scene_descriptions.json` / `scene_tags.json` through the new
   describer, as the end-to-end acceptance test.
4. Remove `transformers`, `pyvips`, `einops` from the dependency tree. No
   `transformers` version pin anywhere.

## Non-goals (explicitly deferred)

- ONNX backends for CLIP / YOLO / MTCNN (`PROTOCOL_OPTION.md` Step 4 / release build).
- int4/int8 GGUF quantization for the <500 MB binary target (a llama.cpp
  quantize follow-up; the design must not block it).
- A `transformers`-5-native VLM (Qwen2.5-VL, SmolVLM2, …) — becomes a clean
  future Protocol backend (one file + one config line), not this effort.
- Multi-film library work (unchanged from the prior FastAPI recovery scope).

---

## Architecture

Faithful execution of `PROTOCOL_OPTION.md` "Design" + "Migration path" Steps 1–2,
adjusted for what the current code actually looks like (verified 2026-05-17).

### `src/cinemateca/models/` package

```
src/cinemateca/models/
  __init__.py
  base.py            # 5 Protocols (verbatim from PROTOCOL_OPTION.md §base.py)
  registry.py        # get_image_embedder / get_face_detector / get_object_detector
                     # / get_scene_describer / get_environment_classifier
  clip/openclip.py       # current CLIPEmbedder, moved + encode_image_single added
  face/mtcnn.py          # current FaceDetector, moved (exact Protocol match)
  objects/yolov8.py      # current ObjectDetector, moved (exact Protocol match)
  environment/opencv_heuristic.py  # current EnvironmentClassifier, moved (exact match)
  describer/
    _common.py           # model-agnostic parsing/tagging extracted from llm_describer.py
    gguf.py              # NEW: llama-cpp-python + Moondream GGUF (the only describer)
```

**Does NOT move** (per `PROTOCOL_OPTION.md`): `SemanticSearch` (pure numpy),
`scene_detector.py`, `annotator.py`, `library.py`, `data_prep.py`.

### Protocols (`models/base.py`)

Five `@runtime_checkable` Protocols, exactly as specified in
`PROTOCOL_OPTION.md` lines 100–211: `ImageEmbedder`, `FaceDetector`,
`ObjectDetector`, `SceneDescriber`, `EnvironmentClassifier`.

Reality check performed against current code:

| Role | Current class | Current methods | Protocol match |
|---|---|---|---|
| FaceDetector | `visual_analyzer.FaceDetector` | `detect`, `detect_batch` | exact |
| ObjectDetector | `visual_analyzer.ObjectDetector` | `detect`, `detect_batch` | exact |
| EnvironmentClassifier | `visual_analyzer.EnvironmentClassifier` | `classify`, `classify_batch` | exact |
| ImageEmbedder | `embeddings.CLIPEmbedder` | `encode_images`, `encode_text` | needs `encode_image_single` |
| SceneDescriber | `llm_describer.LLMDescriber` | `describe_keyframes(df,…)` | needs `describe` + `describe_batch` rename |

### Registry (`models/registry.py`)

Factory functions per role, reading `cfg.models.*`, lazy-importing the chosen
backend. Unknown name → `ValueError`. `pipeline.py` calls
`registry.get_*()`; it never imports a concrete backend.

### `VisualAnalyzer` facade

`visual_analyzer.VisualAnalyzer` is an orchestrator the pipeline consumes
directly (it composes Face + Object + Environment). It **stays** as a facade but
is refactored to receive its three sub-detectors via constructor injection from
the registry, instead of constructing them itself. Pipeline code that uses
`VisualAnalyzer` is otherwise unchanged.

### `config/default.yaml` addition

```yaml
models:
  image_embedder: clip_openclip          # clip_openclip (only impl today)
  face_detector: mtcnn_pytorch           # mtcnn_pytorch (only impl today)
  object_detector: yolov8                # yolov8 (only impl today)
  scene_describer: moondream_gguf        # moondream_gguf (only impl)
  environment_classifier: opencv_heuristic
```

Overridable in `config/local.yaml`.

---

## Per-role changes

### ImageEmbedder — close the Protocol leak (in scope, required)

`SemanticSearch.by_image` currently reaches into embedder privates
(`embedder._model.encode_image`, `_preprocess`, `_device`). Any non-openclip
backend would silently break image search, making the Protocol fiction.

- Add `encode_image_single(image_path: Path) -> np.ndarray` to
  `OpenClipEmbedder` (one image → L2-normalised `(D,)` float32; extract the
  logic currently inlined in `by_image`).
- Refactor `SemanticSearch.by_image` to call `embedder.encode_image_single(path)`.
  Math/normalisation identical for the openclip backend — behaviour-preserving.

### SceneDescriber — replace, not move

- `describer/_common.py`: extract the **model-agnostic** logic from
  `llm_describer.py` unchanged — `PROMPTS`, `LOCATION_MAP`, `TIME_MAP`,
  `_parse_num_people`, `_parse_objects`, `_generate_tags`, `_build_metadata`.
- `describer/gguf.py`: `MoondreamGGUFDescriber` implementing `SceneDescriber`:
  - `llama_cpp.Llama` + `llama_cpp.llama_chat_format.MoondreamChatHandler`,
    loading the Moondream 2 **`2025-01-09`** GGUF pair
    (`moondream2-text-model-f16.gguf` + `moondream2-mmproj-f16.gguf`) from the
    `vikhyatk/moondream2` repo at that revision (same snapshot the project
    already targeted → no behaviour drift versus the originally-intended model).
  - `describe(image_path) -> dict`: run the 6 `PROMPTS` for the image, feed raw
    answers through the shared `_build_metadata` / `_generate_tags`. Output dict
    schema is byte-for-byte the contract in `PROTOCOL_OPTION.md` §SceneDescriber
    (description, location, setting, time_of_day, num_people, people_action,
    objects, tags, plus scene/keyframe ids and timings).
  - `describe_batch(keyframes_df, existing_results=None, checkpoint_path=None)
    -> list[dict]`: same resume/checkpoint behaviour as the current
    `describe_keyframes`, **with the resume bug fixed** (below).
- Delete the `transformers`/`AutoModelForCausalLM`/`encode_image`+`answer_question`
  loading path entirely. `llm_describer.py` is removed once `describer/` replaces it.

**Resume-bug fix (mandatory, in blast radius).** Current
`llm_describer.describe_keyframes` builds `processed_ids` from *all* rows,
counting error rows as "done" — so resuming over an all-error file processes
zero scenes. `pipeline.py:332`'s skip-check already excludes errors correctly;
`describe_batch` must mirror it:
`processed_ids = {r["scene_id"] for r in all_results if "error" not in r}`.
Without this, acceptance regeneration is a no-op.

**Performance note (implementation concern, not a blocker).**
`MoondreamChatHandler` re-embeds the image on every `create_chat_completion`
call, so 6 prompts/frame = 6 image embeddings/frame (the old code embedded once
and reused via `answer_question`). Mitigation options for the implementation
plan, in preference order: (a) collapse the 6 prompts into fewer combined
prompts where parsing still recovers the fields reliably; (b) use the
lower-level `llama_cpp` multimodal API to embed once and reuse; (c) accept the
~6× embed cost (still CPU-viable on a small VLM). Decision deferred to the plan,
measured against the 412-scene acceptance run.

### Face / Object / Environment — pure moves

Move + rename only. Methods already match the Protocols exactly. No logic
changes. `VisualAnalyzer` updated to take them injected.

---

## Dependencies

`pyproject.toml`:

- `full` extra: **add** `llama-cpp-python>=0.3,<0.4`.
- `full` extra: **remove** `transformers>=4.35`. Also remove the
  session-added `pyvips>=2.2` / `einops>=0.7`.
  - **Verification gate (must run before deletion):** grep the codebase to
    confirm Moondream was the only consumer of `transformers` (expected: CLIP
    uses `open-clip-torch`, facenet/ultralytics are independent — confirmed by
    inspection 2026-05-17, re-verified in the plan). If anything else imports
    `transformers`, keep it and narrow the change.
- No new optional extra. No `transformers` pin anywhere (the whole reason a pin
  was considered — Moondream-on-PyTorch — is being removed).
- `torch` / `torchvision` stay (CLIP, YOLO, facenet still need them).
- GGUF weights download on first run (~3.7 GB f16, comparable to the prior
  ~2 GB safetensors order of magnitude) — gated like any model download per
  CLAUDE.md "Constraints and care".
- `llama-cpp-python` installs prebuilt CPU wheels. CUDA acceleration needs a
  prebuilt-CUDA wheel or a source build with CMake flags — out of scope here;
  CPU is the offline-archive deployment baseline. The plan documents the GPU
  build recipe as an optional note.

---

## `--steps` alias bug (in scope — it bit this investigation)

`__main__.cmd_process` compares `--steps` tokens directly against full step
names, but its `--help` text and the `CLAUDE.md` examples document short
aliases (`frames,scenes,visual,embeddings,llm`). Result: any short alias
silently disables every step and exits 0 (observed this session — a "successful"
run that did nothing). Fix: map the documented short aliases to the canonical
step attribute names (accept both). Add a regression test asserting
`--steps llm` enables `llm_description`.

---

## Testing

Per `CLAUDE.md`: `pytest tests/test_smoke.py` and the full suite (currently
208 tests) must be green **before and after each move commit**.

- **Behaviour preservation:** Face/Object/Env/CLIP are pure moves — their
  existing tests pass unmodified (update import paths only).
- **Protocol conformance:** cheap `runtime_checkable` `isinstance` test per
  backend, no model load.
- **Registry:** each config name → correct concrete type; unknown → `ValueError`.
- **Resume-bug regression:** an all-error `existing_results` must cause all
  scenes to be reprocessed (locks the SceneDescriber fix).
- **`--steps` regression:** `--steps llm` enables the LLM step.
- **`by_image` parity:** image-search result equivalence before/after the
  `encode_image_single` refactor (openclip backend).
- **End-to-end acceptance (sole quality gate — no fallback by maintainer
  decision):** regenerate the 412-scene *Jeca Tatu* film through
  `registry.get_scene_describer()` → `moondream_gguf`:
  - zero `"error"` rows in `scene_descriptions.json`
  - `scene_tags.json` non-empty
  - run in background, checkpointed every 25 scenes
  - precondition: delete the partial error-only `scene_descriptions.json` left
    by the stopped investigation run so `existing` starts empty.

---

## Sequencing (green tree at every commit)

1. `pyproject` deps: add `llama-cpp-python`; remove `transformers`/`pyvips`/
   `einops` after the grep-verification gate. `uv sync`.
2. `models/base.py` (5 Protocols) + conformance tests. No behaviour change.
3. Move Face/Object/Environment → `models/{face,objects,environment}/`;
   `VisualAnalyzer` constructor injection. Suite green.
4. Move CLIP → `models/clip/openclip.py`; add `encode_image_single`; refactor
   `SemanticSearch.by_image`. Suite green (incl. by_image parity).
5. `describer/_common.py` (extract) + `describer/gguf.py` (new
   `MoondreamGGUFDescriber`, resume-bug fixed). Remove `llm_describer.py`.
   Unit tests with the model mocked.
6. `registry.py`; repoint `pipeline.py` to `registry.get_*()`; add `[models]`
   config block. Full pipeline/integration tests green.
7. Fix the `--steps` alias bug + regression test.
8. Acceptance: regenerate the 412-scene film via `moondream_gguf`; assert
   zero-error / non-empty tags.
9. Docs: mark `PROTOCOL_OPTION.md` Steps 1–2 done (and the keyless-GGUF
   describer landing); update the `CLAUDE.md` migration tracker; `CHANGELOG.md`.

Steps 1–7 are logic-preserving except the deliberate, tested changes
(`encode_image_single`/`by_image`, describer replacement + resume fix,
`--steps`). Step 8 is the expensive, CLAUDE.md-gated regeneration.

---

## Risks & decisions

| Risk | Disposition |
|---|---|
| Single describer backend, no fallback | **Accepted by maintainer.** Step-8 regeneration is the sole gate; no A/B baseline. |
| GGUF f16 ≈ same size as safetensors | Acknowledged. Size win is the deferred int-quant follow-up, not automatic. |
| `llama-cpp-python` CPU-only by default | Fine for the offline-archive baseline; GPU build documented as optional. |
| 6× image re-embed per frame via chat handler | Mitigation options recorded; decided in the plan against measured acceptance-run cost. |
| Removing `transformers` breaks an unseen consumer | Grep-verification gate before deletion (step 1). |
| GGUF Moondream output quality vs. PyTorch path | Different inference engine; same `2025-01-09` snapshot. Quality judged on the 412-scene acceptance run; prior artefacts were 100% corrupt so there is nothing of value to regress against. |

---

## Acceptance criteria (definition of done)

- `models/` package with 5 Protocols, registry, all roles moved; `pipeline.py`
  uses the registry only.
- `scene_describer: moondream_gguf` is the only describer; no `transformers` /
  `pyvips` / `einops` in the dependency tree; no version pin.
- Full test suite green (≥ prior 208) including new conformance/registry/
  resume/`--steps`/`by_image` tests.
- 412-scene *Jeca Tatu* `scene_descriptions.json` regenerated with zero error
  rows and non-empty `scene_tags.json`, via the registry path.
- `PROTOCOL_OPTION.md`, `CLAUDE.md` tracker, `CHANGELOG.md` updated.
