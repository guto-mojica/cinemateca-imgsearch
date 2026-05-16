# FastAPI Regression Recovery Plan

> **For agentic workers:** execute this plan task-by-task. Keep the checkbox state current as work lands. Do not mix broad refactors with unrelated feature work.

**Goal:** Recover the FastAPI/HTMX app after the feature work added after `4118275`, then refactor the web layer so Search, Scenes, Annotate, Processing, Library, and i18n share reliable data contracts and are covered by tests.

**Baseline inspected:** `4118275..HEAD` on `main` / `setup_uv`.

**Current verification snapshot:**
- `uv run pytest` passes: 18 smoke tests.
- `uv run ruff check .` fails with 103 findings; many are style/modernization, but there are real failures such as undefined `torch` annotation references reported by ruff and unused route imports.
- FastAPI route smoke checks return 200 for empty-data tab routes, but full-page routes render incomplete content because their template context is missing tab-specific variables.
- A dummy active Processing job crashes `GET /tab/processing` with `jinja2.exceptions.TemplateAssertionError: No filter named 'split'`.
- Direct probes show scene/search tag filtering fails when scene IDs are mixed between `int` and `str`.

---

## Diagnosis

The commits after `4118275` changed the app from a FastAPI shell into a mostly functional HTMX interface, but feature code was added directly in route modules and templates without a shared service layer or web regression tests. The result is duplicated metadata loading, inconsistent scene-id types, divergent full-page vs HTMX-tab context, and Processing state spread across templates, SSE, route code, and private pipeline methods.

The most important architectural problem is that the app now has several implicit contracts:

- `base.html` includes tab partials directly.
- `/tab/*` routes build rich tab-specific context.
- `/search`, `/scenes`, `/annotate`, and `/processing` full-page routes do not build the same context.
- templates assume variable names such as `cards`, `available_tags`, `step_defs`, `steps`, `progress`, and `status`, but not every caller supplies them.
- LLM tags use integer scene IDs, manual annotations use string scene IDs, and each feature normalizes differently.
- library selection advertises multi-film navigation, while the underlying metadata/index layout is still global single-film state.

---

## Known Regressions And Risks

### 1. Full-page routes do not match tab routes

Files:
- `api/server.py`
- `api/routes/search.py`
- `api/routes/scenes.py`
- `api/routes/annotate.py`
- `api/routes/processing.py`
- `web/templates/base.html`

Evidence:
- `/tab/scenes` correctly reports missing scene metadata, but `/scenes` renders the included `scenes.html` partial without `no_data`, `cards`, or `available_tags`, so the UI can say "No scenes match the filters" instead of "run scene detection first".
- `/processing` includes `processing.html` without `step_defs` or `jobs`, so the direct URL misses the step checkboxes and job state that `/tab/processing` supplies.
- `/search` direct render lacks `available_tags`.
- `/annotate` direct render lacks annotation state.

Fix direction:
- Make full-page routes call the same context builders as `/tab/*` routes.
- Prefer one `render_page(active_tab, tab_context)` helper over letting `base.html` include partials with incomplete data.

### 2. Processing tab has multiple broken contracts

Files:
- `api/jobs.py`
- `api/routes/processing.py`
- `web/templates/partials/processing.html`
- `web/templates/partials/processing_job.html`
- `web/templates/partials/processing_stepper.html`
- `web/static/js/htmx-ext-sse.js`

Evidence:
- `processing_job.html` uses `{{ job.video_path | replace('\\', '/') | split('/') | last }}`, but the shared Jinja environment does not define a `split` filter.
- `processing_job.html` includes `processing_stepper.html`, but only passes `job`; the stepper expects top-level `steps`, `progress`, `status`, `error_msg`, and `total_duration_s`.
- The SSE extension registers one close event named `done,error`; it does not split the value. The server only emits generic `message` events anyway, so the EventSource can reconnect after completion.
- `api.jobs._run_pipeline()` calls private `CatalogPipeline._step_*` methods directly and can continue downstream after failed prerequisites, producing stale mixed outputs when `stop_on_error` is false.
- active jobs live in a process-global dict with no retention cleanup, cancellation, concurrency limit, or persistence.

Fix direction:
- Fix the template to compute a display filename in Python or with a supported helper.
- Pass a single `job` object to the stepper and have the template read `job.steps`, `job.progress`, etc., or consistently flatten job state everywhere.
- Emit explicit SSE events: `event: update`, `event: done`, `event: error`; update `sse-swap` / `sse-close` accordingly.
- Replace private pipeline calls with a public pipeline API that accepts selected steps and a progress callback.
- Add dependency-aware step gating so downstream steps are skipped with a clear reason when prerequisites fail or are missing.

### 3. Tag filtering fails with mixed scene-id types

Files:
- `api/routes/scenes.py`
- `api/routes/search.py`
- `src/cinemateca/annotator.py`
- `src/cinemateca/embeddings.py`
- `src/cinemateca/llm_describer.py`

Evidence:
- `LLMDescriber.build_tag_index()` stores `scene_id` values as ints.
- manual annotations are stored as JSON object keys, therefore strings.
- `scenes._build_cards()` compares `str(scene_id)` to a raw `valid_ids` set, so LLM tag filters with integer IDs return no cards.
- `SemanticSearch.combined()` compares a DataFrame integer `scene_id` column to manual string IDs, so manual tag filters return no search rows.

Fix direction:
- Introduce one canonical scene ID representation at the service boundary. Prefer string keys for JSON/index interoperability, with helper functions:
  - `scene_id_key(value) -> str`
  - `normalize_tag_index(index) -> dict[str, set[str]]`
  - `normalize_scene_record(record) -> SceneRecord`
- Convert DataFrame `scene_id` values to the canonical key before filtering.
- Add tests for int-only LLM tags, string-only manual tags, and mixed merged tags.

### 4. Library selection advertises multi-film behavior that does not exist yet

Files:
- `src/cinemateca/library.py`
- `api/routes/library.py`
- `api/routes/search.py`
- `api/routes/scenes.py`
- `api/routes/annotate.py`
- `api/jobs.py`

Evidence:
- `scan_library()` scans only top-level `raw_dir` video files.
- every film is marked processed if global `data/metadata/keyframes_metadata.json` exists.
- `api_library_select()` ignores `slug` and returns the generic Search tab without selected-film context.
- all metadata, frames, embeddings, tags, and annotations are global paths, so multiple films cannot be selected independently.
- local data currently has keyframes and embeddings but no metadata/raw video in the configured `data/raw`, which makes the sidebar empty while search artifacts may still exist.

Fix direction:
- Decide explicitly whether v0.3 is single-film with a library placeholder or true multi-film.
- If true multi-film: introduce `FilmContext(slug, raw_path, metadata_dir, frames_dir, embeddings_dir)` and migrate outputs to per-film directories.
- If single-film for now: remove or disable misleading selection behavior and make the sidebar report global artifact state honestly.

### 5. Search/index loading needs safer cache and data integrity handling

Files:
- `api/routes/search.py`
- `src/cinemateca/embeddings.py`

Risks:
- `_load_index()` is cached forever by path strings, so a newly generated index may not be visible until process restart.
- no validation checks that embedding row count matches mapping rows.
- no fallback if `index_mapping.json` refers to moved files.
- image upload accepts any size and suffix; no size/type validation beyond `accept="image/*"` in the browser.

Fix direction:
- Move index loading to a service with mtime/size-aware cache invalidation.
- Validate embeddings/mapping shape and report a useful no-index/corrupt-index UI state.
- Add upload size and content-type checks.

### 6. Annotation workflow needs a durable data contract

Files:
- `api/routes/annotate.py`
- `src/cinemateca/annotator.py`
- `web/templates/partials/annotate.html`
- `web/templates/partials/annotate_scene.html`

Risks:
- after save/clear, only `#annotate-scene` is swapped, so outer counts and jump options can become stale.
- the "Without LLM description" filter and "annotated" count are conceptually mixed; manual tags do not remove a scene from `no_llm`, which may or may not be intended.
- tag normalization is inline in routes and duplicated with `merge_tag_index()`.
- writes are simple JSON rewrites with no atomic temp-file replace.

Fix direction:
- Define annotation semantics: manual tags are either supplemental only, or they can mark a scene as handled.
- Add a shared tag-normalization helper.
- Save annotations atomically.
- Return either the full Annotate tab or HTMX out-of-band swaps for counts/jump state after save/clear.

### 7. i18n and UI accessibility are incomplete

Files:
- `api/jobs.py`
- `api/routes/about.py`
- `api/deps.py`
- `web/templates/base.html`
- `web/locales/*`

Risks:
- `STEP_DEFS` hardcodes Portuguese labels in Python.
- `<html lang="pt-BR">` does not reflect the active locale.
- locale switching always redirects to `/`, losing the active tab/page.
- several HTMX anchors have no `href`, which hurts keyboard/accessibility behavior.
- Phosphor icons are loaded from an external CDN, so the UI is not fully offline.

Fix direction:
- Store step IDs in Python and translate labels in templates.
- set `<html lang="{{ locale }}">`.
- preserve current URL on locale switch.
- use real links with `href` fallback for tab/about/library actions.
- consider vendoring icons or replacing them with local assets.

### 8. Tooling and repo hygiene need cleanup

Files:
- `pyproject.toml`
- `.gitignore`
- `gitignore`
- `src/cinemateca/device.py`
- changed route/source files reported by ruff

Risks:
- `ruff` config uses deprecated top-level `select`/`ignore`.
- current ruff run reports 103 findings.
- there is a tracked `gitignore` file in addition to `.gitignore`; this is easy to mistake for active ignore rules.
- `uv.lock` is ignored and exists locally, which is expected by docs but should be called out in setup guidance.
- repeated `uv run` calls warned about packages in `.venv` missing `RECORD`; avoid parallel `uv` invocations and consider a clean env rebuild after code work.

Fix direction:
- Move ruff settings to `[tool.ruff.lint]`.
- Fix real F/E findings before style-only modernization.
- Decide whether the tracked `gitignore` file is documentation or an accidental file; if accidental, merge useful rules into `.gitignore` and remove it.
- Do not run `uv` environment-mutating commands in parallel.

---

## Target Architecture

### Service layer

Add a thin service layer under `api/services/`:

- `catalog.py`
  - load and validate metadata artifacts
  - build scene cards
  - normalize scene IDs and tags
  - resolve `/media/...` URLs
  - own `FilmContext`
- `search.py`
  - load/validate embeddings and mappings
  - handle mtime-aware cache invalidation
  - perform text/image/tag searches
- `annotations.py`
  - load/save manual annotations atomically
  - normalize tags and IDs
  - compute annotate view models
- `jobs.py` or refactor existing `api/jobs.py`
  - job registry
  - event model
  - dependency-aware execution

Route modules should become request parsing plus template rendering. They should not duplicate JSON loading, ID normalization, path resolution, or pipeline orchestration.

### View models

Use explicit dataclasses or typed dicts for template contracts:

- `BasePageContext`
- `SearchContext`
- `ScenesContext`
- `AnnotateContext`
- `ProcessingContext`
- `SceneCard`
- `JobView`

Templates should receive one predictable object per partial instead of many loosely named globals.

### Tests

Add web tests that fail on missing template variables and broken HTMX partials:

- route smoke tests for every full page and every `/tab/*` route
- active Processing job render test
- SSE stream contract test
- scenes tag filtering with int/string/mixed IDs
- search combined filtering with int/string/mixed IDs
- annotation save/clear JSON tests
- library select behavior test
- no-index/corrupt-index search tests

Use small temp directories and monkeypatched config; do not require real video, GPU, CLIP, or network.

---

## Execution Plan

### Phase 0: Safety net and baseline

- [ ] Create a branch for the recovery work.
- [ ] Capture `git status --short` and avoid touching unrelated local data.
- [ ] Add a minimal `tests/test_web_routes.py` with route smoke coverage for:
  - `/`
  - `/search`
  - `/scenes`
  - `/annotate`
  - `/processing`
  - `/tab/search`
  - `/tab/scenes`
  - `/tab/annotate`
  - `/tab/processing`
  - `/api/about`
- [ ] Add a test that injects a dummy active job and proves `/tab/processing` renders without a Jinja error.
- [ ] Add tests that document the current full-page context bug before fixing it.

Acceptance:
- tests reproduce the Processing `split` failure and at least one full-page context failure.

### Phase 1: Immediate user-visible bug fixes

- [ ] Extract tab context builders so full-page and `/tab/*` routes share data.
- [ ] Update `api/server.py` to render full pages with the same context as HTMX tab endpoints.
- [ ] Fix `processing_job.html` filename rendering without unsupported Jinja filters.
- [ ] Fix `processing_stepper.html` to consistently read either `job.*` or flattened values.
- [ ] Normalize scene IDs in tag filtering for Scenes and Search.
- [ ] Fix SSE completion semantics:
  - server emits `event: update` for progress
  - server emits `event: done` or `event: error` before closing
  - client listens/swaps/closes using matching event names
- [ ] Add regression tests for all fixes above.

Acceptance:
- `uv run pytest` passes.
- route smoke checks pass with empty data and with a dummy active job.
- scenes/search tag filters pass for int, string, and mixed scene IDs.

### Phase 2: Web test coverage expansion

- [ ] Add temp-config fixtures that build isolated `data/raw`, `data/metadata`, `data/frames`, and `data/embeddings` directories.
- [ ] Add fixture metadata:
  - 2 scenes
  - one fake keyframe path
  - one LLM tag index with integer IDs
  - one manual annotation file with string IDs
  - one visual analysis record
- [ ] Add no-data tests for Search, Scenes, Annotate, and Processing.
- [ ] Add annotation save/clear tests and verify JSON output.
- [ ] Add library filter/select tests.
- [ ] Add a corrupt-index test where embeddings count differs from mapping count.

Acceptance:
- web tests run without GPU, CLIP model load, or real video.
- no test mutates repository `data/`.

### Phase 3: Service extraction

- [ ] Create `api/services/catalog.py`.
- [ ] Move `_load_json`, `_keyframe_url`, `_load_metadata`, `_build_cards`, tag-index merge/normalization, and scene-card construction into catalog services.
- [ ] Create `api/services/annotations.py`.
- [ ] Move annotation scene-list/context building and tag normalization there.
- [ ] Create `api/services/search.py`.
- [ ] Move index loading, cache invalidation, and result conversion there.
- [ ] Keep route functions thin and focused on HTTP inputs/outputs.
- [ ] Keep backward-compatible templates during the extraction; do not redesign UI in this phase.

Acceptance:
- route modules no longer duplicate JSON loading or scene-id normalization.
- tests from Phases 1-2 still pass.

### Phase 4: Processing runner refactor

- [ ] Add a public pipeline API for selected-step execution with progress callbacks, instead of calling `_step_*` from `api/jobs.py`.
- [ ] Represent steps as a dependency graph:
  - `frame_extraction`
  - `scene_detection`
  - `visual_analysis` depends on keyframes
  - `embeddings` depends on keyframe metadata and keyframe files
  - `llm_description` depends on keyframe metadata and keyframe files
- [ ] When prerequisites fail, mark downstream steps `blocked` or `skipped` with a reason.
- [ ] Add job lifecycle controls:
  - created/running/done/error/cancelled
  - bounded retention
  - thread-safe registry access
  - one active job per film or explicit concurrency policy
- [ ] Add SSE tests for update/done/error event frames.

Acceptance:
- Processing cannot silently combine stale outputs after an upstream failure.
- active/completed jobs render correctly from direct `/processing` and HTMX `/tab/processing`.

### Phase 5: Library and per-film data model

- [ ] Decide and document one of:
  - v0.3 remains single-film: simplify sidebar and remove fake per-film processed state.
  - v0.3 supports multi-film: implement per-film output directories.
- [ ] If multi-film:
  - add `FilmContext`
  - change processing outputs to `data/<artifact>/<slug>/...` or `data/catalog/<slug>/...`
  - scope Search, Scenes, Annotate, and Processing by selected slug
  - add migration/read fallback for existing global artifacts
  - make `/api/library/{slug}/select` load the selected film context
- [ ] If single-film:
  - make sidebar show configured raw/index/metadata state without pretending each video has independent scene counts
  - disable or remove slug selection until per-film work lands.

Acceptance:
- selecting a film changes the actual metadata/search/annotation context, or the UI no longer implies that it will.

### Phase 6: i18n, accessibility, and offline polish

- [ ] Move Processing step labels out of Python hardcoded Portuguese and into templates/translations.
- [ ] Set `<html lang="{{ locale }}">`.
- [ ] Preserve the current path when switching locale.
- [ ] Give HTMX anchors real `href` fallbacks.
- [ ] Audit visible strings and update `.po` / `.mo` files.
- [ ] Decide whether Phosphor icons are acceptable as CDN dependency; vendor or replace if the app must be fully offline.

Acceptance:
- locale switch does not reset the user's active tab.
- direct links still work with JS disabled for core navigation.

### Phase 7: Tooling and repository cleanup

- [ ] Move ruff settings to `[tool.ruff.lint]`.
- [ ] Fix F/E ruff findings first; defer mass UP modernization if it creates noisy churn.
- [ ] Fix `src/cinemateca/device.py` annotations so ruff no longer reports undefined `torch`.
- [ ] Decide the fate of tracked `gitignore`; if accidental, merge useful rules into `.gitignore` and remove `gitignore`.
- [ ] Keep `.gitignore` aligned with docs around `uv.lock`, local configs, models, and data artifacts.
- [ ] Rebuild `.venv` only if package `RECORD` warnings persist and the user approves the environment churn.

Acceptance:
- `uv run ruff check .` passes, or any remaining findings are documented and intentionally deferred.
- `uv run pytest` passes.

### Phase 8: Release verification

- [ ] Run `uv run pytest`.
- [ ] Run `uv run ruff check .`.
- [ ] Run route smoke checks with `fastapi.testclient`.
- [ ] Manually verify in-browser:
  - direct `/search`, `/scenes`, `/annotate`, `/processing`
  - HTMX tab switches
  - Search no-index and index-present states
  - Scenes tag/keyword filters
  - Annotate save/clear
  - Processing start and SSE completion
  - locale switch from each tab
- [ ] Update `CHANGELOG.md` with the regression recovery summary.

Acceptance:
- No known Jinja template errors.
- No known direct-vs-HTMX route divergence.
- Search, Scenes, Annotate, Processing, and Library behavior matches the documented v0.3 scope.

---

## Suggested First PR Scope

Keep the first PR small enough to review:

1. Add failing web regression tests.
2. Fix full-page/tab context divergence.
3. Fix Processing template render.
4. Normalize scene IDs for tag filtering.
5. Fix SSE done/error close contract.

Defer the larger service extraction and per-film data model until the user-visible regressions are covered by tests.
