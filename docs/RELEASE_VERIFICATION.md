# Release verification — v0.3.0 (FastAPI + HTMX)

Pre-release gate for the FastAPI interface after the regression-recovery effort
(Phases 0–8). Run the automated gates first; then a human runs the manual
in-browser checklist for the surfaces automation cannot exercise (visual
rendering, real interaction, browser-offline mode, JS-disabled fallback).

Scope of this release is single-film v0.3 by maintainer decision (Phase 5).
Multi-film support is intentionally deferred (see CHANGELOG `[Não lançado]`).

---

## 1. Automated gates (run from the repo root)

| Gate | Command | Expected |
|---|---|---|
| Test suite | `uv run pytest -q` | `208 passed`, 0 failed/xfailed/xpassed |
| Lint | `uv run ruff check .` | `All checks passed!` |
| Types | `uv run mypy src` | `Found 5 errors in 3 files` — known pre-existing only |

Last recorded run (2026-05-16, HEAD `5e93b8a`):

- `uv run pytest -q` → `208 passed in 2.73s`
- `uv run ruff check .` → `All checks passed!`
- `uv run mypy src` → `Found 5 errors in 3 files (checked 13 source files)`

Known mypy errors — pre-existing, NOT recovery regressions, do not block release:

- `src/cinemateca/config.py:133` / `:139` — `_Namespace` has no attribute
  `logging` / `paths` (dynamic attribute access on the config namespace).
- `src/cinemateca/__main__.py:44` / `:49` — `_Namespace` has no attribute
  `pipeline` / `hardware` (same dynamic-namespace pattern).
- `src/cinemateca/embeddings.py:319` — numpy ndarray vs `list[Any]` assignment.

If the count changes from 5, investigate before release.

### What automation already covers (do NOT re-verify by hand)

- Route smoke — every full-page route and HTMX tab fragment returns 200 via
  `fastapi.testclient` (`tests/test_web_routes.py`).
- Full-page vs HTMX-tab parity — `TestFullPageContextDivergence`
  (`tests/test_web_routes.py`). No known direct-vs-HTMX divergence.
- Processing template render — `split`-filter crash + stepper checklist,
  active/errored jobs, Windows-path basenames
  (`tests/test_web_routes.py::TestProcessingSplitFilterCrash`).
- Scene-id tag filtering — int/str/numpy-int/float/mixed normalization
  (`tests/test_scene_id_filtering.py`, `tests/test_catalog_service.py`).
- SSE close contract — typed `update`→`done`, single `error`, single
  `cancelled`; stream closes on terminal; vendored sse-ext split
  (`tests/test_sse.py`).
- Pipeline gating/cancellation — upstream failure blocks downstream with
  explicit reason, no stale output, no silent success; cancel mid-run yields
  `cancelled`; single global active job (`tests/test_processing_runner.py`).
- Search service — no-index state, corrupt-index degradation, mtime cache
  invalidation, upload validation (`tests/test_search_service.py`,
  `tests/test_web_coverage.py`).
- Annotate save/clear — normalized atomic persist, single-scene clear, missing
  no-op (`tests/test_web_coverage.py`, `tests/test_annotations_service.py`).
- i18n/a11y/offline — locale→lang, `<html lang>` flip, locale switch preserves
  path (HTMX + JS-off) + open-redirect rejected, real `href`s, translated step
  labels, no external CDN anywhere, self-hosted fonts, vendored Phosphor
  (`tests/test_i18n_a11y.py`).

No known Jinja template errors: the suite renders every template/fragment
(Phase 0/1b/1d) and would fail on a `TemplateError`.

---

## 2. Manual in-browser checklist (human, pre-release)

Start: `uv run app.py` → `http://localhost:8501`. Use the test film
**Jeca Tatu (1959)** for populated states. Open devtools Network tab for the
offline check.

### 2.1 Direct full-page routes (type URL, hard reload)

- [ ] `/search` renders the full page (sidebar + main), not a bare fragment
- [ ] `/scenes` renders the full page
- [ ] `/annotate` renders the full page
- [ ] `/processing` renders the full page (step checklist visible, no 500)
- [ ] `/about` renders the full About page (not only the modal)
- [ ] No console errors; layout intact; CSS and icons load

### 2.2 HTMX tab switching (from within the app)

- [ ] Click each sidebar tab: Buscar / Cenas / Anotar / Processamento / Sobre
- [ ] Each swaps the main area without full reload; URL updates
- [ ] Back-and-forth keeps state coherent (no duplicated panels)
- [ ] Full-page render of a tab and the in-app swap look identical

### 2.3 Search

- [ ] No index present: text search shows the no-index/hint message, not a crash
- [ ] Index present (after processing Jeca Tatu): text query returns a results
      grid with keyframes and metadata
- [ ] Very short query returns an empty body (no spurious results)
- [ ] Image upload rejection: non-image (e.g. `.txt`) rejected with clear
      message; oversize rejected; valid JPEG/PNG accepted and runs image search

### 2.4 Scenes (library browsing)

- [ ] Grid lists scenes with keyframes
- [ ] Tag filter narrows results (try an automatic/LLM tag)
- [ ] Manual string tag targeting an integer scene-id still matches (mixed
      int/str scene-id correctness)
- [ ] Keyword filter on description narrows results
- [ ] Empty/no-data state shows the honest "no scenes" message, not an error

### 2.5 Annotate

- [ ] Scene panel shows current tags and the annotated count
- [ ] Save: add tags, save → feedback; reload → tags persisted and normalized
      (lowercase, spaces→hyphens)
- [ ] Clear: clear one scene → only that scene loses tags; counts update
- [ ] Jump/next/prev moves correctly; nav edges (first/last) behave at bounds

### 2.6 Processing (SSE)

- [ ] Start a job (short/subset run) → status streams live
- [ ] SSE: `update` frames progress the stepper, then a single `done`; the
      stream closes after `done` (EventStream connection ends in Network tab,
      no infinite reconnect)
- [ ] Cancellation: start a job, cancel → `cancelled` terminal, stream closes,
      no infinite reconnect loop
- [ ] Dependency gating: upstream step fails (or subset missing inputs) →
      downstream blocked with explicit reason, job ends `error`, no stale/partial
      output presented as success
- [ ] Only one active job — starting a second while one runs is rejected

### 2.7 Locale switch preserves the current tab (per-tab)

For EACH tab — Buscar, Cenas, Anotar, Processamento, Sobre:

- [ ] Navigate to the tab, then switch locale (PT↔EN)
- [ ] You stay on the same tab (does NOT reset to `/` / home)
- [ ] Visible strings switch language
- [ ] `<html lang>` flips (`pt-br` ↔ `en`) — check devtools Elements

### 2.8 JS-disabled core navigation

- [ ] Disable JavaScript
- [ ] Sidebar nav links and the About link still work via real `href`s
      (full-page navigation, not HTMX)
- [ ] Locale switch still works with JS off (server-side redirect, path
      preserved, open-redirect not honored)

### 2.9 Fully offline (no network / no CDN)

- [ ] Devtools → Network → Offline
- [ ] Hard-reload: app loads and renders (HTML, CSS, icons, fonts)
- [ ] No failed requests to any external host/CDN in the Network tab
- [ ] Phosphor icons render (vendored), IBM Plex fonts render (self-hosted),
      `htmx.min.js` and the SSE extension load locally

---

## 3. Consistency statements (release sign-off)

- No known Jinja template errors. Every full page and HTMX fragment is rendered
  by the automated suite (Phase 0/1b/1d in `test_web_routes.py`, `test_sse.py`,
  `test_web_coverage.py`); a template error would fail those tests.
  Confirmed at HEAD `5e93b8a`.
- No known direct-vs-HTMX divergence. Phase-1a parity tests
  (`test_web_routes.py::TestFullPageContextDivergence`) assert the full-page
  render equals the tab fragment for scenes/annotate/processing.
- Behavior matches the documented v0.3 single-film scope (Phase-5 maintainer
  decision). The sidebar reports honest global state and the inventory is not a
  fake per-film selector (`test_web_coverage.py::TestLibrary`). Multi-film is
  deferred and listed under CHANGELOG `[Não lançado]`.

---

## 4. If a human verifier finds a NEW regression

Recovery Phases 0–8 are closed and reviewed. Do not patch in this phase. File
the finding for the maintainer with: exact route/interaction, expected vs
actual, browser + JS-on/off + online/offline state, minimal repro. Continue the
rest of the checklist.
