# uv Migration Scaffold — Design

**Date:** 2026-05-16
**Branch:** `setup_uv`
**Status:** Approved, ready for implementation plan.
**Scope:** Configuration-only adoption of [uv](https://docs.astral.sh/uv/) for
environment and dependency management. No lockfile, no environment rebuild,
no dependency version changes.

---

## Goal

Make uv the primary, documented workflow for creating the dev environment and
running the project, without changing what gets installed or how the package is
built. This is a mechanics change, not a dependency change.

## Decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Deliverable depth | Config only — no committed `uv.lock` | User deferred lockfile to avoid resolving/downloading the ~1 GB torch stack now. |
| Build backend | `setuptools` unchanged | uv drives a setuptools project natively; no backend swap means no packaging-behavior risk. |
| `dev` dependencies | Move to `[dependency-groups]` (PEP 735) | uv-idiomatic; `uv sync` installs it by default; excluded from the wheel. |
| Functional extras | `full`/`web`/`gui`/`search` stay as `[project.optional-dependencies]` | They are install-target selectors users actively choose. |
| Python pin | `.python-version` = `3.11` | Matches existing `.venv` (CPython 3.11) and satisfies `requires-python = ">=3.10"`. |
| `uv.lock` | gitignored with explanatory comment | Self-documents the "deliberately unlocked" state; reversible by deleting one line + `uv lock`. |

## Changes

### `pyproject.toml`
- Build system unchanged (`setuptools>=68`, `wheel`,
  `[tool.setuptools.packages.find] where=["src"]`).
- Add top-level `[dependency-groups]` table with `dev` (pytest, pytest-cov,
  black, ruff, mypy) — versions copied byte-identical from the current
  `[project.optional-dependencies].dev`.
- Remove `dev` from `[project.optional-dependencies]`. `full`, `web`, `gui`,
  `search` untouched.
- Add a minimal `[tool.uv]` block containing only a commented placeholder
  documenting that torch CPU/CUDA index pinning
  (`[tool.uv.sources]` / `[[tool.uv.index]]`) is intentionally deferred until a
  lockfile is wanted. Gives the next person the exact hook to fill in.
- `requires-python` stays `>=3.10`.

### `.python-version` (new)
Single line: `3.11`.

### `.gitignore`
Append `uv.lock` with a comment explaining it is deliberately unlocked for the
config-only pass. Local `uv sync` will still create an untracked lock — expected.

### `SETUP.md` and `CLAUDE.md`
uv becomes the primary path; one-line pip fallback preserved (no lockfile means
`pip install -e ".[...]"` still works identically).

Command mapping:

| Old | New |
|---|---|
| `python3 -m venv .venv && source .venv/bin/activate` | `uv venv` |
| `pip install -e ".[full,dev]"` | `uv sync --extra full` (dev group auto-installed) |
| `python app.py` | `uv run app.py` |
| `python -m cinemateca process ...` | `uv run python -m cinemateca process ...` |
| `pytest tests/ -q` | `uv run pytest tests/ -q` |
| `black .` / `ruff check .` / `mypy src` | `uv run black .` / `uv run ruff check .` / `uv run mypy src` |
| `pybabel ...` | `uv run pybabel ...` |

In `CLAUDE.md`, also replace the stale `setup_uv`-branch observation with the
real completed state.

### `CHANGELOG.md`
One entry: `chore: adopt uv for environment and dependency management
(lockfile deferred)`.

## Explicitly out of scope

- No `uv.lock` committed; no `uv sync`/env rebuild; no torch index resolution.
- No build backend change.
- No dependency version bumps (CLAUDE.md's "ask before version bumps" rule is
  satisfied — versions are relocated, not changed).
- The `full` extra still bundles legacy `streamlit` + `rich`. Pre-existing,
  untouched — not tidied in a uv-mechanics pass.

## Success criteria

- `pyproject.toml` parses; `uv sync --extra full` resolves the same dependency
  set as the prior `pip install -e ".[full,dev]"` (verified by inspection, not
  by running a full install this pass).
- `pip install -e ".[full]"` still works (backward-compat fallback intact).
- Docs contain no remaining pip-only setup instructions except the explicit
  fallback note.
- No file under `data/`, `models/`, or `config/local.yaml` touched.
