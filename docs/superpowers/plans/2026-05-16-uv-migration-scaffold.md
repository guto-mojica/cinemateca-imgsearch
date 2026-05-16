# uv Migration Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make uv the primary documented workflow for environment and dependency management without changing what gets installed or how the package is built.

**Architecture:** Pure configuration/docs change. `setuptools` build backend stays. `dev` deps move from `[project.optional-dependencies]` to a PEP 735 `[dependency-groups]` table. Functional extras (`full`/`web`/`gui`/`search`) are untouched. No `uv.lock` is committed; the heavy ML stack is never resolved this pass. Verification is structural (parse `pyproject.toml` with `tomllib`, assert shape) — no install, no `uv lock`, no `uv sync`.

**Tech Stack:** uv, PEP 735 dependency groups, setuptools (unchanged), Python 3.11 pin.

**Spec:** `docs/superpowers/specs/2026-05-16-uv-migration-scaffold-design.md`

**Spec reconciliation note:** The spec line "replace the stale `setup_uv`-branch observation in CLAUDE.md" is **void** — that observation was only ever stated in chat, never written into `CLAUDE.md`. Task 4 instead updates the CLAUDE.md command blocks and adds one migration-tracker line. No behavior in the spec is lost.

**Precondition:** All tasks run on branch `setup_uv` (already checked out). Do not push.

---

### Task 1: Move `dev` to `[dependency-groups]`, add `[tool.uv]` placeholder

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Capture the baseline structure (verification target)**

Run:
```bash
python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(sorted(d['project']['optional-dependencies'])); print(d['project']['optional-dependencies']['dev'])"
```
Expected output:
```
['dev', 'full', 'gui', 'search', 'web']
['pytest>=7.4', 'pytest-cov>=4.1', 'black>=23.0', 'ruff>=0.1', 'mypy>=1.5']
```
Record these — Task 1 must preserve the exact `dev` version list and leave the other four extras unchanged.

- [ ] **Step 2: Remove the `dev` extra from `[project.optional-dependencies]`**

In `pyproject.toml`, delete exactly this block (it is the last entry before `[project.scripts]`):

```toml
# Desenvolvimento: pip install -e ".[dev]"
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.5",
]
```

Leave the blank line / `[project.scripts]` that followed it intact. Do **not** touch `full`, `web`, `gui`, or `search`.

- [ ] **Step 3: Add the `[dependency-groups]` table**

Insert this block immediately **after** the closing of `[project.optional-dependencies]` (i.e. after the `search` extra block and before `[project.scripts]`):

```toml
# ── Dependency groups (PEP 735) ───────────────────────────────────────────────
# Dev tooling. Installed by default with `uv sync`; excluded from the wheel.
# Not a [project.optional-dependencies] extra — `pip install -e ".[dev]"`
# no longer works; use `uv sync` (uv) or install these manually under pip.
[dependency-groups]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.5",
]
```

- [ ] **Step 4: Add the minimal `[tool.uv]` placeholder**

At the end of `pyproject.toml` (after `[tool.pytest.ini_options]`), append:

```toml

# ── uv ────────────────────────────────────────────────────────────────────────
[tool.uv]
# No uv.lock is committed yet (config-only adoption — see
# docs/superpowers/specs/2026-05-16-uv-migration-scaffold-design.md).
#
# When a committed lockfile is wanted, the torch stack needs an explicit CPU
# (or CUDA) index to keep the lock reproducible and small. Fill in then:
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
```

- [ ] **Step 5: Verify structure with `tomllib`**

Run:
```bash
python -c "
import tomllib
d = tomllib.load(open('pyproject.toml','rb'))
od = d['project']['optional-dependencies']
assert sorted(od) == ['full','gui','search','web'], sorted(od)
assert 'dev' not in od, 'dev still an extra'
dg = d['dependency-groups']['dev']
assert dg == ['pytest>=7.4','pytest-cov>=4.1','black>=23.0','ruff>=0.1','mypy>=1.5'], dg
assert d['build-system']['build-backend'] == 'setuptools.build_meta'
assert 'uv' in d['tool']
print('OK: dev moved, extras intact, backend unchanged, [tool.uv] present')
"
```
Expected: `OK: dev moved, extras intact, backend unchanged, [tool.uv] present`

- [ ] **Step 6: Confirm uv accepts the manifest without resolving**

Run:
```bash
uv --version && uv tree --no-dedupe --frozen 2>&1 | head -1 || true
```
Expected: a uv version line prints. `uv tree --frozen` will report there is no lockfile — that is the intended state; treat any "no lock"/"resolution" message as expected, do **not** run `uv lock` or `uv sync` to "fix" it.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml
git commit -m "chore(deps): move dev deps to PEP 735 dependency-groups for uv

setuptools backend unchanged; functional extras untouched.
Adds commented [tool.uv] hook for deferred torch index config.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Pin Python with `.python-version`

**Files:**
- Create: `.python-version`

- [ ] **Step 1: Create the file**

Create `.python-version` with exactly this single line (no trailing blank line beyond the newline):

```
3.11
```

- [ ] **Step 2: Verify it is consistent with `requires-python`**

Run:
```bash
test "$(cat .python-version)" = "3.11" && python -c "import tomllib; assert tomllib.load(open('pyproject.toml','rb'))['project']['requires-python']=='>=3.10'; print('3.11 satisfies >=3.10')"
```
Expected: `3.11 satisfies >=3.10`

- [ ] **Step 3: Commit**

```bash
git add .python-version
git commit -m "chore: pin Python 3.11 via .python-version

Matches existing .venv (CPython 3.11); satisfies requires-python >=3.10.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Gitignore `uv.lock` with rationale

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Append the uv block**

Add to the end of `.gitignore` (after the `# IDE` block):

```
# uv — lockfile deliberately not committed (config-only uv adoption).
# Local `uv sync` creates an untracked uv.lock; that is expected.
# To start committing locks: delete the next line, run `uv lock`.
uv.lock
```

- [ ] **Step 2: Verify the pattern is active**

Run:
```bash
git check-ignore -q uv.lock && echo "uv.lock is ignored" || echo "FAIL: not ignored"
```
Expected: `uv.lock is ignored`

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore uv.lock (lockfile deferred, see spec)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Update `CLAUDE.md` commands to uv

**Files:**
- Modify: `CLAUDE.md` (Common commands block; migration tracker)

- [ ] **Step 1: Replace the one-time setup + run commands**

In the ` ```bash ` block under `## Common commands`, replace exactly:

```bash
# One-time setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[full,dev]"

# Run the FastAPI interface (v0.3.0+)
python app.py
# Opens at http://localhost:8501

# Run the legacy Streamlit interface (during migration)
streamlit run app_streamlit.py
```

with:

```bash
# One-time setup (uv — primary)
uv venv                       # creates .venv using the .python-version pin
uv sync --extra full          # full extra + dev group (auto-installed)
# Fallback without uv (no lockfile, so this still works identically):
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -e ".[full]" && pip install pytest pytest-cov black ruff mypy

# Run the FastAPI interface (v0.3.0+)
uv run app.py
# Opens at http://localhost:8501

# Run the legacy Streamlit interface (during migration)
uv run streamlit run app_streamlit.py
```

- [ ] **Step 2: Prefix the remaining command groups with `uv run`**

Replace exactly:

```bash
# Tests
pytest tests/ -q
pytest tests/test_smoke.py -v   # smoke only, no heavy models

# CLI
python -m cinemateca info --video data/raw/jeca_tatu.mp4
python -m cinemateca process --video data/raw/jeca_tatu.mp4
python -m cinemateca process --video data/raw/jeca_tatu.mp4 --steps scenes,embeddings

# i18n
pybabel extract -F web/babel.cfg -o web/locales/messages.pot web/
pybabel update -i web/locales/messages.pot -d web/locales/
pybabel compile -d web/locales/

# Lint / format / typecheck (run before committing)
black .
ruff check .
mypy src
```

with:

```bash
# Tests
uv run pytest tests/ -q
uv run pytest tests/test_smoke.py -v   # smoke only, no heavy models

# CLI
uv run python -m cinemateca info --video data/raw/jeca_tatu.mp4
uv run python -m cinemateca process --video data/raw/jeca_tatu.mp4
uv run python -m cinemateca process --video data/raw/jeca_tatu.mp4 --steps scenes,embeddings

# i18n
uv run pybabel extract -F web/babel.cfg -o web/locales/messages.pot web/
uv run pybabel update -i web/locales/messages.pot -d web/locales/
uv run pybabel compile -d web/locales/

# Lint / format / typecheck (run before committing)
uv run black .
uv run ruff check .
uv run mypy src
```

- [ ] **Step 3: Add a migration-tracker line**

In the `## v0.2.1 → v0.3.0 migration tracker` checklist, add this line immediately after the `[x] Streamlit parity confirmed ...` line:

```
- [x] uv adopted for env/deps (config-only, lockfile deferred — see docs/superpowers/specs/2026-05-16-uv-migration-scaffold-design.md)
```

- [ ] **Step 4: Verify no stale pip-only setup remains**

Run:
```bash
grep -n 'pip install -e' CLAUDE.md
```
Expected: exactly one match, and it is on the commented `#   pip install -e ".[full]" ...` fallback line inside the setup block. No uncommented `pip install` setup line.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: switch CLAUDE.md commands to uv, note adoption in tracker

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Update `SETUP.md` install/run/update commands to uv

**Files:**
- Modify: `SETUP.md` (sections 3, 4, 11)

Scope guard: only the command mechanics change. Do **not** rewrite the Streamlit-vs-FastAPI prose, the model-size tables, or section numbering — that is pre-existing migration debt, out of scope here.

- [ ] **Step 1: Replace Section 3 venv body**

Replace exactly the fenced block in `## 3. Ambiente Python` that begins `# Criar ambiente virtual` and ends with the `(.venv) usuario@maquina...` comment:

```bash
# Criar ambiente virtual (faça isso UMA vez, dentro da pasta do projeto)
python3 -m venv .venv

# Ativar o ambiente
# macOS / Linux:
source .venv/bin/activate

# Você verá (.venv) aparecer no início do seu prompt:
# (.venv) usuario@maquina:~/cinemateca-ai$
```

with:

```bash
# uv cria e gerencia o ambiente automaticamente (faça isso UMA vez):
uv venv
# uv usa a versão fixada em .python-version (3.11). Se não tiver o
# Python 3.11, o uv baixa automaticamente.

# Não é necessário "ativar" o ambiente: use `uv run <comando>`.
# Se preferir ativar manualmente mesmo assim:
# macOS / Linux:
source .venv/bin/activate
```

- [ ] **Step 2: Replace Section 4 install command**

Replace exactly:

```bash
# Instalar o pacote em modo desenvolvimento + todas as dependências
pip install -e ".[full]"
```

with:

```bash
# Instalar o pacote + todas as dependências (extra "full") + ferramentas
# de desenvolvimento (grupo "dev", instalado automaticamente pelo uv):
uv sync --extra full

# Sem uv (alternativa — funciona porque não há lockfile):
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -e ".[full]"
```

- [ ] **Step 3: Replace Section 4 partial-install and verify commands**

Replace exactly:

```bash
pip install -e ".[search]"    # só CLIP e busca
pip install -e ".[gui]"       # só Streamlit
```

with:

```bash
uv sync --extra search    # só CLIP e busca
uv sync --extra gui       # só Streamlit
```

Then replace exactly:

```bash
python -c "import cinemateca; print(cinemateca.__version__)"
# Deve imprimir: 0.1.0-alpha
```

with:

```bash
uv run python -c "import cinemateca; print(cinemateca.__version__)"
# Deve imprimir: 0.1.0-alpha
```

- [ ] **Step 4: Replace Section 11 update commands**

Replace exactly:

```bash
# Dentro da pasta do projeto, com .venv ativo:
git pull origin main
pip install -e ".[full]"   # reinstalar caso pyproject.toml tenha mudado
```

with:

```bash
# Dentro da pasta do projeto:
git pull origin main
uv sync --extra full   # re-sincroniza caso pyproject.toml tenha mudado
```

- [ ] **Step 5: Verify no uncommented pip-only setup remains in SETUP.md**

Run:
```bash
grep -n 'pip install -e' SETUP.md
```
Expected: matches appear **only** on commented fallback lines (lines whose first non-space char is `#`) and the troubleshooting block in Section 10 (left intentionally as a recovery hint). No uncommented `pip install -e` in Sections 3, 4, or 11.

Run:
```bash
grep -n 'uv venv\|uv sync\|uv run' SETUP.md | head
```
Expected: at least the new commands from Steps 1–4 appear.

- [ ] **Step 6: Commit**

```bash
git add SETUP.md
git commit -m "docs(setup): switch install/run/update commands to uv

pip fallback kept inline (no lockfile). Streamlit/FastAPI prose
untouched — pre-existing migration debt, out of scope.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Changelog entry + final cross-file verification

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the changelog entry**

In `CHANGELOG.md`, under `## [Não lançado]`, immediately after the planned-features bullet list and before the `---` that closes the section, insert:

```markdown

### Ferramentas de desenvolvimento

- Adoção do **uv** para gerenciamento de ambiente e dependências.
  Dependências de desenvolvimento movidas para `[dependency-groups]`
  (PEP 735). Backend de build (`setuptools`) inalterado. Lockfile
  (`uv.lock`) adiado — instalação via `pip` continua funcionando.
```

- [ ] **Step 2: Verify the changelog still parses as the same section structure**

Run:
```bash
grep -n '^## \[Não lançado\]\|^### Ferramentas de desenvolvimento\|^## \[0.1.0-alpha\]' CHANGELOG.md
```
Expected: three lines, in this order — `## [Não lançado]`, then `### Ferramentas de desenvolvimento`, then `## [0.1.0-alpha] — 2025`.

- [ ] **Step 3: Full cross-file consistency check**

Run:
```bash
python -c "
import tomllib
d = tomllib.load(open('pyproject.toml','rb'))
assert 'dev' not in d['project']['optional-dependencies']
assert d['dependency-groups']['dev'][0] == 'pytest>=7.4'
assert d['build-system']['build-backend'] == 'setuptools.build_meta'
print('pyproject OK')
"
test "$(cat .python-version)" = "3.11" && echo ".python-version OK"
git check-ignore -q uv.lock && echo ".gitignore OK"
grep -q 'uv sync --extra full' CLAUDE.md && echo "CLAUDE.md OK"
grep -q 'uv sync --extra full' SETUP.md && echo "SETUP.md OK"
grep -q 'Ferramentas de desenvolvimento' CHANGELOG.md && echo "CHANGELOG OK"
```
Expected: `pyproject OK`, `.python-version OK`, `.gitignore OK`, `CLAUDE.md OK`, `SETUP.md OK`, `CHANGELOG OK` — all six lines.

- [ ] **Step 4: Confirm no protected paths were touched**

Run:
```bash
git diff --name-only origin/setup_uv...HEAD 2>/dev/null || git log --name-only --oneline -8
```
Expected: only `pyproject.toml`, `.python-version`, `.gitignore`, `CLAUDE.md`, `SETUP.md`, `CHANGELOG.md`, and the two `docs/superpowers/` files appear. **No** path under `data/`, `models/`, `config/local.yaml`, or `src/cinemateca/`.

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record uv adoption (lockfile deferred)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Verification Summary

| Spec requirement | Task |
|---|---|
| `setuptools` backend unchanged | Task 1 Step 5 (asserts build-backend) |
| `dev` → `[dependency-groups]` | Task 1 Steps 2–3, 5 |
| Functional extras untouched | Task 1 Step 5 (asserts `['full','gui','search','web']`) |
| `[tool.uv]` deferred-torch placeholder | Task 1 Step 4 |
| `.python-version` = 3.11 | Task 2 |
| `uv.lock` gitignored w/ rationale | Task 3 |
| Docs uv-primary, pip fallback kept | Tasks 4, 5 |
| No `uv.lock` / no env build / no resolution | No task runs `uv lock`/`uv sync`/install; Task 1 Step 6 explicitly forbids it |
| No version bumps (relocated only) | Task 1 Steps 1, 5 assert identical version list |
| `full` extra streamlit/rich untouched | Task 1 Step 2 scope guard |
| No protected paths touched | Task 6 Step 4 |
| CHANGELOG entry | Task 6 |

## Post-implementation

After all tasks: this is feature-branch work on `setup_uv`. Do **not** push or merge — surface completion to the user and let them decide. The deferred lockfile + torch-index config is the documented next step (in `[tool.uv]` and the spec) when they want reproducible locks.
