# CLAUDE.md — DreaMS (chemrich fork)

Guidance for Claude Code working in this repo. This is a **fork** of
[`pluskal-lab/DreaMS`](https://github.com/pluskal-lab/DreaMS) at
`chemrich/DreaMS` (`origin`), modernized off the original research code.
`upstream` is `pluskal-lab/DreaMS`.

## Platform: modern Linux x86_64 ONLY

This project runs only on modern Linux and **never on macOS**. `[tool.uv]
environments` pins resolution to Linux, so `uv sync` intentionally **refuses to
run on macOS/Windows**. Do not re-add cross-platform (macOS) accommodations.

## Package management: uv (not pip/conda)

uv is the only supported workflow; conda has been abandoned.

```bash
uv sync --extra dev        # create/refresh .venv from the committed uv.lock
uv run pytest -m "not slow" # run the fast test suite
uv run python ...          # run anything inside the env
uv lock                    # re-resolve after editing deps; commit uv.lock
```

Python 3.13/3.14 are the targets — uv downloads the interpreter itself, so the
system Python (Mint ships 3.10/3.12) is irrelevant; don't use it.

## Linting & type-checking: run ISOLATED

ruff and mypy must run as **isolated tools** (no project deps installed),
because a full env makes mypy load rdkit's bundled stubs, which contain a syntax
error it can't parse. CI and the pre-commit hook both do this:

```bash
uv tool run ruff@0.15.21 check .   # NOT `uv run ruff`
uv tool run mypy@2.2.0             # NOT `uv run mypy`
```

Install hooks once with `uv tool install pre-commit && pre-commit install`.

### mypy is gradual-typing with a quarantine

`[tool.mypy]` type-checks the whole `dreams` package except a shrinking
`ignore_errors` quarantine list of legacy modules (see the overrides block).
"Making progress" = fixing a quarantined module's errors and removing it from
the list. Get a module's real errors with:
`uv tool run mypy@2.2.0 --config-file=/dev/null --ignore-missing-imports --follow-imports=silent <file.py>`.

## Testing

- **Fast tests** (default CI): `uv run pytest -m "not slow"`.
- **Slow tests** (`@pytest.mark.slow`): download model weights / datasets from
  HuggingFace; excluded from CI. Run the end-to-end model check locally with
  `uv run pytest -m slow`. `test_api_characterization.py` is the real safety net
  for the flagship `dreams_embeddings` API — run it after any dependency bump.
- Characterization tests hold **golden values**; a diff means behavior changed
  and must be reviewed, not blindly re-baselined.

## Load-bearing dependency facts (don't regress these)

- **`msml`** git dep points to our fork
  `chemrich/dreams_legacy_architectures` (not upstream). It's needed only to
  unpickle the pretrained checkpoint, and our fork patches the same matchms API
  drift. Keep it pinned to an immutable commit.
- **torch**: checkpoints must load with `weights_only=False` (torch 2.6 flipped
  the default; the DreaMS checkpoint pickles Path/Namespace globals). See the 4
  `load_from_checkpoint` sites in `api.py`/`heads.py`.
- **matchms** ≥0.30 renamed `ModifiedCosine` → `ModifiedCosineGreedy`.
- **HDF5 string columns**: pandas 2 uses pyarrow-backed string arrays; write them
  with `h5py.string_dtype()` and coerce via `[str(x) for x in v]` (see
  `MSData.from_pandas` in `utils/data.py`).
- **Removed from deps**: `molplotly` (pulls dead `rdkit-pypi`) and
  `SpectralEntropy` (unused, unbuildable) — both notebook-only. Don't re-add to core.

## Branches

- `main` — CPU torch, the general/CI branch.
- `gpu-cuda` — CUDA 12.6 torch for Linux x86_64 GPU VMs; pinned to Python 3.13
  (no CUDA cp314 wheels). Standing branch, not meant to merge into `main`. See
  `GPU_TRAINING.md`. Needs a real NVIDIA GPU (`nvidia-smi`); validate with
  `dreams/training/smoke_test.sh`.

## CI & conventions

- CI (`.github/workflows/ci.yml`): `ubuntu-latest` × {3.13, 3.14}; lint job runs
  ruff+mypy isolated; test jobs `uv sync --locked` + `uv run pytest -m "not slow"`.
- CI only runs on `push` to `main` and on PRs — push a feature branch and open a
  PR to trigger it.
- End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
