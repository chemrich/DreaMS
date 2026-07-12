# CLAUDE.md — DreaMS (chemrich fork)

Guidance for Claude Code working in this repo. This is a **fork** of
[`pluskal-lab/DreaMS`](https://github.com/pluskal-lab/DreaMS) at
`chemrich/DreaMS` (`origin`), modernized off the original research code.
`upstream` is `pluskal-lab/DreaMS`.

## ALL work targets `chemrich/DreaMS` — never `pluskal-lab/DreaMS`

Push, open PRs, and file issues **only** against the `chemrich/DreaMS` fork
(`origin`). `upstream` is fetch-only: we track it, we never write to it.

Because this repo is a *fork*, `gh` targets the **parent** (`pluskal-lab`) by
default, and a PR against upstream fails with a misleading
`Resource not accessible by personal access token` — it looks like a token-scope
problem but is really a wrong-repo problem. Two guards are in place; keep them:

```bash
gh repo set-default chemrich/DreaMS   # persisted as remote.origin.gh-resolved=base
git remote set-url --push upstream DISABLED_use_origin_chemrich  # push to upstream fails
```

A fresh clone has **neither** guard — re-apply both. When in doubt, pass
`--repo chemrich/DreaMS` explicitly to every `gh` command.

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

### mypy: the quarantine is gone — keep it that way

`[tool.mypy]` type-checks the **whole** `dreams` and `experiments` tree (52
files) with no `ignore_errors` overrides. The old quarantine (13 modules → 0) is
fully retired. **Do not re-add an overrides block** to silence a new error — fix
the error.

`check_untyped_defs = true` is ON and must stay on. mypy **skips the bodies of
unannotated functions by default**, and this codebase has ~500 of them — so
without this flag most of the code is not actually checked, and "all files clean"
is illusory. `disallow_untyped_defs` is deliberately **off**: it only *demands*
506 annotations and finds no bugs by itself.

Get a single module's errors in isolation with:
`uv tool run mypy@2.2.0 --config-file=/dev/null --ignore-missing-imports --follow-imports=silent <file.py>`.

### ruff: never re-add a wildcard import

`F821` (undefined name) and `F841` (unused local) are enforced. They used to sit
in the ignore list — not because they were noisy, but because `from x import *`
defeats ruff's name resolution: with a star import present ruff downgrades every
unresolvable name to `F405` ("may be undefined"), which **silently disarms
F821**. The rules were disabled exactly where they mattered. All wildcard imports
in `dreams/`, `experiments/*.py` and `tutorials/` are gone — do not bring one
back. (`experiments/**/*.ipynb` are research artifacts and still exempt from
F403/F405/F821; see `[tool.ruff.lint.per-file-ignores]`.)

`experiments/` and all 38 notebooks are linted. Both were excluded as "not
library code", which had been hiding a plain syntax error in
`mol_props/baselines.ipynb`.

### The bug shape this codebase keeps producing

Arming the linters surfaced **nine** latent bugs that no test caught. Almost all
are one of two shapes — **a caller passing a keyword the callee doesn't accept**,
or **an operation on a `str`/`None`/wrong-shaped value**. Several were functions
that could *never* have run:

- `IntRegressionHead` (`backbone_pth=` vs `backbone`) — never constructible.
- `dreams_attn_scores` (`attention_matrices=` vs `compute_attn_matrices`).
- `DeepSetsPeaksFingerprint(args.train_objective, lr=...)` — duplicate `lr`.
- `MSData.get_formulas()` — no such method.
- **`charge_feature=True` never worked**: `forward()` appended the charge column
  *before* `__normalize_spec` (which divides by a length-2 tensor), and
  `get_embeddings()` discarded `charge` and passed `None`.
- `mols.py` used `urllib.request` behind a bare `import urllib`.
- `DreaMSAtlas(nist20=True)` did `None / str`.
- `construct_knn.py` read `embs[i]` after `del embs`.

**Suspect this shape first** when something "has never been run". The cheap guard
is `tests/test_heads_construction.py`: it builds every head against a stub
backbone with no weights, in CI.

## Testing

- **Fast tests** (default CI): `uv run pytest -m "not slow"`.
- **Slow tests** (`@pytest.mark.slow`): download model weights / datasets from
  HuggingFace; excluded from CI. Run the end-to-end model check locally with
  `uv run pytest -m slow`. `test_api_characterization.py` is the real safety net
  for the flagship `dreams_embeddings` API — run it after any dependency bump.
- Characterization tests hold **golden values**; a diff means behavior changed
  and must be reviewed, not blindly re-baselined.

### If tests die with `Fatal Python error: Illegal instruction`, check for AVX

torch's official wheels are MKL-backed, and MKL's VML (which implements
`torch.cos`/`torch.sin`, hit on every forward pass via `FourierFeatures`) emits
**VEX-encoded AVX instructions**. On a CPU without AVX the process takes an
illegal-opcode trap and dies with SIGILL (exit 132) — *intermittently*, roughly
1 run in 5, because MKL's kernel choice depends on buffer alignment.

This is an **environment problem, not a code bug** — CI is green because GitHub
runners have real AVX2 CPUs. Diagnose with:

```bash
grep -w avx2 /proc/cpuinfo || echo "no AVX2 — expect SIGILL"
dmesg | grep 'trap invalid opcode'   # will name libtorch_cpu.so
```

The usual cause is a VM with an emulated CPU model that masks host features
(e.g. `QEMU Virtual CPU version 2.5+`). Fix it on the **hypervisor**, not in
code: set the CPU model to host-passthrough (libvirt/virt-manager: "Copy host
CPU configuration"; Proxmox: CPU type `host`; QEMU: `-cpu host`). Do not add
code workarounds for this. Note that AVX2/FMA also make torch far faster, so a
masked CPU is a large silent performance tax.

**Verify the hypervisor fix landed** (the dev VM was reconfigured for this on
2026-07-12):

```bash
grep -w avx2 /proc/cpuinfo >/dev/null && echo "AVX2 present — good" || echo "still masked"
lscpu | grep 'Model name'        # should no longer say "QEMU Virtual CPU"
for i in 1 2 3 4 5; do uv run pytest -m slow -q | tail -1; done   # 5/5 clean
```

Five consecutive clean slow runs is the bar: the crash was ~1-in-5, so a single
green run proves nothing.

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
