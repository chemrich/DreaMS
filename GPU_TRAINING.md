# GPU / CUDA training branch

This branch (`gpu-cuda`) configures DreaMS for GPU training on a Linux VM. It
differs from `main` only in the packaging/training setup — no library code
changes — so it can be rebased on `main` as the project evolves.

**What's different from `main`:**
- `torch` is pulled as the **CUDA 12.6** build (`[tool.uv.sources]` → the `cu126`
  PyTorch index) instead of `main`'s CPU build. Both branches are Linux x86_64 only.
- `requires-python` is pinned to **3.13** (PyTorch has no CUDA `cp314` wheels yet).
- `uv.lock` is regenerated against the CUDA index.
- `fine_tune.sh` uses `--train_precision 32` (see note below).
- Adds `dreams/training/smoke_test.sh` — a fast single-GPU end-to-end check.

## VM sizing

The model is small (**96.6M params**); the paper's 8-GPU rig was for wall-clock,
not capacity.

| Use case | GPU | RAM | Disk |
|---|---|---|---|
| Fine-tune / experiment | 1× A100 40GB (or L4/A10 24GB) | 64 GB | 100 GB SSD |
| Single-GPU pre-train | 1× A100 80GB | 128 GB | 200 GB NVMe |
| Paper-scale pre-train | 4–8× A100 80GB | 256 GB+ | 500 GB+ NVMe |

Data: pre-training reads `GeMS_A10.hdf5` (14.6 GB); the full GeMS is 528 GB.
Smaller subsets exist for quick runs (`GeMS_A1_rand50k` = 221 MB, `rand5M` = 22 GB).

## Setup on a fresh GPU VM

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone this branch and sync the locked CUDA environment
git clone -b gpu-cuda https://github.com/chemrich/DreaMS.git
cd DreaMS
uv sync                      # installs torch 2.13.0+cu126 + CUDA runtime libs

# 3. Verify the GPU is visible (must print a device name, not an error)
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"

# 4. Fast end-to-end smoke test (downloads a 221 MB subset, ~few min on 1 GPU)
bash dreams/training/smoke_test.sh
```

Then run the real jobs with `dreams/training/pre_train.sh` / `fine_tune.sh`
(SLURM directives are at the top; drop them and use `bash` for a plain VM).

## Notes & gotchas

- **CUDA toolkit**: this branch targets CUDA 12.6. For a different toolkit change
  `cu126` → `cu128`/`cu129` in the `[[tool.uv.index]]` url in `pyproject.toml`,
  then `uv lock`. Your NVIDIA **driver** must support the chosen toolkit
  (`nvidia-smi` shows the max CUDA version the driver allows).
- **Precision**: `fine_tune.sh` was `--train_precision 64` in the paper. fp64 is
  very slow on GPUs without strong FP64 units (L4, RTX 40xx). It's set to `32`
  here; revert to `64` on A100/H100 to reproduce the paper exactly.
- **Checkpoints**: `pre_train.sh` uses `--save_top_k -1` (saves *every* epoch's
  checkpoint, ~386 MB each). Lower it or size your disk accordingly.
- **W&B**: the training scripts log to Weights & Biases by default. `smoke_test.sh`
  passes `--no_wandb`; add it to the other scripts, or `wandb login`, as needed.
