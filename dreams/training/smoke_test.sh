#!/bin/bash
# Single-GPU smoke test for the GPU/CUDA branch.
#
# Validates the whole training path end-to-end on a GPU VM using the small
# GeMS_A1 rand50k subset (~221 MB, 50k spectra) — one epoch, one device, no
# Weights & Biases account required. Finishes in a few minutes on an A100/L4.
# If this runs clean, `pre_train.sh` / `fine_tune.sh` will too.
#
# Usage (from repo root, after `uv sync`):
#   bash dreams/training/smoke_test.sh
set -euo pipefail

# Sanity check: make sure torch actually sees CUDA (not the CPU wheel).
uv run python -c "import torch; assert torch.cuda.is_available(), \
'CUDA not available — is this a GPU host, and did uv install the +cu126 torch? \
See the GPU note in README / pyproject [tool.uv.sources].'; \
print('CUDA OK:', torch.cuda.get_device_name(0))"

# Export project definitions (DREAMS_DIR, GEMS_DIR, ...).
$(uv run python -c "from dreams.definitions import export; export()")

# Download the 50k-spectra subset (idempotent; cached by huggingface_hub).
DATASET=$(uv run python -c \
  "from dreams.utils.misc import gems_hf_download; print(gems_hf_download('GeMS_A/GeMS_A1_DreaMS_rand50k.hdf5'))")
echo "Smoke-test dataset: ${DATASET}"

cd "${DREAMS_DIR}" || exit 3

# Same architecture as pre_train.sh; only the run size differs (1 device, 1 epoch,
# smaller batch, no wandb). Keeping the full arg set avoids missing required flags.
uv run python3 training/train.py \
 --no_wandb \
 --project_name DreaMS_smoke \
 --job_key smoke \
 --run_name smoke \
 --frac_masks 0.3 \
 --train_regime pre-training \
 --dataset_pth "${DATASET}" \
 --val_check_interval 1.0 \
 --train_objective mask_mz_hot \
 --hot_mz_bin_size 0.05 \
 --dformat A \
 --model DreaMS \
 --ff_peak_depth 1 \
 --ff_fourier_depth 5 \
 --ff_fourier_d 512 \
 --ff_out_depth 1 \
 --prec_intens 1.1 \
 --num_devices 1 \
 --max_epochs 1 \
 --log_every_n_steps 20 \
 --seed 3402 \
 --n_layers 7 \
 --n_heads 8 \
 --d_peak 44 \
 --d_fourier 980 \
 --lr 1e-4 \
 --batch_size 64 \
 --dropout 0.1 \
 --save_top_k 1 \
 --att_dropout 0.1 \
 --residual_dropout 0.1 \
 --ff_dropout 0.1 \
 --weight_decay 0 \
 --attn_mech dot-product \
 --train_precision 32 \
 --mask_peaks \
 --mask_intens_strategy intens_p \
 --max_peaks_n 60 \
 --ssl_probing_depth 0 \
 --focal_loss_gamma 5 \
 --no_transformer_bias \
 --n_warmup_steps 5000 \
 --fourier_strategy lin_float_int \
 --mz_shift_aug_p 0.2 \
 --mz_shift_aug_max 50 \
 --pre_norm \
 --graphormer_mz_diffs \
 --ret_order_loss_w 0.2

echo "Smoke test finished OK."
