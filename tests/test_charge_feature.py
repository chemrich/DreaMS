"""`charge_feature=True` must actually work.

It never did. Two bugs stacked:

1. `DreaMS.forward` concatenated the charge column onto `spec` *before* calling
   `__normalize_spec`, which divides by a length-2 tensor ``[max_mz, 1.]``. With
   charge appended the tensor has 3 columns, so it raised
   ``RuntimeError: The size of tensor a (3) must match the size of tensor b (2)``.
   (`ff_peak` is built with ``in_dim=token_dim``, which is 3 exactly when
   `charge_feature` is set -- so the concat was meant to feed `ff_peak`, after
   normalization, not before.)

2. The module-level `get_embeddings` computed `charge` and then passed `None` to
   the model, so `forward` hit ``if charge is None: raise ValueError``.

Neither ever fired in practice because the released DreaMS backbone ships with
``charge_feature=False``, and nothing tested the flag. `get_embeddings` is called
from `DreaMS`'s own `validation_step`, so training with `--charge_feature` would
have crashed at the first validation pass.

These build a tiny model straight from the argparse defaults -- no checkpoint, no
download -- so they run in the fast suite.
"""
import sys

import pytest
import torch

import dreams.utils.data as du
import dreams.utils.dformats as dformats
from dreams.models.dreams.dreams import DreaMS, get_embeddings
from dreams.training.train_argparse import parse_args

N_SPECTRA, N_PEAKS, D_PEAK = 4, 60, 16


def build_model(charge_feature: bool) -> DreaMS:
    sys.argv = [
        "x", "--run_name", "t", "--project_name", "t", "--job_key", "t",
        "--dataset_pth", "/dev/null", "--dformat", "A", "--batch_size", "2",
        "--train_regime", "pre-training", "--train_objective", "mask_mz", "--lr", "1e-4",
        "--no_wandb", "--n_layers", "1", "--n_heads", "2", "--d_peak", str(D_PEAK),
        "--ff_peak_depth", "1", "--ff_out_depth", "1", "--ff_fourier_depth", "1",
    ] + (["--charge_feature"] if charge_feature else [])
    args = parse_args()
    # Not exposed by argparse; the checkpoint's Namespace carries them.
    args.gains_dir = None
    args.dformat = dformats.DataFormatA()
    spec_preproc = du.SpectrumPreprocessor(dformat=dformats.DataFormatA(), n_highest_peaks=N_PEAKS)
    return DreaMS(args, spec_preproc).eval()


def batch():
    # spec is (batch, n_peaks, 2) -> [m/z, intensity]
    return {"spec": torch.rand(N_SPECTRA, N_PEAKS, 2), "charge": torch.ones(N_SPECTRA)}


@pytest.mark.parametrize("charge_feature", [True, False])
def test_get_embeddings_runs(charge_feature):
    """Regression: charge_feature=True used to raise before producing any embedding."""
    embs = get_embeddings(build_model(charge_feature), batch(), batch_size=2)
    assert embs.shape == (N_SPECTRA, D_PEAK)
    assert torch.isfinite(embs).all()


def test_forward_accepts_charge_when_flag_set():
    """The charge column must be appended after normalization, not before."""
    model, data = build_model(charge_feature=True), batch()
    out = model(data["spec"], data["charge"])
    assert out.shape[0] == N_SPECTRA
    assert torch.isfinite(out).all()


def test_forward_rejects_missing_charge_when_flag_set():
    """The flag makes charge mandatory -- that guard should stay."""
    model, data = build_model(charge_feature=True), batch()
    with pytest.raises(ValueError):
        model(data["spec"], None)


def test_charge_is_ignored_when_flag_unset():
    """With the flag off (as in the released backbone), charge must not affect the output."""
    model, data = build_model(charge_feature=False), batch()
    with torch.inference_mode():
        a = model(data["spec"], None)
        b = model(data["spec"], data["charge"] * 7)
    assert torch.equal(a, b)
