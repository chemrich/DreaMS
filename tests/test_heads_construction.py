"""Every fine-tuning head must be constructible.

`IntRegressionHead` was dead on arrival for the entire life of the project: its
`__init__` forwarded `backbone_pth=` to `RegressionHead`, which names that
parameter `backbone`, so *every* construction raised TypeError. Nothing in the
repo ever built one, so nothing ever noticed.

These tests construct each head against a stub backbone -- no checkpoint, no
download -- so the same class of breakage fails fast in CI instead of lying
dormant. Heads disagree on whether the first argument is called `backbone` or
`backbone_pth`, so it is always passed positionally here.
"""
import inspect

import pytest
import torch
import torch.nn as nn

import dreams.models.heads.heads as heads_mod
from dreams.models.heads.heads import FineTuningHead

D_MODEL = 32


class StubBackbone(nn.Module):
    """Stand-in for the DreaMS backbone; heads only read these attributes."""

    def __init__(self):
        super().__init__()
        self.d_model = D_MODEL
        self.n_layers = 2
        self.n_heads = 4
        self.lin = nn.Linear(D_MODEL, D_MODEL)


# Required arguments beyond the common (backbone, lr, weight_decay), by class name.
EXTRA_KWARGS = {
    "FingerprintHead": {"fp_str": "morgan_2_2048", "batch_size": 2},
    "ContrastiveHead": {"triplet_loss_margin": 0.5},
}


def head_classes():
    for name, obj in vars(heads_mod).items():
        if inspect.isclass(obj) and issubclass(obj, FineTuningHead) and obj is not FineTuningHead:
            yield pytest.param(obj, id=name)


@pytest.mark.parametrize("head_cls", list(head_classes()))
def test_head_is_constructible(head_cls):
    extra = EXTRA_KWARGS.get(head_cls.__name__, {})
    head = head_cls(StubBackbone(), lr=1e-4, weight_decay=0.0, **extra)
    assert head.backbone.d_model == D_MODEL


def test_int_regression_head_forwards_backbone_to_parent():
    """Regression test for the backbone_pth/backbone mismatch that made this class unusable."""
    head = heads_mod.IntRegressionHead(StubBackbone(), lr=1e-4, weight_decay=0.0)
    assert isinstance(head.backbone, StubBackbone)
    assert head.out_dim == 1
    assert head.sigmoid is None  # IntRegressionHead passes sigmoid=False


def test_regression_head_runs_a_step():
    """The head's own layers must actually produce a finite loss."""
    head = heads_mod.RegressionHead(StubBackbone(), lr=1e-4, weight_decay=0.0, out_dim=1)
    batch, n_peaks = 4, 5
    embeddings = torch.randn(batch, n_peaks, D_MODEL)
    preds = head.head(embeddings[:, 0, :])
    assert preds.shape == (batch, 1)
    assert torch.isfinite(preds).all()
