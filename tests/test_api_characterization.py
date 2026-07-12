"""End-to-end characterization test for the public DreaMS embedding API.

Marked `slow` because it downloads the pre-trained model weights from Hugging
Face and runs a forward pass — excluded from the default CI run
(`pytest -m "not slow"`). Run it locally before/after a dependency bump (e.g. the
NumPy 2 / torch upgrade) to confirm the full pipeline still produces the same
shaped, finite, deterministic embeddings:

    uv run pytest tests/test_api_characterization.py -m slow

We deliberately assert structural + determinism invariants rather than exact
float values: the latter legitimately vary across torch/BLAS builds, so pinning
them would produce false failures. Shape/finiteness/determinism still catch a
genuinely broken pipeline.
"""
import numpy as np
import pytest

EXAMPLE_MGF = "data/examples/example_5_spectra.mgf"
N_SPECTRA = 5
EMB_DIM = 1024


@pytest.mark.slow
def test_dreams_embeddings_shape_and_finiteness():
    from dreams.api import dreams_embeddings

    embs = np.asarray(dreams_embeddings(EXAMPLE_MGF))
    assert embs.shape == (N_SPECTRA, EMB_DIM)
    assert np.issubdtype(embs.dtype, np.floating)
    assert np.isfinite(embs).all()
    # Embeddings should be non-degenerate (not all identical / all zero).
    assert embs.std() > 0


@pytest.mark.slow
def test_dreams_embeddings_deterministic():
    from dreams.api import dreams_embeddings

    a = np.asarray(dreams_embeddings(EXAMPLE_MGF))
    b = np.asarray(dreams_embeddings(EXAMPLE_MGF))
    # Model runs in eval mode (no dropout) -> repeated calls must match closely.
    assert np.allclose(a, b, atol=1e-5)


def _backbone_ckpt():
    """Path to the pre-trained DreaMS backbone, downloading it if absent."""
    import dreams.utils.misc as utils
    from dreams.definitions import PRETRAINED

    ckpt = PRETRAINED / "ssl_model.ckpt"
    if not ckpt.exists():
        utils.download_pretrained_model("ssl_model.ckpt")
    return ckpt


@pytest.mark.slow
def test_dreams_attn_scores_shape_and_normalization():
    """`dreams_attn_scores` could never run: it passed `attention_matrices=` to
    `dreams_intermediates`, whose parameter is `compute_attn_matrices`, so every
    call raised TypeError. It also indexed the result with a bogus `[1]`, which
    would have returned one spectrum's matrix instead of all of them.
    """
    from dreams.api import dreams_attn_scores

    attn = np.asarray(dreams_attn_scores(model=_backbone_ckpt(), msdata=EXAMPLE_MGF, progress_bar=False))

    # (n_spectra, n_heads, n_tokens, n_tokens) -- one matrix per spectrum, not one spectrum's.
    assert attn.ndim == 4
    assert attn.shape[0] == N_SPECTRA
    assert attn.shape[2] == attn.shape[3], "attention matrices must be square"
    assert np.isfinite(attn).all()
    # Attention is a softmax over the last axis, so every row sums to 1.
    assert np.allclose(attn.sum(axis=-1), 1.0, atol=1e-3)


@pytest.mark.slow
def test_dreams_intermediates_returns_embeddings_and_attention():
    """Both outputs together is the only path that returns a tuple."""
    from dreams.api import dreams_intermediates

    embs, attn = dreams_intermediates(
        model=_backbone_ckpt(),
        msdata=EXAMPLE_MGF,
        compute_embeddings=True,
        compute_attn_matrices=True,
        progress_bar=False,
    )
    embs, attn = np.asarray(embs), np.asarray(attn)

    assert embs.shape[0] == N_SPECTRA
    assert attn.shape[0] == N_SPECTRA
    assert np.isfinite(embs).all()
    assert np.isfinite(attn).all()
    assert embs.std() > 0
