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
