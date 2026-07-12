"""k-NN graph construction for the DreaMS Atlas (experiments/dreams_atlas/construct_knn.py).

SKIPPED in CI, and that is expected: `ngt` publishes no cp314 wheel and no sdist, so it
cannot be installed on this project's Python (>=3.13, venv is 3.14). It is also not a
declared dependency — the script only ever ran on the authors' HPC cluster. The test still
earns its place: it runs for anyone who does have ngt (py<=3.13), and it pins the bug that
made this script silently wrong, namely that `ngt_index.get_object(i)` returns corrupt
vectors on an object_type='Float16' index.
"""
import importlib.util
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ngtpy = pytest.importorskip("ngtpy", reason="ngt has no cp314 wheel; not a project dep")

D, K = 16, 3
N_LIB, N_CHUNKS, N_PER_CHUNK = 12, 3, 8
BLANK_ROWS = (1, 4)  # rows named blank_*/wash_*, which load_gems_embs must drop


def _load_construct_knn():
    pth = Path(__file__).parent.parent / "experiments" / "dreams_atlas" / "construct_knn.py"
    spec = importlib.util.spec_from_file_location("construct_knn", pth)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["construct_knn"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def synthetic_embs(tmp_path):
    """Library file + GeMS chunks, mirroring the real HDF5 layout."""
    rng = np.random.default_rng(7)

    lib = rng.normal(size=(N_LIB, D)).astype(np.float32)
    lib_pth = tmp_path / "lib.hdf5"
    with h5py.File(lib_pth, "w") as f:
        f.create_dataset("DreaMS_embedding", data=lib)

    gems_pths, kept = [], []
    for c in range(N_CHUNKS):
        embs = rng.normal(size=(N_PER_CHUNK, D)).astype(np.float32)
        names = [f"sample_{c}_{r}" for r in range(N_PER_CHUNK)]
        names[BLANK_ROWS[0]] = f"blank_{c}"
        names[BLANK_ROWS[1]] = f"wash_{c}"
        p = tmp_path / f"msvn_C_H1000_KK1.{c}.hdf5"
        with h5py.File(p, "w") as f:
            f.create_dataset("DreaMS_Embedding", data=embs)
            f.create_dataset("name", data=[n.encode() for n in names])
        gems_pths.append(p)
        kept.append(embs[[r for r in range(N_PER_CHUNK) if r not in BLANK_ROWS]])

    # Object id i in the index == i-th vector here, in this order.
    expected = np.concatenate(
        [lib.astype(np.float16)] + [g.astype(np.float16) for g in kept]
    ).astype(np.float32)
    return lib_pth, gems_pths, expected


def _build_index(tmp_path, ck, lib_pth, gems_pths, logger):
    idx_pth = str(tmp_path / "idx")
    ngtpy.create(idx_pth, dimension=D, distance_type="Cosine", object_type="Float16",
                 edge_size_for_creation=30, edge_size_for_search=60)
    index = ngtpy.Index(idx_pth)
    for chunk in ck.iter_embs(lib_pth, gems_pths, logger):
        index.batch_insert(chunk)
        index.save()
    return index


def test_iter_embs_matches_insertion_order(synthetic_embs):
    """The query pass must re-yield exactly what the insert pass inserted, in order."""
    ck = _load_construct_knn()
    lib_pth, gems_pths, expected = synthetic_embs
    streamed = np.concatenate(
        list(ck.iter_embs(lib_pth, gems_pths, logging.getLogger("t")))
    ).astype(np.float32)
    assert streamed.shape == expected.shape  # blanks dropped
    assert np.array_equal(streamed, expected)


def test_get_object_is_unusable_on_float16_index(synthetic_embs, tmp_path):
    """Regression pin: the reason build_knn_graph must not call get_object().

    On object_type='Float16', get_object returns corrupt (often all-zero) vectors, so
    searching with them yields garbage or nothing at all. If a future ngt release fixes
    this, this test fails and the workaround can be revisited.
    """
    ck = _load_construct_knn()
    lib_pth, gems_pths, expected = synthetic_embs
    index = _build_index(tmp_path, ck, lib_pth, gems_pths, logging.getLogger("t"))

    corrupt = 0
    for i in range(index.get_num_of_objects()):
        got = np.array(index.get_object(i), dtype=np.float32)
        want = expected[i] / np.linalg.norm(expected[i])  # Cosine => stored normalised
        if not np.allclose(got, want, atol=1e-2):
            corrupt += 1
    assert corrupt > 0, "get_object() now round-trips Float16 — the workaround may be droppable"


def test_build_knn_graph_matches_brute_force(synthetic_embs, tmp_path):
    ck = _load_construct_knn()
    lib_pth, gems_pths, expected = synthetic_embs
    logger = logging.getLogger("t")
    index = _build_index(tmp_path, ck, lib_pth, gems_pths, logger)

    n = index.get_num_of_objects()
    assert n == len(expected)

    knn_i, knn_j, knn_w = ck.build_knn_graph(
        index, ck.iter_embs(lib_pth, gems_pths, logger), K, n
    )
    assert knn_i.shape == knn_j.shape == knn_w.shape == (n * K,)

    # Brute-force cosine k-NN, self excluded.
    x = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    truth = np.argsort(-sim, axis=1)[:, :K]

    assert not (knn_i == knn_j).any(), "a node is its own neighbour"

    hits = sum(
        len(set(knn_j[knn_i == i].tolist()) & set(truth[i].tolist())) for i in range(n)
    )
    assert hits / (n * K) > 0.9, "recall against brute-force k-NN too low"

    # Reported weights are true cosine similarities, not distances.
    worst = max(
        abs(w - sim[i, j])
        for i in range(n)
        for j, w in zip(knn_j[knn_i == i], knn_w[knn_i == i])
    )
    assert worst < 0.05
    assert knn_w.max() <= 1.0 + 1e-6
