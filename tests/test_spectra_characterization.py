"""Characterization (regression) tests for dreams.utils.spectra.

These lock in the *current* numerical behavior of the spectral-processing
primitives so that dependency bumps — especially the NumPy 1.x -> 2.x jump and
the numba-jitted kernels — can't silently change results. The golden values were
captured from the code as of the Python 3.13+/uv modernization; a diff here means
behavior changed and must be reviewed, not blindly re-baselined.
"""
import numpy as np
import pytest
import torch

import dreams.utils.spectra as su


def test_bin_peak_list_golden():
    # numba @njit kernel — the primary NumPy-2 drift risk.
    pl = np.array([[10.0, 55.0, 90.0, 150.0, 299.0], [3.0, 1.0, 4.0, 1.5, 2.0]])
    binned = su.bin_peak_list(pl, max_mz=300.0, bin_step=50.0)
    # bins: [0,50)->3 ; [50,100)->1+4=5 ; [100,150)->0 ; [150,200)->1.5 ; [200,250)->0 ; [250,300)->2
    assert binned.tolist() == [3.0, 5.0, 0.0, 1.5, 0.0, 2.0]


def test_bin_peak_lists_batch_matches_single():
    pl = np.array([[10.0, 55.0, 90.0, 150.0, 299.0], [3.0, 1.0, 4.0, 1.5, 2.0]])
    batch = su.bin_peak_lists(np.stack([pl, pl]), 300.0, 50.0)
    assert batch.shape == (2, 6)
    assert np.array_equal(batch[0], su.bin_peak_list(pl, 300.0, 50.0))
    assert np.array_equal(batch[0], batch[1])


def test_to_rel_intensity_golden():
    pl = np.array([[10.0, 55.0, 90.0, 150.0, 299.0], [3.0, 1.0, 4.0, 1.5, 2.0]])
    rel = su.to_rel_intensity(pl)
    assert rel[0].tolist() == pl[0].tolist()  # m/z untouched
    assert rel[1].tolist() == [0.75, 0.25, 1.0, 0.375, 0.5]  # intensities / max(=4)


def test_merge_peak_lists_golden():
    # Peaks within eps in m/z are summed onto the highest-intensity peak's m/z.
    a = np.array([[100.0, 150.0, 200.0], [0.4, 1.0, 0.2]])
    b = np.array([[100.005, 150.0, 205.0], [0.5, 0.9, 0.3]])
    merged = su.merge_peak_lists([a, b], eps=1e-2)
    # 100.0/100.005 merge -> m/z of the 0.5 peak, intensity 0.9; 150.0 -> 1.9; others untouched.
    assert np.allclose(merged[0], [100.005, 150.0, 200.0, 205.0])
    assert np.allclose(merged[1], [0.9, 1.9, 0.2, 0.3])


def test_merge_single_peak_list_is_identity():
    a = np.array([[100.0, 150.0], [0.4, 1.0]])
    assert np.array_equal(su.merge_peak_lists([a]), a)


def test_to_classes_golden():
    vals = torch.tensor([[0.0], [0.049], [0.05], [1.0], [9.99], [10.0]])
    classes = su.to_classes(vals, max_val=10.0, bin_size=0.05)
    # round(v/0.05), clamped to num_classes-1 (=199) so max_val doesn't get its own class.
    assert classes.squeeze(-1).tolist() == [0, 1, 1, 20, 199, 199]


def test_from_hot_golden():
    hots = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    vals = su.from_hot(hots, bin_size=0.05)
    # argmax * bin_size passes through float32 before the double cast, so values
    # carry float32 rounding (0.05 -> 0.0500000007) — assert with tolerance.
    assert vals.squeeze(-1).tolist() == pytest.approx([0.05, 0.0, 0.10], abs=1e-6)


def test_pad_unpad_roundtrip():
    a = np.array([[100.0, 150.0, 200.0], [0.4, 1.0, 0.2]])
    padded = su.pad_peak_list(a, target_len=5)
    assert padded.shape == (2, 5)
    assert np.array_equal(su.unpad_peak_list(padded), a)


def test_trim_peak_list_golden():
    pl = np.array([[10.0, 20.0, 30.0, 40.0, 50.0], [3.0, 1.0, 5.0, 2.0, 4.0]])
    trimmed = su.trim_peak_list(pl, n_highest=2)
    # two highest intensities are 5.0 (m/z 30) and 4.0 (m/z 50), returned in original m/z order.
    assert trimmed.tolist() == [[30.0, 50.0], [5.0, 4.0]]


def test_parse_raw_peak_list_golden():
    parsed = su.parse_raw_peak_list("100.0 0.5\n200.0 1.0")
    assert parsed.tolist() == [[100.0, 200.0], [0.5, 1.0]]


def test_peaklist_modified_cosine_golden():
    # Locks in the matchms 0.33 ModifiedCosineGreedy behavior (post API-drift fix).
    plmc = su.PeakListModifiedCosine(mz_tolerance=0.05)
    s1 = np.array([[100.0, 150.0, 200.0], [0.4, 1.0, 0.2]])
    s2 = np.array([[100.0, 150.0, 205.0], [0.5, 0.9, 0.3]])
    score = plmc.compute(s1, s2, prec_mz1=300.0, prec_mz2=300.0)
    assert score == pytest.approx(0.9363821838346235, rel=1e-6)
    # Identical spectra score 1.0.
    assert plmc.compute(s1, s1, 300.0, 300.0) == pytest.approx(1.0, abs=1e-9)
