"""Characterization tests for the MS-data loading path (dreams.utils.data).

Fast (no model download) — runs in CI. Exercises `MSData.from_mgf`, i.e. the
.mgf -> pandas -> HDF5 conversion that regressed on the modernized stack:
NumPy 2 + pandas 2's pyarrow-backed string columns broke the HDF5 writer
("Object dtype has no native HDF5 equivalent"). This locks that path in.
"""
import shutil
from pathlib import Path

import numpy as np

from dreams.utils import data as du

EXAMPLE_MGF = Path("data/examples/example_5_spectra.mgf")
NAMES = ["DMAPT", "Mirk-IN-1", "1373215-15-6", "IPSU", "Vadimezan"]


def _load_from_tmp(tmp_path):
    # from_mgf writes <mgf>.hdf5 next to the input, so copy into a tmp dir to
    # avoid polluting the repo and to isolate the generated HDF5 per test run.
    local = tmp_path / EXAMPLE_MGF.name
    shutil.copy(EXAMPLE_MGF, local)
    return du.MSData.from_mgf(local)


def test_from_mgf_roundtrip_structure(tmp_path):
    msd = _load_from_tmp(tmp_path)
    assert msd.num_spectra == 5
    spectra = np.asarray(msd.get_values("spectrum"))
    assert spectra.shape[:2] == (5, 2)  # (n_spectra, [mz, intensity], n_peaks)
    assert np.issubdtype(spectra.dtype, np.floating)
    assert np.isfinite(spectra).all()


def test_from_mgf_string_columns_roundtrip(tmp_path):
    # The exact regression: pyarrow-backed string columns must survive the HDF5
    # write and read back as the original text.
    msd = _load_from_tmp(tmp_path)
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in msd.get_values("name")]
    assert names == NAMES


def test_from_mgf_numeric_columns_roundtrip(tmp_path):
    msd = _load_from_tmp(tmp_path)
    prec = np.asarray(msd.get_values("precursor_mz"), dtype=float)
    assert prec.shape == (5,)
    assert np.isfinite(prec).all()
    assert (prec > 0).all()
