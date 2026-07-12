"""pyopenms must not be imported just because DreaMS was.

pyopenms is an mzML-only dependency, but it used to be imported at module load in
dreams/utils/io.py and dreams/utils/lcms.py — so `import dreams.api` pulled a large native
extension into every process, including the MGF embedding path that never uses it. On some
Windows machines its DLL init fails hard (a SIGILL that try/except cannot catch), which took
DreaMS down with it.

These run in a FRESH interpreter on purpose: sys.modules is process-wide, so another test
importing pyopenms first would mask a regression here.
"""
import subprocess
import sys

import pytest


def _fresh(code: str) -> str:
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout.strip()


def test_importing_dreams_api_does_not_import_pyopenms():
    out = _fresh(
        "import sys; import dreams.api; print('pyopenms' in sys.modules)"
    )
    assert out == "False", "importing dreams.api eagerly imported pyopenms again"


@pytest.mark.parametrize("module", ["dreams.utils.io", "dreams.utils.lcms"])
def test_utils_modules_do_not_import_pyopenms(module):
    out = _fresh(f"import sys; import {module}; print('pyopenms' in sys.modules)")
    assert out == "False", f"importing {module} eagerly imported pyopenms"


def test_lazy_module_still_resolves_the_real_module():
    """Deferring must not break the mzML path: the first attribute access imports for real."""
    out = _fresh(
        "import sys\n"
        "from dreams.utils.io import pyms\n"
        "assert 'pyopenms' not in sys.modules, 'imported too early'\n"
        "spec = pyms.MSSpectrum()\n"                      # first real use -> triggers import
        "assert 'pyopenms' in sys.modules, 'never imported'\n"
        "print(type(spec).__module__)"
    )
    assert out.startswith("pyopenms"), f"pyms did not resolve to the real module: {out!r}"


def test_introspection_does_not_trigger_the_import():
    """repr()/dunder probing must not drag pyopenms in — only a genuine pyms.X use should."""
    out = _fresh(
        "import sys\n"
        "from dreams.utils.io import pyms\n"
        "repr(pyms)\n"
        "print('pyopenms' in sys.modules)"
    )
    assert out == "False", "introspecting the lazy module imported pyopenms"
