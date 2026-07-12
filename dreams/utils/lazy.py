"""Deferred module import.

Used for pyopenms, which is a large native extension needed *only* for mzML/LC-MS I/O —
yet importing it at module load dragged it into every `import dreams.api`, including the
MGF embedding path that never touches it. On some Windows machines its DLL init fails
hard enough to take the process down (a SIGILL that `try/except` cannot catch), so
deferring the import is what lets DreaMS import at all there.
"""
import contextlib
import importlib
import io as std_io
import typing as T
from types import ModuleType


class LazyModule:
    """A stand-in for a module that imports it on first attribute access.

    `pyms = LazyModule('pyopenms')` behaves like the module for every `pyms.X` use, but
    the import does not happen until the first such use.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._mod: T.Optional[ModuleType] = None

    def __getattr__(self, attr: str) -> T.Any:
        # Never let introspection (copy, pickle, inspect, a debugger) trigger the import —
        # only a real `pyms.Something` use should.
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        if self._mod is None:
            # pyopenms chatters on stderr at import; preserve the original suppression.
            with contextlib.redirect_stderr(std_io.StringIO()):
                self._mod = importlib.import_module(self._name)
        return getattr(self._mod, attr)

    def __repr__(self) -> str:
        state = "loaded" if self._mod is not None else "not yet imported"
        return f"<LazyModule {self._name!r} ({state})>"
