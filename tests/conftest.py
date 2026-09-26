import importlib
import importlib.util
import pathlib
import sys
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_LIB = _ROOT / "dist" / "libmojo-biosppy.so"

if not _LIB.exists():
    pytest.skip(
        "libmojo-biosppy.so not built; run `bash build/build.sh`",
        allow_module_level=True,
    )


def _register_real_source_tree():
    """Make the upstream `biosppy` source importable in this venv.

    `biosppy/__init__.py` pulls in every signal module and `signals/ecg.py`
    imports `peakutils`, which is not installed in this test venv.  The
    functions under test live in `biosppy.signals.tools`, whose own imports
    are all satisfiable, so register the real package directories as
    namespace packages and let the normal finder load the real `utils.py` and
    `signals/tools.py` from site-packages.  No upstream numerics are replaced.
    """
    spec = importlib.util.find_spec("biosppy")
    root = pathlib.Path(list(spec.submodule_search_locations)[0])
    for name, path in (("biosppy", root), ("biosppy.signals", root / "signals")):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module


try:
    importlib.import_module("biosppy.signals.tools")
    HOW = "import biosppy.signals.tools"
except ImportError:
    _register_real_source_tree()
    importlib.import_module("biosppy.utils")
    importlib.import_module("biosppy.signals.tools")
    HOW = "real biosppy source loaded from site-packages (peakutils absent)"


@pytest.fixture(scope="session")
def biosppy_load_path():
    """How the real upstream module was made importable."""
    return HOW


@pytest.fixture(scope="session")
def tools():
    """The real `biosppy.signals.tools` module."""
    return importlib.import_module("biosppy.signals.tools")


@pytest.fixture(scope="session")
def utils():
    """The real `biosppy.utils` module."""
    return importlib.import_module("biosppy.utils")
