"""Runtime discovery for optional CLI conversion to BUCToolkit."""

import importlib
import importlib.util
import os
from pathlib import Path
import sys
from importlib.machinery import PathFinder


def _candidate_paths(search_paths):
    """Build ordered runtime search paths for optional BUCToolkit loading.

    Args:
        search_paths: Paths configured by the application settings.

    Return:
        Unique existing directories, ordered before the ambient interpreter.
    """
    values = []
    if isinstance(search_paths, str):
        values.extend(search_paths.split(os.pathsep))
    elif search_paths is not None:
        values.extend(search_paths)
    for variable in ("BTGUI_BUCTOOLKIT_PATH", "BTGUI_BUCTOOLKIT_PATHS"):
        value = os.environ.get(variable, "")
        if value:
            values.extend(value.split(os.pathsep))
    result = []
    seen = set()
    for value in values:
        try:
            path = Path(value).expanduser().resolve()
        except (OSError, TypeError, ValueError):
            continue
        if path.name.lower() == "buctoolkit" and (path / "__init__.py").is_file():
            path = path.parent
        if path.is_dir() and path not in seen:
            seen.add(path)
            result.append(path)
    return result


def _clear_partial_import():
    """Remove a failed partial BUCToolkit import from ``sys.modules``.

    Args:
        None.

    Return:
        None. Subsequent search paths can be attempted cleanly.
    """
    for name in list(sys.modules):
        if name == "BUCToolkit" or name.startswith("BUCToolkit."):
            sys.modules.pop(name, None)


def load_buctoolkit(search_paths=None):
    """Load optional BUCToolkit from settings paths, environment, or Python.

    Args:
        search_paths: Settings-provided directories tried first.

    Return:
        The imported BUCToolkit package, or ``None`` when unavailable.
    """
    existing = sys.modules.get("BUCToolkit")
    if existing is not None:
        return existing
    for path in _candidate_paths(search_paths):
        location = str(path)
        spec = PathFinder.find_spec("BUCToolkit", [location])
        if spec is None:
            continue
        if location not in sys.path:
            sys.path.insert(0, location)
        try:
            return importlib.import_module("BUCToolkit")
        except (ImportError, ModuleNotFoundError, OSError):
            _clear_partial_import()
            try:
                sys.path.remove(location)
            except ValueError:
                pass
    try:
        if importlib.util.find_spec("BUCToolkit") is None:
            return None
        return importlib.import_module("BUCToolkit")
    except (ImportError, ModuleNotFoundError, OSError):
        _clear_partial_import()
        return None
