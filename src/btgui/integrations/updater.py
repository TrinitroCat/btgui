"""Dynamic loading for user-provided structure updater modules."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_MODULE_NAMES = {}


def load_updater_module(path, base_directory=None):
    """Load a source module with a registered identity for function metadata.

    Args:
        path: Python source path, optionally relative to ``base_directory``.
        base_directory: Optional base directory for a relative source path.

    Return:
        The loaded module, or ``None`` when ``path`` is empty.
    """
    if path is None or not str(path).strip():
        return None
    source_path = Path(str(path)).expanduser()
    if not source_path.is_absolute() and base_directory is not None:
        source_path = Path(base_directory) / source_path
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"updater file does not exist: {source_path}")

    # Section: Give each source a distinct identity, reused when refreshing it.
    if source_path not in _MODULE_NAMES:
        _MODULE_NAMES[source_path] = f"_btgui_external_updater_{len(_MODULE_NAMES)}"
    module_name = _MODULE_NAMES[source_path]
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load updater module: {source_path}")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Restore metadata lookup for any previously loaded functions on failure.
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise
    return module


def load_structure_updater(path, function_name, base_directory=None):
    """Load one updater callable from a standalone Python source file.

    Args:
        path: Python source path, optionally relative to ``base_directory``.
        function_name: Named callable to retrieve from the loaded module.
        base_directory: Optional base directory for a relative source path.

    Return:
        The requested callable, or ``None`` when ``path`` is empty.

    Note:
        The callable protocol is validated when the updater is invoked.
    """
    if path is None or not str(path).strip():
        return None
    name = str(function_name or "").strip()
    if not name or not name.isidentifier():
        raise ValueError("updater_function must be a valid Python function name")
    module = load_updater_module(path, base_directory)
    updater = getattr(module, name, None)
    if not callable(updater):
        raise TypeError(f"updater function {name!r} is not callable")
    return updater
