"""Per-file frame state with explicit lifetime and window reference counts."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np


class FileState:
    """Own every frame-aligned value for one opened structure file.

    Args:
        path: Canonical source path, or ``None`` for untitled CLI data.
        history_limit: Maximum retained Undo entries.

    Return:
        A mutable container. ``close`` clears all owned references explicitly.
    """

    def __init__(self, path: Path | None, history_limit: int):
        self.path = path
        self.elements = []
        self.coordinates = []
        self.lattices = []
        self.frame_names = []
        self._frame_name_serial = 0
        self.atom_frame_styles = []
        self.frame_metrics = []
        self.bond_adjacencies = []
        self.bond_orders = []
        self.bond_frame_indices = []
        self.supercell_bases = {}
        self.supercell_transforms = {}
        self.current_index = 0
        self.undo_stack = deque(maxlen=max(1, int(history_limit)))
        self.redo_stack = deque()
        self.window_refs = 0
        self.border_color = None
        self.closed = False

    def retain(self) -> int:
        """Register one display window that references this state."""
        self.window_refs += 1
        return self.window_refs

    def release(self) -> int:
        """Drop one display-window reference and return the remaining count."""
        self.window_refs = max(0, self.window_refs - 1)
        return self.window_refs

    def nbytes(self) -> int:
        """Estimate owned memory, including structure arrays and history."""
        total = 0
        for values in (
            self.elements,
            self.coordinates,
            self.lattices,
            self.bond_adjacencies,
            self.bond_orders,
        ):
            for value in values:
                if isinstance(value, np.ndarray):
                    total += int(value.nbytes)
        for base in self.supercell_bases.values():
            for value in base.values():
                if isinstance(value, np.ndarray):
                    total += int(value.nbytes)
        for transform in self.supercell_transforms.values():
            for value in transform:
                if isinstance(value, np.ndarray):
                    total += int(value.nbytes)
        for frame_styles in self.atom_frame_styles:
            total += sum(256 for _ in frame_styles)
        total += sum(256 for _ in self.frame_metrics)
        for stack in (self.undo_stack, self.redo_stack):
            for change in stack:
                total += _change_nbytes(change)
        return total

    def close(self) -> None:
        """Release every large or frame-owned reference explicitly."""
        if self.closed:
            return
        for values in (
            self.elements,
            self.coordinates,
            self.lattices,
            self.frame_names,
            self.atom_frame_styles,
            self.frame_metrics,
            self.bond_adjacencies,
            self.bond_orders,
            self.bond_frame_indices,
        ):
            values.clear()
        self.supercell_bases.clear()
        self.supercell_transforms.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.current_index = 0
        self._frame_name_serial = 0
        self.window_refs = 0
        self.border_color = None
        self.path = None
        self.closed = True


def _change_nbytes(change) -> int:
    """Approximate the arrays retained by one history event or compound."""
    total = 0
    if isinstance(change, tuple):
        return sum(_change_nbytes(item) for item in change)
    for value in getattr(change, "__dict__", {}).values():
        total += _object_nbytes(value)
    return total


def _object_nbytes(value) -> int:
    """Recursively count NumPy arrays and small containers in one object."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_object_nbytes(item) for item in value.values()) + 64 * len(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(_object_nbytes(item) for item in value) + 32 * len(value)
    if hasattr(value, "__dict__"):
        return _object_nbytes(vars(value))
    return 0
