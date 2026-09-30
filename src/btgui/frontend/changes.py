"""Unified reversible changes for the three raw structure lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _copy_arrays(values: list[np.ndarray]) -> list[np.ndarray]:
    """Copy every array in one frame-ordered list.

    Args:
        values: Arrays to detach from mutable GUI data.

    Returns:
        Independent arrays in the same order.
    """
    return [np.array(value, copy=True) for value in values]


def _restore_arrays(target: list[np.ndarray], source: list[np.ndarray]) -> None:
    """Restore arrays while retaining compatible allocations.

    Args:
        target: Mutable destination list retained by the GUI.
        source: Snapshot arrays to restore.

    Returns:
        None. The outer target list identity is preserved.
    """
    if len(target) != len(source):
        target[:] = _copy_arrays(source)
        return
    for index, value in enumerate(source):
        current = target[index]
        if (
            isinstance(current, np.ndarray)
            and current.shape == value.shape
            and current.dtype == value.dtype
        ):
            np.copyto(current, value)
        else:
            target[index] = np.array(value, copy=True)


@dataclass
class StructureChange:
    """Store one reversible update to the raw structure lists.

    Args:
        description: Short operation name used for diagnostics.
        before_elements: Element arrays before the operation.
        before_coordinates: Coordinate arrays before the operation.
        before_lattices: Lattice arrays before the operation.
        after_elements: Element arrays after the operation.
        after_coordinates: Coordinate arrays after the operation.
        after_lattices: Lattice arrays after the operation.
        frame_indices: Target frames, or ``None`` when snapshots cover all frames.
        before_context: Optional owner-defined state restored by Undo.
        after_context: Optional owner-defined state restored by Redo.

    Notes:
        Frame-scoped changes store only their affected frames. Batch changes
        retain complete lists so list insertion, deletion, and reordering remain
        reversible.
    """

    description: str
    before_elements: list[np.ndarray]
    before_coordinates: list[np.ndarray]
    before_lattices: list[np.ndarray]
    after_elements: list[np.ndarray]
    after_coordinates: list[np.ndarray]
    after_lattices: list[np.ndarray]
    frame_indices: tuple[int, ...] | None = None
    before_context: Any = None
    after_context: Any = None

    @classmethod
    def for_frame(
        cls,
        description: str,
        frame_index: int,
        before_elements: np.ndarray,
        before_coordinates: np.ndarray,
        before_lattice: np.ndarray,
        after_elements: np.ndarray,
        after_coordinates: np.ndarray,
        after_lattice: np.ndarray,
        before_context: Any = None,
        after_context: Any = None,
    ) -> "StructureChange":
        """Create a memory-efficient change for one frame.

        Args:
            description: Short operation name.
            frame_index: Changed frame index.
            before_elements: Element array before the operation.
            before_coordinates: Coordinate array before the operation.
            before_lattice: Lattice before the operation.
            after_elements: Element array after the operation.
            after_coordinates: Coordinate array after the operation.
            after_lattice: Lattice after the operation.
            before_context: Optional context restored by Undo.
            after_context: Optional context restored by Redo.

        Returns:
            A detached reversible frame change.
        """
        return cls(
            description=description,
            before_elements=_copy_arrays([before_elements]),
            before_coordinates=_copy_arrays([before_coordinates]),
            before_lattices=_copy_arrays([before_lattice]),
            after_elements=_copy_arrays([after_elements]),
            after_coordinates=_copy_arrays([after_coordinates]),
            after_lattices=_copy_arrays([after_lattice]),
            frame_indices=(int(frame_index),),
            before_context=before_context,
            after_context=after_context,
        )

    @classmethod
    def for_batch(
        cls,
        description: str,
        before_elements: list[np.ndarray],
        before_coordinates: list[np.ndarray],
        before_lattices: list[np.ndarray],
        after_elements: list[np.ndarray],
        after_coordinates: list[np.ndarray],
        after_lattices: list[np.ndarray],
        before_context: Any = None,
        after_context: Any = None,
    ) -> "StructureChange":
        """Create a reversible change covering complete structure lists.

        Args:
            description: Short operation name.
            before_elements: Element lists before the operation.
            before_coordinates: Coordinate lists before the operation.
            before_lattices: Lattice lists before the operation.
            after_elements: Element lists after the operation.
            after_coordinates: Coordinate lists after the operation.
            after_lattices: Lattice lists after the operation.
            before_context: Optional context restored by Undo.
            after_context: Optional context restored by Redo.

        Returns:
            A detached reversible batch change.
        """
        return cls(
            description=description,
            before_elements=_copy_arrays(before_elements),
            before_coordinates=_copy_arrays(before_coordinates),
            before_lattices=_copy_arrays(before_lattices),
            after_elements=_copy_arrays(after_elements),
            after_coordinates=_copy_arrays(after_coordinates),
            after_lattices=_copy_arrays(after_lattices),
            before_context=before_context,
            after_context=after_context,
        )

    def apply(
        self,
        elements: list[np.ndarray],
        coordinates: list[np.ndarray],
        lattices: list[np.ndarray],
        use_after: bool,
    ) -> Any:
        """Apply the before or after snapshot to retained raw lists.

        Args:
            elements: GUI-owned element lists.
            coordinates: GUI-owned coordinate lists.
            lattices: GUI-owned lattice lists.
            use_after: Apply the after snapshot when true, otherwise before.

        Returns:
            The matching owner-defined context snapshot.

        Raises:
            IndexError: If a frame-scoped change no longer has a valid target.
        """
        source_elements = self.after_elements if use_after else self.before_elements
        source_coordinates = self.after_coordinates if use_after else self.before_coordinates
        source_lattices = self.after_lattices if use_after else self.before_lattices
        if self.frame_indices is None:
            _restore_arrays(elements, source_elements)
            _restore_arrays(coordinates, source_coordinates)
            _restore_arrays(lattices, source_lattices)
        else:
            for source_index, frame_index in enumerate(self.frame_indices):
                if not 0 <= frame_index < len(coordinates):
                    raise IndexError(f"history frame {frame_index} is no longer available")
                for target, source in (
                    (elements, source_elements),
                    (coordinates, source_coordinates),
                    (lattices, source_lattices),
                ):
                    value = source[source_index]
                    current = target[frame_index]
                    if (
                        isinstance(current, np.ndarray)
                        and current.shape == value.shape
                        and current.dtype == value.dtype
                    ):
                        np.copyto(current, value)
                    else:
                        target[frame_index] = np.array(value, copy=True)
        return self.after_context if use_after else self.before_context
