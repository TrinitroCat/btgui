"""One-slot application clipboard for copied atom fragments."""

from __future__ import annotations

import numpy as np


class AtomClipboard:
    """Hold the most recently copied atom fragment for cross-file pasting.

    Args:
        None.

    Return:
        A mutable single-slot buffer. ``store`` replaces any previous content,
        so only the latest copied fragment is retained.

    Note:
        The buffer owns detached copies of both arrays. Later edits to the
        source frame never change a stored fragment, and pasting never shares
        memory with the clipboard.
    """

    def __init__(self):
        self.elements = None
        self.coordinates = None
        self.source = ""

    def is_empty(self):
        """Return whether no atom fragment is currently stored."""
        return self.elements is None or len(self.elements) == 0

    def count(self):
        """Return the number of stored atoms, or zero for an empty slot."""
        return 0 if self.is_empty() else int(len(self.elements))

    def store(self, elements, coordinates, source=""):
        """Replace the slot with detached copies of one atom fragment.

        Args:
            elements: One-dimensional element symbols to retain.
            coordinates: Cartesian positions shaped ``(N, 3)``.
            source: Optional origin label used by caller diagnostics.

        Return:
            The number of stored atoms.

        Raises:
            ValueError: If the two arrays disagree on the atom count.
        """
        stored_elements = np.array(elements, dtype="<U3", copy=True).reshape(-1)
        stored_coordinates = np.array(coordinates, dtype=float, copy=True).reshape((-1, 3))
        if len(stored_elements) != len(stored_coordinates):
            raise ValueError("clipboard elements and coordinates must have equal lengths")
        self.elements = stored_elements
        self.coordinates = stored_coordinates
        self.source = str(source)
        return len(stored_elements)

    def content(self):
        """Return detached copies of the stored fragment, or ``None``.

        Return:
            An ``(elements, coordinates)`` pair of fresh arrays, or ``None``
            when the slot is empty.
        """
        if self.is_empty():
            return None
        return np.array(self.elements, copy=True), np.array(self.coordinates, copy=True)
