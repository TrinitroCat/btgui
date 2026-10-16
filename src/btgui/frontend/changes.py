"""Reversible structure, sparse bond, appearance and frame-order changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from copy import deepcopy

import numpy as np

from btgui.backend.core import normalize_bond_adjacency


@dataclass
class BondChange:
    """Store only changed COO triples and optional atom-index mappings.

    Args:
        description: History label.
        removed: Per-frame int64 ``(K, 3)`` triples in the old index space.
        added: Per-frame int64 ``(K, 3)`` triples in the new index space.
        before_missing: Whether a frame previously had an uncomputed graph.
        mappings: Optional old-to-new arrays; deleted atoms map to ``-1``.
        after_missing: Optional uncomputed state after a graph/topology reset.

    Return:
        A reversible graph delta, never a pair of full graph snapshots.

    Note:
        The graph is currently unweighted: ``info == 1`` denotes an edge.
        Endpoint indices are zero-based; console formatting alone is one-based.
        ``None`` is uncomputed, whereas empty ``(2, 0)`` is an explicit graph.
    """

    description: str
    removed: dict
    added: dict
    before_missing: dict
    mappings: dict | None = None
    after_missing: dict | None = None

    @property
    def frame_indices(self):
        """Return affected frames for consolidated cache invalidation.

        Args:
            None.

        Return:
            Sorted frame indices.
        """
        return tuple(sorted(self.before_missing))

    @classmethod
    def between(cls, description, before, after):
        """Compare canonical edge coordinates using vectorized sorted matching.

        Args:
            description: History label.
            before: Mapping from frame to its old COO or ``None``.
            after: Mapping from frame to its replacement COO or ``None``.

        Return:
            A delta containing only removed/added ``(i, j, 1)`` triples.
        """
        removed, added, missing, after_missing = {}, {}, {}, {}
        for frame, replacement in after.items():
            old = normalize_bond_adjacency(before[frame])
            new = normalize_bond_adjacency(replacement)
            # Section: Structured pairs compare lexicographically without i*N+j
            # packing, which would introduce an atom-count/overflow dependency.
            pair_dtype = np.dtype([("src", np.int64), ("dst", np.int64)])
            old_pairs = np.ascontiguousarray(old.T).view(pair_dtype).reshape(-1)
            new_pairs = np.ascontiguousarray(new.T).view(pair_dtype).reshape(-1)
            old_only = ~np.isin(old_pairs, new_pairs, assume_unique=True)
            new_only = ~np.isin(new_pairs, old_pairs, assume_unique=True)
            if (
                not old_only.any() and not new_only.any()
                and (before[frame] is None) == (replacement is None)
            ):
                continue
            removed[frame] = np.column_stack((
                old[:, old_only].T, np.ones(old_only.sum(), dtype=np.int64),
            ))
            added[frame] = np.column_stack((
                new[:, new_only].T, np.ones(new_only.sum(), dtype=np.int64),
            ))
            missing[frame] = before[frame] is None
            after_missing[frame] = replacement is None
        return cls(description, removed, added, missing, after_missing=after_missing)

    def apply(self, window, use_after):
        """Apply a delta or its inverse without touching unrelated graph edges.

        Args:
            window: Owner of frame-aligned ``bond_adjacencies``.
            use_after: True for Redo, False for Undo.

        Return:
            None. Every replaced graph remains canonical.
        """
        for frame in self.frame_indices:
            graph = normalize_bond_adjacency(window.bond_adjacencies[frame])
            drop = self.removed[frame] if use_after else self.added[frame]
            insert = self.added[frame] if use_after else self.removed[frame]
            # Section: Remove triples in the current index space BEFORE mapping.
            if len(drop):
                pair_dtype = np.dtype([("src", np.int64), ("dst", np.int64)])
                current_pairs = np.ascontiguousarray(graph.T).view(pair_dtype).reshape(-1)
                drop_pairs = np.ascontiguousarray(drop[:, :2]).view(pair_dtype).reshape(-1)
                graph = graph[:, ~np.isin(current_pairs, drop_pairs, assume_unique=True)]
            if self.mappings is not None and frame in self.mappings:
                mapping = self.mappings[frame]
                if use_after:
                    graph = mapping[graph]
                    graph = graph[:, np.all(graph >= 0, axis=0)]
                else:
                    # Invert the old->new map explicitly so both endpoints use
                    # the same inverse even if retained nodes were reordered.
                    retained = np.flatnonzero(mapping >= 0)
                    inverse = np.empty(len(retained), dtype=np.int64)
                    inverse[mapping[retained]] = retained
                    graph = inverse[graph]
            if len(insert):
                graph = np.concatenate((graph, insert[:, :2].T), axis=1)
            # Missing/explicit-empty is metadata, not an edge delta. Retain it
            # separately so Undo of lazy generation or a topology reset is exact.
            if use_after:
                missing = self.after_missing is not None and self.after_missing.get(frame, False)
            else:
                missing = self.before_missing[frame]
            window.bond_adjacencies[frame] = (
                None if missing
                else normalize_bond_adjacency(graph)
            )


@dataclass
class ClearBondsChange:
    """Retain complete original graphs specifically for explicit Clear bonds.

    Args:
        description: History label.
        originals: Detached per-frame COO snapshots, including ``None``.

    Return:
        A clear/restore event; other bond edits use sparse deltas.
    """

    description: str
    originals: dict

    @property
    def frame_indices(self):
        """Return frames whose graphs were explicitly cleared.

        Args:
            None.

        Return:
            Sorted affected frame indices.
        """
        return tuple(sorted(self.originals))

    def apply(self, window, use_after):
        """Clear graphs on Redo and restore their exact snapshots on Undo.

        Args:
            window: Owner of the graph list.
            use_after: Whether to clear rather than restore.

        Return:
            None. Empty graphs prevent Show bonds from regenerating connections.
        """
        for frame, original in self.originals.items():
            if use_after:
                window.bond_adjacencies[frame] = np.empty((2, 0), dtype=np.int64)
            else:
                window.bond_adjacencies[frame] = (
                    None if original is None else np.array(original, copy=True)
                )


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


@dataclass
class AppearanceChange:
    """Store a renderer appearance edit without copying structure arrays.

    Args:
        description: Short operation name used by the history UI.
        before_global: Renderer attributes before the edit.
        after_global: Renderer attributes after the edit.
        before_atom_styles: Sparse ``(frame, atom)`` style snapshots before edit.
        after_atom_styles: Sparse ``(frame, atom)`` style snapshots after edit.

    Returns:
        A lightweight history entry for global or selected-atom appearance edits.
    """

    description: str
    before_global: dict[str, Any]
    after_global: dict[str, Any]
    before_atom_styles: dict[tuple[int, int], dict]
    after_atom_styles: dict[tuple[int, int], dict]

    def apply(self, window, use_after: bool) -> None:
        """Apply the selected renderer and sparse atom-style snapshot.

        Args:
            window: Main window owning renderer and per-frame atom styles.
            use_after: Whether to apply the state after the edit.

        Returns:
            None. Only named renderer fields and sparse atom styles are changed.
        """
        global_values = self.after_global if use_after else self.before_global
        atom_values = self.after_atom_styles if use_after else self.before_atom_styles
        for name, value in global_values.items():
            setattr(window.viewer, name, deepcopy(value))
        for (frame, atom), style in atom_values.items():
            if 0 <= frame < len(window.atom_frame_styles):
                styles = window.atom_frame_styles[frame]
                if 0 <= atom < len(styles):
                    styles[atom] = deepcopy(style)
        window.viewer.renderer.set_background(window.viewer._background_color_value)


@dataclass
class FramePermutationChange:
    """Store only the forward and inverse indices for a frame reorder."""

    description: str
    permutation: tuple[int, ...]
    inverse: tuple[int, ...]
    before_current: int
    after_current: int

    def apply(self, window, use_after: bool) -> None:
        """Apply one reference-only frame permutation to the main window.

        Args:
            window: Main window owning every frame-aligned list.
            use_after: Whether to apply the forward or inverse permutation.

        Returns:
            None. Array contents are not copied.
        """
        indices = self.permutation if use_after else self.inverse
        window._apply_frame_permutation(indices, update_current=False)
        window._apply_frame_list_permutation(indices)
        window.current_index = self.after_current if use_after else self.before_current
