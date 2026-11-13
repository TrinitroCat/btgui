"""Reversible structure, sparse bond, appearance and frame-order changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from copy import deepcopy

import numpy as np

from btgui.backend.core import normalize_bond_adjacency


def _bond_order_map(graph, orders):
    """Return a canonical endpoint-pair to bond-order mapping."""
    graph = normalize_bond_adjacency(graph)
    pairs = graph.T
    if orders is None or len(orders) != len(pairs):
        values = np.ones(len(pairs), dtype=np.int64)
    else:
        values = np.asarray(orders, dtype=np.int64)
    return {
        (int(source), int(target)): int(order)
        for (source, target), order in zip(pairs, values)
    }


def _mapped_order_map(order_map, mapping, inverse=False):
    """Remap endpoint keys, dropping deleted atoms and canonicalizing pairs."""
    if inverse:
        retained = np.flatnonzero(mapping >= 0)
        inverse_map = np.empty(len(retained), dtype=np.int64)
        inverse_map[mapping[retained]] = retained
        mapping = inverse_map
    result = {}
    for (source, target), order in order_map.items():
        new_source = int(mapping[source])
        new_target = int(mapping[target])
        if new_source >= 0 and new_target >= 0:
            result[tuple(sorted((new_source, new_target)))] = order
    return result


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
        The third triple column is the bond order (1, 2 or 3). Endpoint indices
        are zero-based; console formatting alone is one-based. ``None`` is
        uncomputed, whereas empty ``(2, 0)`` is an explicit graph.
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
    def between(cls, description, before, after, before_orders=None, after_orders=None):
        """Compare canonical edge coordinates and bond orders.

        Args:
            description: History label.
            before: Mapping from frame to its old COO or ``None``.
            after: Mapping from frame to its replacement COO or ``None``.
            before_orders: Optional mapping from frame to old order arrays.
            after_orders: Optional mapping from frame to new order arrays.

        Return:
            A delta containing only removed/added ``(i, j, order)`` triples.
        """
        before_orders = before_orders or {}
        after_orders = after_orders or {}
        removed, added, missing, after_missing = {}, {}, {}, {}
        for frame, replacement in after.items():
            old_orders = _bond_order_map(before[frame], before_orders.get(frame))
            new_orders = _bond_order_map(replacement, after_orders.get(frame))
            removed_rows = []
            added_rows = []
            for key in sorted(set(old_orders) - set(new_orders)):
                removed_rows.append((*key, old_orders[key]))
            for key in sorted(set(new_orders) - set(old_orders)):
                added_rows.append((*key, new_orders[key]))
            for key in sorted(set(old_orders) & set(new_orders)):
                if old_orders[key] != new_orders[key]:
                    removed_rows.append((*key, old_orders[key]))
                    added_rows.append((*key, new_orders[key]))
            if (
                not removed_rows and not added_rows
                and (before[frame] is None) == (replacement is None)
            ):
                continue
            removed[frame] = np.asarray(removed_rows, dtype=np.int64).reshape(-1, 3)
            added[frame] = np.asarray(added_rows, dtype=np.int64).reshape(-1, 3)
            missing[frame] = before[frame] is None
            after_missing[frame] = replacement is None
        return cls(description, removed, added, missing, after_missing=after_missing)

    def apply(self, window, use_after):
        """Apply a delta or its inverse without touching unrelated graph edges.

        Args:
            window: Owner of frame-aligned ``bond_adjacencies`` and
                ``bond_orders``.
            use_after: True for Redo, False for Undo.

        Return:
            None. Every replaced graph remains canonical and orders stay aligned.
        """
        for frame in self.frame_indices:
            graph = normalize_bond_adjacency(window.bond_adjacencies[frame])
            orders = getattr(window, "bond_orders", None)
            current_orders = (
                orders[frame] if orders is not None and frame < len(orders) else None
            )
            order_map = _bond_order_map(graph, current_orders)
            drop = self.removed[frame] if use_after else self.added[frame]
            insert = self.added[frame] if use_after else self.removed[frame]
            # Section: Remove triples in the current index space BEFORE mapping.
            for source, target, _order in drop:
                order_map.pop((int(source), int(target)), None)
            if self.mappings is not None and frame in self.mappings:
                order_map = _mapped_order_map(
                    order_map, self.mappings[frame], inverse=not use_after
                )
            for source, target, order in insert:
                pair = tuple(sorted((int(source), int(target))))
                order_map[pair] = int(order)
            if use_after:
                missing = self.after_missing is not None and self.after_missing.get(frame, False)
            else:
                missing = self.before_missing[frame]
            if missing:
                window.bond_adjacencies[frame] = None
                if orders is not None and frame < len(orders):
                    orders[frame] = None
                continue
            pairs = sorted(order_map)
            if pairs:
                replacement = np.asarray(pairs, dtype=np.int64).T
                replacement_orders = np.asarray(
                    [order_map[pair] for pair in pairs], dtype=np.int8
                )
            else:
                replacement = np.empty((2, 0), dtype=np.int64)
                replacement_orders = np.empty(0, dtype=np.int8)
            window.bond_adjacencies[frame] = normalize_bond_adjacency(replacement)
            if orders is not None and frame < len(orders):
                orders[frame] = replacement_orders


@dataclass
class ClearBondsChange:
    """Retain complete original graphs specifically for explicit Clear bonds.

    Args:
        description: History label.
        originals: Detached per-frame COO snapshots, including ``None``.
        orders: Optional detached per-frame bond-order snapshots.

    Return:
        A clear/restore event; other bond edits use sparse deltas.
    """

    description: str
    originals: dict
    orders: dict | None = None

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
            window: Owner of the graph and order lists.
            use_after: Whether to clear rather than restore.

        Return:
            None. Empty graphs prevent Show bonds from regenerating connections.
        """
        for frame, original in self.originals.items():
            if use_after:
                window.bond_adjacencies[frame] = np.empty((2, 0), dtype=np.int64)
                window.bond_orders[frame] = np.empty(0, dtype=np.int8)
            else:
                window.bond_adjacencies[frame] = (
                    None if original is None else np.array(original, copy=True)
                )
                saved = None if self.orders is None else self.orders.get(frame)
                if saved is not None:
                    window.bond_orders[frame] = np.array(saved, dtype=np.int8, copy=True)
                elif original is None:
                    window.bond_orders[frame] = None
                else:
                    window.bond_orders[frame] = np.ones(
                        len(np.asarray(original).reshape(2, -1)[0]), dtype=np.int8
                    )


@dataclass
class InplaceUpdaterChange:
    """Swap a sparse four-field snapshot for one multi-frame updater step.

    Args:
        description: History label.
        frame_indices: Frames represented by each aligned snapshot slot.
        snapshot: ``(cell, elem, coo, bond_graph)``. Each field is either None
            (unchanged for every frame) or a list aligned with frame_indices,
            with None slots for unchanged frames. Arrays are detached copies.
        graph_changed: Mask distinguishing unchanged graph slots from a saved
            uncomputed graph (both would otherwise be None).
        topology_frames: Frames whose atom count changed.
        contexts: Saved supercell bases/transforms and, only for topology
            changes, saved per-atom styles and measurements.

    Return:
        A reversible event holding only one saved side, exchanged on Undo/Redo.

    Note:
        A bare tuple denotes compound history elsewhere. This wrapper keeps
        the four-field data format distinct from compound event execution.
    """

    description: str
    frame_indices: tuple[int, ...]
    snapshot: tuple
    graph_changed: tuple[bool, ...]
    topology_frames: tuple[int, ...]
    contexts: dict

    def apply(self, window, use_after: bool) -> None:
        """Exchange saved/current fields; redo never calls the plugin again.

        Args:
            window: Main window owning frame-aligned structure and graph lists.
            use_after: History protocol flag; both directions perform a swap.

        Return:
            None. Only changed arrays and graph states are replaced.
        """
        del use_after
        targets = (
            window.lattices, window.elements, window.coordinates, window.bond_adjacencies,
        )
        for slot, frame in enumerate(self.frame_indices):
            for field, (target, saved) in enumerate(zip(targets, self.snapshot)):
                if saved is None or (saved[slot] is None and not (
                    field == 3 and self.graph_changed[slot]
                )):
                    continue
                # Section: Save only the field being overwritten. Keep GUI list
                # identities and compatible array allocations, but detach the
                # inverse snapshot from mutable CLI/viewer references.
                current, value = target[frame], saved[slot]
                saved[slot] = None if current is None else np.array(current, copy=True)
                if value is None:
                    target[frame] = None
                elif (
                    current is not None and current.shape == value.shape
                    and current.dtype == value.dtype
                ):
                    np.copyto(current, value)
                else:
                    target[frame] = np.array(value, copy=True)
        for frame, saved in self.contexts.items():
            # These objects were detached from live state at commit. Swap their
            # ownership rather than cloning all base arrays for every undo.
            for key, target in (
                ("base", window.supercell_bases),
                ("transform", window.supercell_transforms),
                ("styles", window.atom_frame_styles),
                ("metrics", window.frame_metrics),
                ("orders", window.bond_orders),
            ):
                if key in saved:
                    target[frame], saved[key] = saved[key], target[frame]


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
            # Section: Batch snapshots cover the frame prefix that existed when
            # recorded. Later updater appends are deliberately not undoable:
            # carry their suffix by reference instead of restoring/deleting it
            # or retaining it in this old history entry.
            extent = len(self.before_elements if use_after else self.after_elements)
            for target, source in (
                (elements, source_elements), (coordinates, source_coordinates),
                (lattices, source_lattices),
            ):
                prefix = target[:extent]
                suffix = target[extent:]
                _restore_arrays(prefix, source)
                target[:] = prefix + suffix
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
        current = window.current_index
        # Non-history updater appends remain outside this recorded permutation.
        indices = indices + tuple(range(len(indices), len(window.elements)))
        window._apply_frame_permutation(indices, update_current=False)
        window._apply_frame_list_permutation(indices)
        if current < len(self.permutation):
            window.current_index = self.after_current if use_after else self.before_current
