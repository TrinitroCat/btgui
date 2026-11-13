"""Hydrogen placement for the preset structure-editing operations."""

from __future__ import annotations

import numpy as np

from btgui.backend.updater_presets.bond_order import UNPAIRED_ELECTRONS
from btgui.data.appearance import covalent_radius


def _unit(vector):
    """Return a normalized vector, or ``None`` for a zero vector."""
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return vector / norm


def _cone_direction(axis, half_angle, azimuth):
    """Return a unit direction on a cone around ``axis``."""
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, reference)
    first = first / np.linalg.norm(first)
    second = np.cross(axis, first)
    return (
        np.cos(half_angle) * axis
        + np.sin(half_angle)
        * (np.cos(azimuth) * first + np.sin(azimuth) * second)
    )


def _directions(axis, count):
    """Return ``count`` roughly uniform directions around ``axis``."""
    axis = _unit(axis)
    if axis is None:
        candidate = np.random.normal(size=3)
        axis = _unit(candidate)
        if axis is None:
            axis = np.array([0.0, 0.0, 1.0])
    if count == 1:
        return [axis]
    # A shared phase keeps the two hydrogen directions of one double-bond end
    # in the same plane as the bond axis instead of twisting each centre.
    phase = 0.0 if count == 2 else float(np.random.uniform(0.0, 2.0 * np.pi))
    if count == 2:
        half = np.radians(54.73561031724535)
        return [
            _cone_direction(axis, half, phase + 0.0),
            _cone_direction(axis, half, phase + np.pi),
        ]
    if count == 3:
        half = np.radians(70.52877936550931)
        return [
            _cone_direction(axis, half, phase + index * 2.0 * np.pi / 3.0)
            for index in range(3)
        ]
    half = np.radians(109.47122063449069)
    directions = [axis]
    for index in range(count - 1):
        directions.append(
            _cone_direction(axis, half, phase + index * 2.0 * np.pi / (count - 1))
        )
    return directions


def add_hydrogens(elements, coordinates, bonds, orders, selected=None):
    """Place hydrogens according to the explicit bond orders.

    Args:
        elements: One-dimensional element symbols.
        coordinates: Cartesian coordinates shaped ``(N, 3)``.
        bonds: Canonical ``(2, E)`` COO bond array.
        orders: Per-edge bond orders aligned with ``bonds``.
        selected: Optional atom indices to consider as centres.

    Return:
        ``(elements, coordinates, bonds, orders)`` with appended hydrogen
        atoms and single-bond edges. The input arrays are not modified.
    """
    elements = np.asarray(elements, dtype=str).reshape(-1)
    coordinates = np.asarray(coordinates, dtype=float)
    bonds = np.asarray(bonds, dtype=np.int64).reshape(2, -1)
    orders = np.asarray(orders, dtype=np.int64).reshape(-1)
    if len(orders) != len(bonds[0]):
        orders = np.ones(len(bonds[0]), dtype=np.int64)
    n_atoms = len(elements)

    neighbours = [[] for _ in range(n_atoms)]
    for (source, target), order in zip(bonds.T, orders):
        source, target = int(source), int(target)
        neighbours[source].append((target, int(order)))
        neighbours[target].append((source, int(order)))

    centres = range(n_atoms)
    if selected is not None:
        centres = [int(index) for index in np.asarray(selected, dtype=int).reshape(-1)]

    new_elements = [str(symbol) for symbol in elements]
    new_coordinates = [np.array(value, copy=True) for value in coordinates]
    new_edges = []
    for centre in centres:
        if not 0 <= centre < n_atoms or elements[centre] == "H":
            continue
        unpaired = UNPAIRED_ELECTRONS.get(str(elements[centre]))
        if unpaired is None:
            continue
        used = sum(order for _, order in neighbours[centre])
        missing = int(unpaired) - used
        if missing <= 0:
            continue
        axis = np.zeros(3, dtype=float)
        for neighbour, _order in neighbours[centre]:
            axis += coordinates[centre] - coordinates[neighbour]
        bond_length = covalent_radius(str(elements[centre])) + covalent_radius("H")
        for direction in _directions(axis, missing):
            position = coordinates[centre] + direction * bond_length
            index = len(new_elements)
            new_elements.append("H")
            new_coordinates.append(position)
            new_edges.append((centre, index))

    order_map = {
        (int(source), int(target)): int(order)
        for (source, target), order in zip(bonds.T, orders)
    }
    for source, target in new_edges:
        order_map[tuple(sorted((source, target)))] = 1
    pairs = sorted(order_map)
    if pairs:
        new_bonds = np.asarray(pairs, dtype=np.int64).T
        new_orders = np.asarray([order_map[pair] for pair in pairs], dtype=np.int8)
    else:
        new_bonds = np.empty((2, 0), dtype=np.int64)
        new_orders = np.empty(0, dtype=np.int8)
    return (
        np.asarray(new_elements, dtype=elements.dtype),
        np.asarray(new_coordinates, dtype=float),
        new_bonds,
        new_orders,
    )
