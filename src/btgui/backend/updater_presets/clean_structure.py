"""Built-in clean updater: one rough force-field relaxation per structure."""

from __future__ import annotations

import numpy as np

from btgui.backend.core import build_bond_adjacency, normalize_bond_adjacency
from btgui.backend.updater_presets.bond_order import (
    SYMBOL_TO_Z,
    bond_order,
    geometry_label,
)
from btgui.backend.updater_presets.build_structure_graph import ChemGraph
from btgui.backend.updater_presets.simple_forcefield import (
    AngleCosineHarmonics,
    Harmonics,
    LennardJones,
)
from btgui.data.appearance import covalent_radius, vdw_radius


BOND_LENGTH_OFFSETS = {1: 0.0, 2: 0.18, 3: 0.30}


def _frame_value(value, index, frames):
    """Select one frame from an optional per-frame sequence."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == frames:
        return value[index]
    return value


def _frame_bonds(value, coordinates, elements, tolerance):
    """Return a canonical ``(2, E)`` bond array for one frame."""
    if value is None:
        return build_bond_adjacency(coordinates, elements, tolerance)
    bonds = np.asarray(value, dtype=np.int64)
    if bonds.ndim != 2 or bonds.shape[0] != 2:
        raise ValueError("bonds must be a (2, E) COO array")
    return normalize_bond_adjacency(bonds)


def _adjacency(bonds, n_atoms):
    """Expand a canonical COO array into a symmetric boolean adjacency."""
    adjacency = np.zeros((n_atoms, n_atoms), dtype=bool)
    if bonds.size:
        adjacency[bonds[0], bonds[1]] = True
        adjacency[bonds[1], bonds[0]] = True
    return adjacency


def _fixed_mask(value, coordinates):
    """Return the per-coordinate free mask (1 free, 0 frozen)."""
    if value is None:
        return np.ones(coordinates.shape, dtype=bool)
    mask = np.asarray(value, dtype=float)
    if mask.shape != coordinates.shape:
        raise ValueError(
            f"fixed must have the same shape as coo {coordinates.shape}, "
            f"but got {mask.shape}"
        )
    return mask != 0.0


def _relax_frame(
    lattice,
    elements,
    coordinates,
    fixed,
    bonds,
    *,
    maxiter,
    gtol,
    bond_k,
    angle_k,
    lj_epsilon,
    lj_equilibrium_scale,
    lj_cutoff,
    optimizer,
):
    """Relax one frame and return new Cartesian coordinates."""
    n_atoms = len(coordinates)
    if n_atoms == 0:
        return coordinates.copy()
    fixed_mask = _fixed_mask(fixed, coordinates)
    free = fixed_mask.reshape(-1).astype(float)
    if not np.any(fixed_mask):
        return coordinates.copy()

    adjacency = _adjacency(bonds, n_atoms)
    atomic_numbers = np.asarray(
        [SYMBOL_TO_Z.get(str(symbol), 0) for symbol in elements], dtype=np.int64
    )
    graph = ChemGraph(lattice, atomic_numbers, coordinates)
    graph.build(edge=adjacency, distance_cutoff=lj_cutoff)

    degree = adjacency.sum(axis=1)
    labels = np.asarray(
        [
            geometry_label(int(degree[index]), str(symbol))
            for index, symbol in enumerate(elements)
        ],
        dtype=object,
    )
    covalent_radii = np.asarray(
        [covalent_radius(str(symbol)) for symbol in elements], dtype=float
    )
    vdw_radii = np.asarray(
        [vdw_radius(str(symbol)) for symbol in elements], dtype=float
    )

    bond_mask = graph.bond_mask
    bond_edges = graph.indices[:, bond_mask]
    orders = np.asarray(
        [
            bond_order(labels[source], labels[target])
            for source, target in bond_edges.T
        ],
        dtype=float,
    )
    k_values = np.zeros(graph.indices.shape[1], dtype=float)
    l_values = np.zeros(graph.indices.shape[1], dtype=float)
    if orders.size:
        offsets = np.asarray(
            [BOND_LENGTH_OFFSETS.get(int(order), 0.0) for order in orders],
            dtype=float,
        )
        k_values[bond_mask] = bond_k * orders
        l_values[bond_mask] = (
            covalent_radii[bond_edges[0]] + covalent_radii[bond_edges[1]] - offsets
        )

    terms = (
        Harmonics(k_values, l_values),
        AngleCosineHarmonics(angle_k, labels),
        LennardJones(
            lj_epsilon, lj_equilibrium_scale, lj_cutoff, vdw_radii
        ),
    )

    def objective(vector):
        graph.coo = vector.reshape(n_atoms, 3)
        graph.r_ij = graph.coo[graph.src] - graph.coo[graph.dst]
        graph.dist_ij = np.linalg.norm(graph.r_ij, axis=1)
        energy = 0.0
        forces = np.zeros((n_atoms, 3), dtype=float)
        for term in terms:
            term_energy, term_forces = term.apply(graph)
            energy += term_energy
            forces += term_forces
        # Terms return forces; scipy asks for the energy gradient.
        return float(energy), -forces.reshape(-1) * free

    initial = coordinates.reshape(-1).astype(float, copy=True)
    energy_before, gradient = objective(initial)
    if not np.isfinite(energy_before) or not np.isfinite(gradient).all():
        raise ValueError("clean produced non-finite force-field values")
    if np.max(np.abs(gradient)) <= gtol:
        return coordinates.copy()

    if optimizer is None:
        try:
            from scipy.optimize import minimize
        except ImportError as error:
            raise RuntimeError(
                "clean requires scipy; install btgui with its core dependencies"
            ) from error
        result = minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": int(maxiter),
                "gtol": float(gtol),
                "ftol": 1e-12,
            },
        )
        relaxed = np.asarray(result.x, dtype=float)
    else:
        relaxed = np.asarray(optimizer(objective, initial), dtype=float)

    if relaxed.shape != initial.shape or not np.isfinite(relaxed).all():
        return coordinates.copy()
    relaxed = relaxed.reshape(n_atoms, 3)
    relaxed = np.where(fixed_mask, relaxed, coordinates)
    energy_after, _ = objective(relaxed.reshape(-1))
    if not np.isfinite(energy_after) or energy_after > energy_before + 1e-9:
        return coordinates.copy()

    displacement = np.linalg.norm(relaxed - coordinates, axis=1)
    bond_count = len({
        tuple(sorted((int(source), int(target))))
        for source, target in graph.indices[:, graph.bond_mask].T
    })
    free_atoms = int(np.count_nonzero(fixed_mask.any(axis=1)))
    print(
        f"clean: atoms={n_atoms}, bonds={bond_count}, "
        f"free_atoms={free_atoms}, "
        f"E={energy_before:.4f}->{energy_after:.4f}, "
        f"max|dx|={float(displacement.max()):.3f}"
    )
    return relaxed.astype(coordinates.dtype, copy=False)


def clean(
    cell,
    elem,
    coo,
    *,
    fixed=None,
    bonds=None,
    maxiter=200,
    gtol=1e-5,
    bond_tolerance=1.2,
    bond_k=10.0,
    angle_k=1.0,
    lj_epsilon=5e-3,
    lj_equilibrium_scale=0.75,
    lj_cutoff=6.0,
    optimizer=None,
):
    """Relax one or more structures with the preset valence force field.

    Args:
        cell: Per-frame row-vector lattice matrices. Stored only; no PBC.
        elem: Per-frame element symbol arrays.
        coo: Per-frame Cartesian coordinate arrays.
        fixed: Optional per-frame free mask with the same shape as ``coo``.
            ``0`` freezes a coordinate, ``1`` releases it. ``None`` releases
            every atom. A sequence selects one mask per frame.
        bonds: Optional per-frame ``(2, E)`` COO bond arrays. ``None`` builds
            bonds with the Check bonds covalent-radius rule; an explicitly
            empty array means no bonds. A sequence selects one array per frame.
        maxiter: Maximum L-BFGS-B iterations.
        gtol: Gradient tolerance used as the early-exit condition.
        bond_tolerance: Covalent-radius multiplier used when ``bonds`` is
            ``None``.
        bond_k: Harmonic bond force constant per unit bond order.
        angle_k: Cosine-angle force constant.
        lj_epsilon: Weak Lennard-Jones well depth.
        lj_equilibrium_scale: Multiplier on the van der Waals radius sum that
            places the Lennard-Jones minimum at
            ``scale * (vdW_i + vdW_j)``.
        lj_cutoff: Lennard-Jones cutoff; candidates are stored to 1.3 times
            this value.
        optimizer: Optional ``callable(objective, x0) -> ndarray`` replacing
            the default scipy L-BFGS-B call.

    Return:
        ``(cells, elements, coordinates)`` lists. Elements, frame count, atom
        count, and dtypes are preserved.
    """
    frames = len(coo)
    if not (len(cell) == len(elem) == frames):
        raise ValueError("cell, elem and coo must have equal frame counts")

    result_cell = []
    result_elem = []
    result_coo = []
    for index in range(frames):
        frame_cell = np.asarray(cell[index], dtype=float)
        frame_elem = np.asarray(elem[index], dtype=str)
        frame_coo = np.asarray(coo[index], dtype=float)
        frame_fixed = _frame_value(fixed, index, frames)
        frame_bonds = _frame_bonds(
            _frame_value(bonds, index, frames),
            frame_coo,
            frame_elem,
            bond_tolerance,
        )
        relaxed = _relax_frame(
            frame_cell,
            frame_elem,
            frame_coo,
            frame_fixed,
            frame_bonds,
            maxiter=maxiter,
            gtol=gtol,
            bond_k=bond_k,
            angle_k=angle_k,
            lj_epsilon=lj_epsilon,
            lj_equilibrium_scale=lj_equilibrium_scale,
            lj_cutoff=lj_cutoff,
            optimizer=optimizer,
        )
        result_cell.append(np.array(frame_cell, copy=True))
        result_elem.append(np.array(frame_elem, copy=True))
        result_coo.append(relaxed)
    return result_cell, result_elem, result_coo
