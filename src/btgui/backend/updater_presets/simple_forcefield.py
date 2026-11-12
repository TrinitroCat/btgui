"""Small valence force field terms used by the preset updaters."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from .bond_order import ideal_angles
from .build_structure_graph import ChemGraph


def coo_row_sum(indices, val, n):
    """
    Solve the row-sum problem.
    indices: (2, nnz) int
    val:     (nnz, d)
    n:       row number
    return:  (n, d)
    """
    nnz, d = val.shape
    if nnz == 0:
        return np.zeros((n, d), dtype=val.dtype)

    row = indices[0]
    order = np.argsort(row, kind='stable')   # stable ensures the order of float additions.
    sorted_row = row[order]
    sorted_val = val[order]

    counts = np.bincount(sorted_row, minlength=n).astype(np.intp)
    indptr = np.empty(n + 1, dtype=np.intp)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    out = np.zeros((n, d), dtype=val.dtype)
    non_empty = counts > 0
    # 关键：只对非空行取起点，否则空行起点可能 == nnz 导致 reduceat 越界
    starts = indptr[:-1][non_empty]
    out[non_empty] = np.add.reduceat(sorted_val, starts, axis=0)

    return out


def _expand(value, mask, edge_count):
    """Broadcast one scalar/array term parameter onto the selected edges."""
    array = np.asarray(value, dtype=float)
    selected = int(np.count_nonzero(mask))
    if array.ndim == 0:
        return np.full(selected, float(array), dtype=float)
    if array.shape[0] == edge_count:
        return array[mask]
    if array.shape[0] == selected:
        return array
    raise ValueError(
        f"term parameter has {array.shape[0]} entries, expected {edge_count} or {selected}"
    )


class StdTerm(ABC):
    """
    A base class of all ff term
    """

    @abstractmethod
    def apply(self, G: ChemGraph) -> Tuple[float, np.ndarray]:
        raise NotImplementedError


class Harmonics(StdTerm):
    """Harmonic bond term over the graph's bond edges.

    ``k`` and ``l_0`` may be scalars, full edge arrays, or arrays already
    restricted to the bond mask. The energy is ``1/2 * sum(k * (d - l_0)^2)``,
    so a symmetric adjacency that stores both ``ij`` and ``ji`` is not
    double-counted.
    """

    def __init__(self, k: np.ndarray | float, l_0: np.ndarray | float):
        self.k = k
        self.l_0 = l_0

    def apply(self, G: ChemGraph):
        mask = G.bond_mask
        if mask is None:
            mask = np.ones(G.indices.shape[1], dtype=bool)
        if not np.any(mask):
            return 0.0, np.zeros((G.n_atoms, 3), dtype=float)

        indices = G.indices[:, mask]
        distance = G.dist_ij[mask]
        vector = G.r_ij[mask]
        k = _expand(self.k, mask, G.indices.shape[1])
        l_0 = _expand(self.l_0, mask, G.indices.shape[1])

        difference = distance - l_0
        energy = 0.5 * float(np.sum(k * difference * difference))
        safe = np.where(distance > 0.0, distance, 1.0)
        edge_force = (k * (l_0 - distance) / safe)[:, None] * vector
        forces = coo_row_sum(indices, edge_force, G.n_atoms)
        forces += coo_row_sum(indices[::-1], -edge_force, G.n_atoms)
        return energy, forces


class AngleCosineHarmonics(StdTerm):
    """Harmonic term on the cosine of each bond angle.

    Angle triplets are rebuilt from the current bond edges on every ``apply``.
    A target cosine is taken as the ideal angle of the centre atom's geometry
    label closest to the current angle; it is cached per triplet so the target
    does not jump during optimization.
    """

    def __init__(self, k: float, labels=None):
        self.k = float(k)
        self.labels = labels
        self._targets = {}

    def _triplets(self, G: ChemGraph):
        mask = G.bond_mask
        if mask is None:
            mask = np.ones(G.indices.shape[1], dtype=bool)
        edges = G.indices[:, mask]
        neighbours = [set() for _ in range(G.n_atoms)]
        for source, target in edges.T:
            source, target = int(source), int(target)
            if source != target:
                neighbours[source].add(target)
                neighbours[target].add(source)
        triplets = []
        for centre, adjacent in enumerate(neighbours):
            ordered = sorted(adjacent)
            for first in range(len(ordered)):
                for second in range(first + 1, len(ordered)):
                    triplets.append((ordered[first], centre, ordered[second]))
        if not triplets:
            return np.empty((3, 0), dtype=np.int64)
        return np.asarray(triplets, dtype=np.int64).T

    def _target_cos(self, key, centre, cosine):
        if key not in self._targets:
            label = "unknown" if self.labels is None else str(self.labels[centre])
            angles = ideal_angles(label)
            if angles:
                target = min(
                    angles,
                    key=lambda angle: abs(np.cos(np.radians(angle)) - cosine),
                )
                self._targets[key] = float(np.cos(np.radians(target)))
            else:
                self._targets[key] = float(cosine)
        return self._targets[key]

    def apply(self, G: ChemGraph):
        forces = np.zeros((G.n_atoms, 3), dtype=float)
        triplets = self._triplets(G)
        if triplets.shape[1] == 0:
            return 0.0, forces

        coo = G.coo
        first = coo[triplets[0]] - coo[triplets[1]]
        second = coo[triplets[2]] - coo[triplets[1]]
        norm_first = np.linalg.norm(first, axis=1)
        norm_second = np.linalg.norm(second, axis=1)
        valid = (norm_first > 0.0) & (norm_second > 0.0)
        if not np.any(valid):
            return 0.0, forces

        triplets = triplets[:, valid]
        first = first[valid]
        second = second[valid]
        norm_first = norm_first[valid]
        norm_second = norm_second[valid]
        denominator = norm_first * norm_second
        cosine = np.clip(
            np.sum(first * second, axis=1) / denominator, -1.0, 1.0
        )
        cos0 = np.asarray(
            [
                self._target_cos(
                    (int(triplets[0, index]), int(triplets[1, index]), int(triplets[2, index])),
                    int(triplets[1, index]),
                    float(cosine[index]),
                )
                for index in range(triplets.shape[1])
            ],
            dtype=float,
        )

        difference = cosine - cos0
        energy = float(self.k * np.sum(difference * difference))
        prefactor = 2.0 * self.k * difference
        grad_first = (
            second / denominator[:, None]
            - cosine[:, None] * first / (norm_first ** 2)[:, None]
        )
        grad_second = (
            first / denominator[:, None]
            - cosine[:, None] * second / (norm_second ** 2)[:, None]
        )
        force_first = -prefactor[:, None] * grad_first
        force_second = -prefactor[:, None] * grad_second
        np.add.at(forces, triplets[0], force_first)
        np.add.at(forces, triplets[2], force_second)
        np.add.at(forces, triplets[1], -(force_first + force_second))
        return energy, forces


class LennardJones(StdTerm):
    """Weak Lennard-Jones term on non-bonded candidate pairs.

    The graph stores non-bonded candidates up to ``1.3 * cutoff``; the term
    only applies the potential to pairs within ``cutoff``. The 1/2 factor
    keeps symmetric ``ij``/``ji`` entries from double counting.
    """

    def __init__(
            self,
            epsilon: float,
            sigma_scale: float,
            cutoff: float,
            radii: np.ndarray,
    ):
        self.epsilon = float(epsilon)
        self.sigma_scale = float(sigma_scale)
        self.cutoff = float(cutoff)
        self.radii = np.asarray(radii, dtype=float)

    def apply(self, G: ChemGraph):
        mask = ~G.bond_mask if G.bond_mask is not None else np.ones(
            G.indices.shape[1], dtype=bool
        )
        mask &= G.dist_ij <= self.cutoff
        if not np.any(mask):
            return 0.0, np.zeros((G.n_atoms, 3), dtype=float)

        indices = G.indices[:, mask]
        distance = G.dist_ij[mask]
        vector = G.r_ij[mask]
        sigma = self.sigma_scale * (self.radii[indices[0]] + self.radii[indices[1]])
        safe = np.where(distance > 0.0, distance, np.inf)
        ratio6 = (sigma / safe) ** 6
        energy = 0.5 * 4.0 * self.epsilon * float(
            np.sum(ratio6 * ratio6 - ratio6)
        )
        derivative = (24.0 * self.epsilon / safe) * (ratio6 - 2.0 * ratio6 * ratio6)
        edge_force = (-0.5 * derivative / safe)[:, None] * vector
        forces = coo_row_sum(indices, edge_force, G.n_atoms)
        forces += coo_row_sum(indices[::-1], -edge_force, G.n_atoms)
        return energy, forces
