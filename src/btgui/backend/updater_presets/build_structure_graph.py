
from typing import Callable, Literal, Tuple
import numpy as np


class ChemGraph:
    def __init__(
            self,
            cells: np.ndarray,
            atomic_numbers: np.ndarray,
            coo: np.ndarray,
    ):
        if coo.ndim != 2:
            raise ValueError(f"coo must be 2-dimensional with shape (n_atom, n_dim), but got {coo.shape}")

        if atomic_numbers.ndim != 1:
            raise ValueError(
                f"atomic_numbers must be 1-dimensional with shape (n_atom), but got {atomic_numbers.shape}"
            )

        if atomic_numbers.shape[0] != coo.shape[0]:
            raise ValueError(
                f"atom numbers of atomic_numbers and coo are not matched: {atomic_numbers.shape} and {coo.shape}"
            )

        if (cells is not None) and cells.shape != (3, 3):
            raise ValueError(f"cells must be 2-dimensional with shape (3, 3), but got {cells.shape}")

        self.cells = cells
        self.coo = coo
        self.atomic_numbers = atomic_numbers
        self.indices = None
        self.src = None
        self.dst = None
        self.dist_ij = None
        self.r_ij = None
        self.bond_mask = None
        self.distance_cutoff = None

    def build(self, edge=None, distance_cutoff: float | None = None):
        """Build the graph from an explicit edge matrix or from distances.

        Args:
            edge: Optional ``(n_atom, n_atom)`` adjacency whose non-zero
                entries define directed edges. Dense arrays and scipy sparse
                matrices are accepted. Self edges are ignored.
            distance_cutoff: Optional total interaction cutoff. Pairs within
                ``1.3 * distance_cutoff`` are retained as non-bonded
                candidates; ``None`` keeps only the explicit edge entries.

        Return:
            None. ``indices``, ``src``, ``dst``, ``dist_ij``, ``r_ij`` and
            ``bond_mask`` are replaced in place.
        """
        self.n_atoms = self.coo.shape[0]
        self.distance_cutoff = (
            None if distance_cutoff is None else float(distance_cutoff)
        )
        pairs = []
        bond_pairs = set()
        if edge is not None:
            if hasattr(edge, "tocoo"):
                sparse_edge = edge.tocoo()
                if sparse_edge.shape != (self.n_atoms, self.n_atoms):
                    raise ValueError(
                        f"edge must have shape ({self.n_atoms}, {self.n_atoms}), "
                        f"but got {sparse_edge.shape}"
                    )
                rows, columns = sparse_edge.row, sparse_edge.col
            else:
                adjacency = np.asarray(edge)
                if adjacency.shape != (self.n_atoms, self.n_atoms):
                    raise ValueError(
                        f"edge must have shape ({self.n_atoms}, {self.n_atoms}), "
                        f"but got {adjacency.shape}"
                    )
                rows, columns = np.nonzero(adjacency)
            for source, target in zip(rows.tolist(), columns.tolist()):
                if source == target:
                    continue
                pair = (int(source), int(target))
                if pair not in bond_pairs:
                    bond_pairs.add(pair)
                    pairs.append(pair)
        if self.distance_cutoff is not None:
            stored_cutoff = 1.3 * self.distance_cutoff
            distance_indices, _ = build_structure_graph(
                self.coo, 'euclidean', stored_cutoff
            )
            existing = set(pairs)
            for source, target in distance_indices.T:
                if source == target:
                    continue
                pair = (int(source), int(target))
                if pair not in existing:
                    existing.add(pair)
                    pairs.append(pair)
        pairs = sorted(set(pairs))
        if pairs:
            self.indices = np.asarray(pairs, dtype=np.int64).T
        else:
            self.indices = np.empty((2, 0), dtype=np.int64)
        self.src = self.indices[0]  # alias
        self.dst = self.indices[1]
        self.r_ij = self.coo[self.src] - self.coo[self.dst]
        self.dist_ij = np.linalg.norm(self.r_ij, axis=1)
        self.bond_mask = np.asarray(
            [pair in bond_pairs for pair in pairs], dtype=bool
        )

    def in_degree(self, i):
        pass

    def out_degree(self, j):
        pass


def _euclidean_metrics(X, ):
    r_ij = X[:, None, :] - X[None, :, :]
    dist_ij = np.linalg.norm(r_ij, axis=-1)  # (n_atom, n_atom)
    return dist_ij

def _cosine_metrics(X, ):
    r_i, r_j = np.broadcast_arrays(X[:, None, :], X[None, :, :])  # (n, n, d)
    inner_prod = np.linalg.vecdot(r_i, r_j)
    norm_i = np.linalg.norm(r_i, axis=-1)
    norm_j = np.linalg.norm(r_j, axis=-1)
    dist_ij = inner_prod/(norm_i * norm_j)  # (n_atom, n_atom)
    return dist_ij


def build_structure_graph(
        coo: np.ndarray,
        metrics: str | Callable[[np.ndarray,], np.ndarray] = 'euclidean',
        cut_off: float | None = None,
        cut_mode: Literal['<', '>'] = '>'
) -> Tuple[np.ndarray, np.ndarray]:
    if coo.ndim != 2:
        raise ValueError(f"`coo` must be 2-dimensional with shape (n_atom, n_dim), but got {coo.shape}")
    if cut_off is not None:
        cut_off = float(cut_off)  # Do not limit the value. There may be some `metrics` given negative values.

    if not callable(metrics):
        if metrics not in _METRICS:
            raise ValueError(f"`metrics` string must be one of {list(_METRICS.keys())}, but got {metrics}")
        dist_func = _METRICS[metrics]
    else:
        dist_func = metrics
    dist_ij = dist_func(coo)

    if cut_off is not None:
        if cut_mode == '<':  # cut the value l.t. `cut_mode`, preserve the large ones.
            indx = np.argwhere(dist_ij > cut_off).T  # (2, E)
        elif cut_mode == '>':
            indx = np.argwhere(dist_ij < cut_off).T  # (2, E)
        else:
            raise ValueError(f"`cut_mode` must be one of {'<', '>'}, but got {cut_mode}")
    else:
        indx = np.argwhere(np.isfinite(dist_ij)).T
    dist_ij = dist_ij[indx[0], indx[1]]

    return indx, dist_ij


_METRICS = {
    'euclidean': _euclidean_metrics,
    'cosine': _cosine_metrics,
}