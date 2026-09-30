"""Integer lattice transformations for one structure frame.

The functions in this module operate directly on the public array interface used
by btgui. They do not own, cache, or wrap structure data.
"""

from __future__ import annotations

from itertools import product

import numpy as np


_TRANSFORM_TOLERANCE = 1.0e-9


def transform_structure(
    elements: np.ndarray,
    coordinates: np.ndarray,
    lattice: np.ndarray,
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform one periodic structure with an integer lattice matrix.

    Parameters
    ----------
    elements
        One-dimensional element symbols, one entry for each atom.
    coordinates
        Cartesian atomic coordinates with shape (atom_count, 3).
    lattice
        Lattice row vectors with shape (3, 3).
    transform
        Non-singular integer matrix. The transformed lattice is transform @ lattice.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Transformed element symbols, Cartesian coordinates, and lattice.

    Raises
    ------
    ValueError
        If an input has an invalid shape, contains non-finite data, the
        transform is not integral, or either matrix is singular.
    """
    element_array = np.asarray(elements)
    coordinate_array = np.asarray(coordinates, dtype=float)
    lattice_array = np.asarray(lattice, dtype=float)
    transform_array = np.asarray(transform, dtype=float)

    if element_array.ndim != 1:
        raise ValueError("Elements must be a one-dimensional array.")
    if coordinate_array.shape != (element_array.size, 3):
        raise ValueError("Coordinates must have shape (atom_count, 3).")
    if lattice_array.shape != (3, 3):
        raise ValueError("The lattice must have shape (3, 3).")
    if transform_array.shape != (3, 3):
        raise ValueError("The lattice transform must have shape (3, 3).")
    if not np.all(np.isfinite(coordinate_array)):
        raise ValueError("Coordinates must contain only finite values.")
    if not np.all(np.isfinite(lattice_array)):
        raise ValueError("The lattice must contain only finite values.")
    if not np.all(np.isfinite(transform_array)):
        raise ValueError("The lattice transform must contain only finite values.")

    rounded_transform = np.rint(transform_array)
    if not np.allclose(transform_array, rounded_transform, atol=_TRANSFORM_TOLERANCE):
        raise ValueError("The lattice transform must contain integers.")
    integer_transform = rounded_transform.astype(np.int64)

    if abs(float(np.linalg.det(lattice_array))) <= _TRANSFORM_TOLERANCE:
        raise ValueError("The source lattice is singular.")
    transform_determinant = int(round(float(np.linalg.det(integer_transform))))
    if transform_determinant == 0:
        raise ValueError("The lattice transform must be non-singular.")

    fractional = coordinate_array @ np.linalg.inv(lattice_array)
    fractional %= 1.0
    fractional[np.isclose(fractional, 1.0, atol=_TRANSFORM_TOLERANCE)] = 0.0
    inverse_transform = np.linalg.inv(integer_transform.astype(float))

    corners = np.asarray(list(product((0, 1), repeat=3)), dtype=int) @ integer_transform
    lower = np.floor(corners.min(axis=0)).astype(int) - 1
    upper = np.ceil(corners.max(axis=0)).astype(int) + 1
    translations = np.asarray(
        list(
            product(
                range(lower[0], upper[0] + 1),
                range(lower[1], upper[1] + 1),
                range(lower[2], upper[2] + 1),
            )
        ),
        dtype=float,
    )

    generated_elements: list[object] = []
    generated_fractional: list[np.ndarray] = []
    symbols = element_array.astype(str)
    for symbol in dict.fromkeys(symbols.tolist()):
        for atom_index in np.flatnonzero(symbols == symbol):
            candidates = (fractional[atom_index] + translations) @ inverse_transform
            inside = np.all(candidates >= -_TRANSFORM_TOLERANCE, axis=1) & np.all(
                candidates < 1.0 - _TRANSFORM_TOLERANCE, axis=1
            )
            accepted = candidates[inside]
            accepted[np.abs(accepted) <= _TRANSFORM_TOLERANCE] = 0.0
            for transformed_fractional in accepted:
                generated_elements.append(element_array[atom_index])
                generated_fractional.append(transformed_fractional)

    expected_count = abs(transform_determinant) * element_array.size
    if len(generated_elements) != expected_count:
        raise ValueError(
            "The lattice transformation generated an unexpected number of atoms "
            f"({len(generated_elements)} instead of {expected_count})."
        )

    new_lattice = integer_transform @ lattice_array
    if generated_fractional:
        new_coordinates = np.asarray(generated_fractional, dtype=float) @ new_lattice
    else:
        new_coordinates = np.empty((0, 3), dtype=float)
    new_elements = np.asarray(generated_elements, dtype=element_array.dtype)
    return new_elements, new_coordinates, new_lattice
