"""Small array-oriented operations used by the graphical interface."""

import numpy as np

from btgui.data.appearance import ELEMENT_COLORS, covalent_radius


def distance_between(coordinates: np.ndarray, first: int, second: int) -> float:
    """Return the Euclidean distance between two atoms in one frame.

    Args:
        coordinates: Cartesian coordinates with shape ``(N, 3)``.
        first: Index of the first atom.
        second: Index of the second atom.

    Return:
        Distance in the coordinate unit.
    """
    vector = np.asarray(coordinates[second], dtype=float) - np.asarray(coordinates[first], dtype=float)
    return float(np.linalg.norm(vector))


def angle_between(coordinates: np.ndarray, vertex: int, first: int, second: int) -> float:
    """Return the angle formed by two atoms around a vertex in degrees.

    Args:
        coordinates: Cartesian coordinates with shape ``(N, 3)``.
        vertex: Index of the central atom.
        first: Index of the first endpoint atom.
        second: Index of the second endpoint atom.

    Return:
        Angle in degrees in the closed interval from zero to 180.
    """
    first_vector = np.asarray(coordinates[first], dtype=float) - coordinates[vertex]
    second_vector = np.asarray(coordinates[second], dtype=float) - coordinates[vertex]
    first_norm = float(np.linalg.norm(first_vector))
    second_norm = float(np.linalg.norm(second_vector))
    if first_norm == 0.0 or second_norm == 0.0:
        raise ValueError("angle is undefined when an endpoint overlaps the vertex")
    cosine = np.dot(first_vector, second_vector) / (first_norm * second_norm)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def parse_frame_selection(specification: str, frame_count: int) -> list[int]:
    """Parse inclusive one-based frame ranges used by batch export.

    Args:
        specification: Comma-separated values such as ``"1-2, 4, 6-8"``.
        frame_count: Number of available frames.

    Return:
        Sorted unique zero-based frame indices.
    """
    if frame_count <= 0:
        raise ValueError("there are no frames to export")
    if not isinstance(specification, str) or not specification.strip():
        raise ValueError("frame selection must not be empty")
    selected = set()
    tokens = specification.split(",")
    if any(not token.strip() for token in tokens):
        raise ValueError("frame selection contains an empty item")
    for token in tokens:
        token = token.strip()
        parts = token.split("-")
        if len(parts) > 2 or any(not part.strip().isdigit() for part in parts):
            raise ValueError(f"invalid frame selection token: {token}")
        start = int(parts[0])
        stop = int(parts[-1])
        if start <= 0 or stop <= 0 or start > stop or stop > frame_count:
            raise ValueError(f"frame selection must stay within 1-{frame_count}")
        selected.update(range(start - 1, stop))
    if not selected:
        raise ValueError("frame selection is empty")
    return sorted(selected)


def parse_frame_scope(
    specification: str,
    frame_count: int,
    current_index: int,
) -> list[int]:
    """Parse a current/all/one-based inclusive frame selector.

    Args:
        specification: Empty text, ``all``, or comma-separated frame ranges.
        frame_count: Number of available frames.
        current_index: Zero-based frame returned for an empty selector.

    Return:
        Sorted unique zero-based frame indices.

    Raises:
        ValueError: If no frame exists or the selector is invalid.
    """
    if frame_count <= 0:
        raise ValueError("there are no frames")
    text = str(specification).strip()
    if not text:
        if not 0 <= current_index < frame_count:
            raise ValueError("the current frame is unavailable")
        return [int(current_index)]
    if text.lower() == "all":
        return list(range(frame_count))
    return parse_frame_selection(text, frame_count)


def build_bond_adjacency(
    coordinates: np.ndarray,
    elements: np.ndarray | list,
    tolerance: float,
) -> np.ndarray:
    """Build a sparse upper-triangle covalent-bond adjacency array.

    Args:
        coordinates: Cartesian coordinates with shape ``(N, 3)``.
        elements: Element symbols aligned with the coordinates.
        tolerance: Nonnegative covalent-radius cutoff multiplier.

    Return:
        An ``int64`` COO index array shaped ``(2, E)`` with ``left < right``.

    Raises:
        ValueError: If array shapes, symbols, or tolerance are invalid.
    """
    positions = np.asarray(coordinates, dtype=float)
    symbols = np.asarray(elements, dtype=str).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    if len(positions) != len(symbols):
        raise ValueError("elements and coordinates must have equal lengths")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("bond tolerance must be a finite nonnegative number")
    radii = np.asarray([covalent_radius(symbol) for symbol in symbols], dtype=float)
    left_parts = []
    right_parts = []
    for left in range(max(len(positions) - 1, 0)):
        vectors = positions[left + 1:] - positions[left]
        distances_squared = np.einsum("ij,ij->i", vectors, vectors)
        cutoffs = (radii[left] + radii[left + 1:]) * tolerance
        matches = np.flatnonzero(
            (distances_squared > 0.0) & (distances_squared <= cutoffs * cutoffs)
        )
        if len(matches):
            left_parts.append(np.full(len(matches), left, dtype=np.int64))
            right_parts.append((matches + left + 1).astype(np.int64, copy=False))
    if not left_parts:
        return np.empty((2, 0), dtype=np.int64)
    return np.vstack((np.concatenate(left_parts), np.concatenate(right_parts)))


def wrap_coordinates(coordinates: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Return Cartesian coordinates wrapped into one row-vector unit cell.

    Args:
        coordinates: Cartesian coordinates with shape ``(N, 3)``.
        lattice: Nonsingular row-vector lattice matrix with shape ``(3, 3)``.

    Return:
        A new Cartesian array whose fractional coordinates lie in ``[0, 1)``.

    Raises:
        ValueError: If input shapes are invalid or the lattice is singular.
    """
    positions = np.asarray(coordinates, dtype=float)
    cell = np.asarray(lattice, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    if cell.shape != (3, 3):
        raise ValueError("lattice must have shape (3, 3)")
    try:
        fractional = np.linalg.solve(cell.T, positions.T).T
    except np.linalg.LinAlgError as error:
        raise ValueError("cannot wrap coordinates in a singular lattice") from error
    fractional -= np.floor(fractional)
    return fractional @ cell


def validate_data(
    elements: list,
    coordinates: list[np.ndarray],
    lattices: list[np.ndarray],
) -> None:
    """Validate three frame-ordered lists of element, position, and cell arrays.

    Args:
        elements: Per-frame per-atom element symbols.
        coordinates: Per-frame Cartesian positions, each with shape ``(N, 3)``.
        lattices: Per-frame row-vector lattice matrices, each with shape ``(3, 3)``.

    Return:
        None. Raises ValueError when the arrays are inconsistent or non-finite.
    """
    if not (len(elements) == len(coordinates) == len(lattices)):
        raise ValueError("elements, coordinates, and lattices must have equal frame counts")
    for frame, (frame_elements, frame_coordinates, lattice) in enumerate(
        zip(elements, coordinates, lattices)
    ):
        if np.asarray(frame_elements).ndim != 1:
            raise ValueError(f"elements[{frame}] must have shape (N,)")
        if np.asarray(frame_coordinates).ndim != 2 or np.asarray(frame_coordinates).shape[1] != 3:
            raise ValueError(f"coordinates[{frame}] must have shape (N, 3)")
        if len(frame_elements) != len(frame_coordinates):
            raise ValueError(f"frame {frame} has different element and coordinate counts")
        if np.asarray(lattice).shape != (3, 3):
            raise ValueError(f"lattices[{frame}] must have shape (3, 3)")
        if not np.isfinite(frame_coordinates).all() or not np.isfinite(lattice).all():
            raise ValueError(f"frame {frame} contains non-finite coordinates or lattice values")


def translate_selected(
    coordinates: np.ndarray,
    selected_indices: np.ndarray,
    displacement: np.ndarray,
) -> None:
    """Translate selected atoms in place.

    Args:
        coordinates: Cartesian positions for one frame.
        selected_indices: Integer indices of the atoms to translate.
        displacement: Cartesian displacement vector with shape ``(3,)``.

    Return:
        None. The provided coordinate array is modified in place.
    """
    if len(selected_indices):
        coordinates[selected_indices] += displacement


def rotate_selected(
    coordinates: np.ndarray,
    selected_indices: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> None:
    """Rotate selected atoms around their geometric center in place.

    Args:
        coordinates: Cartesian positions for one frame.
        selected_indices: Integer indices of the atoms to rotate.
        axis: Cartesian rotation axis.
        angle: Rotation angle in radians.

    Return:
        None. The provided coordinate array is modified in place.
    """
    if not len(selected_indices):
        return
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return
    axis /= norm
    cosine = np.cos(angle)
    sine = np.sin(angle)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = cosine * np.eye(3) + (1.0 - cosine) * np.outer(axis, axis) + sine * cross
    center = coordinates[selected_indices].mean(axis=0)
    coordinates[selected_indices] = (coordinates[selected_indices] - center) @ rotation.T + center


def parse_frame_range(specification: str, frame_count: int) -> list[int]:
    """Convert Python-style ``start:stop:step`` text to valid frame indices.

    Args:
        specification: Slice text; omitted fields use Python slice defaults.
        frame_count: Number of available frames.

    Return:
        Frame indices produced by the slice, clamped to available frames.
    """
    parts = specification.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("frame range must use start:stop[:step] syntax")
    values = [int(part) if part.strip() else None for part in parts]
    if len(values) == 2:
        values.append(None)
    step = values[2] or 1
    if step == 0:
        raise ValueError("frame step cannot be zero")
    start, stop, step = slice(*values).indices(frame_count)
    return list(range(start, stop, step))


def parse_atom_selection(specification: str, elements: np.ndarray | list) -> list[int]:
    """Parse one-based atom numbers, ranges, and exact element-stub keys.

    Args:
        specification: Comma-separated selector text; empty selects all atoms.
        elements: Current-frame element symbols.

    Return:
        Sorted unique zero-based atom indices.

    Raises:
        ValueError: If a token is malformed, unknown, reversed, or out of range.
    """
    values = np.asarray(elements, dtype=str).reshape(-1)
    text = str(specification).strip()
    if not text:
        return list(range(len(values)))
    selected = set()
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError("atom selector contains an empty item")
        if token in ELEMENT_COLORS:
            selected.update(np.flatnonzero(values == token).tolist())
            continue
        parts = token.split("-")
        if len(parts) > 2 or any(not part.strip().isdigit() for part in parts):
            raise ValueError(f"invalid atom selector token: {token}")
        start = int(parts[0])
        stop = int(parts[-1])
        if start <= 0 or stop <= 0 or start > stop or stop > len(values):
            raise ValueError(f"atom selector must stay within 1-{len(values)}")
        selected.update(range(start - 1, stop))
    return sorted(selected)


def format_atom_selection(indices: list[int] | np.ndarray, elements: np.ndarray | list) -> str:
    """Format selected one-based atom indices with compact runs of three or more.

    Args:
        indices: Zero-based selected atom indices.
        elements: Current-frame element symbols used for optional grouping context.

    Return:
        Canonical comma-separated one-based selector text.
    """
    del elements
    values = sorted(set(int(index) for index in np.asarray(indices, dtype=int).tolist()))
    if not values:
        return ""
    result = []
    start = previous = values[0]
    for value in values[1:] + [None]:
        if value is not None and value == previous + 1:
            previous = value
            continue
        length = previous - start + 1
        if length >= 3:
            result.append(f"{start + 1}-{previous + 1}")
        else:
            result.extend(str(index + 1) for index in range(start, previous + 1))
        if value is not None:
            start = previous = value
    return ",".join(result)
