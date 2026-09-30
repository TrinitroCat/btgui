"""Small serial structure readers and writers used by the standalone GUI.

Portions of the parsing and text-output behavior are adapted from BUCToolkit,
Copyright (c) 2024-2026 Pu Pengxin and Song Xin, under the MIT License.
"""

from __future__ import annotations

from pathlib import Path
import re
import shlex
from typing import Iterable

import numpy as np


_SUPPORTED_SUFFIXES = {".cif", ".extxyz", ".xyz", ".vasp", ".poscar", ".outcar"}
_UNKNOWN_FORMAT_MESSAGE = (
    "Unknown file format! Please check your input file. "
    "Support cif, xyz, vasp files."
)


def _normalize_paths(paths) -> list[Path]:
    """Expand selected files and directories into an ordered file list.

    Args:
        paths: One filesystem path or an iterable of paths.

    Return:
        Resolved files accepted by the built-in structure readers.
    """
    if isinstance(paths, (str, Path)):
        selected = [Path(paths)]
    else:
        selected = [Path(path) for path in paths]
    files = []
    for selected_path in selected:
        selected_path = selected_path.expanduser().resolve()
        if selected_path.is_dir():
            files.extend(
                sorted(
                    candidate
                    for candidate in selected_path.iterdir()
                    if candidate.is_file() and _is_supported_path(candidate)
                )
            )
        elif selected_path.is_file():
            files.append(selected_path)
    if len(files) == 0:
        raise ValueError("the selected paths contain no supported structure files")
    return files


def _is_supported_path(path: Path) -> bool:
    """Return whether one path is handled by a built-in reader.

    Args:
        path: Candidate structure path.

    Return:
        True for known suffixes and POSCAR/OUTCAR-style file names.
    """
    name = path.name.upper()
    return path.suffix.lower() in _SUPPORTED_SUFFIXES or name == "POSCAR" or "OUTCAR" in name


def _reader_kind(path: Path) -> str:
    """Classify a structure path for reader dispatch.

    Args:
        path: Resolved input file.

    Return:
        Reader name used by read_structure_files.
    """
    suffix = path.suffix.lower()
    name = path.name.upper()
    if name == "POSCAR" or suffix in {".vasp", ".poscar"}:
        return "poscar"
    if suffix == ".cif":
        return "cif"
    if "OUTCAR" in name or suffix == ".outcar":
        return "outcar"
    if suffix == ".extxyz":
        return "extxyz"
    if suffix == ".xyz":
        return "xyz"
    return "poscar"


def _cell_from_lengths_angles(lengths: Iterable[float], angles: Iterable[float]) -> np.ndarray:
    """Construct row-vector lattice vectors from lengths and angles.

    Args:
        lengths: Cell lengths a, b, and c.
        angles: Cell angles alpha, beta, and gamma in degrees.

    Return:
        A Cartesian lattice matrix with shape (3, 3).
    """
    a, b, c = [float(value) for value in lengths]
    alpha, beta, gamma = np.deg2rad([float(value) for value in angles])
    b_x = b * np.cos(gamma)
    b_y = b * np.sin(gamma)
    if abs(float(b_y)) < 1.0e-12:
        raise ValueError("CIF gamma angle produces a singular lattice")
    c_x = c * np.cos(beta)
    c_y = (b * c * np.cos(alpha) - b_x * c_x) / b_y
    c_z_squared = c * c - c_x * c_x - c_y * c_y
    if c_z_squared < -1.0e-8:
        raise ValueError("CIF cell lengths and angles are inconsistent")
    return np.asarray(
        [[a, 0.0, 0.0], [b_x, b_y, 0.0], [c_x, c_y, np.sqrt(max(c_z_squared, 0.0))]],
        dtype=float,
    )


def _parse_poscar(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Read one VASP 5 style POSCAR into raw GUI lists.

    Args:
        path: POSCAR-like input file.

    Return:
        One-frame element, Cartesian-coordinate, and lattice lists.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 8:
        raise ValueError("POSCAR contains too few lines")
    scale = float(lines[1].split()[0])
    if scale == 0.0:
        raise ValueError("POSCAR scale factor must be non-zero")
    lattice = np.asarray([lines[index].split()[:3] for index in range(2, 5)], dtype=float)
    if scale < 0.0:
        determinant = abs(float(np.linalg.det(lattice)))
        if determinant == 0.0:
            raise ValueError("negative POSCAR volume requires a non-singular lattice")
        scale = (abs(scale) / determinant) ** (1.0 / 3.0)
    lattice *= scale
    symbols = lines[5].split()
    try:
        counts = [int(value) for value in lines[6].split()]
    except ValueError as error:
        raise ValueError("VASP 4 POSCAR files without element symbols are unsupported") from error
    if len(symbols) != len(counts) or sum(counts) <= 0:
        raise ValueError("POSCAR element symbols and counts are inconsistent")
    coordinate_line = 7
    if lines[coordinate_line].lstrip().lower().startswith("s"):
        coordinate_line += 1
    coordinate_kind = lines[coordinate_line].strip().lower()
    coordinate_start = coordinate_line + 1
    atom_count = sum(counts)
    coordinate_lines = lines[coordinate_start : coordinate_start + atom_count]
    if len(coordinate_lines) != atom_count:
        raise ValueError("POSCAR coordinate count is incomplete")
    coordinates = np.asarray([line.split()[:3] for line in coordinate_lines], dtype=float)
    if coordinate_kind.startswith(("d", "f")):
        coordinates = coordinates @ lattice
    elif coordinate_kind.startswith(("c", "k")):
        coordinates *= scale
    else:
        raise ValueError("POSCAR coordinate mode must be Direct or Cartesian")
    elements = np.asarray(
        [symbol for symbol, count in zip(symbols, counts) for _ in range(count)],
        dtype=str,
    )
    return [elements], [coordinates], [lattice]


def _parse_xyz(path: Path, extended: bool) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Read standard or extended XYZ frames.

    Args:
        path: XYZ or EXTXYZ input file.
        extended: Whether to interpret Lattice and Properties metadata.

    Return:
        Frame-ordered element, Cartesian-coordinate, and lattice lists.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    elements = []
    coordinates = []
    lattices = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        try:
            atom_count = int(lines[cursor].strip())
        except ValueError as error:
            raise ValueError(f"invalid XYZ atom count at line {cursor + 1}") from error
        if atom_count < 0 or cursor + atom_count + 1 >= len(lines):
            raise ValueError("XYZ frame is incomplete")
        comment = lines[cursor + 1]
        property_columns = {"species": (0, 1), "pos": (1, 4)}
        lattice = np.zeros((3, 3), dtype=float)
        if extended:
            metadata = {}
            for token in shlex.split(comment):
                if "=" in token:
                    key, value = token.split("=", 1)
                    metadata[key.lower()] = value
            if "lattice" in metadata:
                lattice_values = np.fromstring(metadata["lattice"], sep=" ", dtype=float)
                if lattice_values.size != 9:
                    raise ValueError("EXTXYZ Lattice must contain nine numbers")
                lattice = lattice_values.reshape(3, 3)
            if "properties" in metadata:
                schema = metadata["properties"].split(":")
                if len(schema) % 3 != 0:
                    raise ValueError("EXTXYZ Properties metadata is malformed")
                property_columns = {}
                start = 0
                for index in range(0, len(schema), 3):
                    width = int(schema[index + 2])
                    property_columns[schema[index].lower()] = (start, start + width)
                    start += width
        atom_lines = [line.split() for line in lines[cursor + 2 : cursor + 2 + atom_count]]
        species_column = property_columns.get("species", property_columns.get("element"))
        position_column = property_columns.get("pos", property_columns.get("positions"))
        if species_column is None or position_column is None:
            raise ValueError("XYZ frame does not define species and positions")
        frame_elements = np.asarray(
            [row[species_column[0]] for row in atom_lines],
            dtype=str,
        )
        frame_coordinates = np.asarray(
            [row[position_column[0] : position_column[1]] for row in atom_lines],
            dtype=float,
        )
        if frame_coordinates.shape != (atom_count, 3):
            raise ValueError("XYZ positions must have three components")
        elements.append(frame_elements)
        coordinates.append(frame_coordinates)
        lattices.append(lattice)
        cursor += atom_count + 2
    if len(elements) == 0:
        raise ValueError("XYZ file contains no structures")
    return elements, coordinates, lattices


def _cif_number(value: str) -> float:
    """Convert one CIF numeric token, discarding uncertainty notation.

    Args:
        value: CIF numeric token such as 5.431(2).

    Return:
        Floating-point value without its parenthesized uncertainty.
    """
    return float(re.sub(r"\([^)]*\)$", "", value.strip()))


def _parse_cif(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Read explicit atom sites from one CIF data block.

    Args:
        path: CIF input file.

    Return:
        One-frame element, Cartesian-coordinate, and lattice lists.

    Note:
        Symmetry operations are not expanded; this matches the former
        BUCToolkit reader used by btgui.
    """
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    values = {}
    atom_headers = []
    atom_rows = []
    cursor = 0
    while cursor < len(raw_lines):
        line = raw_lines[cursor].strip()
        if not line or line.startswith("#"):
            cursor += 1
            continue
        if line.lower() == "loop_":
            cursor += 1
            headers = []
            while cursor < len(raw_lines) and raw_lines[cursor].strip().startswith("_"):
                headers.append(raw_lines[cursor].strip().split()[0].lower())
                cursor += 1
            rows = []
            while cursor < len(raw_lines):
                row = raw_lines[cursor].strip()
                if not row or row.startswith("#"):
                    cursor += 1
                    continue
                lower = row.lower()
                if lower == "loop_" or row.startswith("_") or lower.startswith("data_"):
                    break
                rows.append(shlex.split(row))
                cursor += 1
            if any(header.startswith("_atom_site_") for header in headers):
                atom_headers = headers
                atom_rows = rows
            continue
        if line.startswith("_"):
            tokens = shlex.split(line)
            if len(tokens) >= 2:
                values[tokens[0].lower()] = tokens[1]
        cursor += 1
    required_cell = (
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    )
    if any(name not in values for name in required_cell):
        raise ValueError("CIF cell parameters are incomplete")
    lattice = _cell_from_lengths_angles(
        [_cif_number(values[name]) for name in required_cell[:3]],
        [_cif_number(values[name]) for name in required_cell[3:]],
    )
    if len(atom_headers) == 0 or len(atom_rows) == 0:
        raise ValueError("CIF atom-site loop was not found")
    header_indices = {name: index for index, name in enumerate(atom_headers)}
    symbol_index = header_indices.get("_atom_site_type_symbol")
    if symbol_index is None:
        symbol_index = header_indices.get("_atom_site_label")
    fractional_names = ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")
    cartesian_names = ("_atom_site_cartn_x", "_atom_site_cartn_y", "_atom_site_cartn_z")
    if symbol_index is None:
        raise ValueError("CIF atom symbols were not found")
    if all(name in header_indices for name in fractional_names):
        coordinate_indices = [header_indices[name] for name in fractional_names]
        fractional = True
    elif all(name in header_indices for name in cartesian_names):
        coordinate_indices = [header_indices[name] for name in cartesian_names]
        fractional = False
    else:
        raise ValueError("CIF atom coordinates were not found")
    frame_elements = []
    frame_coordinates = []
    for row in atom_rows:
        if len(row) < len(atom_headers):
            continue
        symbol = re.match(r"[A-Za-z]+", row[symbol_index])
        if symbol is None:
            raise ValueError("CIF contains an invalid atom symbol")
        frame_elements.append(symbol.group())
        frame_coordinates.append([_cif_number(row[index]) for index in coordinate_indices])
    coordinates = np.asarray(frame_coordinates, dtype=float)
    if fractional:
        coordinates = coordinates @ lattice
    return [np.asarray(frame_elements, dtype=str)], [coordinates], [lattice]


def _parse_outcar(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Read all ionic POSITION/TOTAL-FORCE frames from one OUTCAR.

    Args:
        path: VASP OUTCAR input file.

    Return:
        Frame-ordered elements, Cartesian coordinates, and lattice matrices.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    counts = None
    potcar_symbols = []
    for line in lines:
        if "ions per type" in line:
            counts = [int(value) for value in line.split("=", 1)[1].split()]
        if "POTCAR:" in line:
            match = re.search(r"POTCAR:\s+\S+\s+([A-Z][a-z]?)(?:_|\s|$)", line)
            if match is not None:
                potcar_symbols.append(match.group(1))
    if counts is None:
        raise ValueError("OUTCAR ions per type information was not found")
    type_count = len(counts)
    symbols = potcar_symbols[:type_count]
    if len(symbols) != type_count:
        raise ValueError("OUTCAR POTCAR element information was not found")
    frame_elements = np.asarray(
        [symbol for symbol, count in zip(symbols, counts) for _ in range(count)],
        dtype=str,
    )
    atom_count = len(frame_elements)
    elements = []
    coordinates = []
    lattices = []
    current_lattice = np.zeros((3, 3), dtype=float)
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor]
        if "direct lattice vectors" in line.lower() and cursor + 3 < len(lines):
            candidate = []
            for offset in range(1, 4):
                numbers = lines[cursor + offset].split()
                if len(numbers) < 3:
                    break
                candidate.append(numbers[:3])
            if len(candidate) == 3:
                current_lattice = np.asarray(candidate, dtype=float)
            cursor += 4
            continue
        if "POSITION" in line and "TOTAL-FORCE" in line:
            data_start = cursor + 1
            while data_start < len(lines) and set(lines[data_start].strip()) <= {"-", " "}:
                data_start += 1
            rows = []
            for atom_index in range(atom_count):
                if data_start + atom_index >= len(lines):
                    break
                fields = lines[data_start + atom_index].split()
                if len(fields) < 3:
                    break
                rows.append(fields[:3])
            if len(rows) == atom_count:
                elements.append(frame_elements.copy())
                coordinates.append(np.asarray(rows, dtype=float))
                lattices.append(current_lattice.copy())
                cursor = data_start + atom_count
                continue
        cursor += 1
    if len(coordinates) == 0:
        raise ValueError("OUTCAR contains no complete ionic frames")
    return elements, coordinates, lattices


def read_structure_files(paths) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Read supported files into the three raw frame lists.

    Args:
        paths: One path or a sequence of selected files and directories.

    Return:
        Complete element, Cartesian-coordinate, and lattice lists.
    """
    elements = []
    coordinates = []
    lattices = []
    for path in _normalize_paths(paths):
        kind = _reader_kind(path)
        try:
            if kind == "poscar":
                loaded = _parse_poscar(path)
            elif kind == "cif":
                loaded = _parse_cif(path)
            elif kind == "outcar":
                loaded = _parse_outcar(path)
            elif kind == "extxyz":
                loaded = _parse_xyz(path, extended=True)
            else:
                loaded = _parse_xyz(path, extended=False)
        except (IndexError, KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError(_UNKNOWN_FORMAT_MESSAGE) from error
        elements.extend(loaded[0])
        coordinates.extend(loaded[1])
        lattices.extend(loaded[2])
    return elements, coordinates, lattices


def _group_for_text_output(
    elements: np.ndarray, coordinates: np.ndarray
) -> tuple[list[str], list[int], np.ndarray]:
    """Group per-atom arrays by first element appearance for text formats.

    Args:
        elements: Per-atom element symbols.
        coordinates: Matching Cartesian coordinates.

    Return:
        Ordered symbols, counts, and coordinates grouped by symbol.
    """
    symbols = list(dict.fromkeys(np.asarray(elements, dtype=str).tolist()))
    indices = [np.flatnonzero(np.asarray(elements, dtype=str) == symbol) for symbol in symbols]
    counts = [len(index) for index in indices]
    grouped = np.concatenate([coordinates[index] for index in indices], axis=0)
    return symbols, counts, grouped


def _write_poscar(path: Path, elements: np.ndarray, coordinates: np.ndarray, lattice: np.ndarray) -> None:
    """Write one frame as a Cartesian POSCAR.

    Args:
        path: Destination file.
        elements: Per-atom element symbols.
        coordinates: Cartesian coordinates.
        lattice: Row-vector lattice matrix.

    Return:
        None. The destination is replaced.
    """
    symbols, counts, grouped = _group_for_text_output(elements, coordinates)
    lines = ["btgui", "1.0"]
    lines.extend("  " + "  ".join(f"{value:.16g}" for value in row) for row in lattice)
    lines.append("  " + "  ".join(symbols))
    lines.append("  " + "  ".join(str(count) for count in counts))
    lines.append("Cartesian")
    lines.extend("  " + "  ".join(f"{value:.16g}" for value in row) for row in grouped)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_xyz(path: Path, elements: np.ndarray, coordinates: np.ndarray, lattice: np.ndarray) -> None:
    """Write one frame as standard XYZ with optional lattice metadata.

    Args:
        path: Destination file.
        elements: Per-atom element symbols.
        coordinates: Cartesian coordinates.
        lattice: Row-vector lattice matrix.

    Return:
        None. The destination is replaced.
    """
    if np.any(np.asarray(lattice, dtype=float)):
        lattice_text = " ".join(f"{value:.16g}" for value in np.asarray(lattice).reshape(-1))
        comment = f'Lattice="{lattice_text}" Properties=species:S:1:pos:R:3'
    else:
        comment = "generated by btgui"
    lines = [str(len(elements)), comment]
    lines.extend(
        f"{symbol}  {position[0]:.16g}  {position[1]:.16g}  {position[2]:.16g}"
        for symbol, position in zip(elements, coordinates)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_cif(path: Path, elements: np.ndarray, coordinates: np.ndarray, lattice: np.ndarray) -> None:
    """Write one frame as a P1 CIF.

    Args:
        path: Destination file.
        elements: Per-atom element symbols.
        coordinates: Cartesian coordinates.
        lattice: Row-vector lattice matrix.

    Return:
        None. The destination is replaced.
    """
    lattice = np.asarray(lattice, dtype=float)
    determinant = float(np.linalg.det(lattice))
    if abs(determinant) < 1.0e-12:
        raise ValueError("CIF output requires a non-singular lattice")
    lengths = np.linalg.norm(lattice, axis=1)
    alpha = np.degrees(np.arccos(np.clip(np.dot(lattice[1], lattice[2]) / (lengths[1] * lengths[2]), -1.0, 1.0)))
    beta = np.degrees(np.arccos(np.clip(np.dot(lattice[0], lattice[2]) / (lengths[0] * lengths[2]), -1.0, 1.0)))
    gamma = np.degrees(np.arccos(np.clip(np.dot(lattice[0], lattice[1]) / (lengths[0] * lengths[1]), -1.0, 1.0)))
    fractional = np.asarray(coordinates, dtype=float) @ np.linalg.inv(lattice)
    lines = [
        "data_btgui",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        f"_cell_length_a {lengths[0]:.16g}",
        f"_cell_length_b {lengths[1]:.16g}",
        f"_cell_length_c {lengths[2]:.16g}",
        f"_cell_angle_alpha {alpha:.16g}",
        f"_cell_angle_beta {beta:.16g}",
        f"_cell_angle_gamma {gamma:.16g}",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]
    symbol_counts = {}
    for symbol, position in zip(np.asarray(elements, dtype=str), fractional):
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        label = f"{symbol}{symbol_counts[symbol]}"
        lines.append(
            f"{label} {symbol} {position[0]:.16g} {position[1]:.16g} {position[2]:.16g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_structure(path, elements, coordinates, lattice, file_format=None) -> None:
    """Write one raw-list frame in a supported text format.

    Args:
        path: Destination file.
        elements: Per-atom element symbols.
        coordinates: Cartesian coordinates.
        lattice: Row-vector lattice matrix.
        file_format: Optional POSCAR, cif, or xyz override.

    Return:
        None. Parent directories are created when needed.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    requested = str(file_format).lower() if file_format is not None else path.suffix.lower().lstrip(".")
    if path.name.upper() == "POSCAR" or requested in {"poscar", "vasp"}:
        _write_poscar(path, np.asarray(elements), np.asarray(coordinates, dtype=float), np.asarray(lattice, dtype=float))
    elif requested == "cif":
        _write_cif(path, np.asarray(elements), np.asarray(coordinates, dtype=float), np.asarray(lattice, dtype=float))
    elif requested in {"xyz", "extxyz"}:
        _write_xyz(path, np.asarray(elements), np.asarray(coordinates, dtype=float), np.asarray(lattice, dtype=float))
    else:
        raise ValueError("save path must end with .vasp, .poscar, .cif, or .xyz")


def make_structure_handlers():
    """Create raw-list open, save, and batch-export callbacks.

    Args:
        None.

    Return:
        An open handler and a save handler with an export_batch attribute.
    """
    def open_handler(paths):
        """Read selected files without retaining a structure manager.

        Args:
            paths: One path or a sequence of files and directories.

        Return:
            Newly read element, coordinate, and lattice frame lists.
        """
        return read_structure_files(paths)

    def save_handler(path, elements, coordinates, lattices):
        """Write exactly one frame supplied by MainWindow.

        Args:
            path: Destination file.
            elements: One-item element list.
            coordinates: One-item coordinate list.
            lattices: One-item lattice list.

        Return:
            None.
        """
        if len(elements) != 1 or len(coordinates) != 1 or len(lattices) != 1:
            raise ValueError("Save As exports exactly one current frame")
        write_structure(path, elements[0], coordinates[0], lattices[0])

    def export_batch(output_directory, indices, file_format, elements, coordinates, lattices):
        """Write selected frames from the authoritative GUI lists.

        Args:
            output_directory: Destination directory.
            indices: Zero-based selected frame indices.
            file_format: POSCAR, cif, or xyz.
            elements: Authoritative per-frame element lists.
            coordinates: Authoritative per-frame coordinate lists.
            lattices: Authoritative per-frame lattice lists.

        Return:
            None. Each selected frame is written to a separate file.
        """
        output_directory = Path(output_directory).expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        normalized_format = str(file_format).lower()
        extension = "vasp" if normalized_format == "poscar" else normalized_format
        for frame_index in indices:
            write_structure(
                output_directory / f"frame_{frame_index + 1:04d}.{extension}",
                elements[frame_index],
                coordinates[frame_index],
                lattices[frame_index],
                file_format=normalized_format,
            )

    open_handler.returns_aggregate = False
    save_handler.export_batch = export_batch
    return open_handler, save_handler
