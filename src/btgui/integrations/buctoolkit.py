"""Explicit file-I/O bridge to BUCToolkit; the GUI does not import this module."""

from pathlib import Path

import numpy as np


def make_buctoolkit_handlers():
    """Create I/O callbacks that retain the loaded BatchStructures instance.

    Args:
        None.

    Return:
        ``(open_handler, save_handler)`` callbacks for MainWindow.
    """
    current_structures = None
    appended_batch_id = 0
    try:
        from BUCToolkit.Preprocessing.load_files import (
            Cif2Feat,
            ExtXyz2Feat,
            OUTCAR2Feat,
            POSCARs2Feat,
            Xyz2Feat,
        )
    except ImportError:
        Cif2Feat = ExtXyz2Feat = OUTCAR2Feat = POSCARs2Feat = Xyz2Feat = None

    def open_handler(paths):
        """Read selected structure files in batches and append them.

        Args:
            paths: One path or a sequence of POSCAR, CIF, XYZ, or EXTXYZ paths.

        Return:
            Complete aggregate per-atom elements, coordinates, and cell lists.
        """
        nonlocal appended_batch_id, current_structures
        if Cif2Feat is None:
            raise RuntimeError("BUCToolkit is not installed; file I/O is unavailable")
        if isinstance(paths, (str, Path)):
            path_list = [Path(paths)]
        else:
            path_list = [Path(path) for path in paths]
        if len(path_list) == 0:
            raise ValueError("at least one structure file must be selected")

        file_groups = {}
        for path in path_list:
            file_path = path.expanduser().resolve()
            if file_path.is_dir():
                directory_files = sorted(
                    item for item in file_path.iterdir()
                    if item.is_file()
                    and (
                        item.suffix.lower() in {".cif", ".extxyz", ".xyz", ".vasp", ".poscar"}
                        or item.name.upper() in {"POSCAR", "OUTCAR"}
                        or "OUTCAR" in item.name.upper()
                    )
                )
                for directory_file in directory_files:
                    file_groups.setdefault(_reader_group(directory_file), []).append(directory_file.name)
            else:
                file_groups.setdefault(_reader_group(file_path), []).append(file_path.name)
        if len(file_groups) == 0:
            raise ValueError("the selected paths contain no supported structure files")

        for (reader_kind, directory), file_names in file_groups.items():
            try:
                loaded_structures = _read_group(reader_kind, directory, file_names)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise ValueError(
                    "Unknown file format! Please check your input file. "
                    "Support cif, xyz, vasp files."
                ) from error
            loaded_structures.direct2cartesian()
            loaded_structures.generate_atom_list()
            if current_structures is None:
                current_structures = loaded_structures
                appended_batch_id = len(current_structures)
            else:
                # Reader sample ids are often file names. Replace them with
                # adapter-owned ids before appending mixed-format groups.
                sample_ids = [
                    f"btgui_append_{appended_batch_id + index}"
                    for index in range(len(loaded_structures.Sample_ids))
                ]
                appended_batch_id += len(sample_ids)
                current_structures.append_from_lists(
                    add_Sample_ids=sample_ids,
                    add_Cells=loaded_structures.Cells,
                    add_Elements=loaded_structures.Elements,
                    add_Numbers=loaded_structures.Numbers,
                    add_Coords_type=loaded_structures.Coords_type,
                    add_Coords=loaded_structures.Coords,
                    add_Fixed=loaded_structures.Fixed,
                    add_Energies=loaded_structures.Energies,
                    add_Forces=loaded_structures.Forces,
                    add_Labels=loaded_structures.Labels,
                )
                current_structures.generate_atom_list(force_update=True)
        return current_structures.Atom_list, current_structures.Coords, current_structures.Cells

    def _reader_group(file_path):
        """Classify one path by reader type and parent directory.

        Args:
            file_path: Resolved structure-file path.

        Return:
            A ``(reader_kind, directory)`` tuple used to batch compatible files.
        """
        suffix = file_path.suffix.lower()
        if file_path.name.upper() == "POSCAR" or suffix in {".vasp", ".poscar"}:
            reader_kind = "poscar"
        elif suffix == ".cif":
            reader_kind = "cif"
        elif "OUTCAR" in file_path.name.upper() or suffix == ".outcar":
            reader_kind = "outcar"
        elif suffix == ".extxyz":
            reader_kind = "extxyz"
        elif suffix == ".xyz":
            reader_kind = "xyz"
        else:
            reader_kind = "poscar"
        return reader_kind, str(file_path.parent)

    def _read_group(reader_kind, directory, file_names):
        """Read one directory group through its BUCToolkit reader.

        Args:
            reader_kind: Reader key returned by ``_reader_group``.
            directory: Parent directory containing all file names.
            file_names: Basenames passed as one reader ``file_list``.

        Return:
            A loaded ``BatchStructures`` instance.
        """
        if reader_kind == "poscar":
            loaded_structures = POSCARs2Feat(directory, verbose=0)
            loaded_structures.read(file_names, output_coord_type="cartesian")
        elif reader_kind == "cif":
            loaded_structures = Cif2Feat(directory, verbose=0)
            loaded_structures.read(file_names, n_core=1)
        elif reader_kind == "extxyz":
            loaded_structures = ExtXyz2Feat(directory, verbose=0)
            loaded_structures.read(file_names, n_core=1)
        elif reader_kind == "outcar":
            loaded_structures = OUTCAR2Feat(directory, verbose=0)
            loaded_structures.read(file_names, n_core=1)
        else:
            loaded_structures = Xyz2Feat(directory, verbose=0)
            loaded_structures.read(file_names)
        return loaded_structures

    def save_handler(path, elements, coordinates, lattices):
        """Write the retained BatchStructures object with its native text writer.

        Args:
            path: Destination path ending in .vasp, .poscar, .cif, or .xyz.
            elements: The exact per-atom element list returned by open_handler.
            coordinates: The exact coordinate list returned by open_handler.
            lattices: The exact lattice list returned by open_handler.

        Return:
            None. Writes exactly the frame represented by the one-item lists.
        """
        if current_structures is None:
            raise RuntimeError("open a BUCToolkit structure before saving")
        if len(coordinates) != 1 or len(elements) != 1 or len(lattices) != 1:
            raise ValueError("Save As exports exactly one current frame")
        matching_indices = [
            index for index, frame in enumerate(current_structures.Coords)
            if frame is coordinates[0]
        ]
        if len(matching_indices) != 1:
            raise ValueError("the current frame is not owned by the loaded BUCToolkit structure")
        frame_index = matching_indices[0]
        frame_elements = np.asarray(elements[0], dtype=str).reshape(-1)
        frame_coordinates = np.asarray(coordinates[0], dtype=float)
        frame_lattice = np.asarray(lattices[0], dtype=float)
        element_order = list(dict.fromkeys(frame_elements.tolist()))
        element_counts = [
            int(np.count_nonzero(frame_elements == symbol)) for symbol in element_order
        ]
        current_structures.Atom_list[frame_index] = frame_elements.copy()
        current_structures.Coords[frame_index] = frame_coordinates.copy()
        current_structures.Cells[frame_index] = frame_lattice.copy()
        current_structures.Elements[frame_index] = element_order
        current_structures.Numbers[frame_index] = element_counts
        file_path = Path(path).expanduser().resolve()
        formats = {".vasp": "POSCAR", ".poscar": "POSCAR", ".cif": "cif", ".xyz": "xyz"}
        if file_path.suffix.lower() not in formats:
            raise ValueError("save path must end with .vasp, .poscar, .cif, or .xyz")
        if formats[file_path.suffix.lower()] == "xyz" and current_structures.Energies is None:
            raise ValueError("BUCToolkit XYZ writing requires energies; save as POSCAR or CIF instead")
        output_names = [file_path.name]
        current_structures.write2text(
            output_path=str(file_path.parent),
            file_format=formats[file_path.suffix.lower()],
            indices=matching_indices,
            file_name_list=output_names,
            n_core=1,
        )

    def export_batch(output_directory, indices, file_format):
        """Export selected frames from the retained BatchStructures object.

        Args:
            output_directory: Directory receiving generated text files.
            indices: Zero-based selected frame indices.
            file_format: BUCToolkit format name.

        Return:
            None. BUCToolkit writes one file per selected frame.
        """
        if current_structures is None:
            raise RuntimeError("open a BUCToolkit structure before exporting")
        formats = {"POSCAR": "POSCAR", "cif": "cif", "xyz": "xyz"}
        if file_format not in formats:
            raise ValueError(f"unsupported export format: {file_format}")
        output_directory = Path(output_directory).expanduser().resolve()
        output_names = [
            f"frame_{index + 1:04d}.{'vasp' if file_format == 'POSCAR' else file_format}"
            for index in indices
        ]
        current_structures.write2text(
            output_path=str(output_directory),
            indices=list(indices),
            file_format=formats[file_format],
            file_name_list=output_names,
            n_core=1,
        )

    open_handler.returns_aggregate = True
    save_handler.export_batch = export_batch
    return open_handler, save_handler
