# btgui

An interactive PySide6 and PyVista viewer for molecular and crystal structures.

## Input API

Pass three frame-ordered lists. Frame `i` in each list describes the same structure:

```python
elements[i]     # N element symbols, one per atom
coordinates[i]  # shape (N, 3), Cartesian coordinates in angstroms
lattices[i]     # shape (3, 3), lattice vectors as rows
```

The lists are retained directly and coordinate editing changes `coordinates[i]` in place. Use `MainWindow.set_data(elements, coordinates, lattices)` when embedding the viewer. The GUI package has no BUCToolkit import dependency. An application can supply `open_handler(path)` and `save_handler(path, elements, coordinates, lattices)` callbacks to connect its preferred I/O implementation.

For BUCToolkit integration, pass its lists explicitly after expanding element counts:

```python
structures.generate_atom_list()
window.set_data(structures.Atom_list, structures.Coords, structures.Cells)
```

## Run

Install the project and launch the viewer with:

```text
pip install -e .
btgui
```

The standalone entry point supplies a BUCToolkit I/O bridge. File Open supports multi-selection and groups files by reader and parent directory before appending them to one retained `BatchStructures` object. The command `open PATH` accepts one path or a path list and reads POSCAR, CIF, XYZ, EXTXYZ, or complete OUTCAR trajectories; `save PATH` writes POSCAR, CIF, or XYZ through the retained batch. File > Export batch writes a selected inclusive one-based frame list such as `1-2, 4, 6,8,10-30` in POSCAR, CIF, or XYZ format. The bridge preloads BUCToolkit at application startup when it is available. Other hosts can supply their own I/O callbacks without adding BUCToolkit imports to the GUI.

## Mouse controls

- Right-drag: orbit the view.
- Shift+right-drag: translate selected atoms.
- Shift+Alt+right-drag near the selection: horizontal motion rotates around the
  screen vertical axis, while vertical motion rotates around the screen
  horizontal axis.
- Shift+Alt+right-drag near a window edge: circular pointer motion rotates
  around the screen-normal axis.
- Left-click: select one atom or clear selection on blank space.
- Ctrl+left-click: toggle an atom in the selection.
- Left-drag: select atoms inside a screen-space rectangle; Ctrl adds/toggles.
- Scroll: zoom.

Metrics > Length and Metrics > Angle temporarily replace normal atom selection. Pick
two atoms for a distance or three atoms in endpoint-vertex-endpoint order for an
angle. Measurements remain visible across frames and update from the current
coordinates; clicking empty space exits the special selector.

Selected-atom transforms are previewed through VTK actors while dragging. The
coordinate array and covalent bonds are updated once on release. Use `Ctrl+Z`
and `Ctrl+Y`, or the toolbar arrows, to undo and redo committed coordinate
edits. The undo limit and default atom material, lighting, and boundary style
are editable through Settings and stored in `src/btgui/settings.json` (or the
equivalent installed package directory).

The toolbar frame controls use zero-based indices. Save As writes only the
currently displayed frame. Opening another file appends
its frames to the current batch; the current frame field rejects invalid or
out-of-range input without changing the displayed frame. Appearance contains
per-element settings and global VTK material, lighting, and atom-boundary
settings. Global atom radius mode and renderer background color are configured
in Appearance; per-element controls retain only color and opacity.

Enter `help` in the bottom command line for available commands. Property overlays can be added with `MainWindow.register_property_renderer(name, callback)`; the callback receives `(plotter, frame_index, positions)`.
