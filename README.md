# btgui

An interactive PySide6 and PyVista viewer for molecular and crystal structures.

## Input API

Pass three frame-ordered lists. Frame `i` in each list describes the same structure:

```python
elements[i]     # N element symbols, one per atom
coordinates[i]  # shape (N, 3), Cartesian coordinates in angstroms
lattices[i]     # shape (3, 3), lattice vectors as rows
```

The lists are retained directly and coordinate editing changes `coordinates[i]` in place. Use `MainWindow.set_data(elements, coordinates, lattices)` when embedding the viewer. The standalone GUI includes serial NumPy/Python readers and writers, so normal structure I/O has no BUCToolkit, PyTorch, or joblib dependency. An application can still supply `open_handler(path)` and `save_handler(path, elements, coordinates, lattices)` callbacks to connect another I/O implementation.

BUCToolkit remains an optional runtime backend only for the CLI `to_bt()` conversion. Add one or more site-package directories (or a directory containing the `BUCToolkit` package) to `buctoolkit_paths` in `src/btgui/settings.json`; these paths are tried first. `BTGUI_BUCTOOLKIT_PATH` or `BTGUI_BUCTOOLKIT_PATHS` may be used for machine-specific startup configuration, followed by the current Python environment. If no installation can be loaded, structure viewing and file I/O remain available and `to_bt()` reports that the optional backend is absent.

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

The standalone entry point uses built-in serial readers and writers. File Open supports multi-selection and appends POSCAR, CIF, XYZ, EXTXYZ, or complete OUTCAR trajectories directly to the three GUI lists. The command `open PATH` accepts one path or a path list; `save PATH` writes the current frame as POSCAR, CIF, or XYZ. File > Export batch writes a selected inclusive one-based frame list such as `1-2, 4, 6,8,10-30` in POSCAR, CIF, or XYZ format. The built-in path contains no multiprocessing branch and always exports the current authoritative GUI data. Portions adapted from BUCToolkit are distributed under its MIT license in `LICENSES/BUCToolkit-MIT.txt`.

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
angle. Measurements belong to their source frame and update from that frame's
coordinates; clicking empty space exits the special selector. Metrics > Style
changes the current frame's line and label appearance.

Selected-atom transforms are previewed through VTK actors while dragging. The
coordinate array and covalent bonds are updated once on release. Use `Ctrl+Z`
and `Ctrl+Y`, or the toolbar arrows, to undo and redo committed coordinate
edits. The undo limit and default atom material, lighting, and boundary style
are editable through Settings and stored in `src/btgui/settings.json` (or the
equivalent installed package directory).

The top menu bar contains File, Settings, Appearance, Windows, and View. The toolbar
below it contains structure operations and current-frame playback/navigation.
Axes and Grid are optional Appearance settings and default to off. Bond display
and its tolerance are grouped under Check Bonds; Resize is available from the
Supercell menu. The View menu contains lattice-axis views and zoom controls. The
toolbar frame controls use zero-based indices. Save As writes only the
currently displayed frame. Opening another file appends
its frames to the current batch; the current frame field rejects invalid or
out-of-range input without changing the displayed frame. `Ctrl+W` and the close
button remove the current frame. Windows > Show Frames opens a frame sidebar for
navigation, duplication, deletion, and undoable drag reordering. File > New adds
an empty or populated frame. Appearance contains an atom selector plus per-element
settings and global VTK material, lighting, and atom-boundary settings. An empty
selector applies atom appearance globally; element symbols and one-based atom
numbers/ranges limit the change to matching atoms in the current frame.

Enter `help` in the bottom command line for available commands. Property overlays can be added with `MainWindow.register_property_renderer(name, callback)`; the callback receives `(plotter, frame_index, positions)`.

Valid atom selections persist after coordinate, element, and other edits until
the user clicks empty space. Replace Atom is enabled only for a valid selection
and records the element replacement in the same undo history as other edits.
Topology-changing operations ask before discarding persistent metrics because
atom indices may no longer refer to the same atoms; Undo restores the structure
and its discarded metrics.

The command pane is a persistent Python-style console: expression results are
shown with `repr`, `print` output and tracebacks stay in the pane, and `>>>` /
`...` prompts support multi-line blocks. `Delete` removes the selected atoms;
`Add Atom` supports direct Cartesian coordinates or a reversible reference-based
placement chain. Files may also be dropped onto the viewer. Supercell changes
are reversible, retain their current multipliers, and `Resize` rebases the
current frame when a new base is explicitly desired.

## CLI data workspace

The command console is implemented in `frontend/cli.py`. Python commands receive
three ordinary editable variables: `elements`, `coo`, and `cell`, each a
`List[np.ndarray]`, plus NumPy as `np`. Every command edits detached working
copies. Shape, dtype, frame-count, and finite-value checks run before a valid
change is committed to the viewer; invalid changes print `Invalid change for
inner variables` and leave the displayed data unchanged. Completed commands also
refresh the renderer so in-place edits become visible immediately. When BUCToolkit is
available at runtime, `to_bt()` returns a copied `BatchStructures` object;
`to_bt("1, 3-4")` uses the one-based inclusive frame syntax and
`to_bt([0, 2])` uses zero-based Python indices. The optional `shortcut` command
uses `pyshortcuts` when installed and reports availability without creating a
file otherwise.

### Supercell transformations

The **Supercell** toolbar action transforms the current frame only. The three axis multipliers are combined with the editable integer matrix as `M @ diag(a, b, c)`; the transformed arrays replace that frame, so **Save As** writes the resulting current structure. Multipliers must be positive and the integer matrix must be non-singular.
