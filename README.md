# btgui
btgui: BUCToolkit Graphical User Interface.

An interactive PySide6 and PyVista viewer for molecular and crystal structures.
It can be viewed as a graphic frontend of [BUCToolkit](https://github.com/TrinitroCat/BUCToolkit), 
but also an independent software that must not require the BUCToolkit package. BUCToolkit or any other calculator/structure updater are applied by registering as plugins. (See [this section](#custom-structure-updaters))

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

Open one or more files directly at startup, in command-line order:

```text
btgui POSCAR
btgui structure.cif trajectory.xyz
btgui "path with spaces/structure.cif"
```

Omitting files opens an empty viewer. Use `btgui --help` for command-line help
and `btgui -- -structure.xyz` for a filename beginning with a hyphen. Startup
files use the same reader as File Open; loading errors appear in the GUI console.

The standalone entry point uses built-in serial readers and writers. File Open supports multi-selection and appends POSCAR, CIF, XYZ, EXTXYZ, or complete OUTCAR trajectories directly to the three GUI lists. The command `open PATH` accepts one path or a path list; `save PATH` writes the current frame as POSCAR, CIF, or XYZ. File > Export batch writes a selected inclusive one-based frame list such as `1-2, 4, 6,8,10-30` in POSCAR, CIF, or XYZ format. The built-in path contains no multiprocessing branch and always exports the current authoritative GUI data. Portions adapted from BUCToolkit are distributed under its MIT license in `LICENSES/BUCToolkit-MIT.txt`.

## Mouse controls

- Right-drag: orbit the view.
- Shift+right-drag: translate selected atoms.
- Shift+arrow key: translate selected atoms in the screen horizontal/vertical
  directions by the Cartesian step configured in Settings (default `0.1 Å`).
- Up/Down arrow key: switch to the previous/next frame.
- Shift+Alt+right-drag near the selection: horizontal motion rotates around the
  screen vertical axis, while vertical motion rotates around the screen
  horizontal axis.
- Shift+Alt+right-drag near a window edge: circular pointer motion rotates
  around the screen-normal axis.
- Left-click: select one atom or visible bond; blank space clears selection.
- Ctrl+left-click: toggle atoms or bonds within the current selection type.
- Left-drag: clear the old selection and select only atoms inside the rectangle.
- Scroll: zoom.

Metrics > Length and Metrics > Angle temporarily replace normal atom selection. Pick
two atoms for a distance or three atoms in endpoint-vertex-endpoint order for an
angle. Measurements belong to their source frame and update from that frame's
coordinates. Temporary metric picks use amber outlines; completing the required
picks immediately returns to normal selection, while `Esc` or clicking empty space
cancels an incomplete pick. Metrics > Style changes the current frame's line and
label appearance.

Selected-atom transforms are previewed through VTK actors while dragging. The
coordinate array and covalent bonds are updated once on release. Use `Ctrl+Z`
and `Ctrl+Y`, or the toolbar arrows, to undo and redo committed coordinate
edits. The undo limit, keyboard translation step, and default atom material,
lighting, and boundary style are editable through Settings and stored in
`src/btgui/settings.json` (or the equivalent installed package directory).

The top menu bar contains File, Settings, Appearance, Windows, and View. View
defaults to orthographic projection; enabling Perspective makes nearby objects
appear larger. Covalent spheres default to `0.6 × covalent radius × scale`, with
scale `1.0`; vdW spheres use `vdW radius × scale`. Appearance shows an editable
Scale for these modes, an absolute Custom radius (Å) for custom spheres, and a
disabled size control for points. Custom radius and Scale retain independent
values. Display scaling does not affect bond detection. Bond width defaults to
`0.3 Å` (the cylinder radius). The toolbar
below it contains structure operations and current-frame playback/navigation.
Axes and Grid are optional Appearance settings and default to off. Bond display
and its tolerance are grouped under Bonds. Add bond temporarily selects two
atoms to connect, independently of distance cutoffs. Check bonds computes the
complete pair-distance matrix and fully rebuilds each selected frame's COO graph;
this may restore manually deleted bonds. Show bonds initializes only uncomputed
graphs and otherwise displays existing connections. Clear bonds explicitly empties
all graphs and keeps full originals in history. Add/Delete/Check use edge deltas
in the shared Undo/Redo history. Atom and bond selections are mutually exclusive;
Delete removes the selected kind. Bond selection prints one-based endpoints,
element symbols and Cartesian length in Å. Resize is available
from the Supercell menu. PBC wrap applies fractional-coordinate wrapping to the
selected frames and is undoable. The View menu contains lattice-axis views and
zoom controls. The
toolbar frame controls use zero-based indices. Save As writes only the
currently displayed frame. Opening another file appends
its frames to the current batch; the current frame field rejects invalid or
out-of-range input without changing the displayed frame. The play/pause icon starts
at the current frame; its adjacent menu controls FPS, an inclusive frame range,
stride, and once/loop completion behavior. Playback updates atom and lattice geometry
incrementally, while the right mouse button remains available for camera rotation.
The display-window close button closes the focused display window without
removing its frame. Windows > Delete removes selected frames, and Windows > Show Frames
opens a frame sidebar for navigation, multi-selection, duplication, deletion, renaming, and
undoable drag reordering. Single-clicking a frame switches the focused viewer;
double-clicking or choosing Open creates a new shared-data viewer window. Frame
names remain attached to their data while reordering, and copies use an `_copy`
suffix. File > New adds an empty or populated frame. Appearance contains an atom
selector plus per-element settings. An empty selector applies color, opacity,
display radius, material, lighting, PBR, and atom-boundary settings to all atoms;
element symbols and one-based atom numbers/ranges limit those settings to matching
atoms in the current frame. Background, axes, grid, and bond settings remain
scene-wide.

Enter `help` in the bottom command line for available commands. Property overlays can be added with `MainWindow.register_property_renderer(name, callback)`; the callback receives `(plotter, frame_index, positions)`.

Valid atom selections persist after coordinate, element, and other edits until
the user clicks empty space. Replace Atom is enabled only for a valid selection
and records the element replacement in the same undo history as other edits.
Topology-changing operations ask before discarding persistent metrics because
atom indices may no longer refer to the same atoms; Undo restores the structure
and its discarded metrics.

The command pane is a persistent Python-style console: expression results are
shown with `repr`, `print` output and tracebacks stay in the pane, and `>>>` /
`...` prompts support multi-line blocks. `Delete` removes selected atoms or bonds;
`Add Atom` supports direct Cartesian coordinates or a reversible reference-based
placement chain, automatically bonded from the reference through consecutive new
atoms; the atoms and bonds undo/redo together. Ordinary atom addition and species
replacement retain existing bonds. Deleting atoms removes incident edges and
simultaneously remaps both COO endpoints; Undo restores their original numbering
and incident edges. Files may also be dropped onto the viewer. Supercell changes
are reversible, retain their current multipliers, and `Resize` rebases the
current frame when a new base is explicitly desired.

Internally each bond graph is an unweighted `int64` COO array of shape `(2, E)`:
zero-based endpoints, `src < dst`, unique edges, lexicographically sorted by
`(src, dst)`. `None` means uncomputed; an empty `(2, 0)` array means explicitly
empty and is not lazily regenerated. Bond history stores changed `(i, j, 1)`
triples; `1` is the current edge-presence information, not a bond order. Atom
deletion additionally stores an old-node-sized old-to-new mapping (`-1` for
deleted nodes), applied simultaneously to both endpoints, plus removed incident
triples. Clear bonds alone retains full original graph snapshots. Compound
history entries execute forward on Redo and in reverse on Undo, so nodes and
edges are restored in their required index spaces before a single redraw.

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
`to_bt([0, 2])` uses zero-based Python indices.

### Supercell transformations

The **Supercell** toolbar action transforms the current frame only. The three axis multipliers are combined with the editable integer matrix as `M @ diag(a, b, c)`; the transformed arrays replace that frame, so **Save As** writes the resulting current structure. Multipliers must be positive and the integer matrix must be non-singular.

## Custom structure updaters

External structure updaters use one stable protocol. Configure independent path
and function-name lists in `settings.json`; paths may be absolute or relative
to that configuration file. The lists need not have equal lengths: one module
can provide multiple functions. Legacy single-string values remain supported:

```python
def update(cell, elem, coo, *args, **kwargs):
    # return (new_cell, new_elem, new_coo), all List[np.ndarray]
    return cell, elem, coo
```

```json
{
  "updater_path": ["updaters/my_optimizer.py"],
  "updater_function": ["relax_atoms", "relax_cell"]
}
```

btgui loads all configured modules first (each path once), then looks up each
requested function name across those modules. Functions are grouped by their
defining module using their `__module__` metadata. Reexports of the same function
are deduplicated; distinct functions with the same name in different modules
are all listed. The toolbar
`Update` button is always available as a menu containing every successfully
loaded function, `More...`, and `Help`. Selecting a function opens the existing
one-based frame-scope prompt, including `all`. `More...` groups functions under
their source files and provides Apply, built-in Python Help, Dry Run, Refresh,
and literal `args`/`kwargs` inputs. Dry Run executes and validates the selected
function without changing frames; plugin output, warnings, and tracebacks are
written to the GUI console. Refresh rereads `settings.json`. The console also
accepts `update` or `update 1,2-5` and uses the first loaded function.

The callback receives cloned arrays in `(cell, elem, coo)` order, so in-place
changes inside the updater cannot alter existing frames. Its result may contain
any number of frames or atoms; each returned frame must still have matching
elements and coordinates, valid cell shape, finite values, and compatible NumPy
dtypes. All returned frames are appended to the existing data and the operation
is undoable. Embedding applications may still call `window.set_updater(fn)` or
`window.apply_updater(fn)` directly.

For a Nuitka standalone build (including onefile), place an editable
`settings.json` beside the executable; it takes precedence over the packaged
defaults. `BTGUI_CONFIG_FILE` can be used to point to another configuration
file.
