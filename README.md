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

The standalone entry point uses built-in serial readers and writers. File Open supports multi-selection and opens each POSCAR, CIF, XYZ, EXTXYZ, or complete OUTCAR trajectory file in its own workspace window. The command `open PATH` accepts one path or a path list; `save PATH` writes the current frame as POSCAR, CIF, or XYZ. File > Export batch writes a selected inclusive one-based frame list such as `1-2, 4, 6,8,10-30` in POSCAR, CIF, or XYZ format. The built-in path contains no multiprocessing branch and always exports the current authoritative GUI data. Portions adapted from BUCToolkit are distributed under its MIT license in `LICENSES/BUCToolkit-MIT.txt`.

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
- Ctrl+left-drag: add atoms inside the rectangle to the current atom selection.
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
Settings also controls the file-cache limit (default `1024 MB`) and provides
**Clear File Cache** for unreferenced files.

## File workspaces

Every opened structure file is a separate workspace. Its frames, playback,
Undo/Redo history, measurements, and CLI variables are independent of other
files. A unique window exists for each `(file, frame)` pair; opening the same
file again focuses its existing window without changing the displayed frame.
Double-clicking an unopened file in the Files tree creates a new window on its
first frame and leaves older windows open.

The left dock stacks the file manager above the Frames list. The tree follows
the focused file's directory, supports normal selection and expansion, and its
context menu provides Open, New Folder, Rename, Delete, Copy Path, and Refresh.
By default it shows only the file-name tree with folders sorted before files.
Settings > File Manager chooses optional Size/Type/Modified columns, folder
ordering, and the sort column/order. Renaming or deleting a file that is
currently open is blocked.

The main window is available in English and 简体中文. Settings > Language
switches immediately without restarting; the choice is stored in
`settings.json`. Menu bar items, toolbars, dock titles, frame/playback/zoom
controls, file headers and visible button labels are localized. Nested dialogs
remain in English in this first pass. Strings live in
`src/btgui/frontend/locales/*.json`; adding another language only requires a new
JSON file with a `_meta` block.

Closed files stay cached until the total estimated memory exceeds the configured
limit. Only files with no display windows are released; if all referenced files
exceed the limit they are kept, and a file is released as soon as its last
window closes. `File > Open` opens each selected file in its own window and
focuses the last one; directories are expanded to their supported files first.
Save As starts from the active file path and writes only the current frame; it
does not rename or split the in-memory workspace.

Each viewer shows `file_name[frame]` in a translucent upper-left overlay, with a
one-based frame number and the full path in its tooltip. Appearance's default-on
**distinguish files by frame colours** option gives every file a pastel border
shared by all of its frame windows; disabling it makes every border white.

The top menu bar contains File, Settings, Appearance, Windows, and View. View
defaults to orthographic projection; enabling Perspective makes nearby objects
appear larger. Covalent spheres default to `0.6 × covalent radius × scale`, with
scale `1.0`; vdW spheres use `vdW radius × scale`. Default covalent and vdW
radii are clamped to `1.15 × bond width` so hydrogen spheres remain thicker than
bonds. Appearance shows an editable Scale for these modes, an absolute Custom
radius (Å) for custom spheres, and a disabled size control for points. Custom
radius and Scale retain independent values. Display scaling does not affect bond
detection. Bond width defaults to `0.3 Å` (the single-bond cylinder radius). The
toolbar below it is split into thin-bordered Edit, Structure, Measure, Update,
and Frames groups. Axes and Grid are optional Appearance settings and default
to off. Bond display
and its tolerance are grouped under Bonds. Add bond selects two atoms or an
existing bond: two atoms receive a single bond, while a click on an existing bond
cycles its order through 1-2-3-1. Hold Ctrl to keep Add bond active for repeated
bond or atom-pair clicks; Escape cancels. Check bonds computes the
complete pair-distance matrix and fully rebuilds each selected frame's COO graph;
this resets that frame's bond orders to single bonds. Show bonds initializes only
uncomputed graphs and otherwise displays existing connections. Visibility is a
per-frame flag: checking Show bonds marks the selected frame scope, and the
action follows the current frame (and file) as focus changes. Clear bonds
explicitly empties all graphs and keeps full originals in history. Each frame
stores a parallel `int8` bond-order array; an order-n bond is drawn as n parallel
cylinders of radius `bond width / n`, keeping the total diameter unchanged.
Orders affect automatic hydrogen addition and rendering only; clean ignores
them. Add/Delete/Check use edge deltas
in the shared Undo/Redo history. Atom and bond clicks pick the nearest rendered
sphere/tube surface, with roughly one physical pixel of edge tolerance, so bond
hit regions follow the displayed width under zoom and either projection.
Ordinary clicks switch between atom and bond selections, releasing the previous
kind. Ctrl toggles within the current kind and ignores hits on the other kind.
Blank clicks clear both kinds; box selection clears the previous selection and
selects atoms only. Ctrl+left-drag adds the boxed atoms to the current atom
selection. Delete removes the selected kind. Bond selection prints one-based endpoints,
element symbols and Cartesian length in Å. Resize is available
from the Supercell menu. PBC wrap applies fractional-coordinate wrapping to the
selected frames and is undoable. The View menu contains lattice-axis views and
zoom controls; these commands target the focused window, and zero-lattice
molecules fall back to Cartesian axes. The
toolbar frame controls use zero-based indices. Save As writes only the
currently displayed frame and suggests the active file path. Opening another
file creates a separate workspace and window; the current frame field rejects
invalid or out-of-range input without changing the displayed frame. The play/pause icon starts
at the current frame; its adjacent menu controls FPS, an inclusive frame range,
stride, and once/loop completion behavior. Playback updates atom and lattice geometry
incrementally, while the right mouse button remains available for camera rotation.
The display-window close button closes the focused display window; its frame
remains cached in the file workspace. Windows > Delete removes selected frames,
and Windows > Show Frames opens the frame sidebar for navigation,
multi-selection, duplication, deletion, renaming, and undoable drag reordering.
Single-clicking a frame switches the focused viewer; double-clicking or choosing
Open creates the unique window for that file and frame. Frame names remain
attached while reordering, and copies use an `_copy` suffix. File > New adds an empty or populated frame. Appearance contains atom and
frame selectors plus per-element settings. An empty atom selector applies color,
opacity, display radius, material, lighting, PBR, and atom-boundary settings to
all atoms; element symbols and one-based atom numbers/ranges limit those settings
to matching atoms in the listed frames. The frame selector defaults to the
current frame, and also accepts `all` or inclusive ranges such as `1,2-5`; invalid
or out-of-range input is rejected with a warning and changes nothing. Renderer-wide
defaults and scene settings ignore the frame selector, and a listed frame that
lacks a selected index keeps its own atoms. Background, axes, grid, and bond
settings remain scene-wide.

### Automatic hydrogen addition

The toolbar **Add H** action appends hydrogens to the focused frame from the
stored bond orders. For each main-group atom, the used valence is the sum of its
incident bond orders; the remaining valence is filled with H (`<= 0` adds none).
Check bonds creates single bonds only, so double and triple bonds must be set
with Add bond before an unsaturated structure is hydrogenated. New hydrogens are
placed around the missing-direction axis with roughly uniform azimuths and then
relaxed by the clean force field while every pre-existing atom stays fixed. The
atom count, bond orders, and hydrogen coordinates are committed as one undo step.

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
`List[np.ndarray]`, plus NumPy as `np`. The console is scoped to the focused
file: its protected variables reference only that file, and switching to a
different file clears the console output and persistent user variables before
rebinding the new file's frames. Every command edits detached working
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
their source files and provides Apply with an `Apply to:` frame selector and
an `in-place` checkbox, built-in Python Help, Dry Run, Refresh, and literal
`args`/`kwargs` inputs. The selector defaults to the focused viewer's one-based
frame number; empty means current, `all` selects all frames, and `1,2-5` uses
the standard inclusive range syntax. Indices are sorted and deduplicated.
Dry Run executes and validates the selected function for this same scope and
mode without changing frames; plugin output, warnings, and tracebacks are
written to the GUI console. Refresh rereads `settings.json`. The console also
accepts `update` or `update 1,2-5` and uses the first loaded function.

The callback receives cloned arrays in `(cell, elem, coo)` order, so in-place
changes inside the updater cannot alter existing frames. Each selected frame
is passed separately as three singleton lists. In append mode its result may
contain any number of frames or atoms; results are appended in source-frame,
then output-frame order, with no Undo/Redo entry or copy of the existing batch.
Earlier operations remain undoable without removing these appended frames.
Each returned frame must have matching elements and coordinates, valid cell
shape, finite values, and compatible NumPy dtypes. All selected calls must pass
validation before any data is committed; an error identifies its source frame.

Checking `in-place` requires exactly one output frame per input and replaces the
selected frames with one multi-frame undo step. History stores only changed
original arrays in `(cell, elem, coo, bond_graph)` order; unchanged fields are
`None`, and mixed fields use aligned lists with `None` for unchanged frames.
Undo and Redo exchange saved/current state without calling the plugin again.
The bond graph is preserved when atom count is unchanged. When the count changes,
an existing graph (including an explicitly empty one) is rebuilt from the full
distance matrix, and its original is saved in the graph field. An uncomputed
graph remains uncomputed. Topology changes reset per-atom styles and metrics;
Undo restores them and the previous supercell base/transform.
Embedding applications may call `window.set_updater(fn)` or
`window.apply_updater(fn)` directly, or use
`window.apply_updater('1,2-5', in_place=True)` for in-place updates.

### Built-in clean updater

btgui ships a `clean` updater built on `btgui.backend.updater_presets`. It is a
small valence force field: harmonic bonds whose target lengths come from
covalent radii and an inferred bond order, cosine-harmonic bond angles whose
targets follow the coordination geometry (sp, sp2, sp3, trigonal bipyramidal,
octahedral, pentagonal bipyramidal, cubic), and a weak Lennard-Jones term for
non-bonded contacts. Bond orders come from the main-group unpaired-electron
count minus the graph degree, taking `min(order_i, order_j)` per pair. This is a
coarse model intended for cleanup, not a valence calculation; periodic images
are not considered.

The updater is registered like any other updater and follows the standard frame
scope and append semantics. The toolbar **Clean** action runs it in place on the
focused frame: with a non-empty atom selection only the selected atoms move, and
the frame's existing bond graph is reused when present. The Lennard-Jones term is
calibrated from the bundled van der Waals radii: its minimum sits at
`lj_equilibrium_scale * (vdW_i + vdW_j)` with the default `0.75`, which keeps
non-bonded contacts near 1.5-2 times the covalent bond length. Bonded and 1-3
pairs are excluded from the non-bonded sum. A separate 1-4 H-H Lennard-Jones
term uses the full van der Waals contact distance (`1.0 * (vdW_H + vdW_H)`) so
alkane conformers prefer staggered over eclipsed/gauche contacts. scipy supplies
the optimizers. Settings > Clean Backend chooses either a local minimizer
(`BFGS` default, `L-BFGS-B`, or `Nelder-Mead`) or a global minimizer
(`basinhopping`, `differential_evolution`, `dual_annealing`, `shgo`, or
`direct`), with separate iteration settings. `workers` is forwarded only to
differential evolution and SHGO; fixed coordinates are removed from the
optimizer variables by a free-coordinate wrapper. Embedding applications can
pass `fixed`, `bonds`, `maxiter`, `bond_k`, `angle_k`, `lj_epsilon`,
`lj_equilibrium_scale`, `lj_hh14_epsilon`, `lj_cutoff`, `mode`,
`local_method`, `global_method`, `workers` and `bounds_span` through
`apply_updater`.

## Nuitka build

A root `Makefile` wraps the standalone and onefile builds around
`src/btgui/app.py`, the same entry point as the `btgui` console script.

Prerequisites:

- a Python environment with this project installed editable (`pip install -e .`);
- Nuitka (`pip install nuitka`) and a working C compiler;
- GNU Make for the `Makefile` targets, or `build.bat` on Windows without GNU Make.

```text
make help
make standalone     # build/nuitka/app.dist/btgui.exe
make onefile        # build/nuitka/btgui.exe
make clean
```

On Linux and macOS the Makefile adds `--static-libpython=no`, because conda's
`libpython*.a` is frequently an LTO/LLVM archive that the system linker rejects
with `File format not recognized`. Override with
`make standalone STATIC_LIBPYTHON=--static-libpython=yes` only if the static
archive is known to be valid.

Variables can be overridden on the command line:

```text
make standalone PYTHON=python JOBS=8 OUTPUT_DIR=build/nuitka
make standalone CONSOLE_MODE=disable
make onefile UPX=C:/tools/upx.exe
make standalone EXTRA_FLAGS="--include-package=scipy"
make standalone FULL_VTK=1
```

On Windows without GNU Make, use `build.bat`, which carries the same options:

```bat
build.bat standalone
build.bat onefile
build.bat clean
build.bat help
```

Override the same settings through environment variables:

```bat
set OUTPUT_DIR=D:\nuitka_build
set CONSOLE_MODE=disable
set UPX_EXE=C:\tools\upx.exe
build.bat standalone
```

`build.bat flags` prints the expanded Nuitka command line for inspection.

By default the build includes only the 36 VTK modules listed in
`vtk_modules.txt`, the set PyVista/PyVistaQt actually loads for btgui. This
drops the unused `vtkAcceleratorsVTKm*` modules and their roughly 39 MB VisKores
dependency, plus other unused VTK readers and filters. Set `FULL_VTK=1` to package
all of `vtkmodules` instead (larger, but useful as a compatibility fallback):

```text
make standalone FULL_VTK=1
```

```bat
set FULL_VTK=1
build.bat standalone
```

> **UPX is optional and off by default.** Windows Defender often flags
> UPX-packed scipy DLLs (the temporary `*.upx` files) as malware and may
> quarantine them while Nuitka is working, which then fails with
> `OSError: [Errno 22]` or a UPX plugin error. Prefer building without
> `UPX_EXE` and compressing the finished `app.dist` with 7z or an installer.
> If UPX is required, exclude the build directory from real-time scanning for
> the duration of the build, but expect the packed binaries to be flagged on
> other machines too.

The Makefile and batch wrapper set `PYTHONPATH=src`, include the complete
`btgui` package, `scipy.optimize`, PyVista/PyVistaQt, the selected VTK modules,
the JSON data tables, and `settings.json`; on Windows they also apply
`gui_logo.ico` and the requested console mode. For an editable configuration,
place a `settings.json` beside the built executable (or point
`BTGUI_CONFIG_FILE` at another file); it takes precedence over the packaged
defaults.
