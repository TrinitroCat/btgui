# Changelog

## Unreleased

- Added an English/简体中文 main-window localization with JSON locale files,
  Settings > Language switching, and translated menus, toolbars, dock titles,
  frame/playback controls, file headers, and visible button labels. Nested
  dialogs remain in English for this first pass.
- Added an independent 1-4 H-H non-bonded term to clean, using the full van der
  Waals H-H contact distance so alkane conformers favor staggered over
  eclipsed/gauche contacts, while ordinary non-bonded pairs keep the existing
  scaling.
- Clean now optimizes only free coordinates through a shared wrapper, defaults
  to scipy BFGS, and supports local BFGS/L-BFGS-B/Nelder-Mead or global
  basinhopping/differential evolution/dual annealing/SHGO/DIRECT backends.
  Settings > Clean Backend configures the mode, method, iterations, bounds span
  and workers; workers are used by differential evolution and SHGO.
- View > Along A/B/C, Reset View, measurement mode and zoom now target the
  focused MDI window; zero-lattice molecules use Cartesian axes for Along, and
  the zoom control follows the active window.
- Show bonds is now a per-frame visibility flag. Its checked state follows the
  current frame and file, and checking/unchecking only changes the selected
  frame scope instead of leaving a global renderer switch enabled.
- The Files tree now defaults to the file-name tree with folders sorted first;
  Settings > File Manager configures optional Size/Type/Modified columns,
  folder ordering, and sort column/order.
- Reworked structure data into per-file workspaces with independent frames,
  playback, Undo/Redo and CLI scope. Windows are unique per `(file, frame)`;
  opening a file creates a new window, while reopening it only focuses its
  existing window.
- Added the left Files dock above the frame list, with directory following,
  double-click opening, and Open/New Folder/Rename/Delete/Copy Path/Refresh
  actions that avoid renaming or deleting open files.
- Added window reference-counted file caching with a configurable 1 GB default,
  automatic release of unreferenced files, and a Settings > Clear File Cache
  action.
- Added a one-based `file[frame]` overlay in each viewer, pastel per-file window
  borders controlled by the default-on Appearance option "distinguish files by
  frame colours", and a default atom-radius floor of `1.15 × bond width`.
- Split the toolbar into thin-bordered Edit, Structure, Measure, Update and
  Frames groups; Bonds belongs to Structure and Clean/Add H to Update.
- Trimmed the default Nuitka VTK payload to the 36 modules listed in
  `vtk_modules.txt`; this removes the unused VTK-m accelerators and their
  roughly 39 MB VisKores dependency. `FULL_VTK=1` restores the complete
  `vtkmodules` package.
- Added root `Makefile` and Windows `build.bat` wrappers with `standalone` and
  `onefile` Nuitka targets, packaging the `btgui` package, data tables, settings,
  icon, and scipy. The batch wrapper uses `UPX_EXE` and clears `UPX` so the UPX
  child process does not parse the executable path as its own configuration.
- Retuned the preset Lennard-Jones term from the bundled van der Waals radii:
  its minimum sits at `0.75 * (vdW_i + vdW_j)`, and bonded plus 1-3 pairs are
  excluded from the non-bonded sum, so non-bonded atoms no longer settle at
  bonding distances.
- Added per-frame explicit bond orders (parallel `int8` arrays), Add bond order
  cycling with Ctrl-persistent repeated clicks, and bond-order rendering as
  parallel cylinders of constant total diameter. Check bonds still creates
  single bonds only; clean ignores explicit orders.
- Added the toolbar Add H action: hydrogens are counted from the valence left by
  the explicit bond orders, placed around the missing-direction axis, and
  relaxed by clean with all pre-existing atoms fixed.
- Ctrl+left-drag box selection now adds the boxed atoms to the current atom
  selection instead of replacing it; an empty Ctrl box leaves the selection
  unchanged.
- Added the built-in `clean` updater and its preset graph, bond-order, and force
  field modules; the toolbar Clean action relaxes the focused frame in place,
  honors an atom selection, and reuses the frame's existing bond graph. scipy is
  now a core dependency.
- Added a frame selector to Appearance so selected-atom overrides can target the
  current frame, `all`, or inclusive ranges; invalid or out-of-range input is
  rejected before any edit, and a listed frame without that atom index keeps its
  own atoms.
- Added built-in serial POSCAR, CIF, XYZ, EXTXYZ, and complete OUTCAR I/O using
  the GUI's raw lists, removing BUCToolkit, PyTorch, and joblib from normal file
  operations while retaining optional BUCToolkit conversion through `to_bt()`.
- Added view-relative selected-atom rotation with separate center and edge gestures.
- Added actor-only drag previews, release-time coordinate and bond updates, and undo/redo history.
- Added persistent rendering and history settings in `btgui/settings.json`.
- Disabled atom mesh boundaries by default while retaining optional boundary controls.
- Added PBR sphere lighting with Ambient/Diffuse/Specular controls, global radius/background controls, persistent distance and angle measurements, OUTCAR import, and selected-frame export.
- Added percentage zoom controls, English frame/playback controls, current-frame Save As behavior, duplicate-safe batch frame parsing, and lightweight glyph-only playback updates.

- Added a Supercell toolbar action with positive a/b/c replication controls and a validated 3 x 3 integer lattice transformation.
- Updated the optional BUCToolkit save path to write transformed current-frame metadata correctly.
- Brightened the Default, Matte, Glossy, and Metallic material presets.
- Extracted the command console into `frontend/cli.py` with validated raw-list
  workspaces and runtime `to_bt` conversion.
- Added reversible supercell rebasing, rectangle selection feedback, file drag
  and drop, Delete atom removal, and direct/reference-based Add Atom placement.
- Added persistent Python-console prompts, protected-list rollback, and concise
  interaction/file error reporting for empty or invalid viewer states.
- Reorganized the top menus and structure toolbar, grouped bond tolerance and
  supercell rebasing in their menus, and moved axes/grid controls into Appearance.
- Fixed lightweight playback updates to notify VTK data objects through their
  native modification API; metrics remain available after supercell transforms.
- Preserved valid atom selections after edits and enabled direct Settings and
  Appearance actions from the menu bar.
- Added confirmation before topology changes discard persistent metrics and added
  a selection-aware Replace Atom toolbar action with undo support.
