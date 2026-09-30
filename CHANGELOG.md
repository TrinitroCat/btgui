# Changelog

## Unreleased

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
  workspaces, runtime `to_bt` conversion, and optional pyshortcuts support.
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
