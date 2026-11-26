"""Main window, file actions, animation timer, and command console."""

from collections import deque
import ast
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import traceback
import warnings

import numpy as np

from ..backend.lattice_transform import transform_structure
from PySide6.QtCore import (
    QSignalBlocker, QTimer, Qt, Signal, QDir, QFile, QModelIndex,
    QAbstractProxyModel, QPersistentModelIndex, QSortFilterProxyModel,
)
from PySide6.QtGui import QAction, QActionGroup, QColor, QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QAbstractItemView,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFileSystemModel,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMdiArea,
    QMdiSubWindow,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSizePolicy,
    QSplitter,
    QStyle,
    QToolBar,
    QToolButton,
    QTreeView,
    QTreeWidget,
    QTreeWidgetItem,
    QWidgetAction,
    QVBoxLayout,
    QWidget,
    QColorDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QRadioButton,
    QButtonGroup,
)

from btgui.backend.core import (
    apply_structure_updater,
    build_bond_adjacency,
    normalize_bond_adjacency,
    parse_frame_range,
    parse_frame_scope,
    parse_frame_selection,
    validate_data,
    wrap_coordinates,
)
from btgui.backend.core import format_atom_selection, parse_atom_selection
from btgui.backend.hydrogenate import add_hydrogens
from btgui.backend.updater_presets.clean_structure import (
    CLEAN_GLOBAL_METHODS,
    CLEAN_LOCAL_METHODS,
    clean as clean_structure,
)
from btgui.data.appearance import ELEMENT_COLORS, default_color
from btgui.frontend.cli import CLI
from btgui.frontend.clipboard import AtomClipboard
from btgui.frontend.file_state import FileState
from btgui.frontend.changes import (
    AppearanceChange, BondChange, ClearBondsChange, FramePermutationChange,
    InplaceUpdaterChange, StructureChange,
)
from btgui.integrations.structure_io import normalize_structure_files
from btgui.integrations.updater import load_updater_module
from btgui.frontend.file_recycler import FileRecycler
from btgui.frontend.i18n import Catalog, available_languages, language_codes
from btgui.frontend.python_runner import ExternalPythonRunner, ExternalRunDialog
from btgui.frontend.text_editor import TextEditorDialog
from btgui.frontend.viewer import StructureViewer


DEFAULT_SETTINGS = {
    "history_limit": 10,
    "default_atom_material": "default",
    "atom_lighting": True,
    "atom_edges": False,
    "atom_edge_color": "#202020",
    "atom_edge_width": 1.0,
    "buctoolkit_paths": [],
    "keyboard_translation_step": 0.1,
    "file_cache_limit_mb": 1024,
    "file_manager_show_size": False,
    "file_manager_show_type": False,
    "file_manager_show_modified": False,
    "file_manager_folders_first": True,
    "file_manager_sort_column": 0,
    "file_manager_sort_order": "asc",
    "clean_mode": "local",
    "clean_local_method": "BFGS",
    "clean_local_maxiter": 200,
    "clean_local_gtol": 1e-5,
    "clean_global_method": "differential_evolution",
    "clean_global_maxiter": 200,
    "clean_workers": 1,
    "clean_bounds_span": 2.0,
    "language": "zh_CN",
    "external_python_path": None,
    "updater_path": [],
    "updater_function": [],
}


def _preset_updater_records():
    """Return the updater records shipped with btgui."""
    return [
        {
            "path": "<preset>",
            "name": "clean",
            "callback": clean_structure,
            "configured": False,
            "preset": True,
        }
    ]


class AppearanceDialog(QDialog):
    """Edit per-element atom styles and global bond appearance."""

    def __init__(
        self,
        viewer,
        elements,
        selected_indices=None,
        parent=None,
        current_frame=None,
        frame_count=None,
    ):
        super().__init__(parent)
        self.viewer = viewer
        self.frame_count = (
            len(viewer.coordinates) if frame_count is None else int(frame_count)
        )
        self.current_frame = (
            int(viewer.current_index) if current_frame is None else int(current_frame)
        )
        self.setWindowTitle("Appearance")
        self.resize(560, 470)
        self.styles = {key: dict(value) for key, value in viewer.atom_styles.items()}
        self.elements = list(dict.fromkeys(np.asarray(elements, dtype=str).tolist()))
        self.frame_elements = np.asarray(elements, dtype=str)
        self.initial_element_styles = {
            symbol: {
                "color": viewer._style_for(symbol).get("color", default_color(symbol)),
                "alpha": float(viewer._style_for(symbol).get("alpha", 1.0)),
            }
            for symbol in self.elements
        }
        self.accepted_values = None
        self.selector_edit = QLineEdit(self)
        self.selector_edit.setPlaceholderText("All atoms")
        self.selector_edit.setText(
            format_atom_selection(
                [] if selected_indices is None else selected_indices,
                self.frame_elements,
            )
        )
        self.frames_edit = QLineEdit(self)
        self.frames_edit.setText(str(self.current_frame + 1))
        self.frames_edit.setPlaceholderText("empty=current, all, 1,2-5")
        self.selector_edit.textChanged.connect(self._update_frame_control)
        self._update_frame_control()
        self.list_widget = QListWidget(self)
        self.list_widget.addItems(self.elements)
        self.opacity_spin = QDoubleSpinBox(self)
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.color_button = QPushButton(self)
        self.color_button.clicked.connect(self._choose_color)
        self.bond_width_spin = QDoubleSpinBox(self)
        self.bond_width_spin.setRange(0.001, 1.0)
        self.bond_width_spin.setSingleStep(0.01)
        self.bond_width_spin.setValue(viewer.bond_width)
        self.bond_alpha_spin = QDoubleSpinBox(self)
        self.bond_alpha_spin.setRange(0.0, 1.0)
        self.bond_alpha_spin.setSingleStep(0.05)
        self.bond_alpha_spin.setValue(viewer.bond_alpha)
        self.material_combo = QComboBox(self)
        self.material_combo.addItem("Default", "default")
        self.material_combo.addItem("Matte", "matte")
        self.material_combo.addItem("Glossy", "glossy")
        self.material_combo.addItem("Metallic", "metallic")
        material_index = self.material_combo.findData(viewer.atom_material)
        self.material_combo.setCurrentIndex(max(0, material_index))
        material_values = viewer._material_properties()
        self.ambient_spin = QDoubleSpinBox(self)
        self.ambient_spin.setRange(0.0, 1.0)
        self.ambient_spin.setSingleStep(0.05)
        self.ambient_spin.setValue(material_values["ambient"])
        self.diffuse_spin = QDoubleSpinBox(self)
        self.diffuse_spin.setRange(0.0, 1.0)
        self.diffuse_spin.setSingleStep(0.05)
        self.diffuse_spin.setValue(material_values["diffuse"])
        self.specular_spin = QDoubleSpinBox(self)
        self.specular_spin.setRange(0.0, 1.0)
        self.specular_spin.setSingleStep(0.05)
        self.specular_spin.setValue(material_values["specular"])
        self.roughness_spin = QDoubleSpinBox(self)
        self.roughness_spin.setRange(0.0, 1.0)
        self.roughness_spin.setSingleStep(0.05)
        self.roughness_spin.setValue(material_values["roughness"])
        self.metallic_spin = QDoubleSpinBox(self)
        self.metallic_spin.setRange(0.0, 1.0)
        self.metallic_spin.setSingleStep(0.05)
        self.metallic_spin.setValue(material_values["metallic"])
        self.material_combo.currentIndexChanged.connect(self._load_material_preset)
        self.lighting_check = QCheckBox("Lighting", self)
        self.lighting_check.setChecked(viewer.atom_lighting)
        self.edges_check = QCheckBox("Atom boundaries", self)
        self.edges_check.setChecked(viewer.atom_edges)
        self.axes_check = QCheckBox("Axes", self)
        self.axes_check.setChecked(viewer.axes_visible)
        self.grid_check = QCheckBox("Grid", self)
        self.grid_check.setChecked(viewer.grid_visible)
        self.distinguish_check = QCheckBox(
            "distinguish files by frame colours", self
        )
        self.distinguish_check.setChecked(viewer.distinguish_files)
        self.edge_width_spin = QDoubleSpinBox(self)
        self.edge_width_spin.setRange(0.1, 10.0)
        self.edge_width_spin.setSingleStep(0.1)
        self.edge_width_spin.setValue(viewer.atom_edge_width)
        self.edge_color_button = QPushButton(self)
        self.edge_color_button.clicked.connect(self._choose_edge_color)
        self._set_color_button(viewer.atom_edge_color, self.edge_color_button)
        self.global_style_combo = QComboBox(self)
        self.global_style_combo.addItem("Point", "point")
        self.global_style_combo.addItem("Covalent radius sphere", "covalent")
        self.global_style_combo.addItem("Van der Waals sphere", "vdw")
        self.global_style_combo.addItem("Custom radius sphere", "custom")
        self.global_style_combo.setCurrentIndex(
            max(0, self.global_style_combo.findData(viewer.default_atom_style))
        )
        self.global_radius_spin = QDoubleSpinBox(self)
        self.global_radius_spin.setRange(0.01, 10.0)
        self.global_radius_spin.setSingleStep(0.01)
        self.global_radius_spin.setSuffix(" Å")
        self.global_radius_spin.setValue(viewer.default_custom_radius)
        # One visible control edits two independent values; switching display
        # modes must not reinterpret an angstrom radius as a dimensionless scale.
        self._custom_radius = viewer.default_custom_radius
        self._atom_scale = viewer.default_atom_scale
        self._radius_mode = "custom"
        self.global_radius_label = QLabel(self)
        self.global_style_combo.currentIndexChanged.connect(self._update_radius_control)
        self._update_radius_control()
        self.background_button = QPushButton(self)
        self._set_color_button(viewer._background_color_value, self.background_button)
        self.background_button.clicked.connect(self._choose_background_color)
        form = QFormLayout()
        form.addRow("Atoms", self.selector_edit)
        form.addRow("Frames", self.frames_edit)
        form.addRow("Atom color", self.color_button)
        form.addRow("Atom opacity", self.opacity_spin)
        form.addRow("Atom display", self.global_style_combo)
        form.addRow(self.global_radius_label, self.global_radius_spin)
        form.addRow("Sphere material", self.material_combo)
        form.addRow("Ambient", self.ambient_spin)
        form.addRow("Diffuse", self.diffuse_spin)
        form.addRow("Specular", self.specular_spin)
        form.addRow("PBR roughness", self.roughness_spin)
        form.addRow("PBR metallic", self.metallic_spin)
        form.addRow(self.lighting_check)
        form.addRow(self.edges_check)
        form.addRow("Boundary width", self.edge_width_spin)
        form.addRow("Boundary color", self.edge_color_button)
        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        form.addRow(separator)
        form.addRow("Background color", self.background_button)
        form.addRow("Bond width", self.bond_width_spin)
        form.addRow("Bond opacity", self.bond_alpha_spin)
        form.addRow(self.axes_check)
        form.addRow(self.grid_check)
        form.addRow(self.distinguish_check)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.list_widget, 1)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.list_widget.currentItemChanged.connect(self._change_element)
        if self.elements:
            self.list_widget.setCurrentRow(0)

    def _update_radius_control(self, index=None):
        """Keep custom radii and display scales independent across mode switches.

        Args:
            index: Unused combo-box index supplied by Qt.

        Return:
            None. The label, units and editable value follow the display mode.
        """
        if self._radius_mode == "custom":
            self._custom_radius = self.global_radius_spin.value()
        else:
            self._atom_scale = self.global_radius_spin.value()
        mode = self.global_style_combo.currentData()
        self._radius_mode = "custom" if mode == "custom" else "scale"
        self.global_radius_label.setText("Custom radius" if mode == "custom" else "Scale")
        self.global_radius_spin.setSuffix(" Å" if mode == "custom" else "")
        self.global_radius_spin.setValue(
            self._custom_radius if mode == "custom" else self._atom_scale
        )
        self.global_radius_spin.setEnabled(mode != "point")

    def _update_frame_control(self):
        """Scope frame selection to explicit atom selections only.

        Args:
            None.

        Return:
            None. The field stays visible but is disabled for all atoms, whose
            renderer-wide defaults are not stored per frame.
        """
        scoped = bool(self.selector_edit.text().strip())
        self.frames_edit.setEnabled(scoped)
        self.frames_edit.setToolTip(
            "Frames that receive these atom overrides. Empty means the current "
            "frame, all means every frame, and 1,2-5 uses inclusive one-based "
            "ranges."
            if scoped
            else "All-atoms and scene settings are renderer-wide; this field is ignored."
        )

    def _load_material_preset(self):
        """Refresh PBR coefficient controls when the material preset changes."""
        presets = {
            "default": (0.24, 0.86, 0.32, 0.34, 0.0),
            "matte": (0.28, 0.72, 0.0, 0.9, 0.0),
            "glossy": (0.18, 0.86, 0.78, 0.16, 0.0),
            "metallic": (0.20, 0.76, 0.88, 0.22, 0.40),
        }
        values = presets.get(self.material_combo.currentData())
        if values is None:
            return
        for control, value in zip(
            (
                self.ambient_spin,
                self.diffuse_spin,
                self.specular_spin,
                self.roughness_spin,
                self.metallic_spin,
            ),
            values,
        ):
            control.setValue(value)

    def _load_element(self, row):
        """Load one element's current style into the lower controls."""
        if row < 0 or row >= len(self.elements):
            return
        symbol = self.elements[row]
        style = self.styles.get(symbol, self.viewer._style_for(symbol))
        controls = (self.opacity_spin, self.color_button)
        blockers = [QSignalBlocker(control) for control in controls]
        self.opacity_spin.setValue(float(style.get("alpha", 1.0)))
        self._set_color_button(style.get("color", default_color(symbol)), self.color_button)
        del blockers

    def _change_element(self, current, previous):
        """Save the previous element style before loading the new row."""
        if previous is not None:
            self._save_element_by_symbol(previous.text())
        self._load_element(self.list_widget.row(current) if current is not None else -1)

    def _save_element_by_symbol(self, symbol):
        """Save controls for a symbol without changing the current list row."""
        if symbol not in self.elements:
            return
        self.styles[symbol] = {
            "alpha": self.opacity_spin.value(),
            "color": self.color_button.property("color") or ELEMENT_COLORS["X"],
        }

    def _save_element(self):
        """Store the lower controls in the currently selected element style."""
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self.elements):
            return
        self.styles[self.elements[row]] = {
            "alpha": self.opacity_spin.value(),
            "color": self.color_button.property("color") or ELEMENT_COLORS["X"],
        }

    def _set_color_button(self, color, button=None):
        """Set the color button style and retained color property."""
        target = button if button is not None else self.color_button
        target.setProperty("color", color)
        target.setStyleSheet(f"background-color: {color};")

    def _choose_color(self):
        """Choose a color for the selected element."""
        color = QColorDialog.getColor(QColor(self.color_button.property("color") or ELEMENT_COLORS["X"]), self)
        if color.isValid():
            self._set_color_button(color.name())

    def _choose_edge_color(self):
        """Choose the global atom-boundary color."""
        color = QColorDialog.getColor(QColor(self.edge_color_button.property("color") or "#202020"), self)
        if color.isValid():
            self._set_color_button(color.name(), self.edge_color_button)

    def _choose_background_color(self):
        """Choose the global renderer background color."""
        color = QColorDialog.getColor(QColor(self.background_button.property("color") or "#f5f7f7"), self)
        if color.isValid():
            self._set_color_button(color.name(), self.background_button)

    def accept(self):
        """Collect current controls for application by the main window."""
        self._save_element()
        self._update_radius_control()
        try:
            indices = parse_atom_selection(self.selector_edit.text(), self.frame_elements)
        except ValueError as error:
            self.selector_edit.clear()
            QMessageBox.warning(self, "Invalid atom selector", str(error))
            return
        # Section: Renderer-wide defaults are not frame-scoped, so the frame
        # field matters only when a concrete atom selection is being styled.
        global_atoms = not bool(self.selector_edit.text().strip())
        if global_atoms:
            frames = [self.current_frame]
        else:
            try:
                frames = parse_frame_scope(
                    self.frames_edit.text(), self.frame_count, self.current_frame
                )
            except ValueError as error:
                QMessageBox.warning(self, "Invalid frame selector", str(error))
                return
        self.accepted_values = {
            "indices": indices,
            "frames": frames,
            "global_atoms": global_atoms,
            "styles": deepcopy(self.styles),
            "material": self.material_combo.currentData(),
            "lighting": self.lighting_check.isChecked(),
            "edges": self.edges_check.isChecked(),
            "edge_color": self.edge_color_button.property("color") or "#202020",
            "edge_width": self.edge_width_spin.value(),
            "atom_style": self.global_style_combo.currentData(),
            "custom_radius": self._custom_radius,
            "atom_scale": self._atom_scale,
            "background": self.background_button.property("color") or "#f5f7f7",
            "ambient": self.ambient_spin.value(),
            "diffuse": self.diffuse_spin.value(),
            "specular": self.specular_spin.value(),
            "roughness": self.roughness_spin.value(),
            "metallic": self.metallic_spin.value(),
            "axes": self.axes_check.isChecked(),
            "grid": self.grid_check.isChecked(),
            "distinguish_files": self.distinguish_check.isChecked(),
            "bond_width": self.bond_width_spin.value(),
            "bond_alpha": self.bond_alpha_spin.value(),
        }
        self.accepted_values["changed_symbols"] = {
            symbol
            for symbol in self.elements
            if {
                "color": self.styles.get(symbol, {}).get(
                    "color", default_color(symbol)
                ),
                "alpha": float(self.styles.get(symbol, {}).get("alpha", 1.0)),
            }
            != self.initial_element_styles[symbol]
        }
        atom_attribute_names = {
            "atom_style": "default_atom_style",
            "custom_radius": "default_custom_radius",
            "atom_scale": "default_atom_scale",
            "material": "atom_material",
            "lighting": "atom_lighting",
            "edges": "atom_edges",
            "edge_color": "atom_edge_color",
            "edge_width": "atom_edge_width",
            "ambient": "material_ambient",
            "diffuse": "material_diffuse",
            "specular": "material_specular",
            "roughness": "material_roughness",
            "metallic": "material_metallic",
        }
        self.accepted_values["changed_atom_values"] = {
            name: self.accepted_values[name]
            for name, attribute in atom_attribute_names.items()
            if self.accepted_values[name] != getattr(self.viewer, attribute)
        }
        scene_attribute_names = {
            "background": "_background_color_value",
            "axes": "axes_visible",
            "grid": "grid_visible",
            "bond_width": "bond_width",
            "bond_alpha": "bond_alpha",
            "distinguish_files": "distinguish_files",
        }
        self.accepted_values["changed_scene_values"] = {
            name: self.accepted_values[name]
            for name, attribute in scene_attribute_names.items()
            if self.accepted_values[name] != getattr(self.viewer, attribute)
        }
        super().accept()


class MetricStyleDialog(QDialog):
    """Collect line and label settings for current-frame measurements."""

    def __init__(self, style, parent=None):
        """Build measurement style controls from one style mapping."""
        super().__init__(parent)
        self.setWindowTitle("Metric Style")
        self.line_color_button = QPushButton(self)
        self.text_color_button = QPushButton(self)
        for button, color in (
            (self.line_color_button, style["line_color"]),
            (self.text_color_button, style["text_color"]),
        ):
            button.setProperty("color", color)
            button.setStyleSheet(f"background-color: {color};")
            button.clicked.connect(lambda _checked=False, target=button: self._choose_color(target))
        self.line_style_combo = QComboBox(self)
        self.line_style_combo.addItem("Dashed", "dashed")
        self.line_style_combo.addItem("Solid", "solid")
        self.line_style_combo.setCurrentIndex(
            max(0, self.line_style_combo.findData(style.get("line_style", "dashed")))
        )
        self.font_size_spin = QSpinBox(self)
        self.font_size_spin.setRange(6, 72)
        self.font_size_spin.setValue(int(style.get("font_size", 14)))
        form = QFormLayout(self)
        form.addRow("Line color", self.line_color_button)
        form.addRow("Text color", self.text_color_button)
        form.addRow("Line style", self.line_style_combo)
        form.addRow("Font size", self.font_size_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _choose_color(self, button):
        """Store one valid chosen color on its button."""
        color = QColorDialog.getColor(QColor(button.property("color")), self)
        if color.isValid():
            button.setProperty("color", color.name())
            button.setStyleSheet(f"background-color: {color.name()};")

    def values(self):
        """Return a detached metric style mapping."""
        return {
            "line_color": self.line_color_button.property("color"),
            "text_color": self.text_color_button.property("color"),
            "line_style": self.line_style_combo.currentData(),
            "font_size": self.font_size_spin.value(),
        }


class NewFrameDialog(QDialog):
    """Collect one lattice and a dynamic list of atoms for a new frame."""

    def __init__(self, parent=None):
        """Build an initially empty structure form."""
        super().__init__(parent)
        self.setWindowTitle("New Frame")
        self.lattice_edits = []
        self.atom_rows = []
        layout = QVBoxLayout(self)
        lattice_grid = QGridLayout()
        for row in range(3):
            entries = []
            for column in range(3):
                edit = QLineEdit("1" if row == column else "0", self)
                lattice_grid.addWidget(edit, row, column)
                entries.append(edit)
            self.lattice_edits.append(entries)
        layout.addLayout(lattice_grid)
        self.atom_layout = QVBoxLayout()
        layout.addLayout(self.atom_layout)
        add_button = QPushButton("Add atom row", self)
        add_button.clicked.connect(self._add_atom_row)
        layout.addWidget(add_button)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._values = None

    def _add_atom_row(self):
        """Append editable element and Cartesian-coordinate controls."""
        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        edits = [QLineEdit(self) for _ in range(4)]
        edits[0].setPlaceholderText("Element")
        for edit, placeholder in zip(edits[1:], ("x", "y", "z")):
            edit.setPlaceholderText(placeholder)
            edit.setText("0")
        remove_button = QToolButton(row_widget)
        remove_button.setText("X")
        remove_button.clicked.connect(lambda: self._remove_atom_row(row_widget, edits))
        for edit in edits:
            row_layout.addWidget(edit)
        row_layout.addWidget(remove_button)
        self.atom_rows.append((row_widget, edits))
        self.atom_layout.addWidget(row_widget)

    def _remove_atom_row(self, widget, edits):
        """Remove one dynamic atom row from the form."""
        self.atom_rows.remove((widget, edits))
        widget.deleteLater()

    def accept(self):
        """Validate finite lattice and atom rows before accepting."""
        try:
            lattice = np.asarray(
                [[float(edit.text()) for edit in row] for row in self.lattice_edits],
                dtype=float,
            )
            symbols = []
            coordinates = []
            for _, edits in self.atom_rows:
                symbol = edits[0].text().strip()
                if symbol not in ELEMENT_COLORS or symbol == "X":
                    raise ValueError(f"unknown element: {symbol or '(empty)'}")
                symbols.append(symbol)
                coordinates.append([float(edit.text()) for edit in edits[1:]])
            coordinate_array = np.asarray(coordinates, dtype=float).reshape((-1, 3))
            if not np.isfinite(lattice).all() or not np.isfinite(coordinate_array).all():
                raise ValueError("lattice and coordinates must be finite")
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "Invalid frame", str(error))
            return
        self._values = (
            np.asarray(symbols, dtype="<U3"),
            coordinate_array,
            lattice,
        )
        super().accept()

    def values(self):
        """Return detached arrays from the validated form."""
        return tuple(np.array(value, copy=True) for value in self._values)


class FrameListWidget(QListWidget):
    """Expose the displayed frame permutation after an internal drop."""

    permutationChanged = Signal(object)

    def dropEvent(self, event):
        """Emit new-row to old-row indices after Qt completes a move."""
        super().dropEvent(event)
        permutation = [int(self.item(row).data(Qt.ItemDataRole.UserRole)) for row in range(self.count())]
        self.permutationChanged.emit(permutation)


class FileSortProxyModel(QSortFilterProxyModel):
    """Sort directories and files with a configurable folder-first rule."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._folders_first = True
        self._header_labels = {}

    def set_header_labels(self, labels):
        """Set translated labels for the horizontal file-tree headers."""
        self._header_labels = dict(labels)
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, 3)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        """Translate the file-tree header while preserving model defaults."""
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.DisplayRole
            and section in self._header_labels
        ):
            return self._header_labels[section]
        return super().headerData(section, orientation, role)

    def set_folders_first(self, enabled):
        """Set whether directories sort before files and refresh the view."""
        enabled = bool(enabled)
        if enabled != self._folders_first:
            self._folders_first = enabled
            self.invalidate()

    def lessThan(self, left, right):
        """Compare two source indexes using the folder-first preference."""
        model = left.model()
        if model is None:
            return super().lessThan(left, right)
        left_dir = bool(model.isDir(left))
        right_dir = bool(model.isDir(right))
        if self._folders_first and left_dir != right_dir:
            return left_dir
        return super().lessThan(left, right)


class FrameSelectionDialog(QDialog):
    """Collect a frame scope using the shared current/all/range syntax."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Delete Frames")
        self.selection = QLineEdit(self)
        self.selection.setPlaceholderText("current, all, or e.g. 1-3, 5")
        layout = QFormLayout(self)
        layout.addRow("Frames (1-based)", self.selection)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self, frame_count, current_index):
        """Return zero-based indices parsed as current/all/range scope."""
        return parse_frame_scope(self.selection.text(), frame_count, current_index)


class UpdaterDialog(QDialog):
    """Present configured updater functions and optional call arguments.

    Args:
        updaters: Loaded updater records containing path, name, and callback.
        parent: Optional Qt parent widget.

    Return:
        A modal updater-management dialog.
    """

    def __init__(self, updaters, parent=None):
        """Build the updater tree, argument fields, and operation buttons.

        Args:
            updaters: Loaded updater records containing path, name, and callback.
            parent: Optional Qt parent widget.

        Return:
            None. Controls are initialized with no selected function.
        """
        super().__init__(parent)
        self.setWindowTitle("Update - More")
        self.resize(760, 420)
        self.updaters = []
        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabel("Updater files and functions")
        self.tree.currentItemChanged.connect(self._sync_selection)
        self.args_edit = QLineEdit(self)
        self.args_edit.setPlaceholderText("e.g. 0.1, 'fast' or [0.1, 'fast']")
        self.kwargs_edit = QLineEdit(self)
        self.kwargs_edit.setPlaceholderText("e.g. {'steps': 20, 'relax_cell': True}")
        self.apply_button = QPushButton("Apply", self)
        self.help_button = QPushButton("Help", self)
        self.dry_run_button = QPushButton("Dry Run", self)
        self.refresh_button = QPushButton("Refresh", self)
        self.close_button = QPushButton("Close", self)
        self.scope_edit = QLineEdit(self)
        owner = self.parent()
        self.scope_edit.setText(str(getattr(owner, "current_index", 0) + 1))
        self.in_place_check = QCheckBox("in-place", self)
        self.close_button.clicked.connect(self.accept)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        form = QFormLayout()
        form.addRow("args:", self.args_edit)
        form.addRow("kwargs:", self.kwargs_edit)
        right_layout.addLayout(form)
        apply_row = QHBoxLayout()
        apply_row.addWidget(self.apply_button)
        apply_row.addWidget(QLabel("Apply to:", self))
        self.scope_edit.setPlaceholderText("empty=current, all, 1,2-5")
        apply_row.addWidget(self.scope_edit, 1)
        apply_row.addWidget(self.in_place_check)
        right_layout.addLayout(apply_row)
        right_layout.addWidget(self.help_button)
        right_layout.addWidget(self.dry_run_button)
        right_layout.addWidget(self.refresh_button)
        right_layout.addStretch(1)
        right_layout.addWidget(self.close_button)

        splitter = QSplitter(self)
        splitter.addWidget(self.tree)
        splitter.addWidget(right)
        splitter.setSizes([360, 400])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self.set_updaters(updaters)

    def set_updaters(self, updaters):
        """Replace the tree with the currently loaded updater records.

        Args:
            updaters: Loaded updater records containing path, name, and callback.

        Return:
            None. Any previous selection is cleared and controls are disabled.
        """
        self.updaters = list(updaters)
        self.tree.clear()
        file_items = {}
        # Show successfully loaded files even when none of the requested names
        # exists there; only child function rows enable invocation controls.
        for path in getattr(self.parent(), "updater_module_paths", []):
            parent_item = QTreeWidgetItem([path])
            self.tree.addTopLevelItem(parent_item)
            file_items[path] = parent_item
        for index, updater in enumerate(self.updaters):
            path = str(updater["path"])
            parent_item = file_items.get(path)
            if parent_item is None:
                parent_item = QTreeWidgetItem([path])
                self.tree.addTopLevelItem(parent_item)
                file_items[path] = parent_item
            function_item = QTreeWidgetItem([updater["name"]])
            function_item.setData(0, Qt.ItemDataRole.UserRole, index)
            parent_item.addChild(function_item)
            parent_item.setExpanded(True)
        self.tree.setCurrentItem(None)
        self._sync_selection(None, None)

    def selected_updater(self):
        """Return the updater record selected in the function tree.

        Args:
            None.

        Return:
            The selected updater record, or ``None`` when no function is selected.
        """
        item = self.tree.currentItem()
        if item is None:
            return None
        index = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(index, int) or not 0 <= index < len(self.updaters):
            return None
        return self.updaters[index]

    def _sync_selection(self, current, previous):
        """Enable function operations only for a selected function item.

        Args:
            current: Newly selected tree item.
            previous: Previously selected tree item.

        Return:
            None. Buttons and argument fields reflect selection availability.
        """
        del current, previous
        enabled = self.selected_updater() is not None
        for widget in (
            self.args_edit,
            self.kwargs_edit,
            self.scope_edit,
            self.in_place_check,
            self.apply_button,
            self.help_button,
            self.dry_run_button,
            self.refresh_button,
        ):
            widget.setEnabled(enabled)


class ManagedSubWindow(QMdiSubWindow):
    """Let the window manager own the viewer shutdown sequence."""

    closed = Signal(object)

    def closeEvent(self, event):
        """Delegate cleanup before accepting the MDI close event."""
        self.closed.emit(self)
        event.accept()


class WindowManager:
    """Manage equal frame-indexed viewers hosted by the MDI area."""

    def __init__(self, owner, area):
        self.owner = owner
        self.area = area
        self.records = []
        self._activating = False
        self._closing = set()
        self.area.subWindowActivated.connect(self._activated)

    def add(self, viewer, frame_index, title="Frame", show=True, file_key=None):
        """Add one viewer subwindow and retain its current frame index."""
        subwindow = ManagedSubWindow(self.area)
        subwindow.setWidget(viewer)
        self.area.addSubWindow(subwindow)
        subwindow.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        flags = (
            Qt.WindowType.SubWindow
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        subwindow.setWindowFlags(flags)
        subwindow.setWindowTitle(title)
        subwindow.setOption(QMdiSubWindow.SubWindowOption.RubberBandMove, True)
        subwindow.setOption(QMdiSubWindow.SubWindowOption.RubberBandResize, True)
        record = {
            "viewer": viewer,
            "subwindow": subwindow,
            "frame_index": int(frame_index),
            "file_key": file_key,
        }
        self.records.append(record)
        subwindow.closed.connect(self._close_record)
        if show:
            viewer.show()
            subwindow.show()
            self.area.setActiveSubWindow(subwindow)
        return record

    def find(self, file_key, frame_index=None):
        """Return the unique record for one file/frame, or ``None``."""
        for record in self.records:
            if record["file_key"] != file_key:
                continue
            if frame_index is None or record["frame_index"] == int(frame_index):
                return record
        return None

    def records_for_file(self, file_key):
        """Return all records displaying one file."""
        return [record for record in self.records if record["file_key"] == file_key]

    def show_hidden(self):
        """Show windows that were populated while hidden without changing focus."""
        for record in self.records:
            if not record["subwindow"].isVisible():
                record["viewer"].show()
                record["subwindow"].show()
            QTimer.singleShot(0, lambda item=record: self._render_if_open(item))

    def _render_if_open(self, record):
        """Render a still-open viewer after Qt has realized its native surface."""
        if record not in self.records or not record["subwindow"].isVisible():
            return
        record["viewer"].render()

    def find_frame(self, frame_index, file_key=None):
        """Return an open window record for one frame, optionally per file."""
        frame_index = int(frame_index)
        return next(
            (
                record for record in self.records
                if record["frame_index"] == frame_index
                and (file_key is None or record["file_key"] == file_key)
            ),
            None,
        )

    def _close_record(self, subwindow):
        """Close and destroy one viewer without leaving queued render callbacks."""
        record = next(
            (item for item in self.records if item["subwindow"] is subwindow),
            None,
        )
        if record is None:
            return
        if subwindow in self._closing:
            return
        self._closing.add(subwindow)
        self.records.remove(record)
        viewer = record["viewer"]
        if self.owner.viewer is viewer and self.owner.timer.isActive():
            self.owner._pause_playback()
        replacement = self.active_record()
        owner_viewer_was_closed = self.owner.viewer is viewer
        if owner_viewer_was_closed:
            if replacement is not None:
                self.owner._activate_viewer(replacement["viewer"])
            else:
                self.owner.viewer = None
                self.owner._activate_empty_state()
        subwindow.setWidget(None)
        subwindow.hide()

        def finalize_close():
            """Release native rendering resources after queued paints drain."""
            viewer.setUpdatesEnabled(False)
            viewer.close()
            viewer.setParent(None)
            viewer.deleteLater()
            subwindow.deleteLater()
            self._closing.discard(subwindow)
            state = self.owner.files.get(record["file_key"])
            if state is not None:
                state.release()
                self.owner._on_file_released(state)

        QTimer.singleShot(0, finalize_close)

    def _activated(self, subwindow):
        """Make the activated viewer the owner's current viewer."""
        if self._activating or subwindow is None:
            return
        for record in self.records:
            if record["subwindow"] is subwindow:
                self.owner._activate_viewer(record["viewer"])
                break

    def active_record(self):
        """Return the record for the active subwindow, if any."""
        active = self.area.activeSubWindow()
        for record in self.records:
            if record["subwindow"] is active:
                return record
        return self.records[0] if self.records else None

    def close_all(self):
        """Close every displayed structure window."""
        for record in list(self.records):
            record["subwindow"].close()

    def remove_frames(self, indices, file_key=None):
        """Drop deleted indices and shift surviving window references."""
        deleted = set(int(index) for index in indices)
        for record in self.records:
            if file_key is not None and record["file_key"] != file_key:
                continue
            index = record["frame_index"]
            record["frame_index"] = index - sum(old < index for old in deleted)

    def apply_permutation(self, inverse, file_key=None):
        """Update every displayed frame through an old-to-new permutation."""
        for record in self.records:
            if file_key is not None and record["file_key"] != file_key:
                continue
            index = record["frame_index"]
            if 0 <= index < len(inverse):
                record["frame_index"] = int(inverse[index])

    def refresh_all(self, reset_camera=False):
        """Redraw every open viewer against its own file state."""
        for record in list(self.records):
            viewer = record["viewer"]
            state = self.owner.files.get(record["file_key"])
            if state is None or state.closed:
                continue
            if not state.coordinates:
                viewer.elements = state.elements
                viewer.coordinates = state.coordinates
                viewer.lattices = state.lattices
                viewer.current_index = 0
                viewer.set_file_header("")
                viewer.set_frame_state(
                    state.atom_frame_styles,
                    state.frame_metrics,
                    state.bond_adjacencies,
                    state.bond_orders,
                )
                viewer.draw_frame(reset_camera=reset_camera)
                continue
            index = min(max(int(record["frame_index"]), 0), len(state.coordinates) - 1)
            record["frame_index"] = index
            self.owner._set_record_header(record)
            viewer.elements = state.elements
            viewer.coordinates = state.coordinates
            viewer.lattices = state.lattices
            viewer.bond_frame_indices = set(state.bond_frame_indices)
            viewer.bond_adjacencies = state.bond_adjacencies
            viewer.bond_orders = state.bond_orders
            viewer.invalidate_bond_meshes()
            viewer.current_index = index
            viewer.set_frame_state(
                state.atom_frame_styles,
                state.frame_metrics,
                state.bond_adjacencies,
                state.bond_orders,
            )
            viewer.selected_indices = viewer.selected_indices[
                (viewer.selected_indices >= 0)
                & (viewer.selected_indices < len(state.elements[index]))
            ]
            graph = state.bond_adjacencies[index]
            # Graph-only events already clear their selections. This also
            # protects views reassigned by frame deletion or batch replacement.
            if graph is None:
                viewer.selected_bonds.clear()
            elif viewer.selected_bonds:
                viewer.selected_bonds = {
                    pair for pair in viewer.selected_bonds
                    if max(pair) < len(state.elements[index])
                    and np.any(np.all(graph == np.asarray(pair)[:, None], axis=0))
                }
            viewer.draw_frame(reset_camera=reset_camera)
        self.show_hidden()
        self.apply_file_colors()
        self.owner._sync_bond_action()

    def apply_file_colors(self):
        """Tint each subwindow border by its file, or make all borders white."""
        enabled = bool(
            self.owner.viewer is not None and self.owner.viewer.distinguish_files
        )
        for record in self.records:
            state = self.owner.files.get(record["file_key"])
            color = "#ffffff"
            if enabled and state is not None and state.border_color is not None:
                source = state.border_color
                color = (
                    f"rgba({source.red()}, {source.green()}, {source.blue()}, 90)"
                )
            record["subwindow"].setStyleSheet(
                f"QMdiSubWindow {{ border: 2px solid {color}; }}"
            )


class SettingsDialog(QDialog):
    """Edit persistent history, rendering, and keyboard preferences.

    Args:
        settings: Validated application settings.
        parent: Optional Qt parent widget.

    Return:
        A modal settings dialog.
    """

    def __init__(self, settings, parent=None):
        """Build controls from one validated settings dictionary.

        Args:
            settings: Validated application settings.
            parent: Optional Qt parent widget.

        Return:
            None. The modal dialog is initialized without changing settings.
        """
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(420, 300)
        self.buctoolkit_paths = list(settings.get("buctoolkit_paths", []))
        self.history_limit_spin = QSpinBox(self)
        self.history_limit_spin.setRange(1, 1000)
        self.history_limit_spin.setValue(settings["history_limit"])
        self.cache_limit_spin = QSpinBox(self)
        self.cache_limit_spin.setRange(64, 65536)
        self.cache_limit_spin.setSingleStep(256)
        self.cache_limit_spin.setSuffix(" MB")
        self.cache_limit_spin.setValue(settings["file_cache_limit_mb"])
        self.material_combo = QComboBox(self)
        self.material_combo.addItem("Default", "default")
        self.material_combo.addItem("Matte", "matte")
        self.material_combo.addItem("Glossy", "glossy")
        self.material_combo.addItem("Metallic", "metallic")
        material_index = self.material_combo.findData(settings["default_atom_material"])
        self.material_combo.setCurrentIndex(max(0, material_index))
        self.lighting_check = QCheckBox("Enable lighting", self)
        self.lighting_check.setChecked(settings["atom_lighting"])
        self.edges_check = QCheckBox("Show atom boundaries", self)
        self.edges_check.setChecked(settings["atom_edges"])
        self.edge_width_spin = QDoubleSpinBox(self)
        self.edge_width_spin.setRange(0.1, 10.0)
        self.edge_width_spin.setSingleStep(0.1)
        self.edge_width_spin.setValue(settings["atom_edge_width"])
        self.edge_color_button = QPushButton(self)
        self.edge_color_button.setProperty("color", settings["atom_edge_color"])
        self.edge_color_button.setStyleSheet(f"background-color: {settings['atom_edge_color']};")
        self.edge_color_button.clicked.connect(self._choose_edge_color)
        self.keyboard_translation_step_spin = QDoubleSpinBox(self)
        self.keyboard_translation_step_spin.setRange(0.001, 100.0)
        self.keyboard_translation_step_spin.setDecimals(4)
        self.keyboard_translation_step_spin.setSingleStep(0.1)
        self.keyboard_translation_step_spin.setSuffix(" Å")
        self.keyboard_translation_step_spin.setValue(
            settings["keyboard_translation_step"]
        )
        self.external_python_edit = QLineEdit(
            settings.get("external_python_path") or "", self
        )
        self.external_python_edit.setPlaceholderText("None (build-in)")
        self.external_python_button = QPushButton("Browse...", self)
        self.external_python_button.clicked.connect(self._choose_external_python)
        external_python_row = QWidget(self)
        external_python_layout = QHBoxLayout(external_python_row)
        external_python_layout.setContentsMargins(0, 0, 0, 0)
        external_python_layout.addWidget(self.external_python_edit, 1)
        external_python_layout.addWidget(self.external_python_button)

        form = QFormLayout()
        form.addRow("Undo history steps", self.history_limit_spin)
        form.addRow("File cache limit", self.cache_limit_spin)
        form.addRow("Default material", self.material_combo)
        form.addRow(self.lighting_check)
        form.addRow(self.edges_check)
        form.addRow("Boundary width", self.edge_width_spin)
        form.addRow("Boundary color", self.edge_color_button)
        form.addRow("Keyboard translation step", self.keyboard_translation_step_spin)
        form.addRow("External Python interpreter", external_python_row)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _choose_edge_color(self):
        """Choose the default atom-boundary color.

        Args:
            None.

        Return:
            None. A valid selected color is retained by the button.
        """
        current_color = self.edge_color_button.property("color") or "#202020"
        color = QColorDialog.getColor(QColor(current_color), self)
        if color.isValid():
            self.edge_color_button.setProperty("color", color.name())
            self.edge_color_button.setStyleSheet(f"background-color: {color.name()};")

    def _choose_external_python(self):
        """Choose a persistent external Python interpreter path."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "External Python interpreter",
            self.external_python_edit.text(),
            "Python (*python*.exe python*);;All files (*)",
        )
        if path:
            self.external_python_edit.setText(path)

    def values(self):
        """Return validated settings represented by the controls.

        Args:
            None.

        Return:
            A JSON-compatible application settings dictionary.
        """
        return {
            "history_limit": self.history_limit_spin.value(),
            "default_atom_material": self.material_combo.currentData(),
            "atom_lighting": self.lighting_check.isChecked(),
            "atom_edges": self.edges_check.isChecked(),
            "atom_edge_color": self.edge_color_button.property("color") or "#202020",
            "atom_edge_width": self.edge_width_spin.value(),
            "buctoolkit_paths": self.buctoolkit_paths,
            "keyboard_translation_step": self.keyboard_translation_step_spin.value(),
            "file_cache_limit_mb": self.cache_limit_spin.value(),
            "external_python_path": self.external_python_edit.text().strip() or None,
        }


class FileManagerDialog(QDialog):
    """Configure visible file-tree metadata and sorting."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("File Manager")
        self.resize(360, 220)
        self.size_check = QCheckBox("Show size", self)
        self.size_check.setChecked(settings["file_manager_show_size"])
        self.type_check = QCheckBox("Show type", self)
        self.type_check.setChecked(settings["file_manager_show_type"])
        self.modified_check = QCheckBox("Show modified date", self)
        self.modified_check.setChecked(settings["file_manager_show_modified"])
        self.folders_check = QCheckBox("Folders before files", self)
        self.folders_check.setChecked(settings["file_manager_folders_first"])
        self.sort_combo = QComboBox(self)
        for label, value in (
            ("Name", 0), ("Size", 1), ("Type", 2), ("Modified", 3)
        ):
            self.sort_combo.addItem(label, value)
        self.sort_combo.setCurrentIndex(
            max(0, self.sort_combo.findData(settings["file_manager_sort_column"]))
        )
        self.order_combo = QComboBox(self)
        self.order_combo.addItem("Ascending", "asc")
        self.order_combo.addItem("Descending", "desc")
        self.order_combo.setCurrentIndex(
            0 if settings["file_manager_sort_order"] == "asc" else 1
        )
        form = QFormLayout()
        form.addRow("Metadata", self.size_check)
        form.addRow("", self.type_check)
        form.addRow("", self.modified_check)
        form.addRow(self.folders_check)
        form.addRow("Sort by", self.sort_combo)
        form.addRow("Order", self.order_combo)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self):
        """Return file-manager settings represented by the dialog."""
        return {
            "file_manager_show_size": self.size_check.isChecked(),
            "file_manager_show_type": self.type_check.isChecked(),
            "file_manager_show_modified": self.modified_check.isChecked(),
            "file_manager_folders_first": self.folders_check.isChecked(),
            "file_manager_sort_column": int(self.sort_combo.currentData()),
            "file_manager_sort_order": self.order_combo.currentData(),
        }


class ParentRowProxyModel(QAbstractProxyModel):
    """Expose a synthetic POSIX-style ``..`` row above the current root."""

    _PARENT_ID = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root = QPersistentModelIndex()
        self._has_parent = False
        self._header_labels = {}
        self._next_id = 2
        self._id_to_source = {}
        self._source_to_id = {}

    def _reset_mapping(self):
        self._next_id = 2
        self._id_to_source = {}
        self._source_to_id = {}

    def set_root_source_index(self, source_index):
        """Set the sorted source index represented by the view root."""
        self.beginResetModel()
        self._root = QPersistentModelIndex(source_index)
        root = QModelIndex(self._root)
        self._has_parent = root.isValid() and root.parent().isValid()
        self._reset_mapping()
        self.endResetModel()

    def setSourceModel(self, model):
        """Forward source changes so asynchronous directory loads repaint."""
        old = self.sourceModel()
        if old is not None:
            for signal in (
                old.rowsInserted,
                old.rowsRemoved,
                old.rowsMoved,
                old.dataChanged,
                old.layoutChanged,
                old.modelReset,
            ):
                try:
                    signal.disconnect()
                except (RuntimeError, TypeError):
                    pass
        super().setSourceModel(model)
        if model is not None:
            for signal in (
                model.rowsInserted,
                model.rowsRemoved,
                model.rowsMoved,
                model.dataChanged,
                model.layoutChanged,
                model.modelReset,
            ):
                signal.connect(self._source_changed)

    def _source_changed(self, *_args):
        """Refresh mapped rows after the file model changes."""
        self.layoutAboutToBeChanged.emit()
        self._reset_mapping()
        self.layoutChanged.emit()

    def is_parent_row(self, index):
        """Return whether a view index is the synthetic parent row."""
        return index.isValid() and index.internalId() == self._PARENT_ID

    def rowCount(self, parent=QModelIndex()):
        if self.sourceModel() is None:
            return 0
        root = QModelIndex(self._root)
        if not root.isValid():
            return 0
        if not parent.isValid():
            return self.sourceModel().rowCount(root) + int(self._has_parent)
        source_parent = self.mapToSource(parent)
        if not source_parent.isValid():
            return 0
        return self.sourceModel().rowCount(source_parent)

    def columnCount(self, parent=QModelIndex()):
        if self.sourceModel() is None:
            return 0
        return self.sourceModel().columnCount()

    def index(self, row, column, parent=QModelIndex()):
        if not parent.isValid():
            if self._has_parent and row == 0:
                return self.createIndex(row, column, self._PARENT_ID)
            source_row = row - int(self._has_parent)
            source_index = self.sourceModel().index(
                source_row, column, QModelIndex(self._root)
            )
            return self.mapFromSource(source_index)
        source_parent = self.mapToSource(parent)
        if not source_parent.isValid():
            return QModelIndex()
        source_index = self.sourceModel().index(row, column, source_parent)
        return self.mapFromSource(source_index)

    def parent(self, index):
        if not index.isValid() or self.is_parent_row(index):
            return QModelIndex()
        source_index = self.mapToSource(index)
        if not source_index.isValid():
            return QModelIndex()
        source_parent = source_index.parent()
        root = QModelIndex(self._root)
        if not source_parent.isValid() or source_parent == root:
            return QModelIndex()
        return self.mapFromSource(source_parent)

    def mapToSource(self, proxy_index):
        if not proxy_index.isValid() or self.is_parent_row(proxy_index):
            return QModelIndex()
        persistent = self._id_to_source.get(int(proxy_index.internalId()))
        if persistent is None or not persistent.isValid():
            return QModelIndex()
        return QModelIndex(persistent)

    def mapFromSource(self, source_index):
        root = QModelIndex(self._root)
        if not source_index.isValid() or not root.isValid():
            return QModelIndex()
        if source_index == root:
            return QModelIndex()
        parent = source_index.parent()
        key = (
            source_index.row(),
            source_index.column(),
            parent.row(),
            parent.column(),
            int(parent.internalId()) if parent.isValid() else -1,
        )
        existing = self._source_to_id.get(key)
        if existing is not None:
            row = source_index.row()
            if parent == root and self._has_parent:
                row += 1
            return self.createIndex(row, source_index.column(), existing)
        proxy_id = self._next_id
        self._next_id += 1
        self._source_to_id[key] = proxy_id
        self._id_to_source[proxy_id] = QPersistentModelIndex(source_index)
        row = source_index.row()
        if parent == root and self._has_parent:
            row += 1
        return self.createIndex(row, source_index.column(), proxy_id)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if self.is_parent_row(index):
            if role == Qt.ItemDataRole.DisplayRole and index.column() == 0:
                return ".."
            if role == Qt.ItemDataRole.ToolTipRole and index.column() == 0:
                return ".."
            return None
        source_index = self.mapToSource(index)
        if not source_index.isValid():
            return None
        return self.sourceModel().data(source_index, role)

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        """Forward inline edits to the source file model."""
        if self.is_parent_row(index):
            return False
        source_index = self.mapToSource(index)
        if not source_index.isValid():
            return False
        return self.sourceModel().setData(source_index, value, role)

    def flags(self, index):
        if self.is_parent_row(index):
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        source_index = self.mapToSource(index)
        if not source_index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return self.sourceModel().flags(source_index)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.DisplayRole
            and section in self._header_labels
        ):
            return self._header_labels[section]
        return self.sourceModel().headerData(section, orientation, role)

    def set_header_labels(self, labels):
        """Set translated labels for the horizontal headers."""
        self._header_labels = dict(labels)
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, 3)

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        """Sort the underlying file model and refresh mapped rows."""
        self.layoutAboutToBeChanged.emit()
        self.sourceModel().sort(column, order)
        self._reset_mapping()
        self.layoutChanged.emit()


class CleanBackendDialog(QDialog):
    """Configure local or global clean optimizer settings."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clean Backend")
        self.resize(440, 390)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)

        self.local_group = QGroupBox("Local minimizer", self)
        local_layout = QFormLayout(self.local_group)
        self.local_radio = QRadioButton("Use local minimizer", self.local_group)
        self.local_method_combo = QComboBox(self.local_group)
        for method in CLEAN_LOCAL_METHODS:
            self.local_method_combo.addItem(method, method)
        self.local_method_combo.setCurrentIndex(
            max(0, self.local_method_combo.findData(settings["clean_local_method"]))
        )
        self.local_maxiter_spin = QSpinBox(self.local_group)
        self.local_maxiter_spin.setRange(1, 100000)
        self.local_maxiter_spin.setValue(settings["clean_local_maxiter"])
        self.local_gtol_spin = QDoubleSpinBox(self.local_group)
        self.local_gtol_spin.setDecimals(8)
        self.local_gtol_spin.setRange(1e-10, 1.0)
        self.local_gtol_spin.setSingleStep(1e-6)
        self.local_gtol_spin.setValue(settings["clean_local_gtol"])
        local_layout.addRow(self.local_radio)
        local_layout.addRow("Method", self.local_method_combo)
        local_layout.addRow("Max iterations", self.local_maxiter_spin)
        local_layout.addRow("Gradient tolerance", self.local_gtol_spin)

        self.global_group = QGroupBox("Global minimizer", self)
        global_layout = QFormLayout(self.global_group)
        self.global_radio = QRadioButton("Use global minimizer", self.global_group)
        self.global_method_combo = QComboBox(self.global_group)
        for method in CLEAN_GLOBAL_METHODS:
            self.global_method_combo.addItem(method, method)
        self.global_method_combo.setCurrentIndex(
            max(0, self.global_method_combo.findData(settings["clean_global_method"]))
        )
        self.global_maxiter_spin = QSpinBox(self.global_group)
        self.global_maxiter_spin.setRange(1, 100000)
        self.global_maxiter_spin.setValue(settings["clean_global_maxiter"])
        self.workers_spin = QSpinBox(self.global_group)
        self.workers_spin.setRange(1, max(1, (os.cpu_count() or 1)))
        self.workers_spin.setValue(settings["clean_workers"])
        self.bounds_spin = QDoubleSpinBox(self.global_group)
        self.bounds_spin.setRange(0.1, 30.0)
        self.bounds_spin.setDecimals(2)
        self.bounds_spin.setSingleStep(0.5)
        self.bounds_spin.setSuffix(" Å")
        self.bounds_spin.setValue(settings["clean_bounds_span"])
        global_layout.addRow(self.global_radio)
        global_layout.addRow("Method", self.global_method_combo)
        global_layout.addRow("Max iterations", self.global_maxiter_spin)
        global_layout.addRow("Workers (DE/SHGO only)", self.workers_spin)
        global_layout.addRow("Bounds span", self.bounds_spin)

        self.mode_group.addButton(self.local_radio)
        self.mode_group.addButton(self.global_radio)
        if settings["clean_mode"] == "global":
            self.global_radio.setChecked(True)
        else:
            self.local_radio.setChecked(True)
        self.local_radio.toggled.connect(self._update_enabled)
        self.global_radio.toggled.connect(self._update_enabled)
        self.global_method_combo.currentIndexChanged.connect(
            self._update_global_controls
        )
        self._update_enabled()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.local_group)
        layout.addWidget(self.global_group)
        layout.addWidget(buttons)

    def _update_enabled(self):
        """Enable only the selected optimizer group's controls."""
        local = self.local_radio.isChecked()
        for widget in (
            self.local_method_combo,
            self.local_maxiter_spin,
            self.local_gtol_spin,
        ):
            widget.setEnabled(local)
        for widget in (
            self.global_method_combo,
            self.global_maxiter_spin,
            self.workers_spin,
            self.bounds_spin,
        ):
            widget.setEnabled(not local)
        self._update_global_controls()

    def _update_global_controls(self):
        """Disable controls that the selected global method does not support."""
        if not self.global_radio.isChecked():
            return
        method = self.global_method_combo.currentData()
        self.workers_spin.setEnabled(
            method in {"differential_evolution", "shgo"}
        )
        self.bounds_spin.setEnabled(method != "basinhopping")

    def values(self):
        """Return the selected clean-backend settings."""
        return {
            "clean_mode": "local" if self.local_radio.isChecked() else "global",
            "clean_local_method": self.local_method_combo.currentData(),
            "clean_local_maxiter": self.local_maxiter_spin.value(),
            "clean_local_gtol": self.local_gtol_spin.value(),
            "clean_global_method": self.global_method_combo.currentData(),
            "clean_global_maxiter": self.global_maxiter_spin.value(),
            "clean_workers": self.workers_spin.value(),
            "clean_bounds_span": self.bounds_spin.value(),
        }


class AddAtomDialog(QDialog):
    """Collect one element and either Cartesian coordinates or a reference mode."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Atom")
        self._reference_requested = False
        layout = QVBoxLayout(self)
        quick_layout = QHBoxLayout()
        quick_layout.addWidget(QLabel("Element:"))
        self.element_edit = QLineEdit(self)
        self.element_edit.setPlaceholderText("C, H, O, N, P, S, ...")
        quick_layout.addWidget(self.element_edit)
        for symbol in ("C", "H", "O", "N", "P", "S"):
            button = QPushButton(symbol, self)
            button.setFixedWidth(34)
            button.clicked.connect(lambda _checked=False, value=symbol: self.element_edit.setText(value))
            quick_layout.addWidget(button)
        layout.addLayout(quick_layout)

        coordinate_layout = QFormLayout()
        self.coordinate_edits = []
        validator = QDoubleValidator(-1.0e9, 1.0e9, 8, self)
        for axis in ("x", "y", "z"):
            edit = QLineEdit("0", self)
            edit.setValidator(validator)
            coordinate_layout.addRow(f"{axis}:", edit)
            self.coordinate_edits.append(edit)
        layout.addLayout(coordinate_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        reference_button = QPushButton("Reference to existing atom", self)
        reference_button.clicked.connect(self._accept_reference)
        layout.addWidget(reference_button)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validated_symbol(self) -> str:
        """Normalize and validate the manually entered element symbol.

        Args:
            None.

        Returns:
            Canonical element symbol.

        Raises:
            ValueError: If the symbol is not in the periodic table.
        """
        text = self.element_edit.text().strip()
        symbol = text[:1].upper() + text[1:].lower() if text else ""
        if symbol not in ELEMENT_COLORS or symbol == "X":
            raise ValueError("element must be a valid periodic-table symbol")
        return symbol

    def _accept_reference(self) -> None:
        """Validate the element and close in reference-placement mode.

        Args:
            None.

        Returns:
            None. The caller enters viewer reference mode after acceptance.
        """
        try:
            self._validated_symbol()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid element", str(error))
            return
        self._reference_requested = True
        super().accept()

    def accept(self) -> None:
        """Validate the direct-coordinate form before accepting the dialog.

        Args:
            None.

        Returns:
            None. Invalid element or coordinate text keeps the dialog open.
        """
        try:
            self._validated_symbol()
            for edit in self.coordinate_edits:
                float(edit.text().strip())
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "Invalid atom", str(error))
            return
        super().accept()

    def values(self) -> tuple[str, np.ndarray, bool]:
        """Return the validated element, coordinates, and placement mode.

        Args:
            None.

        Returns:
            ``(symbol, xyz, reference_requested)``.
        """
        symbol = self._validated_symbol()
        xyz = np.asarray([float(edit.text()) for edit in self.coordinate_edits], dtype=float)
        return symbol, xyz, self._reference_requested


class SupercellDialog(QDialog):
    """Collect integer replication factors and a lattice transformation."""

    _MAX_MULTIPLIER = 999
    _MAX_MATRIX_VALUE = 999

    def __init__(
        self,
        multipliers: np.ndarray | None = None,
        matrix: np.ndarray | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Supercell")
        self._multipliers: list[QLineEdit] = []
        self._matrix_entries: list[list[QLineEdit]] = []
        self._accepted_values: tuple[np.ndarray, np.ndarray] | None = None
        self.resize_requested = False
        initial_multipliers = np.ones(3, dtype=np.int64) if multipliers is None else np.asarray(multipliers, dtype=np.int64)
        initial_matrix = np.eye(3, dtype=np.int64) if matrix is None else np.asarray(matrix, dtype=np.int64)

        layout = QVBoxLayout(self)
        multiplier_grid = QGridLayout()
        for column, axis_name in enumerate(("a", "b", "c")):
            axis_label = QLabel(axis_name)
            axis_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            multiplier_grid.addWidget(axis_label, 0, column)
            increase_button = QPushButton("▲")
            value_edit = QLineEdit(str(int(initial_multipliers[column])))
            value_edit.setValidator(QIntValidator(1, self._MAX_MULTIPLIER, value_edit))
            value_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            decrease_button = QPushButton("▼")
            for widget in (increase_button, value_edit, decrease_button):
                widget.setFixedWidth(52)
            increase_button.clicked.connect(
                lambda _checked=False, index=column: self._change_multiplier(index, 1)
            )
            decrease_button.clicked.connect(
                lambda _checked=False, index=column: self._change_multiplier(index, -1)
            )
            multiplier_grid.addWidget(increase_button, 1, column)
            multiplier_grid.addWidget(value_edit, 2, column)
            multiplier_grid.addWidget(decrease_button, 3, column)
            self._multipliers.append(value_edit)
        layout.addLayout(multiplier_grid)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)
        layout.addWidget(QLabel("Lattice transform:"))

        matrix_grid = QGridLayout()
        for row in range(3):
            matrix_row: list[QLineEdit] = []
            for column in range(3):
                entry = QLineEdit(str(int(initial_matrix[row, column])))
                entry.setValidator(
                    QIntValidator(-self._MAX_MATRIX_VALUE, self._MAX_MATRIX_VALUE, entry)
                )
                entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
                entry.setFixedWidth(64)
                matrix_grid.addWidget(entry, row, column)
                matrix_row.append(entry)
            self._matrix_entries.append(matrix_row)
        layout.addLayout(matrix_grid)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        resize_button = buttons.addButton("Resize", QDialogButtonBox.ButtonRole.ActionRole)
        resize_button.setToolTip("Use the current structure as a new 1 x 1 x 1 base")
        resize_button.clicked.connect(self._request_resize)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _request_resize(self) -> None:
        """Accept the dialog as a request to rebase the current supercell."""
        self.resize_requested = True
        QDialog.accept(self)

    def _change_multiplier(self, index: int, delta: int) -> None:
        """Change one replication factor while preserving its lower bound."""
        text = self._multipliers[index].text().strip()
        current = int(text) if text.isdigit() and int(text) >= 1 else 1
        value = min(self._MAX_MULTIPLIER, max(1, current + delta))
        self._multipliers[index].setText(str(value))

    def _validated_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Return validated replication factors and the integer matrix."""
        multiplier_values: list[int] = []
        for axis_name, entry in zip(("a", "b", "c"), self._multipliers):
            try:
                value = int(entry.text().strip())
            except ValueError as error:
                raise ValueError(f"The {axis_name}-axis multiplier must be an integer.") from error
            if value < 1:
                raise ValueError(
                    f"The {axis_name}-axis multiplier must be at least 1; zero is not valid."
                )
            multiplier_values.append(value)

        matrix = np.empty((3, 3), dtype=np.int64)
        for row in range(3):
            for column in range(3):
                try:
                    matrix[row, column] = int(
                        self._matrix_entries[row][column].text().strip()
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Transform entry ({row + 1}, {column + 1}) must be an integer."
                    ) from error
        if int(round(float(np.linalg.det(matrix)))) == 0:
            raise ValueError("The lattice transformation matrix must be non-singular.")
        return np.asarray(multiplier_values, dtype=np.int64), matrix

    def accept(self) -> None:
        """Validate the dialog before accepting it."""
        try:
            self._accepted_values = self._validated_values()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid supercell", str(error))
            return
        super().accept()

    def values(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the accepted replication factors and transformation matrix."""
        if self._accepted_values is None:
            return self._validated_values()
        multipliers, matrix = self._accepted_values
        return multipliers.copy(), matrix.copy()


class MainWindow(QMainWindow):
    """Present the viewer, extensible toolbar, and bottom command console.

    Args:
        parent: Optional Qt parent widget.

    Return:
        A main window with an empty toolbar and an initially empty viewer.
    """

    _STATE_ATTRS = frozenset({
        "elements",
        "coordinates",
        "lattices",
        "frame_names",
        "_frame_name_serial",
        "atom_frame_styles",
        "frame_metrics",
        "bond_adjacencies",
        "bond_orders",
        "bond_frame_indices",
        "supercell_bases",
        "supercell_transforms",
        "current_index",
        "undo_stack",
        "redo_stack",
    })

    def __getattr__(self, name):
        """Read one frame-owned attribute from the active file state."""
        if name in type(self)._STATE_ATTRS:
            state = self.__dict__.get("_active_state")
            if state is not None:
                return getattr(state, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __setattr__(self, name, value):
        """Write one frame-owned attribute into the active file state.

        Note:
            Ordinary attributes delegate to ``super().__setattr__``. PySide6
            6.11 implements ``QObject.__setattr__`` and rejects
            ``object.__setattr__`` on QObject subclasses with
            ``TypeError: can't apply this __setattr__ to MainWindow object``.
        """
        if name in type(self)._STATE_ATTRS:
            state = self.__dict__.get("_active_state")
            if state is not None:
                setattr(state, name, value)
                return
        super().__setattr__(name, value)

    def __init__(self, open_handler=None, save_handler=None, parent=None, updater=None):
        super().__init__(parent)
        self.setWindowTitle("btgui - Structure Viewer")
        self.resize(1100, 760)
        self.files = {}
        self.active_key = None
        self._active_state = None
        self._file_color_index = 0
        self.open_handler = open_handler
        self.save_handler = save_handler
        self.updater = None
        self.updaters = []
        self.play_indices = []
        self.play_position = 0
        self.playback_loop = False
        package_settings = Path(__file__).resolve().parents[1] / "settings.json"
        executable_settings = Path(sys.argv[0]).resolve().with_name("settings.json")
        configured_settings = os.environ.get("BTGUI_CONFIG_FILE", "").strip()
        compiled_application = bool(getattr(sys, "frozen", False)) or "__compiled__" in globals()
        if configured_settings:
            self.settings_path = Path(configured_settings).expanduser().resolve()
        elif compiled_application and executable_settings.is_file():
            self.settings_path = executable_settings
        else:
            self.settings_path = package_settings
        self.settings_warning = None
        self.settings = self._load_settings()
        self.catalog = Catalog(self.settings["language"])
        self.recycler = FileRecycler()
        self.atom_clipboard = AtomClipboard()
        self._text_editors = {}
        self._python_runner = ExternalPythonRunner(self)
        self._python_runner.output.connect(self._log)
        self._python_runner.finished.connect(self._on_external_python_finished)
        self.language_actions = {}
        self.updater_warning = None
        self._load_configured_updaters()
        self._active_state = FileState(None, self.settings["history_limit"])
        self.files[None] = self._active_state
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_frame)

        self.toolbar_groups = {}
        for key, title in (
            ("edit", "Edit"),
            ("structure", "Structure"),
            ("measure", "Measure"),
            ("update", "Update"),
            ("frames", "Frames"),
        ):
            self._add_toolbar_group(key, title)
        self.toolbar = self.toolbar_groups["edit"]
        self.viewer = StructureViewer(self)
        self._configure_viewer(self.viewer)
        self.viewer.hide()

        self.undo_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack),
            "Undo",
            self,
        )
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.setToolTip("Undo edit")
        self.undo_action.triggered.connect(self._undo)
        self.toolbar_groups["edit"].addAction(self.undo_action)
        self.redo_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward),
            "Redo",
            self,
        )
        self.redo_action.setShortcut("Ctrl+Y")
        self.redo_action.setToolTip("Redo edit")
        self.redo_action.triggered.connect(self._redo)
        self.toolbar_groups["edit"].addAction(self.redo_action)
        # Section: Copy/Paste deliberately carry no QAction shortcut. A window
        # shortcut would hijack native Ctrl+C/Ctrl+V text editing in the command
        # console and dialogs; the focused viewer handles those keys instead.
        self.copy_action = QAction("Copy", self)
        self.copy_action.setToolTip("Copy the selected atoms (Ctrl+C)")
        self.copy_action.setEnabled(False)
        self.copy_action.triggered.connect(self._copy_selected_atoms)
        self.toolbar_groups["edit"].addAction(self.copy_action)
        self.paste_action = QAction("Paste", self)
        self.paste_action.setToolTip(
            "Append the copied atoms at their original coordinates (Ctrl+V)"
        )
        self.paste_action.setEnabled(False)
        self.paste_action.triggered.connect(self._paste_atoms)
        self.toolbar_groups["edit"].addAction(self.paste_action)

        self.supercell_action = QAction("Build / Transform...", self)
        self.supercell_action.triggered.connect(self._show_supercell)
        self.supercell_menu = QMenu(self)
        self.supercell_menu.addAction(self.supercell_action)
        self.supercell_button = QToolButton(self)
        self.supercell_button.setText("Supercell")
        self.supercell_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.supercell_button.setMenu(self.supercell_menu)
        self.toolbar_groups["structure"].addWidget(self.supercell_button)
        self.add_atom_action = QAction("Add Atom", self)
        self.add_atom_action.triggered.connect(self._show_add_atom)
        self.toolbar_groups["structure"].addAction(self.add_atom_action)
        self.replace_atom_action = QAction("Replace Atom", self)
        self.replace_atom_action.setEnabled(False)
        self.replace_atom_action.triggered.connect(self._show_replace_atom)
        self.toolbar_groups["structure"].addAction(self.replace_atom_action)
        self.clean_action = QAction("Clean", self)
        self.clean_action.setToolTip(
            "Relax the focused frame with the preset force field"
        )
        self.clean_action.triggered.connect(self._clean_current_frame)
        self.hydrogen_action = QAction("Add H", self)
        self.hydrogen_action.setToolTip(
            "Add hydrogens to the focused frame from the explicit bond orders"
        )
        self.hydrogen_action.triggered.connect(self._add_hydrogens)

        self.bond_menu = QMenu(self)
        self.add_bond_action = self.bond_menu.addAction("Add bond")
        self.add_bond_action.triggered.connect(self._begin_bond_addition)
        bond_frames_widget = QWidget(self.bond_menu)
        bond_frames_layout = QHBoxLayout(bond_frames_widget)
        bond_frames_layout.setContentsMargins(8, 4, 8, 4)
        self.bond_frames_label = QLabel("Frames:", bond_frames_widget)
        bond_frames_layout.addWidget(self.bond_frames_label)
        self.bond_frames_edit = QLineEdit(bond_frames_widget)
        self.bond_frames_edit.setPlaceholderText("current or all")
        self.bond_frames_edit.setMinimumWidth(160)
        bond_frames_layout.addWidget(self.bond_frames_edit)
        bond_frames_action = QWidgetAction(self.bond_menu)
        bond_frames_action.setDefaultWidget(bond_frames_widget)
        self.bond_menu.addAction(bond_frames_action)
        self.check_bonds_action = self.bond_menu.addAction("Check bonds")
        self.check_bonds_action.triggered.connect(self._check_bonds)
        self.bond_action = self.bond_menu.addAction("Show bonds")
        self.bond_action.setCheckable(True)
        self.bond_action.toggled.connect(self.viewer_set_bonds)
        self.length_measure_action = QAction("Length", self)
        self.length_measure_action.setCheckable(True)
        self.length_measure_action.toggled.connect(
            lambda checked: self._set_measurement_mode("length" if checked else None)
        )
        self.angle_measure_action = QAction("Angle", self)
        self.angle_measure_action.setCheckable(True)
        self.angle_measure_action.toggled.connect(
            lambda checked: self._set_measurement_mode("angle" if checked else None)
        )
        self.tolerance_spin = QDoubleSpinBox(self)
        self.tolerance_spin.setRange(0.1, 5.0)
        self.tolerance_spin.setSingleStep(0.05)
        self.tolerance_spin.setValue(1.2)
        self.tolerance_spin.setPrefix("Tolerance ")
        self.tolerance_spin.setSuffix("x")
        self.tolerance_spin.valueChanged.connect(self.viewer_set_tolerance)
        tolerance_action = QWidgetAction(self.bond_menu)
        tolerance_action.setDefaultWidget(self.tolerance_spin)
        self.bond_menu.addAction(tolerance_action)
        self.clear_bonds_action = self.bond_menu.addAction("Clear bonds")
        self.clear_bonds_action.triggered.connect(self._clear_bonds)
        self.bond_button = QToolButton(self)
        self.bond_button.setText("Bonds")
        self.bond_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.bond_button.setMenu(self.bond_menu)
        self.toolbar_groups["structure"].addWidget(self.bond_button)

        self.pbc_wrap_button = QToolButton(self)
        self.pbc_wrap_button.setText("PBC wrap")
        self.pbc_wrap_button.clicked.connect(self._wrap_pbc)
        self.toolbar_groups["structure"].addWidget(self.pbc_wrap_button)
        self.pbc_frames_edit = QLineEdit(self)
        self.pbc_frames_edit.setPlaceholderText("current or all")
        self.pbc_frames_edit.setMaximumWidth(130)
        self.toolbar_groups["structure"].addWidget(self.pbc_frames_edit)

        self.metrics_button = QToolButton(self)
        self.metrics_button.setText("Metrics")
        self.metrics_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.metrics_menu = QMenu(self.metrics_button)
        self.metrics_menu.addAction(self.length_measure_action)
        self.metrics_menu.addAction(self.angle_measure_action)
        self.metrics_menu.addSeparator()
        self.metric_style_action = self.metrics_menu.addAction("Style...")
        self.metric_style_action.triggered.connect(self._show_metric_style)
        self.metrics_button.setMenu(self.metrics_menu)
        self.toolbar_groups["measure"].addWidget(self.metrics_button)

        self.update_button = QToolButton(self)
        self.update_button.setText("Update")
        self.update_button.setToolTip("Choose or manage a structure updater")
        self.update_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.update_menu = QMenu(self.update_button)
        self.update_button.setMenu(self.update_menu)
        self._rebuild_update_menu()
        self.toolbar_groups["update"].addWidget(self.update_button)
        self.toolbar_groups["update"].addAction(self.clean_action)
        self.toolbar_groups["update"].addAction(self.hydrogen_action)
        self.previous_frame_button = QToolButton(self)
        self.previous_frame_button.setText("↑")
        self.previous_frame_button.setToolTip("Previous frame")
        self.previous_frame_button.clicked.connect(lambda: self._step_frame(-1))
        self.toolbar_groups["frames"].addWidget(self.previous_frame_button)
        self.frame_label = QLabel("Current Frame", self)
        self.toolbar_groups["frames"].addWidget(self.frame_label)
        self.frame_input = QLineEdit("0", self)
        self.frame_input.setMaximumWidth(58)
        self.frame_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_input.editingFinished.connect(self._on_frame_input)
        self.toolbar_groups["frames"].addWidget(self.frame_input)
        self.frame_total_label = QLabel("/ 0", self)
        self.toolbar_groups["frames"].addWidget(self.frame_total_label)
        self.next_frame_button = QToolButton(self)
        self.next_frame_button.setText("↓")
        self.next_frame_button.setToolTip("Next frame")
        self.next_frame_button.clicked.connect(lambda: self._step_frame(1))
        self.toolbar_groups["frames"].addWidget(self.next_frame_button)
        self.play_button = QToolButton(self)
        self.play_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.play_button.setToolTip("Play")
        self.play_button.clicked.connect(self._toggle_playback)
        self.toolbar_groups["frames"].addWidget(self.play_button)
        self._set_play_button_state(False)

        self.playback_menu = QMenu(self)
        fps_widget = QWidget(self.playback_menu)
        fps_layout = QHBoxLayout(fps_widget)
        fps_layout.setContentsMargins(8, 4, 8, 4)
        self.fps_caption = QLabel("fps:", fps_widget)
        fps_layout.addWidget(self.fps_caption)
        self.playback_fps_slider = QSlider(Qt.Orientation.Horizontal, fps_widget)
        self.playback_fps_slider.setRange(1, 60)
        self.playback_fps_slider.setValue(3)
        self.playback_fps_slider.setMinimumWidth(150)
        fps_layout.addWidget(self.playback_fps_slider)
        self.playback_fps_label = QLabel("3", fps_widget)
        self.playback_fps_slider.valueChanged.connect(
            lambda value: self.playback_fps_label.setText(str(value))
        )
        fps_layout.addWidget(self.playback_fps_label)
        fps_action = QWidgetAction(self.playback_menu)
        fps_action.setDefaultWidget(fps_widget)
        self.playback_menu.addAction(fps_action)

        range_widget = QWidget(self.playback_menu)
        range_layout = QHBoxLayout(range_widget)
        range_layout.setContentsMargins(8, 4, 8, 4)
        self.range_caption = QLabel("range:", range_widget)
        range_layout.addWidget(self.range_caption)
        self.playback_range_start = QLineEdit(range_widget)
        self.playback_range_end = QLineEdit(range_widget)
        for edit in (self.playback_range_start, self.playback_range_end):
            edit.setFixedWidth(54)
            edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        range_layout.addWidget(self.playback_range_start)
        range_layout.addWidget(QLabel("~", range_widget))
        range_layout.addWidget(self.playback_range_end)
        self.playback_range_hint = QLabel("0 ~ 0", range_widget)
        range_layout.addWidget(self.playback_range_hint)
        range_action = QWidgetAction(self.playback_menu)
        range_action.setDefaultWidget(range_widget)
        self.playback_menu.addAction(range_action)

        stride_widget = QWidget(self.playback_menu)
        stride_layout = QHBoxLayout(stride_widget)
        stride_layout.setContentsMargins(8, 4, 8, 4)
        self.stride_caption = QLabel("stride:", stride_widget)
        stride_layout.addWidget(self.stride_caption)
        self.playback_stride = QLineEdit("1", stride_widget)
        self.playback_stride.setFixedWidth(54)
        self.playback_stride.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stride_layout.addWidget(self.playback_stride)
        stride_layout.addStretch(1)
        self.playback_repeat_button = QPushButton("once", stride_widget)
        self.playback_repeat_button.clicked.connect(self._toggle_playback_repeat)
        stride_layout.addWidget(self.playback_repeat_button)
        stride_action = QWidgetAction(self.playback_menu)
        stride_action.setDefaultWidget(stride_widget)
        self.playback_menu.addAction(stride_action)

        self.playback_options_button = QToolButton(self)
        self.playback_options_button.setText("V")
        self.playback_options_button.setFixedWidth(42)
        self.playback_options_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.playback_options_button.setMenu(self.playback_menu)
        self.toolbar_groups["frames"].addWidget(self.playback_options_button)
        self.zoom_minus_button = QToolButton(self)
        self.zoom_minus_button.setText("-")
        self.zoom_minus_button.setToolTip("Decrease zoom")
        self.zoom_minus_button.clicked.connect(lambda: self._change_zoom(-10))
        self.zoom_input = QSpinBox(self)
        self.zoom_input.setRange(10, 1000)
        self.zoom_input.setSingleStep(10)
        self.zoom_input.setValue(100)
        self.zoom_input.setSuffix("%")
        self.zoom_input.setToolTip("Zoom percentage")
        self.zoom_input.valueChanged.connect(self._on_zoom_changed)
        self.zoom_plus_button = QToolButton(self)
        self.zoom_plus_button.setText("+")
        self.zoom_plus_button.setToolTip("Increase zoom")
        self.zoom_plus_button.clicked.connect(lambda: self._change_zoom(10))

        self.output = QPlainTextEdit(self)
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(1000)
        self.command = QLineEdit(self)
        self.command.setPlaceholderText(">>> Python or GUI command")
        self.command.returnPressed.connect(self._run_command)
        self.cli = CLI(
            log=self._log,
            on_open=self._open,
            on_save=self._save,
            on_frame=self._change_frame,
            current_index=lambda: self.current_index,
            on_play=self._play,
            on_pause=self._pause_playback,
            on_select=self._select_indices,
            on_clear=self._clear_selection,
            on_view_reset=self._reset_view,
            on_data_changed=self._on_cli_data_changed,
            on_update=self._run_default_updater,
            on_executed=self._refresh_after_cli_execution,
            on_python_file=self._run_external_python,
            external_python=self.settings.get("external_python_path"),
            buctoolkit_paths=self.settings["buctoolkit_paths"],
        )
        self._cli_file_key = object()
        self.viewer.set_error_handler(self._log)
        self.setAcceptDrops(True)
        if updater is not None:
            self.set_updater(updater)

        console = QWidget(self)
        console_layout = QVBoxLayout(console)
        console_layout.setContentsMargins(6, 4, 6, 6)
        console_layout.addWidget(self.output)
        console_layout.addWidget(self.command)
        splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.mdi_area = QMdiArea(self)
        self.mdi_area.setViewMode(QMdiArea.ViewMode.SubWindowView)
        self.mdi_area.setTabsClosable(False)
        self.window_manager = WindowManager(self, self.mdi_area)
        splitter.addWidget(self.mdi_area)
        splitter.addWidget(console)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([590, 150])
        self.setCentralWidget(splitter)
        self.frame_dock = QDockWidget("Frames", self)
        self.frame_dock.setObjectName("framesDock")
        self.frame_list = FrameListWidget(self.frame_dock)
        self.frame_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.frame_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.frame_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.frame_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._pending_frame_row = None
        self._frame_click_timer = QTimer(self)
        self._frame_click_timer.setSingleShot(True)
        self._frame_click_timer.timeout.connect(self._commit_frame_click)
        self.frame_list.itemClicked.connect(self._queue_frame_item_click)
        self.frame_list.itemDoubleClicked.connect(self._open_frame_item)
        self.frame_list.customContextMenuRequested.connect(self._show_frame_context_menu)
        self.frame_list.permutationChanged.connect(self._reorder_frames)
        self.frame_dock.setWidget(self.frame_list)
        self.file_dock = QDockWidget("Files", self)
        self.file_dock.setObjectName("filesDock")
        self.file_model = QFileSystemModel(self)
        self.file_model.setFilter(
            QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot
        )
        self.file_model.setReadOnly(False)
        self.file_sort_proxy = FileSortProxyModel(self)
        self.file_sort_proxy.setSourceModel(self.file_model)
        self.file_proxy = ParentRowProxyModel(self)
        self.file_proxy.setSourceModel(self.file_sort_proxy)
        self.file_tree = QTreeView(self.file_dock)
        self.file_tree.setModel(self.file_proxy)
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setSortingEnabled(True)
        self.file_tree.setRootIndex(QModelIndex())
        self.file_tree.setEditTriggers(
            QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.file_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_tree.doubleClicked.connect(self._file_tree_double_clicked)
        self.file_tree.customContextMenuRequested.connect(self._show_file_context_menu)
        self.file_model.fileRenamed.connect(self._on_file_renamed)
        file_nav = QToolBar(self.file_dock)
        file_nav.setMovable(False)
        file_nav.setFloatable(False)
        self.file_nav = file_nav
        self.file_up_button = QToolButton(self.file_dock)
        self.file_up_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp)
        )
        self.file_up_button.setToolTip("Parent directory")
        self.file_up_button.clicked.connect(self._open_file_manager_parent)
        self.file_back_button = QToolButton(self.file_dock)
        self.file_back_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack)
        )
        self.file_back_button.setToolTip("Back")
        self.file_back_button.clicked.connect(self._file_manager_go_back)
        self.file_forward_button = QToolButton(self.file_dock)
        self.file_forward_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward)
        )
        self.file_forward_button.setToolTip("Forward")
        self.file_forward_button.clicked.connect(self._file_manager_go_forward)
        self.file_recycle_button = QToolButton(self.file_dock)
        trash_icon = getattr(QStyle.StandardPixmap, "SP_TrashIcon", None)
        if trash_icon is not None:
            self.file_recycle_button.setIcon(self.style().standardIcon(trash_icon))
        else:
            self.file_recycle_button.setText("Recycle Bin")
        self.file_recycle_button.setToolTip("Recycle Bin")
        self.file_recycle_button.clicked.connect(self._open_file_manager_recycle_bin)
        for button in (
            self.file_up_button,
            self.file_back_button,
            self.file_forward_button,
            self.file_recycle_button,
        ):
            file_nav.addWidget(button)
        self.file_placeholder = QLabel(
            "Open a structure file to browse its directory.", self.file_dock
        )
        self.file_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        file_container = QWidget(self.file_dock)
        file_layout = QVBoxLayout(file_container)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.addWidget(file_nav)
        file_layout.addWidget(self.file_placeholder)
        file_layout.addWidget(self.file_tree, 1)
        self.file_dock.setWidget(file_container)
        self._file_manager_dir = None
        self._file_manager_path = None
        self._file_manager_file_key = None
        self._file_manager_manual = False
        self._file_manager_back = deque(maxlen=self.settings["history_limit"])
        self._file_manager_forward = deque(maxlen=self.settings["history_limit"])
        self._recycle_root = None
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.file_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.frame_dock)
        self.splitDockWidget(self.file_dock, self.frame_dock, Qt.Orientation.Vertical)
        self.file_dock.show()
        self.frame_dock.show()
        self._apply_file_manager_settings()
        self._sync_file_manager()
        self._create_menus()
        self._sync_frame_controls()
        self._sync_history_actions()
        self._log("Ready. Supply structure lists directly or connect file I/O handlers.")
        if self.settings_warning is not None:
            self._log(self.settings_warning)
        if self.updater_warning is not None:
            self._log(self.updater_warning)

    def _load_settings(self):
        """Load and validate persistent settings from the package directory.

        Args:
            None.

        Return:
            A complete settings dictionary; invalid files fall back to defaults.
        """
        settings = deepcopy(DEFAULT_SETTINGS)
        warning_messages = []
        self.settings_warning = None
        if not self.settings_path.exists():
            return settings
        try:
            loaded = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise TypeError("settings root must be a JSON object")
        except (json.JSONDecodeError, OSError, TypeError) as error:
            self.settings_warning = f"Settings warning: {error}; using defaults."
            return settings

        settings_directory = self.settings_path.parent
        try:
            history_limit = int(loaded.get("history_limit", settings["history_limit"]))
            if not 1 <= history_limit <= 1000:
                raise ValueError("history_limit must be between 1 and 1000")
            material = loaded.get("default_atom_material", settings["default_atom_material"])
            if material not in {"default", "matte", "glossy", "metallic"}:
                raise ValueError("default_atom_material is unsupported")
            lighting = loaded.get("atom_lighting", settings["atom_lighting"])
            edges = loaded.get("atom_edges", settings["atom_edges"])
            if not isinstance(lighting, bool) or not isinstance(edges, bool):
                raise TypeError("atom_lighting and atom_edges must be Boolean values")
            edge_color = str(loaded.get("atom_edge_color", settings["atom_edge_color"]))
            if not QColor(edge_color).isValid():
                raise ValueError("atom_edge_color must be a valid color")
            edge_width = float(loaded.get("atom_edge_width", settings["atom_edge_width"]))
            if not 0.1 <= edge_width <= 10.0:
                raise ValueError("atom_edge_width must be between 0.1 and 10.0")
            translation_step = float(
                loaded.get(
                    "keyboard_translation_step",
                    settings["keyboard_translation_step"],
                )
            )
            if not 0.001 <= translation_step <= 100.0:
                raise ValueError("keyboard_translation_step must be between 0.001 and 100")
            cache_limit_mb = int(
                loaded.get("file_cache_limit_mb", settings["file_cache_limit_mb"])
            )
            if not 64 <= cache_limit_mb <= 65536:
                raise ValueError("file_cache_limit_mb must be between 64 and 65536")
            file_manager_flags = {}
            for name in (
                "file_manager_show_size",
                "file_manager_show_type",
                "file_manager_show_modified",
                "file_manager_folders_first",
            ):
                value = loaded.get(name, settings[name])
                if not isinstance(value, bool):
                    raise TypeError(f"{name} must be a Boolean value")
                file_manager_flags[name] = value
            sort_column = int(
                loaded.get("file_manager_sort_column", settings["file_manager_sort_column"])
            )
            if sort_column not in {0, 1, 2, 3}:
                raise ValueError("file_manager_sort_column must be 0..3")
            sort_order = str(
                loaded.get("file_manager_sort_order", settings["file_manager_sort_order"])
            )
            if sort_order not in {"asc", "desc"}:
                raise ValueError("file_manager_sort_order must be asc or desc")
            clean_mode = str(loaded.get("clean_mode", settings["clean_mode"]))
            if clean_mode not in {"local", "global"}:
                raise ValueError("clean_mode must be local or global")
            clean_local_method = str(
                loaded.get("clean_local_method", settings["clean_local_method"])
            )
            if clean_local_method not in CLEAN_LOCAL_METHODS:
                raise ValueError("clean_local_method is unsupported")
            clean_global_method = str(
                loaded.get("clean_global_method", settings["clean_global_method"])
            )
            if clean_global_method not in CLEAN_GLOBAL_METHODS:
                raise ValueError("clean_global_method is unsupported")
            clean_local_maxiter = int(
                loaded.get("clean_local_maxiter", settings["clean_local_maxiter"])
            )
            clean_global_maxiter = int(
                loaded.get("clean_global_maxiter", settings["clean_global_maxiter"])
            )
            if not 1 <= clean_local_maxiter <= 100000 or not 1 <= clean_global_maxiter <= 100000:
                raise ValueError("clean iteration limits must be between 1 and 100000")
            clean_local_gtol = float(
                loaded.get("clean_local_gtol", settings["clean_local_gtol"])
            )
            if not 1e-10 <= clean_local_gtol <= 1.0:
                raise ValueError("clean_local_gtol must be between 1e-10 and 1")
            clean_workers = int(loaded.get("clean_workers", settings["clean_workers"]))
            if not 1 <= clean_workers <= max(1, (os.cpu_count() or 1)):
                raise ValueError("clean_workers is out of range")
            clean_bounds_span = float(
                loaded.get("clean_bounds_span", settings["clean_bounds_span"])
            )
            if not 0.1 <= clean_bounds_span <= 30.0:
                raise ValueError("clean_bounds_span must be between 0.1 and 30")
            language = str(loaded.get("language", settings["language"]))
            if language not in language_codes():
                language = "en"
            external_python_path = loaded.get(
                "external_python_path", settings["external_python_path"]
            )
            if external_python_path is not None and not isinstance(
                external_python_path, str
            ):
                raise TypeError("external_python_path must be a string or null")
            buctoolkit_paths = loaded.get("buctoolkit_paths", settings["buctoolkit_paths"])
            if isinstance(buctoolkit_paths, str):
                buctoolkit_paths = buctoolkit_paths.split(os.pathsep)
            if not isinstance(buctoolkit_paths, list) or any(
                not isinstance(path, str) for path in buctoolkit_paths
            ):
                raise TypeError("buctoolkit_paths must be a string or a list of strings")
            normalized_paths = []
            for path in buctoolkit_paths:
                path = path.strip()
                if not path:
                    continue
                candidate = Path(path).expanduser()
                if not candidate.is_absolute():
                    candidate = settings_directory / candidate
                normalized_paths.append(str(candidate))
            buctoolkit_paths = normalized_paths
            settings.update(
                history_limit=history_limit,
                default_atom_material=material,
                atom_lighting=lighting,
                atom_edges=edges,
                atom_edge_color=edge_color,
                atom_edge_width=edge_width,
                buctoolkit_paths=buctoolkit_paths,
                keyboard_translation_step=translation_step,
                file_cache_limit_mb=cache_limit_mb,
                file_manager_sort_column=sort_column,
                file_manager_sort_order=sort_order,
                clean_mode=clean_mode,
                clean_local_method=clean_local_method,
                clean_global_method=clean_global_method,
                clean_local_maxiter=clean_local_maxiter,
                clean_global_maxiter=clean_global_maxiter,
                clean_local_gtol=clean_local_gtol,
                clean_workers=clean_workers,
                clean_bounds_span=clean_bounds_span,
                language=language,
                external_python_path=external_python_path,
                **file_manager_flags,
            )
        except (TypeError, ValueError) as error:
            warning_messages.append(f"base settings: {error}; using defaults")

        try:
            raw_paths = loaded.get("updater_path", [])
            raw_functions = loaded.get("updater_function", [])
            if isinstance(raw_paths, str):
                path_values = [raw_paths] if raw_paths.strip() else []
            elif isinstance(raw_paths, list) and all(
                isinstance(path, str) for path in raw_paths
            ):
                path_values = raw_paths
            else:
                raise TypeError("updater_path must be a string or a list of strings")
            if isinstance(raw_functions, str):
                function_values = [raw_functions] if raw_functions.strip() else []
            elif isinstance(raw_functions, list) and all(
                isinstance(name, str) for name in raw_functions
            ):
                function_values = raw_functions
            else:
                raise TypeError("updater_function must be a string or a list of strings")
            # Section: Module paths and requested names are independent lists.
            updater_paths = []
            for path in path_values:
                path = path.strip()
                if not path:
                    continue
                candidate = Path(path).expanduser()
                if not candidate.is_absolute():
                    candidate = settings_directory / candidate
                updater_paths.append(str(candidate.resolve()))
            updater_functions = []
            for function_name in function_values:
                function_name = function_name.strip()
                if not function_name:
                    continue
                if not function_name.isidentifier():
                    raise ValueError(
                        "each updater_function must be a valid Python function name"
                    )
                updater_functions.append(function_name)
            settings.update(
                updater_path=updater_paths,
                updater_function=updater_functions,
            )
        except (TypeError, ValueError) as error:
            warning_messages.append(f"updater settings: {error}; no updater loaded")
        if warning_messages:
            self.settings_warning = "Settings warning: " + "; ".join(warning_messages) + "."
        return settings

    def _load_configured_updaters(self):
        """Load configured modules, then find requested functions across them.

        Args:
            None.

        Return:
            None. Valid plugins are retained while individual failures are reported.
        """
        embedded = [
            updater for updater in self.updaters
            if not updater.get("configured", False) and not updater.get("preset", False)
        ]
        configured = []
        load_errors = []
        modules = []
        self.updater_module_paths = []
        # Section: Execute each configured module once before looking up names.
        for path in dict.fromkeys(self.settings.get("updater_path", [])):
            try:
                modules.append(
                    load_updater_module(path, self.settings_path.parent)
                )
                self.updater_module_paths.append(str(Path(modules[-1].__file__).resolve()))
            except Exception as error:
                load_errors.append(f"{path}: {error}")

        # Section: Group functions by their defining module, including reexports.
        for function_name in dict.fromkeys(self.settings.get("updater_function", [])):
            seen_callbacks = set()
            for module in modules:
                callback = getattr(module, function_name, None)
                if not callable(callback) or id(callback) in seen_callbacks:
                    continue
                seen_callbacks.add(id(callback))
                owner = sys.modules.get(getattr(callback, "__module__", None))
                source_path = getattr(owner, "__file__", None) or module.__file__
                configured.append(
                    {
                        "path": str(Path(source_path).resolve()),
                        "name": function_name,
                        "callback": callback,
                        "configured": True,
                    }
                )
            if not seen_callbacks:
                load_errors.append(
                    f"{function_name}: no callable found in loaded updater modules"
                )
        self.updaters = embedded + configured + _preset_updater_records()
        self.updater = self.updaters[0]["callback"] if self.updaters else None
        self.updater_warning = (
            "Updater load warning: " + "; ".join(load_errors)
            if load_errors
            else None
        )
        if hasattr(self, "update_menu"):
            self._rebuild_update_menu()

    def _configure_viewer(self, viewer):
        """Apply shared viewer settings and connect the standard callbacks."""
        viewer.atom_material = self.settings["default_atom_material"]
        viewer._set_material_preset(viewer.atom_material)
        viewer.atom_lighting = self.settings["atom_lighting"]
        viewer.atom_edges = self.settings["atom_edges"]
        viewer.atom_edge_color = self.settings["atom_edge_color"]
        viewer.atom_edge_width = self.settings["atom_edge_width"]
        viewer.keyboard_translation_step = self.settings["keyboard_translation_step"]
        if hasattr(self, "perspective_action"):
            viewer.perspective_enabled = self.perspective_action.isChecked()
            viewer.camera.SetParallelProjection(not viewer.perspective_enabled)
        viewer.coordinatesEdited.connect(self._record_coordinate_edit)
        viewer.deleteRequested.connect(self._delete_selected_atoms)
        viewer.atomsAdded.connect(self._commit_reference_atoms)
        viewer.bondAdded.connect(self._add_bond)
        viewer.bondsDeleted.connect(self._delete_selected_bonds)
        viewer.filesDropped.connect(self._open_dropped_files)
        viewer.measurementModeChanged.connect(self._sync_measurement_actions)
        viewer.selectionChanged.connect(self._on_selection_changed)
        viewer.frameStepRequested.connect(self._step_frame)
        viewer.set_error_handler(self._log)
        self._connect_clipboard_signals(viewer)

    def _connect_clipboard_signals(self, viewer):
        """Route one viewer's Copy, Paste, and context-menu requests.

        Args:
            viewer: Viewer that emits the atom clipboard signals.

        Returns:
            None. Every handler activates the emitting viewer first so the
            operation always targets the selection the user is looking at.
        """
        viewer.copyRequested.connect(
            lambda view=viewer: (self._activate_viewer(view), self._copy_selected_atoms())
        )
        viewer.pasteRequested.connect(
            lambda view=viewer: (self._activate_viewer(view), self._paste_atoms())
        )
        viewer.contextMenuRequested.connect(
            lambda position, view=viewer: self._show_viewer_context_menu(view, position)
        )

    def _activate_viewer(self, viewer):
        """Make one displayed viewer the target of toolbar and edit actions."""
        old_key = self.active_key
        self.viewer = viewer
        record = next(
            (item for item in self.window_manager.records if item["viewer"] is viewer),
            None,
        )
        if record is not None:
            if record["file_key"] != old_key and self.timer.isActive():
                self._pause_playback()
            state = self.files.get(record["file_key"])
            if state is not None and not state.closed:
                self.active_key = record["file_key"]
                self._active_state = state
                self.current_index = min(
                    max(int(record["frame_index"]), 0), max(0, len(state.coordinates) - 1)
                )
                state.current_index = self.current_index
        if hasattr(self, "perspective_action"):
            blocker = QSignalBlocker(self.perspective_action)
            self.perspective_action.setChecked(viewer.perspective_enabled)
            del blocker
        blocker = QSignalBlocker(self.zoom_input)
        self.zoom_input.setValue(int(viewer.zoom_percent))
        del blocker
        self._sync_measurement_actions(viewer.measurement_mode)
        self._sync_frame_controls()
        self._sync_frame_list()
        self._sync_history_actions()
        self._sync_file_manager()
        self._sync_cli_scope()
        self._sync_bond_action()

    def _sync_cli_scope(self, force=False):
        """Reset CLI variables and output when the focused file changes."""
        if not hasattr(self, "cli"):
            return
        key = self.active_key
        if not force and key == self._cli_file_key:
            return
        self._cli_file_key = key
        self.cli.reset_scope(self.elements, self.coordinates, self.lattices)
        self.output.clear()
        self.command.clear()
        name = key.name if key is not None else "Untitled"
        self._log(f"CLI scope: {name} ({len(self.elements)} frame(s)).")

    def _activate_empty_state(self):
        """Replace the active state with an empty untitled state."""
        if self.timer.isActive():
            self._pause_playback()
        old = self.files.get(None)
        empty = FileState(None, self.settings["history_limit"])
        self.files[None] = empty
        self.active_key = None
        self._active_state = empty
        if old is not None and old is not empty and old.window_refs == 0:
            old.close()
        self._sync_frame_list()
        self._sync_frame_controls()
        self._sync_history_actions()
        self._sync_file_manager()
        self._sync_cli_scope()
        self._sync_bond_action()

    def _ensure_data_window(self):
        """Create the first structure window when data exists but none is open."""
        if self.window_manager.records or not self.coordinates:
            return
        if self.viewer is None:
            self.viewer = StructureViewer(self)
            self._configure_viewer(self.viewer)
        frame_index = min(max(self.current_index, 0), len(self.coordinates) - 1)
        record = self.window_manager.add(
            self.viewer,
            frame_index,
            self._file_title(self._active_state, frame_index),
            show=True,
            file_key=self.active_key,
        )
        self._active_state.retain()
        QApplication.processEvents()
        self.viewer.set_data(self.elements, self.coordinates, self.lattices)
        self.viewer.set_frame_state(
            self.atom_frame_styles, self.frame_metrics, self.bond_adjacencies,
            self.bond_orders,
        )
        self.viewer.current_index = frame_index
        self.viewer.set_frame(frame_index)
        self.window_manager.refresh_all()

    def _connect_extra_viewer(self, viewer):
        """Connect shared edit and drop signals for an additional viewer."""
        viewer.coordinatesEdited.connect(
            lambda frame, before, after, view=viewer: (
                self._activate_viewer(view),
                self._record_coordinate_edit(frame, before, after),
            )
        )
        viewer.deleteRequested.connect(
            lambda frame, indices, view=viewer: (
                self._activate_viewer(view),
                self._delete_selected_atoms(frame, indices),
            )
        )
        viewer.atomsAdded.connect(
            lambda frame, symbols, positions, reference, view=viewer: (
                self._activate_viewer(view),
                self._commit_reference_atoms(frame, symbols, positions, reference),
            )
        )
        viewer.bondAdded.connect(
            lambda frame, left, right, view=viewer: (
                self._activate_viewer(view), self._add_bond(frame, left, right),
            )
        )
        viewer.bondsDeleted.connect(
            lambda frame, pairs, view=viewer: (
                self._activate_viewer(view), self._delete_selected_bonds(frame, pairs),
            )
        )
        viewer.measurementModeChanged.connect(self._sync_measurement_actions)
        viewer.filesDropped.connect(self._open_dropped_files)
        viewer.set_error_handler(self._log)
        viewer.selectionChanged.connect(
            lambda indices, view=viewer: (
                self._activate_viewer(view),
                self._on_selection_changed(indices),
            )
        )
        viewer.frameStepRequested.connect(
            lambda step, view=viewer: (
                self._activate_viewer(view),
                self._step_frame(step),
            )
        )
        self._connect_clipboard_signals(viewer)

    def _persist_settings(self):
        """Write current settings to ``settings.json`` beside the package.

        Args:
            None.

        Return:
            None. Write failures are reported in the command console.
        """
        try:
            self.settings_path.write_text(
                json.dumps(self.settings, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            self._log(f"Settings save error: {error}")

    def _show_settings(self):
        """Open settings and apply accepted history and rendering defaults.

        Args:
            None.

        Return:
            None. Accepted values are persisted and applied to the current view.
        """
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.settings.update(dialog.values())
        self.cli.set_external_python(self.settings.get("external_python_path"))
        self.undo_stack = deque(self.undo_stack, maxlen=self.settings["history_limit"])
        self._file_manager_back = deque(
            list(self._file_manager_back)[-self.settings["history_limit"] :],
            maxlen=self.settings["history_limit"],
        )
        self._file_manager_forward = deque(
            list(self._file_manager_forward)[-self.settings["history_limit"] :],
            maxlen=self.settings["history_limit"],
        )
        self._update_file_manager_buttons()
        if self.viewer is not None:
            self.viewer.set_appearance(
                self.viewer.atom_styles,
                self.settings["default_atom_material"],
                self.settings["atom_lighting"],
                self.settings["atom_edges"],
                self.settings["atom_edge_color"],
                self.settings["atom_edge_width"],
            )
        viewers = {record["viewer"] for record in self.window_manager.records}
        if self.viewer is not None:
            viewers.add(self.viewer)
        for viewer in viewers:
            viewer.keyboard_translation_step = self.settings[
                "keyboard_translation_step"
            ]
        self._sync_history_actions()
        self._trim_file_cache()
        self._persist_settings()

    def _sync_history_actions(self):
        """Enable history actions only when their corresponding stack has data.

        Args:
            None.

        Return:
            None. Undo and redo action states are updated.
        """
        self.undo_action.setEnabled(len(self.undo_stack) > 0)
        self.redo_action.setEnabled(len(self.redo_stack) > 0)

    def _initialize_supercell_state(self):
        """Create identity supercell state for every retained frame.

        Args:
            None.

        Returns:
            None. Each current frame becomes its own transform base.
        """
        self.supercell_bases = {
            frame: {
                "elements": np.array(self.elements[frame], copy=True),
                "coordinates": np.array(self.coordinates[frame], copy=True),
                "lattice": np.array(self.lattices[frame], copy=True),
            }
            for frame in range(len(self.coordinates))
        }
        self.supercell_transforms = {
            frame: (np.ones(3, dtype=np.int64), np.eye(3, dtype=np.int64))
            for frame in range(len(self.coordinates))
        }

    def _initialize_frame_state(self):
        """Create independent atom-style and metric containers for every frame.

        Args:
            None.

        Returns:
            None. Mutable defaults are never shared across atoms or frames.
        """
        self.atom_frame_styles = [
            [dict() for _ in frame_elements] for frame_elements in self.elements
        ]
        self.frame_metrics = [
            {"length": {}, "angle": {}} for _ in self.coordinates
        ]
        self.bond_adjacencies = [None for _ in self.coordinates]
        self.bond_orders = [None for _ in self.coordinates]

    def _ensure_frame_names(self):
        """Keep fixed frame names aligned with the structure lists.

        Args:
            None.

        Returns:
            None. Existing names are retained and only new frames receive names.
        """
        del self.frame_names[len(self.elements):]
        while len(self.frame_names) < len(self.elements):
            self._frame_name_serial += 1
            self.frame_names.append(f"Frame {self._frame_name_serial}")

    def _ensure_frame_state(self):
        """Align auxiliary frame and atom containers with structure lists.

        Args:
            None.

        Returns:
            None. Existing compatible style and metric entries are preserved.
        """
        while len(self.atom_frame_styles) < len(self.elements):
            frame = len(self.atom_frame_styles)
            self.atom_frame_styles.append([dict() for _ in self.elements[frame]])
        while len(self.frame_metrics) < len(self.elements):
            self.frame_metrics.append({"length": {}, "angle": {}})
        while len(self.bond_adjacencies) < len(self.elements):
            self.bond_adjacencies.append(None)
        while len(self.bond_orders) < len(self.elements):
            self.bond_orders.append(None)
        del self.atom_frame_styles[len(self.elements):]
        del self.frame_metrics[len(self.elements):]
        del self.bond_adjacencies[len(self.elements):]
        del self.bond_orders[len(self.elements):]
        for frame, graph in enumerate(self.bond_adjacencies):
            orders = self.bond_orders[frame]
            if graph is None:
                self.bond_orders[frame] = None
            elif orders is None or len(orders) != len(graph[0]):
                self.bond_orders[frame] = np.ones(len(graph[0]), dtype=np.int8)
        for frame, frame_elements in enumerate(self.elements):
            styles = self.atom_frame_styles[frame]
            if len(styles) < len(frame_elements):
                styles.extend(dict() for _ in range(len(frame_elements) - len(styles)))
            elif len(styles) > len(frame_elements):
                del styles[len(frame_elements):]

    def _ensure_supercell_state(self):
        """Add identity state for frames introduced by file or CLI operations.

        Args:
            None.

        Returns:
            None. Existing frame state is preserved.
        """
        for frame in range(len(self.coordinates)):
            if frame not in self.supercell_bases:
                self.supercell_bases[frame] = {
                    "elements": np.array(self.elements[frame], copy=True),
                    "coordinates": np.array(self.coordinates[frame], copy=True),
                    "lattice": np.array(self.lattices[frame], copy=True),
                }
                self.supercell_transforms[frame] = (
                    np.ones(3, dtype=np.int64),
                    np.eye(3, dtype=np.int64),
                )
        for frame in list(self.supercell_bases):
            if frame >= len(self.coordinates):
                self.supercell_bases.pop(frame, None)
                self.supercell_transforms.pop(frame, None)

    def _supercell_context(self, frame_index):
        """Copy one frame's supercell base and transform state.

        Args:
            frame_index: Frame whose state is copied.

        Returns:
            A detached context dictionary suitable for a history entry.
        """
        self._ensure_supercell_state()
        base = self.supercell_bases[frame_index]
        multipliers, matrix = self.supercell_transforms[frame_index]
        return {
            "base": {
                "elements": np.array(base["elements"], copy=True),
                "coordinates": np.array(base["coordinates"], copy=True),
                "lattice": np.array(base["lattice"], copy=True),
            },
            "transform": (
                np.array(multipliers, copy=True),
                np.array(matrix, copy=True),
            ),
            "atom_styles": deepcopy(self.atom_frame_styles[frame_index]),
            "metrics": deepcopy(self.frame_metrics[frame_index]),
            "appearance": self._appearance_context(),
        }

    def _restore_supercell_context(self, frame_index, context):
        """Restore one frame's remembered supercell state.

        Args:
            frame_index: Frame receiving the state.
            context: Context returned by `_supercell_context`, or ``None``.

        Returns:
            None. Missing context leaves the current state unchanged.
        """
        if context is None:
            return
        self.supercell_bases[frame_index] = {
            "elements": np.array(context["base"]["elements"], copy=True),
            "coordinates": np.array(context["base"]["coordinates"], copy=True),
            "lattice": np.array(context["base"]["lattice"], copy=True),
        }
        multipliers, matrix = context["transform"]
        self.supercell_transforms[frame_index] = (
            np.array(multipliers, copy=True),
            np.array(matrix, copy=True),
        )
        if "atom_styles" in context:
            self.atom_frame_styles[frame_index] = deepcopy(context["atom_styles"])
        if "metrics" in context:
            self.frame_metrics[frame_index] = deepcopy(context["metrics"])
        if "appearance" in context:
            self._restore_appearance_context(context["appearance"])

    def _all_supercell_context(self):
        """Copy remembered supercell state for every retained frame.

        Args:
            None.

        Returns:
            Detached dictionaries suitable for a batch history entry.
        """
        self._ensure_supercell_state()
        return {
            "bases": {
                frame: {
                    "elements": np.array(value["elements"], copy=True),
                    "coordinates": np.array(value["coordinates"], copy=True),
                    "lattice": np.array(value["lattice"], copy=True),
                }
                for frame, value in self.supercell_bases.items()
            },
            "transforms": {
                frame: (
                    np.array(value[0], copy=True),
                    np.array(value[1], copy=True),
                )
                for frame, value in self.supercell_transforms.items()
            },
            "atom_styles": deepcopy(self.atom_frame_styles),
            "metrics": deepcopy(self.frame_metrics),
            "appearance": self._appearance_context(),
            "current_index": int(self.current_index),
            "frame_names": list(self.frame_names),
            "frame_name_serial": int(self._frame_name_serial),
            "bond_frames": list(self.bond_frame_indices),
        }

    def _restore_all_supercell_context(self, context):
        """Restore complete remembered supercell state after a batch change.

        Args:
            context: Dictionary produced by `_all_supercell_context`.

        Returns:
            None. State is detached from the history entry.
        """
        self.supercell_bases = {
            frame: {
                "elements": np.array(value["elements"], copy=True),
                "coordinates": np.array(value["coordinates"], copy=True),
                "lattice": np.array(value["lattice"], copy=True),
            }
            for frame, value in context["bases"].items()
        }
        self.supercell_transforms = {
            frame: (np.array(value[0], copy=True), np.array(value[1], copy=True))
            for frame, value in context["transforms"].items()
        }
        if "atom_styles" in context:
            self.atom_frame_styles = deepcopy(context["atom_styles"])
        if "metrics" in context:
            self.frame_metrics = deepcopy(context["metrics"])
        if "appearance" in context:
            self._restore_appearance_context(context["appearance"])
        if "current_index" in context:
            self.current_index = int(context["current_index"])
        if "frame_names" in context:
            self.frame_names = list(context["frame_names"])
            self._frame_name_serial = int(
                context.get("frame_name_serial", len(self.frame_names))
            )
        if "bond_frames" in context:
            self.bond_frame_indices = list(context["bond_frames"])

    def _appearance_context(self):
        """Copy renderer-wide appearance values for unified history.

        Args:
            None.

        Returns:
            Detached values needed to restore an appearance edit.
        """
        names = (
            "atom_styles", "atom_material", "atom_lighting", "atom_edges",
            "atom_edge_color", "atom_edge_width", "default_atom_style",
            "default_custom_radius", "default_atom_scale", "_background_color_value", "material_ambient",
            "material_diffuse", "material_specular", "material_roughness",
            "material_metallic", "axes_visible", "grid_visible", "bond_width",
            "bond_alpha", "distinguish_files",
        )
        return {name: deepcopy(getattr(self.viewer, name)) for name in names}

    def _restore_appearance_context(self, context):
        """Restore renderer-wide appearance values without drawing.

        Args:
            context: Mapping produced by :meth:`_appearance_context`.

        Returns:
            None. The caller performs the consolidated redraw.
        """
        for name, value in context.items():
            setattr(self.viewer, name, deepcopy(value))
        self.viewer.set_background(self.viewer._background_color_value)
        self.window_manager.apply_file_colors()

    def _refresh_after_change(
        self,
        frame_index=None,
        reset_camera=False,
        changed_frame_indices=None,
    ):
        """Refresh viewer references and selection after a committed change.

        Args:
            frame_index: Preferred frame to display, or ``None`` for current.
            reset_camera: Whether the scene should fit a new camera.
            changed_frame_indices: Frames whose cached bond geometry is stale,
                or ``None`` when the complete frame collection changed.

        Returns:
            None. The viewer is redrawn with the current raw lists.
        """
        if self.viewer is not None:
            self.viewer.invalidate_bond_meshes(changed_frame_indices)
        self._ensure_supercell_state()
        self._ensure_frame_state()
        self._ensure_frame_names()
        if self.coordinates:
            target = self.current_index if frame_index is None else int(frame_index)
            self.current_index = min(max(target, 0), len(self.coordinates) - 1)
        else:
            self.current_index = 0
        self._ensure_data_window()
        active = self.window_manager.active_record()
        if active is not None:
            active["frame_index"] = self.current_index
        self.window_manager.refresh_all(reset_camera=reset_camera)
        self._sync_frame_controls()
        self._sync_frame_list()
        self._sync_atom_actions(
            self.viewer.selected_indices if self.viewer is not None else np.empty(0, dtype=int)
        )
        self.cli.bind_data(self.elements, self.coordinates, self.lattices)

    def _apply_history_entry(self, change, use_after):
        """Apply one event or an ordered tuple without intermediate redraws.

        Args:
            change: History event or tuple in forward execution order.
            use_after: True for execution/Redo, False for Undo.

        Return:
            None. Tuple Undo reverses dependency order before applying inverses.
        """
        # Section: A tuple is ONE deque entry. For (atoms, bonds), Redo creates
        # nodes first; Undo removes their bonds first. Deletion uses (bonds, atoms).
        entries = change if isinstance(change, tuple) else (change,)
        for entry in entries if use_after else reversed(entries):
            if isinstance(entry, StructureChange):
                # Section: An old batch snapshot owns only its recorded prefix.
                # Preserve later non-history updater appends and their aligned
                # metadata by reference; no appended arrays enter the queues.
                suffix = None
                if entry.frame_indices is None:
                    extent = len(entry.before_elements if use_after else entry.after_elements)
                    if len(self.coordinates) > extent:
                        suffix = {
                            name: getattr(self, name)[extent:]
                            for name in ("atom_frame_styles", "frame_metrics", "bond_adjacencies", "bond_orders", "frame_names")
                        }
                        suffix["bases"] = {frame - extent: value for frame, value in self.supercell_bases.items() if frame >= extent}
                        suffix["transforms"] = {frame - extent: value for frame, value in self.supercell_transforms.items() if frame >= extent}
                        suffix["bond_frames"] = [frame - extent for frame in self.bond_frame_indices if frame >= extent]
                        suffix["current"] = self.current_index - extent if self.current_index >= extent else None
                        suffix["serial"] = self._frame_name_serial
                        for name in ("atom_frame_styles", "frame_metrics", "bond_adjacencies", "bond_orders", "frame_names"):
                            del getattr(self, name)[extent:]
                context = entry.apply(
                    self.elements, self.coordinates, self.lattices, use_after
                )
                if entry.frame_indices is not None and context is not None:
                    self._ensure_frame_state()
                    self._restore_supercell_context(entry.frame_indices[0], context)
                elif entry.frame_indices is None and context is not None:
                    self._restore_all_supercell_context(context)
                if suffix is not None:
                    prefix_count = len(entry.after_elements if use_after else entry.before_elements)
                    # Pad only restored-prefix metadata before attaching the
                    # suffix, avoiding per-atom allocation for appended frames.
                    for frame in range(len(self.atom_frame_styles), prefix_count):
                        self.atom_frame_styles.append([dict() for _ in self.elements[frame]])
                    while len(self.frame_metrics) < prefix_count:
                        self.frame_metrics.append({"length": {}, "angle": {}})
                    while len(self.bond_adjacencies) < prefix_count:
                        self.bond_adjacencies.append(None)
                    while len(self.bond_orders) < prefix_count:
                        self.bond_orders.append(None)
                    while len(self.frame_names) < prefix_count:
                        self._frame_name_serial += 1
                        self.frame_names.append(f"Frame {self._frame_name_serial}")
                    for name in ("atom_frame_styles", "frame_metrics", "bond_adjacencies", "bond_orders", "frame_names"):
                        getattr(self, name)[prefix_count:] = suffix[name]
                    for target, saved in ((self.supercell_bases, suffix["bases"]), (self.supercell_transforms, suffix["transforms"])):
                        for frame in list(target):
                            if frame >= prefix_count:
                                del target[frame]
                        target.update({prefix_count + offset: value for offset, value in saved.items()})
                    self.bond_frame_indices = [frame for frame in self.bond_frame_indices if frame < prefix_count]
                    self.bond_frame_indices.extend(prefix_count + offset for offset in suffix["bond_frames"])
                    self._frame_name_serial = max(self._frame_name_serial, suffix["serial"])
                    if suffix["current"] is not None:
                        self.current_index = prefix_count + suffix["current"]
                    for record in self.window_manager.records:
                        if record["frame_index"] >= extent:
                            record["frame_index"] += prefix_count - extent
                self._ensure_frame_state()
            else:
                entry.apply(self, use_after)

    def _refresh_history_entry(self, change, frame_index=None):
        """Refresh once after a complete single or compound history operation.

        Args:
            change: Applied history event or ordered tuple.
            frame_index: Optional preferred frame to display.

        Return:
            None. Graph-only changes avoid rebuilding the atom scene.
        """
        entries = change if isinstance(change, tuple) else (change,)
        frames = set()
        all_frames = False
        for entry in entries:
            indices = getattr(entry, "frame_indices", None)
            if indices is None:
                all_frames = True
            else:
                frames.update(indices)
        changed_frames = None if all_frames else tuple(sorted(frames))
        if all(isinstance(entry, (BondChange, ClearBondsChange)) for entry in entries):
            for record in self.window_manager.records_for_file(self.active_key):
                viewer = record["viewer"]
                viewer.bond_adjacencies = self.bond_adjacencies
                viewer.bond_orders = self.bond_orders
                viewer.bond_frame_indices = set(self.bond_frame_indices)
                viewer.selected_bonds.clear()
                viewer.invalidate_bond_meshes(changed_frames)
                viewer.refresh_bonds()
        elif isinstance(change, FramePermutationChange):
            self.viewer.invalidate_bond_meshes()
            self._refresh_after_permutation(sync_frame_list=False)
        elif isinstance(change, AppearanceChange):
            self._refresh_after_appearance(change)
        else:
            # Coordinate/style edits retain selection; only changed node counts
            # or a compound graph edit invalidate the selected endpoint space.
            topology_changed = any(
                (isinstance(entry, StructureChange) and (
                    entry.frame_indices is None
                    or any(len(before) != len(after) for before, after in zip(
                        entry.before_elements, entry.after_elements
                    ))
                )) or (isinstance(entry, InplaceUpdaterChange) and bool(entry.topology_frames))
                for entry in entries
            )
            for record in self.window_manager.records_for_file(self.active_key):
                # In-place multi-frame edits only invalidate selections in views
                # whose atom-count/index space actually changed.
                frame_topology_changed = topology_changed
                if isinstance(change, InplaceUpdaterChange):
                    frame_topology_changed = record["frame_index"] in change.topology_frames
                if frame_topology_changed:
                    record["viewer"].selected_indices = np.empty(0, dtype=int)
                if frame_topology_changed or any(isinstance(entry, BondChange) for entry in entries):
                    record["viewer"].selected_bonds.clear()
            if frame_index is None and changed_frames and len(changed_frames) == 1:
                frame_index = changed_frames[0]
            self._refresh_after_change(
                frame_index=frame_index, changed_frame_indices=changed_frames
            )

    def _record_change(self, change, already_applied=False, frame_index=None):
        """Record one unified change and optionally apply its after snapshot.

        Args:
            change: History event or tuple retained as one undo step.
            already_applied: Whether the caller already wrote the after state.
            frame_index: Frame to display after applying the change.

        Returns:
            None. Redo history is discarded after a new operation.
        """
        if not already_applied:
            self._apply_history_entry(change, use_after=True)
        self.undo_stack.append(change)
        self.redo_stack.clear()
        self._refresh_history_entry(change, frame_index)
        self._sync_history_actions()

    def _refresh_after_appearance(self, change):
        """Refresh only renderer state affected by an appearance change.

        Args:
            change: Applied appearance history entry.

        Returns:
            None. Structure arrays and unrelated actors remain untouched.
        """
        changed_global = set(change.before_global) | set(change.after_global)
        atom_fields = {
            "atom_styles",
            "atom_material",
            "atom_lighting",
            "atom_edges",
            "atom_edge_color",
            "atom_edge_width",
            "default_atom_style",
            "default_custom_radius",
            "default_atom_scale",
            "material_ambient",
            "material_diffuse",
            "material_specular",
            "material_roughness",
            "material_metallic",
        }
        if change.before_atom_styles or changed_global & atom_fields:
            for record in self.window_manager.records:
                viewer = record["viewer"]
                for name in atom_fields | changed_global:
                    if hasattr(self.viewer, name):
                        setattr(viewer, name, deepcopy(getattr(self.viewer, name)))
                viewer.set_background(viewer._background_color_value)
            self.window_manager.refresh_all()
            if "distinguish_files" in changed_global:
                self.window_manager.apply_file_colors()
            return

        for record in self.window_manager.records:
            viewer = record["viewer"]
            for name in changed_global:
                setattr(viewer, name, deepcopy(getattr(self.viewer, name)))
            viewer.set_background(viewer._background_color_value)
            state = self.files.get(record["file_key"])
            frame = int(record["frame_index"])
            if (
                {"axes_visible", "grid_visible"} & changed_global
                and state is not None
                and 0 <= frame < len(state.coordinates)
            ):
                viewer._apply_reference_widgets(
                    np.asarray(state.coordinates[frame], dtype=float),
                    np.asarray(state.lattices[frame], dtype=float),
                )
            for actor in viewer.bond_actors:
                actor.GetProperty().SetOpacity(viewer.bond_alpha)
            if "bond_width" in changed_global:
                viewer.invalidate_bond_meshes()
                viewer.refresh_bonds()
            else:
                viewer.render()
        if "distinguish_files" in changed_global:
            self.window_manager.apply_file_colors()

    def _record_coordinate_edit(self, frame_index, before_coordinates, after_coordinates):
        """Record one completed mouse coordinate edit.

        Args:
            frame_index: Frame modified by the gesture.
            before_coordinates: Complete coordinates before the gesture.
            after_coordinates: Complete coordinates after the gesture.

        Returns:
            None. The already-applied edit enters unified history.
        """
        if not 0 <= int(frame_index) < len(self.coordinates):
            return
        change = StructureChange.for_frame(
            "Move atoms",
            int(frame_index),
            self.elements[frame_index],
            before_coordinates,
            self.lattices[frame_index],
            self.elements[frame_index],
            after_coordinates,
            self.lattices[frame_index],
            before_context=self._supercell_context(frame_index),
            after_context=self._supercell_context(frame_index),
        )
        self._record_change(change, already_applied=True, frame_index=frame_index)

    def _undo(self, checked=False):
        """Undo the latest structure, graph, appearance or compound edit.

        Args:
            checked: Ignored QAction checked state.

        Return:
            None. An empty stack is a no-op.
        """
        if len(self.undo_stack) == 0:
            return
        change = self.undo_stack[-1]
        try:
            self._apply_history_entry(change, use_after=False)
            self.undo_stack.pop()
            self.redo_stack.append(change)
            self._refresh_history_entry(change)
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            self._log(f"Undo error: {error}")
        self._sync_history_actions()

    def _redo(self, checked=False):
        """Replay the latest undone edit in its forward dependency order.

        Args:
            checked: Ignored QAction checked state.

        Return:
            None. An empty stack is a no-op.
        """
        if len(self.redo_stack) == 0:
            return
        change = self.redo_stack[-1]
        try:
            self._apply_history_entry(change, use_after=True)
            self.redo_stack.pop()
            self.undo_stack.append(change)
            self._refresh_history_entry(change)
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            self._log(f"Redo error: {error}")
        self._sync_history_actions()

    def _new_menu(self, parent):
        """Create a menu for a toolbar button without adding another top-level menu."""
        from PySide6.QtWidgets import QMenu
        return QMenu(parent)

    def _toggle_axes(self, visible):
        """Toggle coordinate axes visibility."""
        self.viewer.set_axes_visible(visible)

    def _toggle_grid(self, visible):
        """Toggle coordinate grid visibility."""
        self.viewer.set_grid_visible(visible)

    def _active_viewer(self):
        """Return the viewer for the active MDI subwindow, if any."""
        record = self.window_manager.active_record()
        return None if record is None else record["viewer"]

    def _set_view(self, direction):
        """Set a lattice-axis view and reset its relative zoom control.

        Args:
            direction: Lattice row index, from zero to two.

        Return:
            None. The viewer camera and toolbar zoom value are synchronized.
        """
        viewer = self._active_viewer()
        if viewer is None:
            return
        if viewer is not self.viewer:
            self._activate_viewer(viewer)
        viewer.set_view(direction)
        blocker = QSignalBlocker(self.zoom_input)
        self.zoom_input.setValue(100)
        del blocker

    def _set_perspective_enabled(self, enabled):
        """Apply the selected projection mode to every open structure view.

        Args:
            enabled: Whether perspective projection should be enabled.

        Return:
            None. Existing and future active views share the selected mode.
        """
        viewers = {record["viewer"] for record in self.window_manager.records}
        if self.viewer is not None:
            viewers.add(self.viewer)
        for viewer in viewers:
            viewer.set_perspective_enabled(enabled)
        blocker = QSignalBlocker(self.zoom_input)
        self.zoom_input.setValue(100)
        del blocker

    def _set_measurement_mode(self, mode):
        """Activate one special measurement selector or restore atom selection."""
        viewer = self._active_viewer()
        if viewer is None:
            return
        if viewer is not self.viewer:
            self._activate_viewer(viewer)
        if mode == "length":
            self.angle_measure_action.setChecked(False)
        elif mode == "angle":
            self.length_measure_action.setChecked(False)
        viewer.set_measurement_mode(mode)
        viewer.draw_frame()

    def _sync_measurement_actions(self, mode):
        """Reflect the viewer's special measurement mode in toolbar buttons."""
        self.length_measure_action.blockSignals(True)
        self.angle_measure_action.blockSignals(True)
        self.length_measure_action.setChecked(mode == "length")
        self.angle_measure_action.setChecked(mode == "angle")
        self.length_measure_action.blockSignals(False)
        self.angle_measure_action.blockSignals(False)

    def _selected_bond_frames(self):
        """Parse the frame scope used by Check/Show bonds.

        Args:
            None.

        Return:
            Zero-based frames selected by the Bonds menu without changing the
            per-frame visibility flags.
        """
        return parse_frame_scope(
            self.bond_frames_edit.text(),
            len(self.coordinates),
            self.current_index,
        )

    def _check_bonds(self, checked=False):
        """Force recalculation of cached bond adjacency for selected frames.

        Args:
            checked: Unused QAction checked state.

        Return:
            None. Invalid frame text is rejected with a warning.
        """
        try:
            indices = self._selected_bond_frames()
            before = {frame: self.bond_adjacencies[frame] for frame in indices}
            before_orders = {frame: self.bond_orders[frame] for frame in indices}
            after = {}
            after_orders = {}
            for frame_index in indices:
                after[frame_index] = build_bond_adjacency(
                    self.coordinates[frame_index],
                    self.elements[frame_index],
                    self.viewer.bond_tolerance,
                )
                after_orders[frame_index] = np.ones(
                    len(after[frame_index][0]), dtype=np.int8
                )
            change = BondChange.between(
                "Check bonds", before, after, before_orders, after_orders
            )
        except (IndexError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "Bonds", str(error))
            return
        # Install every freshly rebuilt graph, including frames whose edge set
        # did not change; history stores only the sparse difference from before.
        for frame in indices:
            self.bond_adjacencies[frame] = after[frame]
            self.bond_orders[frame] = after_orders[frame]
        if change.frame_indices:
            self._record_change(change, already_applied=True)
        else:
            self._refresh_history_entry(change)

    def viewer_set_bonds(self, enabled):
        """Toggle per-frame cached bond rendering for the Bonds frame scope.

        Args:
            enabled: Requested Show bonds check state.

        Return:
            None. Invalid input restores the previous check state.
        """
        try:
            indices = self._selected_bond_frames()
            before, after = {}, {}
            before_orders, after_orders = {}, {}
            if enabled:
                for frame_index in indices:
                    if self.bond_adjacencies[frame_index] is None:
                        before[frame_index] = None
                        before_orders[frame_index] = None
                        after[frame_index] = build_bond_adjacency(
                            self.coordinates[frame_index],
                            self.elements[frame_index],
                            self.viewer.bond_tolerance,
                        )
                        after_orders[frame_index] = np.ones(
                            len(after[frame_index][0]), dtype=np.int8
                        )
        except (IndexError, TypeError, ValueError) as error:
            self._sync_bond_action()
            QMessageBox.warning(self, "Bonds", str(error))
            return
        visible = set(self.bond_frame_indices)
        if enabled:
            visible.update(indices)
        else:
            visible.difference_update(indices)
        self.bond_frame_indices = sorted(visible)
        for record in self.window_manager.records_for_file(self.active_key):
            record["viewer"].bond_frame_indices = set(visible)
            if record["frame_index"] not in visible:
                record["viewer"].selected_bonds.clear()
            record["viewer"].refresh_bonds()
        # Lazy initialization changes the graph too; record it so older graph
        # deltas can never be replayed against an unrecorded replacement graph.
        if after:
            for frame_index in after:
                self.bond_orders[frame_index] = after_orders[frame_index]
            self._record_change(
                BondChange.between(
                    "Generate bonds", before, after, before_orders, after_orders
                )
            )
        self._sync_bond_action()

    def _sync_bond_action(self):
        """Reflect the current frame's bond-visibility flag in the menu action."""
        if not hasattr(self, "bond_action"):
            return
        graph = (
            self.bond_adjacencies[self.current_index]
            if 0 <= self.current_index < len(self.bond_adjacencies)
            else None
        )
        checked = (
            self.current_index in self.bond_frame_indices
            and graph is not None
            and len(graph[0]) > 0
        )
        blocker = QSignalBlocker(self.bond_action)
        self.bond_action.setChecked(checked)
        del blocker

    def _clear_bonds(self, checked=False):
        """Clear all graphs while retaining their complete originals in history.

        Args:
            checked: Unused QAction checked state.

        Return:
            None. Explicit empty graphs remain empty until an explicit rebuild.
        """
        originals = {
            frame: None if graph is None else np.array(graph, copy=True)
            for frame, graph in enumerate(self.bond_adjacencies)
        }
        originals_orders = {
            frame: None if orders is None else np.array(orders, dtype=np.int8, copy=True)
            for frame, orders in enumerate(self.bond_orders)
        }
        if originals:
            self._record_change(
                ClearBondsChange("Clear bonds", originals, originals_orders)
            )

    def _begin_bond_addition(self, checked=False):
        """Start the temporary two-atom picker used for manual bonds.

        Args:
            checked: Unused QAction state.

        Return:
            None. Escape cancels without changing any graph.
        """
        if self.viewer is None or not self.coordinates:
            return
        self.viewer.cancel_atom_addition()
        self.viewer.set_measurement_mode("bond")
        self.viewer.draw_frame()
        self.viewer.setFocus(Qt.FocusReason.OtherFocusReason)
        self._log(
            "Add bond: click a bond to cycle 1-2-3, or two atoms; "
            "hold Ctrl to continue; Escape cancels."
        )

    def _add_bond(self, frame_index, left, right):
        """Append one canonical manual edge independently of the cutoff.

        Args:
            frame_index: Frame receiving the bond.
            left: First zero-based atom index.
            right: Second zero-based atom index.

        Return:
            None. Duplicate edges and self edges do not create history entries.
        """
        if not 0 <= frame_index < len(self.elements):
            return
        if left == right or min(left, right) < 0 or max(left, right) >= len(self.elements[frame_index]):
            return
        pair = tuple(sorted((int(left), int(right))))
        original = self.bond_adjacencies[frame_index]
        graph = normalize_bond_adjacency(original)
        orders = self.bond_orders[frame_index]
        if orders is None or len(orders) != len(graph[0]):
            orders = np.ones(len(graph[0]), dtype=np.int8)
        order_map = {
            (int(source), int(target)): int(order)
            for (source, target), order in zip(graph.T, orders)
        }
        if pair in order_map:
            order_map[pair] = order_map[pair] % 3 + 1
            description = "Change bond order"
        else:
            order_map[pair] = 1
            description = "Add bond"
        pairs = sorted(order_map)
        replacement = np.asarray(pairs, dtype=np.int64).T.reshape(2, -1)
        replacement_orders = np.asarray(
            [order_map[item] for item in pairs], dtype=np.int8
        )
        change = BondChange.between(
            description,
            {frame_index: original},
            {frame_index: replacement},
            {frame_index: orders},
            {frame_index: replacement_orders},
        )
        if change.frame_indices:
            # Make an explicitly added bond immediately visible on this frame,
            # including a previously hidden one, without computing any others.
            self.bond_frame_indices = sorted(
                set(self.bond_frame_indices) | {frame_index}
            )
            for record in self.window_manager.records_for_file(self.active_key):
                record["viewer"].bond_frame_indices = set(self.bond_frame_indices)
            self._record_change(change)
            self._sync_bond_action()

    def _delete_selected_bonds(self, frame_index, pairs):
        """Remove only selected endpoint triples from a frame's COO graph.

        Args:
            frame_index: Frame containing the selected bonds.
            pairs: Canonical zero-based endpoint pairs selected by the viewer.

        Return:
            None. Unselected graph edges remain untouched.
        """
        if not 0 <= frame_index < len(self.bond_adjacencies) or not len(pairs):
            return
        graph = normalize_bond_adjacency(self.bond_adjacencies[frame_index])
        orders = self.bond_orders[frame_index]
        if orders is None or len(orders) != len(graph[0]):
            orders = np.ones(len(graph[0]), dtype=np.int8)
        order_map = {
            (int(source), int(target)): int(order)
            for (source, target), order in zip(graph.T, orders)
        }
        pair_dtype = np.dtype([("src", np.int64), ("dst", np.int64)])
        current = np.ascontiguousarray(graph.T).view(pair_dtype).reshape(-1)
        selected = np.ascontiguousarray(np.asarray(pairs, dtype=np.int64)).view(pair_dtype).reshape(-1)
        after = graph[:, ~np.isin(current, selected)]
        after_orders = np.asarray(
            [order_map[tuple(sorted((int(a), int(b))))] for a, b in after.T],
            dtype=np.int8,
        )
        change = BondChange.between(
            "Delete bonds",
            {frame_index: graph},
            {frame_index: after},
            {frame_index: orders},
            {frame_index: after_orders},
        )
        if change.frame_indices:
            self._record_change(change)

    def viewer_set_tolerance(self, tolerance):
        """Update the cutoff used by the next explicit or lazy bond check."""
        self.viewer.set_bond_tolerance(tolerance)

    def _wrap_pbc(self, checked=False):
        """Wrap atoms in selected frames into their displayed unit cells.

        Args:
            checked: Unused toolbar button state.

        Return:
            None. One multi-frame structure change is added to undo history.
        """
        try:
            indices = parse_frame_scope(
                self.pbc_frames_edit.text(),
                len(self.coordinates),
                self.current_index,
            )
            wrapped = [
                wrap_coordinates(self.coordinates[index], self.lattices[index])
                for index in indices
            ]
        except (IndexError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "PBC wrap", str(error))
            return
        changed = [
            (frame_index, after_coordinates)
            for frame_index, after_coordinates in zip(indices, wrapped)
            if not np.allclose(
                self.coordinates[frame_index],
                after_coordinates,
                rtol=0.0,
                atol=1.0e-12,
            )
        ]
        if not changed:
            return
        frame_indices = tuple(frame_index for frame_index, _ in changed)
        change = StructureChange(
            description="PBC wrap",
            before_elements=[
                np.array(self.elements[index], copy=True) for index in frame_indices
            ],
            before_coordinates=[
                np.array(self.coordinates[index], copy=True) for index in frame_indices
            ],
            before_lattices=[
                np.array(self.lattices[index], copy=True) for index in frame_indices
            ],
            after_elements=[
                np.array(self.elements[index], copy=True) for index in frame_indices
            ],
            after_coordinates=[after_coordinates for _, after_coordinates in changed],
            after_lattices=[
                np.array(self.lattices[index], copy=True) for index in frame_indices
            ],
            frame_indices=frame_indices,
        )
        self._record_change(change, frame_index=self.current_index)

    def _show_supercell(self) -> None:
        """Apply the selected transform to the current frame's base cell.

        Args:
            None.

        Returns:
            None. The committed transform is recorded in unified history.
        """
        if not self.coordinates:
            QMessageBox.information(
                self, "Supercell", "Open structure data before building a supercell."
            )
            return
        frame_index = int(self.viewer.current_index)
        if not 0 <= frame_index < len(self.coordinates):
            QMessageBox.warning(self, "Supercell", "The current frame is not available.")
            return

        self._ensure_supercell_state()
        multipliers, matrix = self.supercell_transforms[frame_index]
        dialog = SupercellDialog(multipliers, matrix, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if dialog.resize_requested:
            self._resize_supercell()
            return
        multipliers, matrix = dialog.values()
        effective_transform = matrix @ np.diag(multipliers)
        try:
            base = self.supercell_bases[frame_index]
            new_elements, new_coordinates, new_lattice = transform_structure(
                base["elements"],
                base["coordinates"],
                base["lattice"],
                effective_transform,
            )
        except ValueError as error:
            QMessageBox.warning(self, "Supercell", str(error))
            return

        topology_changed = (
            len(new_elements) != len(self.elements[frame_index])
            or not np.array_equal(new_elements, self.elements[frame_index])
        )
        if topology_changed and not self._confirm_metrics_topology_change():
            return

        previous_atom_count = len(self.elements[frame_index])
        before_context = self._supercell_context(frame_index)
        self.supercell_transforms[frame_index] = (
            np.array(multipliers, copy=True),
            np.array(matrix, copy=True),
        )
        after_context = self._supercell_context(frame_index)
        if topology_changed:
            after_context["atom_styles"] = [dict() for _ in new_elements]
            after_context["metrics"] = {"length": {}, "angle": {}}
        change = StructureChange.for_frame(
            "Supercell",
            frame_index,
            self.elements[frame_index],
            self.coordinates[frame_index],
            self.lattices[frame_index],
            new_elements,
            new_coordinates,
            new_lattice,
            before_context=before_context,
            after_context=after_context,
        )
        if topology_changed:
            # Drop old endpoints before changing node topology. Undo restores
            # nodes first, then restores the removed edge triples and None state.
            bonds = BondChange.between(
                "Reset supercell bonds",
                {frame_index: self.bond_adjacencies[frame_index]}, {frame_index: None},
                {frame_index: self.bond_orders[frame_index]}, {frame_index: None},
            )
            if bonds.frame_indices:
                change = (bonds, change)
        self._record_change(change, frame_index=frame_index)
        blocker = QSignalBlocker(self.zoom_input)
        self.zoom_input.setValue(100)
        del blocker
        QMessageBox.information(
            self,
            "Supercell",
            "Transformed the current frame from "
            f"{previous_atom_count} to {len(new_elements)} atoms.",
        )

    def _resize_supercell(self) -> None:
        """Rebase the current frame and reset its remembered transform.

        Args:
            None.

        Returns:
            None. The visible structure is unchanged and future transforms start
            from the current frame with identity settings.
        """
        if not self.coordinates:
            self._log("Open structure data before resizing a supercell.")
            return
        frame_index = int(self.current_index)
        before_context = self._supercell_context(frame_index)
        self.supercell_bases[frame_index] = {
            "elements": np.array(self.elements[frame_index], copy=True),
            "coordinates": np.array(self.coordinates[frame_index], copy=True),
            "lattice": np.array(self.lattices[frame_index], copy=True),
        }
        self.supercell_transforms[frame_index] = (
            np.ones(3, dtype=np.int64),
            np.eye(3, dtype=np.int64),
        )
        after_context = self._supercell_context(frame_index)
        change = StructureChange.for_frame(
            "Resize supercell base",
            frame_index,
            self.elements[frame_index],
            self.coordinates[frame_index],
            self.lattices[frame_index],
            self.elements[frame_index],
            self.coordinates[frame_index],
            self.lattices[frame_index],
            before_context=before_context,
            after_context=after_context,
        )
        self._record_change(change, already_applied=True, frame_index=frame_index)
        self._log("Current frame is now the supercell base (1 1 1).")

    @staticmethod
    def _identity_supercell_context(elements, coordinates, lattice):
        """Build a new identity supercell context for changed topology.

        Args:
            elements: Element array becoming the new base.
            coordinates: Coordinate array becoming the new base.
            lattice: Lattice array becoming the new base.

        Returns:
            Detached identity-base context.
        """
        return {
            "base": {
                "elements": np.array(elements, copy=True),
                "coordinates": np.array(coordinates, copy=True),
                "lattice": np.array(lattice, copy=True),
            },
            "transform": (
                np.ones(3, dtype=np.int64),
                np.eye(3, dtype=np.int64),
            ),
        }

    def _show_add_atom(self) -> None:
        """Open the atom dialog and start direct or reference placement.

        Args:
            None.

        Returns:
            None. Direct coordinates commit immediately; reference mode delegates
            placement to the viewer.
        """
        if not self.coordinates:
            self._log("Open structure data before adding an atom.")
            return
        dialog = AddAtomDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            symbol, coordinates, reference_requested = dialog.values()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid atom", str(error))
            return
        if reference_requested:
            self.viewer.begin_atom_addition(symbol)
            return
        self._commit_added_atoms(
            int(self.current_index),
            np.asarray([symbol], dtype="<U3"),
            np.asarray([coordinates], dtype=float),
            "Add atom",
        )

    def _confirm_metrics_topology_change(self) -> bool:
        """Ask whether persistent measurements may be removed before topology edits.

        Args:
            None.

        Returns:
            ``True`` when the edit may continue, otherwise ``False``.
        """
        has_metrics = len(self.viewer.length_measurements) > 0 or len(
            self.viewer.angle_measurements
        ) > 0
        if not has_metrics:
            return True
        answer = QMessageBox.question(
            self,
            "Remove Metrics",
            "This topology change will remove the current metrics. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        return True

    def _commit_added_atoms(self, frame_index, symbols, coordinates, description, reference_index=None):
        """Append atoms to one frame through unified history.

        Args:
            frame_index: Frame receiving the new atoms.
            symbols: One-dimensional element symbols.
            coordinates: Cartesian positions shaped ``(N, 3)``.
            description: History description.
            reference_index: Original reference atom for an automatic bond chain.

        Returns:
            True when the change was committed, otherwise False for an invalid
            request or a cancelled metrics confirmation. The full scene is
            rebuilt after one atomic commit.
        """
        if not 0 <= int(frame_index) < len(self.coordinates):
            return False
        frame_index = int(frame_index)
        symbols = np.asarray(symbols, dtype="<U3").reshape(-1)
        coordinates = np.asarray(coordinates, dtype=float).reshape((-1, 3))
        if len(symbols) != len(coordinates) or len(symbols) == 0:
            return False
        if not self._confirm_metrics_topology_change():
            return False
        before_elements = self.elements[frame_index]
        before_coordinates = self.coordinates[frame_index]
        before_lattice = self.lattices[frame_index]
        after_elements = np.concatenate((before_elements, symbols))
        after_coordinates = np.vstack((before_coordinates, coordinates))
        before_context = self._supercell_context(frame_index)
        after_context = self._identity_supercell_context(
            after_elements, after_coordinates, before_lattice
        )
        after_context["atom_styles"] = (
            deepcopy(self.atom_frame_styles[frame_index])
            + [dict() for _ in symbols]
        )
        after_context["metrics"] = {"length": {}, "angle": {}}
        after_context["appearance"] = self._appearance_context()
        change = StructureChange.for_frame(
            description,
            frame_index,
            before_elements,
            before_coordinates,
            before_lattice,
            after_elements,
            after_coordinates,
            before_lattice,
            before_context=before_context,
            after_context=after_context,
        )
        if reference_index is not None:
            if not 0 <= int(reference_index) < len(before_elements):
                self._log("Add ignored: reference atom is out of range.")
                return False
            original = self.bond_adjacencies[frame_index]
            graph = normalize_bond_adjacency(original)
            orders = self.bond_orders[frame_index]
            if orders is None or len(orders) != len(graph[0]):
                orders = np.ones(len(graph[0]), dtype=np.int8)
            order_map = {
                (int(source), int(target)): int(order)
                for (source, target), order in zip(graph.T, orders)
            }
            new_indices = np.arange(len(before_elements), len(after_elements), dtype=np.int64)
            # The original reference index survives every preview; subsequent
            # chain links join consecutive appended atoms, not nearest neighbors.
            chain = np.vstack((np.r_[int(reference_index), new_indices[:-1]], new_indices))
            for source, target in chain.T:
                order_map[tuple(sorted((int(source), int(target))))] = 1
            pairs = sorted(order_map)
            after_graph = np.asarray(pairs, dtype=np.int64).T.reshape(2, -1)
            after_orders = np.asarray(
                [order_map[item] for item in pairs], dtype=np.int8
            )
            bonds = BondChange.between(
                "Reference atom bonds",
                {frame_index: original}, {frame_index: after_graph},
                {frame_index: orders}, {frame_index: after_orders},
            )
            change = (change, bonds)
        self._record_change(change, frame_index=frame_index)
        return True

    def _commit_reference_atoms(self, frame_index, symbols, coordinates, reference_index):
        """Commit a complete reference-placement chain as one change.

        Args:
            frame_index: Frame receiving the pending atoms.
            symbols: Pending element symbols.
            coordinates: Pending Cartesian coordinates.
            reference_index: First existing atom selected before placing the chain.

        Returns:
            None. Delegates to the common atom-appending path.
        """
        self._commit_added_atoms(frame_index, symbols, coordinates, "Add atoms", reference_index)

    def _delete_selected_atoms(self, frame_index, indices):
        """Delete selected atoms and record one reversible topology change.

        Args:
            frame_index: Frame containing the selection.
            indices: Integer atom indices to remove.

        Returns:
            None. Invalid selections are ignored after a concise log message.
        """
        if not 0 <= int(frame_index) < len(self.coordinates):
            return
        frame_index = int(frame_index)
        indices = np.unique(np.asarray(indices, dtype=int))
        if len(indices) == 0 or np.any(indices < 0) or np.any(
            indices >= len(self.elements[frame_index])
        ):
            self._log("Delete ignored: atom selection is out of range.")
            return
        if not self._confirm_metrics_topology_change():
            return
        before_elements = self.elements[frame_index]
        before_coordinates = self.coordinates[frame_index]
        before_lattice = self.lattices[frame_index]
        after_elements = np.delete(before_elements, indices)
        after_coordinates = np.delete(before_coordinates, indices, axis=0)
        before_context = self._supercell_context(frame_index)
        after_context = self._identity_supercell_context(
            after_elements, after_coordinates, before_lattice
        )
        deleted = set(indices.tolist())
        after_context["atom_styles"] = [
            deepcopy(style)
            for index, style in enumerate(self.atom_frame_styles[frame_index])
            if index not in deleted
        ]
        after_context["metrics"] = {"length": {}, "angle": {}}
        after_context["appearance"] = self._appearance_context()
        change = StructureChange.for_frame(
            "Delete atoms",
            frame_index,
            before_elements,
            before_coordinates,
            before_lattice,
            after_elements,
            after_coordinates,
            before_lattice,
            before_context=before_context,
            after_context=after_context,
        )
        # Section: Keep an old-node-sized simultaneous endpoint mapping. Mapping
        # alone cannot restore incident edges, so retain their triples as well.
        original = self.bond_adjacencies[frame_index]
        graph = normalize_bond_adjacency(original)
        orders = self.bond_orders[frame_index]
        if orders is None or len(orders) != len(graph[0]):
            orders = np.ones(len(graph[0]), dtype=np.int8)
        order_map = {
            (int(source), int(target)): int(order)
            for (source, target), order in zip(graph.T, orders)
        }
        keep = np.ones(len(before_elements), dtype=bool)
        keep[indices] = False
        mapping = np.full(len(before_elements), -1, dtype=np.int64)
        mapping[keep] = np.arange(np.count_nonzero(keep), dtype=np.int64)
        removed = graph[:, ~np.all(keep[graph], axis=0)].T
        triples = np.asarray(
            [
                (int(source), int(target), order_map[(int(source), int(target))])
                for source, target in removed
            ],
            dtype=np.int64,
        ).reshape(-1, 3)
        bonds = BondChange(
            "Delete atom bonds", {frame_index: triples},
            {frame_index: np.empty((0, 3), dtype=np.int64)},
            {frame_index: original is None}, {frame_index: mapping},
            after_missing={frame_index: original is None},
        )
        self._record_change((bonds, change), frame_index=frame_index)

    def _copy_selected_atoms(self, checked=False):
        """Copy the selected atoms of the focused frame into the clipboard.

        Args:
            checked: Ignored Qt action state.

        Return:
            None. The one-slot buffer retains detached copies of species and
            absolute coordinates; no history entry is created.
        """
        if self.viewer is None or not self.coordinates:
            self._log("Copy: no structure is displayed.")
            return
        frame_index = int(self.current_index)
        if not 0 <= frame_index < len(self.coordinates):
            return
        elements = np.asarray(self.elements[frame_index])
        indices = np.asarray(self.viewer.selected_indices, dtype=int)
        indices = indices[(indices >= 0) & (indices < len(elements))]
        if len(indices) == 0:
            self._log("Copy: select at least one atom first.")
            return
        count = self.atom_clipboard.store(
            elements[indices],
            self.coordinates[frame_index][indices],
            self._file_title(self._active_state, frame_index),
        )
        self._sync_clipboard_actions()
        self._log(f"Copy: {count} atom(s) from {self.atom_clipboard.source}.")

    def _paste_atoms(self, checked=False):
        """Append the clipboard fragment to the focused frame as one change.

        Args:
            checked: Ignored Qt action state.

        Return:
            None. Atoms keep their absolute coordinates and carry no bond
            information; the appended indices become the new selection.
        """
        if self.viewer is None or not self.coordinates:
            self._log("Paste: no structure is displayed.")
            return
        if self.atom_clipboard.is_empty():
            self._log("Paste: the atom clipboard is empty.")
            return
        # Section: pausing first makes the displayed frame authoritative, so a
        # paste never targets a playback frame that is already gone.
        self._pause_playback()
        frame_index = int(self.current_index)
        if not 0 <= frame_index < len(self.coordinates):
            return
        fragment = self.atom_clipboard.content()
        if fragment is None:
            return
        elements, coordinates = fragment
        added = len(elements)
        first_new = len(self.elements[frame_index])
        if not self._commit_added_atoms(
            frame_index, elements, coordinates, "Paste atoms"
        ):
            return
        self.viewer.set_selection(np.arange(first_new, first_new + added, dtype=int))
        self._sync_clipboard_actions()
        self._log(f"Paste: appended {added} atom(s) from {self.atom_clipboard.source}.")

    def _show_viewer_context_menu(self, viewer, position):
        """Offer atom Copy and Paste for one right-clicked viewer.

        Args:
            viewer: Viewer that received the right click.
            position: Global screen position of the click.

        Returns:
            None. The menu reuses the toolbar actions so enabled states agree.
        """
        if viewer is not self.viewer:
            self._activate_viewer(viewer)
        self._sync_clipboard_actions()
        menu = QMenu(self)
        menu.addAction(self.copy_action)
        menu.addAction(self.paste_action)
        menu.exec(position)

    def _sync_clipboard_actions(self):
        """Enable Copy for a valid selection and Paste for a filled clipboard.

        Args:
            None.

        Return:
            None. Both toolbar actions and their context-menu copies follow the
            focused viewer, current frame, and clipboard state.
        """
        if not hasattr(self, "copy_action"):
            return
        selected = (
            np.asarray(self.viewer.selected_indices, dtype=int)
            if self.viewer is not None
            else np.empty(0, dtype=int)
        )
        frame_valid = 0 <= self.current_index < len(self.elements)
        atom_count = len(self.elements[self.current_index]) if frame_valid else 0
        has_selection = (
            len(selected) > 0
            and bool(np.all(selected >= 0))
            and bool(np.all(selected < atom_count))
        )
        self.copy_action.setEnabled(bool(has_selection))
        self.paste_action.setEnabled(
            bool(frame_valid and not self.atom_clipboard.is_empty())
        )

    def _show_appearance(self):
        """Open the atom and bond appearance editor for the current frame."""
        if not self.elements:
            return
        frame_index = int(self.current_index)
        dialog = AppearanceDialog(
            self.viewer,
            self.elements[frame_index],
            self.viewer.selected_indices,
            self,
            current_frame=frame_index,
            frame_count=len(self.elements),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.accepted_values is None:
            return
        values = dialog.accepted_values
        atom_fields = {
            "atom_style": ("default_atom_style", "style"),
            "custom_radius": ("default_custom_radius", "radius"),
            "atom_scale": ("default_atom_scale", "scale"),
            "material": ("atom_material", "material"),
            "lighting": ("atom_lighting", "lighting"),
            "edges": ("atom_edges", "edges"),
            "edge_color": ("atom_edge_color", "edge_color"),
            "edge_width": ("atom_edge_width", "edge_width"),
            "ambient": ("material_ambient", "ambient"),
            "diffuse": ("material_diffuse", "diffuse"),
            "specular": ("material_specular", "specular"),
            "roughness": ("material_roughness", "roughness"),
            "metallic": ("material_metallic", "metallic"),
        }
        scene_fields = {
            "background": "_background_color_value",
            "axes": "axes_visible",
            "grid": "grid_visible",
            "bond_width": "bond_width",
            "bond_alpha": "bond_alpha",
            "distinguish_files": "distinguish_files",
        }
        before_global = {}
        after_global = {}
        before_atom_styles = {}
        after_atom_styles = {}

        # Section: scene settings are independent of the atom selector.
        for name, value in values["changed_scene_values"].items():
            attribute = scene_fields[name]
            before_global[attribute] = deepcopy(getattr(self.viewer, attribute))
            after_global[attribute] = deepcopy(value)

        changed_style_keys = {
            atom_fields[name][1]: value
            for name, value in values["changed_atom_values"].items()
        }
        changed_symbols = values["changed_symbols"]
        if values["global_atoms"]:
            # Section: All atoms updates compact renderer defaults. Existing sparse
            # overrides for changed fields are removed so the new default reaches all.
            for name, value in values["changed_atom_values"].items():
                attribute = atom_fields[name][0]
                before_global[attribute] = deepcopy(getattr(self.viewer, attribute))
                after_global[attribute] = deepcopy(value)
            if changed_symbols:
                before_global["atom_styles"] = deepcopy(self.viewer.atom_styles)
                new_styles = deepcopy(self.viewer.atom_styles)
                for symbol in changed_symbols:
                    new_styles[symbol] = {
                        "color": values["styles"][symbol]["color"],
                        "alpha": float(values["styles"][symbol]["alpha"]),
                    }
                after_global["atom_styles"] = new_styles
            if changed_style_keys or changed_symbols:
                for target_frame, frame_styles in enumerate(self.atom_frame_styles):
                    for atom_index, old_style in enumerate(frame_styles):
                        symbol = str(self.elements[target_frame][atom_index])
                        new_style = dict(old_style)
                        for style_key in changed_style_keys:
                            new_style.pop(style_key, None)
                        if symbol in changed_symbols:
                            new_style.pop("color", None)
                            new_style.pop("alpha", None)
                        if new_style != old_style:
                            key = (target_frame, atom_index)
                            before_atom_styles[key] = deepcopy(old_style)
                            after_atom_styles[key] = new_style
        else:
            # Section: a non-empty selector writes only changed fields to its
            # atoms in every listed frame. A frame without that index keeps its
            # own atoms, and element colors follow each frame's own symbol.
            for target_frame in values["frames"]:
                if target_frame >= len(self.atom_frame_styles):
                    continue
                target_elements = self.elements[target_frame]
                for atom_index in values["indices"]:
                    if not 0 <= atom_index < len(target_elements):
                        continue
                    symbol = str(target_elements[atom_index])
                    old_style = self.atom_frame_styles[target_frame][atom_index]
                    new_style = dict(old_style)
                    new_style.update(changed_style_keys)
                    if symbol in changed_symbols and symbol in values["styles"]:
                        new_style["color"] = values["styles"][symbol]["color"]
                        new_style["alpha"] = float(values["styles"][symbol]["alpha"])
                    if new_style != old_style:
                        key = (target_frame, atom_index)
                        before_atom_styles[key] = deepcopy(old_style)
                        after_atom_styles[key] = new_style

        if not before_global and not before_atom_styles:
            return
        change = AppearanceChange(
            "Appearance",
            before_global,
            after_global,
            before_atom_styles,
            after_atom_styles,
        )
        self._record_change(change, frame_index=frame_index)

    def _show_metric_style(self):
        """Apply one style to all current-frame metrics through unified history."""
        if not self.frame_metrics:
            return
        frame_index = int(self.current_index)
        metrics = self.frame_metrics[frame_index]
        style = StructureViewer.default_metric_style()
        for mapping in (metrics["length"], metrics["angle"]):
            if mapping:
                style.update(next(iter(mapping.values())).get("style", {}))
                break
        dialog = MetricStyleDialog(style, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        before_context = self._supercell_context(frame_index)
        style = dialog.values()
        for mapping in (metrics["length"], metrics["angle"]):
            for metric in mapping.values():
                metric["style"] = dict(style)
        after_context = self._supercell_context(frame_index)
        change = StructureChange.for_frame(
            "Metric style", frame_index,
            self.elements[frame_index], self.coordinates[frame_index], self.lattices[frame_index],
            self.elements[frame_index], self.coordinates[frame_index], self.lattices[frame_index],
            before_context=before_context, after_context=after_context,
        )
        self._record_change(change, already_applied=True, frame_index=frame_index)

    def _show_replace_atom(self) -> None:
        """Replace the element type of all currently selected atoms.

        Args:
            None.

        Returns:
            None. The replacement is recorded as one reversible structure change.
        """
        frame_index = int(self.current_index)
        indices = np.asarray(self.viewer.selected_indices, dtype=int)
        if not 0 <= frame_index < len(self.elements) or len(indices) == 0:
            return
        dialog = AddAtomDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            symbol, _, reference_requested = dialog.values()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid atom", str(error))
            return
        if reference_requested:
            QMessageBox.information(
                self, "Replace Atom", "Reference placement is not available for replacement."
            )
            return
        before_elements = self.elements[frame_index]
        before_coordinates = self.coordinates[frame_index]
        before_lattice = self.lattices[frame_index]
        after_elements = before_elements.copy()
        after_elements[indices] = symbol
        before_context = self._supercell_context(frame_index)
        change = StructureChange.for_frame(
            "Replace atoms",
            frame_index,
            before_elements,
            before_coordinates,
            before_lattice,
            after_elements,
            before_coordinates,
            before_lattice,
            before_context=before_context,
            after_context=before_context,
        )
        self._record_change(change, frame_index=frame_index)

    def _clean_option_kwargs(self):
        """Return clean() optimizer keywords from persistent settings."""
        if self.settings["clean_mode"] == "local":
            return {
                "mode": "local",
                "local_method": self.settings["clean_local_method"],
                "maxiter": int(self.settings["clean_local_maxiter"]),
                "gtol": float(self.settings["clean_local_gtol"]),
            }
        return {
            "mode": "global",
            "global_method": self.settings["clean_global_method"],
            "maxiter": int(self.settings["clean_global_maxiter"]),
            "workers": int(self.settings["clean_workers"]),
            "bounds_span": float(self.settings["clean_bounds_span"]),
        }

    def _clean_current_frame(self):
        """Relax the focused frame in place with the preset clean updater.

        Args:
            None.

        Return:
            None. Selected atoms stay mobile and unselected atoms are frozen;
            the current frame's existing bond graph is reused when available.
        """
        if not self.coordinates or self.viewer is None:
            return
        self._pause_playback()
        frame_index = int(self.current_index)
        if not 0 <= frame_index < len(self.coordinates):
            return
        selected = np.asarray(self.viewer.selected_indices, dtype=int)
        selected = selected[
            (selected >= 0) & (selected < len(self.coordinates[frame_index]))
        ]
        fixed = np.zeros_like(self.coordinates[frame_index])
        if len(selected):
            fixed[selected] = 1.0
        else:
            fixed[:] = 1.0
        record = next(
            (
                updater for updater in self.updaters
                if updater.get("preset") and updater.get("name") == "clean"
            ),
            None,
        )
        if record is None:
            self._log("Clean error: the preset updater is unavailable")
            return
        bonds = None
        if frame_index < len(self.bond_adjacencies):
            bonds = self.bond_adjacencies[frame_index]
        self._run_updater(
            record,
            str(frame_index + 1),
            kwargs={
                "fixed": fixed,
                "bonds": bonds,
                **self._clean_option_kwargs(),
            },
            in_place=True,
        )

    def _add_hydrogens(self):
        """Append hydrogens from explicit bond orders and relax only those H.

        Args:
            None.

        Return:
            None. One compound history entry restores atoms, orders and bonds.
        """
        if not self.coordinates or self.viewer is None:
            return
        self._pause_playback()
        frame_index = int(self.current_index)
        if not 0 <= frame_index < len(self.coordinates):
            return
        original = self.bond_adjacencies[frame_index]
        if original is None:
            graph = build_bond_adjacency(
                self.coordinates[frame_index],
                self.elements[frame_index],
                self.viewer.bond_tolerance,
            )
            orders = np.ones(len(graph[0]), dtype=np.int8)
            original_orders = None
            self._log("Add H: bonds were uncomputed; using Check bonds.")
        else:
            graph = normalize_bond_adjacency(original)
            orders = self.bond_orders[frame_index]
            if orders is None or len(orders) != len(graph[0]):
                orders = np.ones(len(graph[0]), dtype=np.int8)
            original_orders = orders
        if len(graph[0]) == 0:
            self._log("Add H: no bonds are available; run Check bonds first.")
            return

        before_count = len(self.elements[frame_index])
        selected = np.asarray(self.viewer.selected_indices, dtype=int)
        selected = selected[(selected >= 0) & (selected < before_count)]
        new_elements, new_coordinates, new_graph, new_orders = add_hydrogens(
            self.elements[frame_index],
            self.coordinates[frame_index],
            graph,
            orders,
            selected=selected if len(selected) else None,
        )
        added = len(new_elements) - before_count
        if added == 0:
            self._log("Add H: no main-group atom needs hydrogen.")
            return

        fixed = np.zeros_like(new_coordinates)
        fixed[before_count:] = 1.0
        try:
            relaxed = clean_structure(
                [self.lattices[frame_index]],
                [new_elements],
                [new_coordinates],
                fixed=fixed,
                bonds=new_graph,
                **self._clean_option_kwargs(),
            )[2][0]
        except Exception as error:
            self._log(f"Add H: hydrogen relaxation skipped ({error}).")
            relaxed = np.array(new_coordinates, copy=True)
        if not self._confirm_metrics_topology_change():
            return

        before_elements = self.elements[frame_index]
        before_coordinates = self.coordinates[frame_index]
        before_lattice = self.lattices[frame_index]
        after_elements = np.asarray(new_elements, dtype=before_elements.dtype)
        after_coordinates = np.asarray(relaxed, dtype=float)
        before_context = self._supercell_context(frame_index)
        after_context = self._identity_supercell_context(
            after_elements, after_coordinates, before_lattice
        )
        after_context["atom_styles"] = (
            deepcopy(self.atom_frame_styles[frame_index])
            + [dict() for _ in range(added)]
        )
        after_context["metrics"] = {"length": {}, "angle": {}}
        after_context["appearance"] = self._appearance_context()
        change = StructureChange.for_frame(
            "Add hydrogens",
            frame_index,
            before_elements,
            before_coordinates,
            before_lattice,
            after_elements,
            after_coordinates,
            before_lattice,
            before_context=before_context,
            after_context=after_context,
        )
        bonds = BondChange.between(
            "Add hydrogen bonds",
            {frame_index: original},
            {frame_index: new_graph},
            {frame_index: original_orders},
            {frame_index: new_orders},
        )
        self._record_change((change, bonds), frame_index=frame_index)
        self._log(f"Add H: appended {added} hydrogen atom(s).")

    def _sync_atom_actions(self, indices=None):
        """Enable selection-dependent actions for the current frame."""
        if not hasattr(self, "replace_atom_action"):
            return
        selected = self.viewer.selected_indices if indices is None else np.asarray(indices, dtype=int)
        has_selection = (
            len(selected) > 0
            and 0 <= self.current_index < len(self.elements)
            and np.all(selected >= 0)
            and np.all(selected < len(self.elements[self.current_index]))
        )
        self.replace_atom_action.setEnabled(bool(has_selection))
        self._sync_clipboard_actions()

    def _on_selection_changed(self, indices):
        """Synchronize actions and print one ordinarily selected atom.

        Args:
            indices: Current ordinary-selection atom indices.

        Returns:
            None. Multiple selections update actions without console output.
        """
        indices = np.asarray(indices, dtype=int)
        self._sync_atom_actions(indices)
        if len(indices) != 1 or not 0 <= self.current_index < len(self.elements):
            return
        atom_index = int(indices[0])
        if not 0 <= atom_index < len(self.elements[self.current_index]):
            return
        symbol = str(self.elements[self.current_index][atom_index])
        x, y, z = np.asarray(
            self.coordinates[self.current_index][atom_index], dtype=float
        )
        index = atom_index + 1
        elem = symbol
        self._log(
            f"{index: <8d} {elem: <4s} "
            f"{x: > 10.4f} {y: > 10.4f} {z: > 10.4f}"
        )

    def _sync_frame_controls(self):
        """Synchronize frame input and maximum-frame display with retained lists."""
        frame_count = len(self.coordinates)
        maximum_index = max(frame_count - 1, 0)
        current_index = min(max(self.current_index, 0), maximum_index)
        self.current_index = current_index
        blocker = QSignalBlocker(self.frame_input)
        self.frame_input.setText(str(current_index))
        del blocker
        self.frame_total_label.setText(f"/ {maximum_index}")
        self.playback_range_hint.setText(f"0 ~ {maximum_index}")
        has_frames = frame_count > 0
        self.previous_frame_button.setEnabled(has_frames)
        self.next_frame_button.setEnabled(has_frames)
        self.play_button.setEnabled(has_frames)
        self.playback_options_button.setEnabled(has_frames)
        self.add_atom_action.setEnabled(has_frames)
        self.metrics_button.setEnabled(has_frames)
        self.bond_button.setEnabled(has_frames)
        self.pbc_wrap_button.setEnabled(has_frames)
        self.pbc_frames_edit.setEnabled(has_frames)
        self._sync_atom_actions(
            self.viewer.selected_indices if self.viewer is not None else np.empty(0, dtype=int)
        )

    def _sync_frame_list(self):
        """Rebuild the file-like frame list while retaining its highlight."""
        selected_row = self.frame_list.currentRow()
        selected_rows = [index.row() for index in self.frame_list.selectedIndexes()]
        blocker = QSignalBlocker(self.frame_list)
        self.frame_list.clear()
        for frame_index, frame_elements in enumerate(self.elements):
            item = QListWidgetItem(
                f"{self.frame_names[frame_index]}  ({len(frame_elements)} atoms)"
            )
            item.setData(Qt.ItemDataRole.UserRole, frame_index)
            self.frame_list.addItem(item)
        if self.frame_list.count():
            target_row = self.current_index if selected_row < 0 else selected_row
            self.frame_list.setCurrentRow(
                min(max(target_row, 0), self.frame_list.count() - 1)
            )
            for row in selected_rows:
                if 0 <= row < self.frame_list.count():
                    self.frame_list.item(row).setSelected(True)
        del blocker

    def _apply_frame_list_permutation(self, permutation):
        """Reorder existing sidebar items without rebuilding their widgets.

        Args:
            permutation: Indices satisfying ``new[k] = old[permutation[k]]``.

        Returns:
            None. Item names and selection state move with their frames.
        """
        if self.frame_list.count() != len(permutation):
            self._sync_frame_list()
            return
        blocker = QSignalBlocker(self.frame_list)
        items = [self.frame_list.takeItem(0) for _ in range(self.frame_list.count())]
        for old_index in permutation:
            self.frame_list.addItem(items[int(old_index)])
        del blocker

    def _refresh_after_permutation(self, sync_frame_list):
        """Synchronize indices after a reference-only frame reorder.

        Args:
            sync_frame_list: Whether sidebar items must be reordered first.

        Returns:
            None. The unchanged logical current frame is not redrawn.
        """
        if sync_frame_list:
            self._sync_frame_list()
        for row in range(self.frame_list.count()):
            self.frame_list.item(row).setData(Qt.ItemDataRole.UserRole, row)
        self.window_manager.refresh_all()
        self._sync_frame_controls()

    def _apply_file_manager_settings(self):
        """Apply metadata columns, folder ordering and sort preferences."""
        columns = {
            1: self.settings["file_manager_show_size"],
            2: self.settings["file_manager_show_type"],
            3: self.settings["file_manager_show_modified"],
        }
        for column, visible in columns.items():
            self.file_tree.setColumnHidden(column, not visible)
        self.file_tree.setHeaderHidden(not any(columns.values()))
        self.file_sort_proxy.set_folders_first(
            self.settings["file_manager_folders_first"]
        )
        order = (
            Qt.SortOrder.AscendingOrder
            if self.settings["file_manager_sort_order"] == "asc"
            else Qt.SortOrder.DescendingOrder
        )
        self.file_tree.sortByColumn(
            int(self.settings["file_manager_sort_column"]), order
        )

    def _show_file_manager_settings(self):
        """Open the file-manager metadata and sorting dialog."""
        dialog = FileManagerDialog(self.settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.settings.update(dialog.values())
        self._apply_file_manager_settings()
        self._persist_settings()

    def _show_clean_backend(self):
        """Open the local/global clean optimizer configuration dialog."""
        dialog = CleanBackendDialog(self.settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.settings.update(dialog.values())
        self._persist_settings()

    def _file_source_to_view(self, source_index):
        """Map a QFileSystemModel index through sorting and the parent row."""
        sorted_index = self.file_sort_proxy.mapFromSource(source_index)
        return self.file_proxy.mapFromSource(sorted_index)

    def _file_view_to_source(self, view_index):
        """Map a view index back to the QFileSystemModel source."""
        sorted_index = self.file_proxy.mapToSource(view_index)
        return self.file_sort_proxy.mapToSource(sorted_index)

    def _update_file_manager_buttons(self):
        """Synchronize navigation and recycle-button availability."""
        directory = self._file_manager_dir
        parent = directory.parent if directory is not None else None
        self.file_up_button.setEnabled(
            directory is not None and parent != directory and parent.is_dir()
        )
        self.file_back_button.setEnabled(bool(self._file_manager_back))
        self.file_forward_button.setEnabled(bool(self._file_manager_forward))
        self.file_recycle_button.setEnabled(self.recycler.has_items())

    def _set_file_manager_root(self, directory, path=None, force=False):
        """Set the visible root without touching navigation history."""
        directory = Path(directory).resolve()
        if not directory.is_dir():
            return False
        if force or directory != self._file_manager_dir:
            self.file_model.setRootPath(str(directory))
            root_source = self.file_model.index(str(directory))
            sorted_root = self.file_sort_proxy.mapFromSource(root_source)
            self.file_proxy.set_root_source_index(sorted_root)
            self.file_tree.setRootIndex(QModelIndex())
            self._file_manager_dir = directory
        if path is not None:
            self._select_file_manager_path(path)
        self.file_placeholder.hide()
        self.file_tree.show()
        self.file_nav.show()
        self._update_file_manager_buttons()
        return True

    def _select_file_manager_path(self, path):
        """Select a file or directory inside the current file-tree root."""
        source_index = self.file_model.index(str(path))
        index = self._file_source_to_view(source_index)
        if index.isValid():
            self.file_tree.setCurrentIndex(index)
            self.file_tree.scrollTo(index)
            self._file_manager_path = Path(path)
            return True
        return False

    def _navigate_file_manager(self, directory, *, record=True, manual=True):
        """Change the visible root and optionally record it in history."""
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            self._log(f"File manager: not a directory: {directory}")
            return False
        previous = self._file_manager_dir
        if record and previous is not None and previous != directory:
            self._file_manager_back.append(previous)
            self._file_manager_forward.clear()
        self._file_manager_manual = manual
        self._file_manager_path = None
        self.file_tree.clearSelection()
        return self._set_file_manager_root(directory, force=True)

    def _open_file_manager_directory(self, directory):
        """Open a directory as the file-tree root, independent of the file."""
        return self._navigate_file_manager(directory, record=True, manual=True)

    def _open_file_manager_parent(self):
        """Move the file tree one directory up."""
        directory = self._file_manager_dir
        if directory is None:
            return
        parent = directory.parent
        if parent != directory and parent.is_dir():
            self._navigate_file_manager(parent, record=True, manual=True)

    def _file_manager_go_back(self):
        """Navigate to the previous directory in the file-manager history."""
        if not self._file_manager_back:
            return
        current = self._file_manager_dir
        target = self._file_manager_back.pop()
        if current is not None:
            self._file_manager_forward.append(current)
        self._navigate_file_manager(target, record=False, manual=True)
        self._update_file_manager_buttons()

    def _file_manager_go_forward(self):
        """Navigate to the next directory in the file-manager history."""
        if not self._file_manager_forward:
            return
        current = self._file_manager_dir
        target = self._file_manager_forward.pop()
        if current is not None:
            self._file_manager_back.append(current)
        self._navigate_file_manager(target, record=False, manual=True)
        self._update_file_manager_buttons()

    def _open_file_manager_recycle_bin(self):
        """Show the session recycle-bin directory in the file tree."""
        if not self.recycler.has_items():
            return
        self._navigate_file_manager(
            self.recycler.ensure(), record=True, manual=True
        )

    @staticmethod
    def _path_is_under(path, directory):
        """Return whether a file path is inside a directory tree."""
        if path is None or directory is None:
            return False
        try:
            Path(path).resolve().relative_to(Path(directory).resolve())
            return True
        except (OSError, ValueError):
            return False

    def _sync_file_manager(self, force=False):
        """Keep a containing root; switch roots only for unrelated files."""
        replace_root = False
        previous_root = self._file_manager_dir
        if self.active_key != self._file_manager_file_key:
            self._file_manager_file_key = self.active_key
            keep_root = self._path_is_under(
                self.active_key, previous_root
            )
            if not keep_root:
                replace_root = True
                self._file_manager_manual = False
                self._file_manager_back.clear()
                self._file_manager_forward.clear()

        if not replace_root and self._path_is_under(
            self.active_key, self._file_manager_dir
        ):
            self._set_file_manager_root(
                self._file_manager_dir,
                path=self.active_key,
                force=force,
            )
            return

        if not replace_root and self._file_manager_manual:
            directory = self._file_manager_dir
            if directory is not None and directory.is_dir():
                selected = (
                    self.active_key
                    if self._path_is_under(self.active_key, directory)
                    else None
                )
                self._set_file_manager_root(
                    directory, path=selected, force=force
                )
                return
            self._file_manager_manual = False

        path = self.active_key
        directory = path.parent if path is not None else None
        if directory is not None and directory.is_dir():
            self._set_file_manager_root(directory, path=path, force=force)
            self._file_manager_path = path
            return
        self.file_tree.hide()
        self.file_nav.hide()
        self.file_placeholder.show()
        self._file_manager_dir = None
        self._file_manager_path = None
        self._file_manager_back.clear()
        self._file_manager_forward.clear()
        self._update_file_manager_buttons()

    def _open_text_editor(self, path):
        """Open a file in the modeless plain-text editor."""
        path = Path(path).expanduser().resolve()
        existing = self._text_editors.get(path)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return existing
        try:
            size = path.stat().st_size
        except OSError as error:
            self._log(f"Text editor: {error}")
            return None
        if size > 50 * 1024 * 1024:
            answer = QMessageBox.warning(
                self,
                "Large file",
                f"{path.name} is larger than 50 MiB. "
                "Loading it into the text editor may be slow. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return None
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            QMessageBox.warning(self, "Text Editor", str(error))
            return None
        dialog = TextEditorDialog(
            path, text, self, before_write=self._backup_overwrite_target
        )
        self._text_editors[path] = dialog
        dialog.destroyed.connect(
            lambda _obj=None, path=path: self._text_editors.pop(path, None)
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def _file_tree_double_clicked(self, index):
        """Let directories expand, open files, and navigate through ``..``."""
        if not index.isValid():
            return
        if self.file_proxy.is_parent_row(index):
            self._open_file_manager_parent()
            return
        source_index = self._file_view_to_source(index)
        path = Path(self.file_model.filePath(source_index))
        if self.file_model.isDir(source_index):
            return
        try:
            self._open_file(path)
        except (ValueError, TypeError, RuntimeError) as error:
            self._log(f"Parse error: {error}")
            self._open_text_editor(path)
        except (ImportError, OSError) as error:
            self._log(f"Open error: {error}")
            QMessageBox.warning(self, "Open structure", self._open_error_message(error))

    def _path_has_open_state(self, path):
        """Return whether a file, or a directory containing files, is in use."""
        path = Path(path)
        for key, state in self.files.items():
            if key is None or state.closed:
                continue
            try:
                key.relative_to(path)
            except ValueError:
                continue
            if path.is_dir() or key == self.active_key or state.window_refs:
                return True
        return False

    def _on_file_renamed(self, directory, old_name, new_name):
        """Move a cached file-state key after an inline filesystem rename."""
        directory = Path(directory)
        old_path = directory / old_name
        new_path = directory / new_name
        state = self.files.pop(old_path, None)
        if state is not None and state.window_refs == 0 and not state.closed:
            state.path = new_path.resolve()
            self.files[state.path] = state
        self._sync_file_manager(force=True)

    def _drop_cached_paths(self, path):
        """Release cached states for a deleted file or directory."""
        path = Path(path)
        for key, state in list(self.files.items()):
            if key is None or state.window_refs or key == self.active_key:
                continue
            try:
                key.relative_to(path)
            except ValueError:
                continue
            self.files.pop(key, None)
            state.close()

    def _show_file_context_menu(self, position):
        """Offer file operations for the selected tree item."""
        index = self.file_tree.indexAt(position)
        if index.isValid() and self.file_proxy.is_parent_row(index):
            return
        selected = index.isValid()
        if not selected:
            path = self._file_manager_dir
        else:
            source_index = self._file_view_to_source(index)
            path = Path(self.file_model.filePath(source_index))
        if path is None:
            return
        menu = QMenu(self.file_tree)
        if selected and path.is_dir():
            menu.addAction(
                self._t("action.context.open"),
                lambda: self._open_file_manager_directory(path),
            )
            menu.addSeparator()
        elif selected and path.is_file():
            menu.addAction(
                self._t("action.context.open"),
                lambda: self._file_tree_double_clicked(index),
            )
            menu.addAction(
                self._t("action.context.edit"),
                lambda: self._open_text_editor(path),
            )
            menu.addSeparator()
        target = path.parent if selected else path
        menu.addAction(
            self._t("action.context.new_folder"),
            lambda: self._new_folder(target),
        )
        if selected:
            menu.addAction(
                self._t("action.context.rename"), lambda: self._rename_file(index)
            )
            menu.addAction(
                self._t("action.context.delete"), lambda: self._delete_file(path)
            )
        menu.addAction(
            self._t("action.context.copy_path"), lambda: self._copy_path(path)
        )
        menu.addSeparator()
        menu.addAction(
            self._t("action.context.refresh"),
            lambda: self._sync_file_manager(force=True),
        )
        menu.exec(self.file_tree.viewport().mapToGlobal(position))

    def _new_folder(self, directory):
        """Create one child folder in the selected directory."""
        directory = Path(directory)
        name, accepted = QInputDialog.getText(self, "New Folder", "Name")
        name = name.strip()
        if not accepted or not name or name in {".", ".."} or Path(name).name != name:
            return
        if not QDir(str(directory)).mkdir(name):
            QMessageBox.warning(self, "New Folder", f"Could not create {name}.")
            return
        self._sync_file_manager(force=True)

    def _rename_file(self, index):
        """Start the model's inline rename when the path is not in use."""
        source_index = self._file_view_to_source(index)
        path = Path(self.file_model.filePath(source_index))
        if self._path_has_open_state(path):
            QMessageBox.warning(
                self, "Rename", "Close the file before renaming it."
            )
            return
        self.file_tree.edit(index)

    def _delete_file(self, path):
        """Move an unused file or directory into the session recycle bin."""
        path = Path(path)
        if self._path_has_open_state(path):
            QMessageBox.warning(
                self, "Delete", "Close the file before deleting it."
            )
            return
        answer = QMessageBox.question(
            self,
            "Delete",
            f"Move {path.name} to the recycle bin?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.recycler.move(path)
        except OSError as error:
            QMessageBox.warning(
                self, "Delete", f"Could not move {path.name}: {error}"
            )
            return
        self._drop_cached_paths(path)
        self._sync_file_manager(force=True)
        self._update_file_manager_buttons()

    def _copy_path(self, path):
        """Copy one absolute path to the clipboard."""
        QApplication.clipboard().setText(str(Path(path)))
        self._log(f"Copied path: {path}")

    def _queue_frame_item_click(self, item):
        """Delay a single-click switch long enough to detect a double-click."""
        self._pending_frame_row = self.frame_list.row(item)
        self._frame_click_timer.start(QApplication.doubleClickInterval())

    def _commit_frame_click(self):
        """Switch the focused viewer after a click is confirmed as single."""
        if self._pending_frame_row is None:
            return
        row = self._pending_frame_row
        self._pending_frame_row = None
        if 0 <= row < len(self.coordinates):
            self._change_frame(row)

    def _open_frame_item(self, item):
        """Open the double-clicked frame in a new structure window."""
        self._frame_click_timer.stop()
        self._pending_frame_row = None
        self._open_frame_window(self.frame_list.row(item))

    @staticmethod
    def _file_signature(path):
        """Return the cache-validation signature of one structure file.

        Args:
            path: Source path, or ``None`` for untitled CLI data.

        Returns:
            A ``(st_mtime_ns, st_size)`` tuple, or ``None`` when the path
            cannot be stat-ed. The pair is captured before the file is read so
            a later modification always produces a mismatch.
        """
        if path is None:
            return None
        try:
            status = Path(path).stat()
        except OSError:
            return None
        return (int(status.st_mtime_ns), int(status.st_size))

    def _cache_limit_bytes(self):
        """Return the configured file-cache limit in bytes."""
        return max(64, int(self.settings["file_cache_limit_mb"])) * 1024 * 1024

    def _cache_total_bytes(self):
        """Estimate memory retained by every live file state."""
        return sum(
            state.nbytes()
            for state in self.files.values()
            if not state.closed
        )

    def _close_file_state(self, key):
        """Close one unreferenced, non-active file state and drop its key."""
        if key == self.active_key:
            return 0
        state = self.files.get(key)
        if state is None or state.closed or state.window_refs:
            return 0
        size = state.nbytes()
        self.files.pop(key, None)
        state.close()
        return size

    def _trim_file_cache(self):
        """Release the largest unreferenced caches while over the limit."""
        limit = self._cache_limit_bytes()
        total = self._cache_total_bytes()
        released = []
        while total > limit:
            candidates = [
                (state.nbytes(), key, state)
                for key, state in self.files.items()
                if key != self.active_key and not state.closed and state.window_refs == 0
            ]
            if not candidates:
                break
            size, key, _state = max(candidates, key=lambda item: item[0])
            freed = self._close_file_state(key)
            total = self._cache_total_bytes()
            released.append((key, freed))
        if released:
            freed = sum(size for _key, size in released)
            self._log(
                f"File cache: released {len(released)} file(s), "
                f"{freed / (1024 * 1024):.1f} MB."
            )
        return total

    def _on_file_released(self, state):
        """Release an unreferenced state immediately when the cache is full."""
        if state is None or state.closed or state.window_refs:
            return
        if self._cache_total_bytes() > self._cache_limit_bytes():
            self._trim_file_cache()

    def _clear_file_cache(self):
        """Manually close every unreferenced, non-active cached file."""
        removed = []
        for key, state in list(self.files.items()):
            if key == self.active_key or state.window_refs:
                continue
            size = state.nbytes()
            self.files.pop(key, None)
            state.close()
            removed.append((key, size))
        freed = sum(size for _key, size in removed)
        self._log(
            f"File cache cleared: {len(removed)} file(s), "
            f"{freed / (1024 * 1024):.1f} MB."
        )

    def _next_file_color(self):
        """Return a pastel border color for one newly opened file."""
        hue = (self._file_color_index * 0.618033988749895) % 1.0
        self._file_color_index += 1
        return QColor.fromHslF(hue, 0.45, 0.72)

    def _prepare_file_state(self, key, elements, coordinates, lattices, signature=None):
        """Build one detached FileState from freshly read raw frame lists.

        Args:
            key: Canonical source path, or None for untitled CLI data.
            elements: Frame-ordered element symbol arrays.
            coordinates: Frame-ordered Cartesian coordinate arrays.
            lattices: Frame-ordered lattice arrays.
            signature: File metadata captured before the read, used to
                invalidate this cache entry when the file changes on disk.

        Return:
            A detached state owning copies of the supplied frame lists.
        """
        state = FileState(key, self.settings["history_limit"])
        state.source_signature = (
            self._file_signature(key) if signature is None else signature
        )
        state.border_color = self._next_file_color() if key is not None else None
        state.elements = list(elements)
        state.coordinates = list(coordinates)
        state.lattices = list(lattices)
        state.frame_names = []
        state._frame_name_serial = 0
        for _ in state.elements:
            state._frame_name_serial += 1
            state.frame_names.append(f"Frame {state._frame_name_serial}")
        state.atom_frame_styles = [
            [dict() for _ in frame_elements] for frame_elements in state.elements
        ]
        state.frame_metrics = [
            {"length": {}, "angle": {}} for _ in state.coordinates
        ]
        state.bond_adjacencies = [None for _ in state.coordinates]
        state.bond_orders = [None for _ in state.coordinates]
        state.bond_frame_indices = []
        state.supercell_bases = {
            frame: {
                "elements": np.array(state.elements[frame], copy=True),
                "coordinates": np.array(state.coordinates[frame], copy=True),
                "lattice": np.array(state.lattices[frame], copy=True),
            }
            for frame in range(len(state.coordinates))
        }
        state.supercell_transforms = {
            frame: (np.ones(3, dtype=np.int64), np.eye(3, dtype=np.int64))
            for frame in range(len(state.coordinates))
        }
        state.current_index = 0
        return state

    def _file_title(self, state, frame_index):
        """Return the visible file-and-frame label for one window."""
        name = state.path.name if state.path is not None else "Untitled"
        return f"{name}[{int(frame_index) + 1}]"

    def _set_record_header(self, record):
        """Refresh one viewer's in-scene file/frame overlay."""
        state = self.files.get(record["file_key"])
        if state is None or state.closed:
            return
        record["viewer"].set_file_header(
            self._file_title(state, record["frame_index"]),
            str(state.path) if state.path is not None else "",
        )

    def _focus_record(self, record):
        """Raise and activate one existing MDI record without changing its frame."""
        record["subwindow"].show()
        self.mdi_area.setActiveSubWindow(record["subwindow"])
        self._activate_viewer(record["viewer"])

    def _create_viewer_for_state(self, state, key, frame_index, show=True):
        """Create one viewer bound to a specific file and frame."""
        frame_index = int(frame_index)
        existing = self.window_manager.find(key, frame_index)
        if existing is not None:
            self._focus_record(existing)
            return existing
        viewer = StructureViewer(self.mdi_area)
        source_viewer = self.viewer
        if source_viewer is None:
            active = self.window_manager.active_record()
            source_viewer = None if active is None else active["viewer"]
        for name in (
            "atom_styles", "atom_material", "atom_lighting", "atom_edges",
            "atom_edge_color", "atom_edge_width", "default_atom_style",
            "default_custom_radius", "default_atom_scale", "_background_color_value", "material_ambient",
            "material_diffuse", "material_specular", "material_roughness",
            "material_metallic", "axes_visible", "grid_visible", "bond_width",
            "bond_alpha", "bond_tolerance",
            "keyboard_translation_step", "perspective_enabled", "distinguish_files",
        ):
            if source_viewer is not None:
                setattr(viewer, name, deepcopy(getattr(source_viewer, name)))
        if source_viewer is None:
            self._configure_viewer(viewer)
        else:
            self._connect_extra_viewer(viewer)
            viewer.camera.SetParallelProjection(not viewer.perspective_enabled)
        # Section: the attribute copy above does not repaint the renderer, and a
        # fresh viewer always starts from the default background, so a new frame
        # window would otherwise ignore the chosen background colour.
        viewer.set_background(viewer._background_color_value)
        record = self.window_manager.add(
            viewer,
            frame_index,
            self._file_title(state, frame_index),
            show=show,
            file_key=key,
        )
        state.retain()
        self.window_manager.apply_file_colors()
        QApplication.processEvents()
        viewer.set_data(state.elements, state.coordinates, state.lattices)
        viewer.set_frame_state(
            state.atom_frame_styles, state.frame_metrics, state.bond_adjacencies,
            state.bond_orders,
        )
        if source_viewer is not None:
            viewer.bonds_enabled = bool(state.bond_frame_indices)
        viewer.bond_frame_indices = set(state.bond_frame_indices)
        viewer.current_index = frame_index
        viewer.set_frame(frame_index)
        self._set_record_header(record)
        QTimer.singleShot(0, viewer.render)
        return record

    def _open_frame_window(self, frame_index):
        """Create one unique viewer for one frame of the active file."""
        if not 0 <= int(frame_index) < len(self.elements):
            return
        frame_index = int(frame_index)
        existing = self.window_manager.find(self.active_key, frame_index)
        if existing is not None:
            self._focus_record(existing)
            return
        self._create_viewer_for_state(
            self._active_state, self.active_key, frame_index, show=True
        )

    def _show_frame_context_menu(self, position):
        """Show multi-frame Copy and Delete actions for the clicked frame."""
        item = self.frame_list.itemAt(position)
        if item is None:
            return
        row = self.frame_list.row(item)
        selected_rows = [index.row() for index in self.frame_list.selectedIndexes()]
        if row not in selected_rows:
            self.frame_list.clearSelection()
            item.setSelected(True)
            selected_rows = [row]
        menu = QMenu(self.frame_list)
        menu.addAction(
            self._t("action.context.open"),
            lambda: self._open_frame_window(row),
        )
        menu.addAction(
            self._t("action.context.copy"),
            lambda: self._duplicate_frames(selected_rows, row),
        )
        menu.addAction(
            self._t("action.context.delete"),
            lambda: self._delete_frames(selected_rows),
        )
        menu.addAction(
            self._t("action.context.rename"),
            lambda: self._rename_frame(row),
        )
        menu.exec(self.frame_list.mapToGlobal(position))

    def _rename_frame(self, frame_index):
        """Rename one frame without changing its data or history.

        Args:
            frame_index: Frame whose fixed display name is edited.

        Returns:
            None. Empty or cancelled names leave the current name unchanged.
        """
        if not 0 <= int(frame_index) < len(self.frame_names):
            return
        frame_index = int(frame_index)
        name, accepted = QInputDialog.getText(
            self,
            "Rename Frame",
            "Name",
            text=self.frame_names[frame_index],
        )
        name = name.strip()
        if not accepted or not name:
            return
        self.frame_names[frame_index] = name
        item = self.frame_list.item(frame_index)
        if item is not None:
            item.setText(f"{name}  ({len(self.elements[frame_index])} atoms)")

    def _append_frame(self, elements, coordinates, lattice, name=None):
        """Append one independent frame and all aligned auxiliary state."""
        self.elements.append(np.array(elements, copy=True))
        self.coordinates.append(np.array(coordinates, dtype=float, copy=True))
        self.lattices.append(np.array(lattice, dtype=float, copy=True))
        frame = len(self.elements) - 1
        self.atom_frame_styles.append([dict() for _ in self.elements[frame]])
        self.frame_metrics.append({"length": {}, "angle": {}})
        self.bond_adjacencies.append(None)
        self.bond_orders.append(None)
        if name is None:
            self._frame_name_serial += 1
            name = f"Frame {self._frame_name_serial}"
        self.frame_names.append(str(name))
        self.supercell_bases[frame] = {
            "elements": np.array(self.elements[frame], copy=True),
            "coordinates": np.array(self.coordinates[frame], copy=True),
            "lattice": np.array(self.lattices[frame], copy=True),
        }
        self.supercell_transforms[frame] = (
            np.ones(3, dtype=np.int64), np.eye(3, dtype=np.int64)
        )

    def _delete_frames(self, frame_indices):
        """Delete several frames with one aligned-list update and one redraw.

        Args:
            frame_indices: Zero-based frame indices to remove.

        Return:
            None. Windows of deleted frames close, the surviving workspace keeps
            displaying a valid frame, and a workspace that loses every frame is
            dropped so the next open reads the file again.
        """
        state = self._active_state
        key = self.active_key
        if state is None or state.closed:
            return
        indices = sorted(set(int(index) for index in frame_indices))
        indices = [index for index in indices if 0 <= index < len(state.elements)]
        if not indices:
            return
        deleted = set(indices)
        # Section: collect the doomed windows before touching any list. Closing
        # the displayed window switches the active workspace to the untitled
        # empty state when it was the last one, so shutdown happens only after
        # this state owns the edited data.
        closing = [
            record["subwindow"]
            for record in self.window_manager.records
            if record["file_key"] == key and record["frame_index"] in deleted
        ]
        retained = [index for index in range(len(state.elements)) if index not in deleted]
        for target in (
            state.elements,
            state.coordinates,
            state.lattices,
            state.atom_frame_styles,
            state.frame_metrics,
            state.bond_adjacencies,
            state.bond_orders,
            state.frame_names,
        ):
            target[:] = [target[index] for index in retained]
        old_bases = state.supercell_bases
        old_transforms = state.supercell_transforms
        state.supercell_bases = {new: old_bases[old] for new, old in enumerate(retained)}
        state.supercell_transforms = {
            new: old_transforms[old] for new, old in enumerate(retained)
        }
        state.bond_frame_indices = [
            index - sum(deleted_index < index for deleted_index in deleted)
            for index in state.bond_frame_indices
            if index not in deleted
        ]
        self.window_manager.remove_frames(indices, key)
        old_current = int(state.current_index)
        if not state.elements:
            state.current_index = 0
        elif old_current in deleted:
            state.current_index = min(old_current, len(state.elements) - 1)
        else:
            state.current_index = old_current - sum(
                deleted_index < old_current for deleted_index in deleted
            )
        for record in self.window_manager.records_for_file(key):
            record["viewer"].selected_indices = np.empty(0, dtype=int)
            record["viewer"].selected_bonds.clear()
        state.undo_stack.clear()
        state.redo_stack.clear()
        for subwindow in closing:
            subwindow.close()
        if not state.elements:
            # Section: the workspace lost every frame. Drop it instead of leaving
            # an empty one, so reopening the path reads the file again.
            if key is not None and self.files.get(key) is state:
                self.files.pop(key, None)
            state.close()
            if self._active_state is state:
                self._activate_empty_state()
            self._refresh_after_change(frame_index=0)
            self._sync_history_actions()
            return
        # Section: window shutdown may have activated another workspace. Restore
        # this file before refreshing so the new current frame gets a window and
        # every later update edits the state that actually lost frames.
        self._active_state = state
        self.active_key = key
        if key is None:
            self.files[None] = state
        self.current_index = int(state.current_index)
        record = self.window_manager.find(key, self.current_index)
        if record is None:
            self._create_viewer_for_state(state, key, self.current_index, show=True)
        elif record["viewer"] is not self.viewer:
            self._focus_record(record)
        self._refresh_after_change(frame_index=self.current_index)
        self._sync_history_actions()

    def closeEvent(self, event):
        """Confirm clearing the recycle bin before closing the main window."""
        self._python_runner.shutdown()
        if self.recycler.has_items():
            path = self.recycler.root
            answer = QMessageBox.question(
                self,
                "Recycle Bin",
                "There are files in the recycle bin "
                f"[path: {path}]. If you exit the program, they will be cleared. "
                "Continue to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            try:
                self.recycler.clear()
            except OSError as error:
                self._log(f"Recycle bin cleanup warning: {error}")
        event.accept()

    def _close_all_views(self):
        """Close every displayed structure window."""
        self.window_manager.close_all()

    def _show_delete_frames(self):
        """Prompt for one-based frame ranges and delete them in one operation."""
        if not self.elements:
            return
        dialog = FrameSelectionDialog(parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            indices = dialog.values(len(self.elements), self.current_index)
        except ValueError as error:
            QMessageBox.warning(self, "Delete Frames", str(error))
            return
        self._delete_frames(indices)

    def _duplicate_frames(self, frame_indices, anchor_index):
        """Copy frames with a reversible graph/structure index-space boundary.

        Args:
            frame_indices: Frames to duplicate in increasing index order.
            anchor_index: Insert the copies immediately after this frame.

        Return:
            None. Older frame-indexed history remains valid after Undo.
        """
        selected = sorted(set(int(index) for index in frame_indices))
        if not selected or not 0 <= int(anchor_index) < len(self.elements):
            return
        anchor_index = int(anchor_index)
        before_elements = list(self.elements)
        before_coordinates = list(self.coordinates)
        before_lattices = list(self.lattices)
        before_context = self._all_supercell_context()
        # Inserting frames shifts all later frame indices. Clear their graphs
        # through deltas before changing the frame list, then restore edges in
        # the new frame space. Undo reverses that boundary before older events.
        before_graphs = {
            frame: self.bond_adjacencies[frame]
            for frame in range(anchor_index + 1, len(self.elements))
        }
        before_orders = {
            frame: self.bond_orders[frame]
            for frame in range(anchor_index + 1, len(self.elements))
        }
        remove_bonds = BondChange.between(
            "Move copied frame bonds", before_graphs,
            {frame: None for frame in before_graphs},
            before_orders, {frame: None for frame in before_graphs},
        )
        snapshots = []
        for index in selected:
            snapshots.append(
                {
                    "elements": np.array(self.elements[index], copy=True),
                    "coordinates": np.array(self.coordinates[index], copy=True),
                    "lattice": np.array(self.lattices[index], copy=True),
                    "styles": deepcopy(self.atom_frame_styles[index]),
                    "metrics": deepcopy(self.frame_metrics[index]),
                    "bonds": None if self.bond_adjacencies[index] is None else np.array(self.bond_adjacencies[index], copy=True),
                    "orders": None if self.bond_orders[index] is None else np.array(self.bond_orders[index], dtype=np.int8, copy=True),
                    "name": f"{self.frame_names[index]}_copy",
                    "base": deepcopy(self.supercell_bases[index]),
                    "transform": deepcopy(self.supercell_transforms[index]),
                }
            )
        old_count = len(self.elements)
        active_before_insert = self.window_manager.active_record()
        insert_at = anchor_index + 1
        for offset, snapshot in enumerate(snapshots):
            position = insert_at + offset
            self.elements.insert(position, snapshot["elements"])
            self.coordinates.insert(position, snapshot["coordinates"])
            self.lattices.insert(position, snapshot["lattice"])
            self.atom_frame_styles.insert(position, snapshot["styles"])
            self.frame_metrics.insert(position, snapshot["metrics"])
            self.bond_adjacencies.insert(position, snapshot["bonds"])
            self.bond_orders.insert(position, snapshot["orders"])
            self.frame_names.insert(position, snapshot["name"])
        self.bond_frame_indices = [
            index + len(snapshots) if index >= insert_at else index
            for index in self.bond_frame_indices
        ]
        self.viewer.bond_frame_indices = set(self.bond_frame_indices)
        for record in self.window_manager.records_for_file(self.active_key):
            if record["frame_index"] >= insert_at:
                record["frame_index"] += len(snapshots)
        old_bases = self.supercell_bases
        old_transforms = self.supercell_transforms
        source_order = list(range(old_count))
        for offset in range(len(snapshots)):
            source_order.insert(insert_at + offset, -(offset + 1))
        self.supercell_bases = {
            new: (snapshots[-old - 1].get("base") if old < 0 else deepcopy(old_bases[old]))
            for new, old in enumerate(source_order)
        }
        self.supercell_transforms = {
            new: (snapshots[-old - 1].get("transform") if old < 0 else deepcopy(old_transforms[old]))
            for new, old in enumerate(source_order)
        }
        self.current_index = (
            int(active_before_insert["frame_index"])
            if active_before_insert is not None
            else insert_at + len(snapshots) - 1
        )
        after_graphs = {
            frame: self.bond_adjacencies[frame]
            for frame in range(insert_at, len(self.elements))
        }
        after_orders = {
            frame: self.bond_orders[frame]
            for frame in range(insert_at, len(self.elements))
        }
        add_bonds = BondChange.between(
            "Restore copied frame bonds", {frame: None for frame in after_graphs}, after_graphs,
            {frame: None for frame in after_graphs}, after_orders,
        )
        structure = StructureChange.for_batch(
            "Copy frames", before_elements, before_coordinates, before_lattices,
            self.elements, self.coordinates, self.lattices,
            before_context=before_context, after_context=self._all_supercell_context(),
        )
        entries = []
        if remove_bonds.frame_indices:
            entries.append(remove_bonds)
        entries.append(structure)
        if add_bonds.frame_indices:
            entries.append(add_bonds)
        self._record_change(tuple(entries), already_applied=True, frame_index=self.current_index)

    def _apply_frame_permutation(self, permutation, update_current=True):
        """Apply new[k] = old[perm[k]] to every frame-aligned list."""
        permutation = np.asarray(permutation, dtype=int)
        if sorted(permutation.tolist()) != list(range(len(self.elements))):
            raise ValueError("frame permutation is invalid")
        inverse = np.argsort(permutation)
        for target in (
            self.elements,
            self.coordinates,
            self.lattices,
            self.atom_frame_styles,
            self.frame_metrics,
            self.bond_adjacencies,
            self.bond_orders,
            self.frame_names,
        ):
            old = list(target)
            target[:] = [old[index] for index in permutation]
        old_bases = self.supercell_bases
        old_transforms = self.supercell_transforms
        self.supercell_bases = {new: old_bases[int(old)] for new, old in enumerate(permutation)}
        self.supercell_transforms = {
            new: old_transforms[int(old)] for new, old in enumerate(permutation)
        }
        self.window_manager.apply_permutation(inverse, self.active_key)
        if update_current and len(permutation):
            self.current_index = int(inverse[min(self.current_index, len(permutation) - 1)])
        self.bond_frame_indices = sorted(
            int(inverse[index])
            for index in self.bond_frame_indices
            if 0 <= index < len(inverse)
        )
        self.viewer.bond_frame_indices = set(self.bond_frame_indices)
        return inverse

    def _reorder_frames(self, permutation):
        """Record one undoable frame permutation from an internal drag."""
        if list(permutation) == list(range(len(self.elements))):
            return
        before_current = int(self.current_index)
        inverse = self._apply_frame_permutation(permutation)
        change = FramePermutationChange(
            "Reorder frames",
            tuple(int(index) for index in permutation),
            tuple(int(index) for index in inverse),
            before_current,
            int(self.current_index),
        )
        self._record_change(change, already_applied=True, frame_index=self.current_index)

    def _new_frame(self):
        """Append a validated user-created frame and display it."""
        dialog = NewFrameDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        elements, coordinates, lattice = dialog.values()
        self._append_frame(elements, coordinates, lattice)
        self.current_index = len(self.elements) - 1
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._refresh_after_change(frame_index=self.current_index, reset_camera=True)
        self._sync_history_actions()

    def _change_frame(self, index):
        """Display a valid frame and update the frame controls.

        Args:
            index: Zero-based frame index.

        Return:
            None. Invalid indices are reported without changing the current frame.
        """
        if not 0 <= index < len(self.coordinates):
            raise ValueError(f"frame index must be between 0 and {max(len(self.coordinates) - 1, 0)}")
        self._pause_playback()
        self._ensure_data_window()
        self.current_index = index
        active = self.window_manager.active_record()
        if active is not None:
            active["frame_index"] = int(index)
            self._set_record_header(active)
        self.viewer.set_frame(index)
        self._sync_frame_controls()
        self._sync_bond_action()

    def _on_frame_input(self):
        """Apply the frame text field when it contains a valid integer index."""
        try:
            index = int(self.frame_input.text().strip())
            self._change_frame(index)
        except (TypeError, ValueError):
            self._sync_frame_controls()
            self._log("Frame must be an integer within the displayed range.")

    def _step_frame(self, step):
        """Move one frame backward or forward without leaving the valid range."""
        if len(self.coordinates) == 0:
            return
        target = min(max(self.current_index + step, 0), len(self.coordinates) - 1)
        self._change_frame(target)

    def _toggle_playback_repeat(self):
        """Toggle the playback completion behavior between once and loop.

        Args:
            None.

        Return:
            None. Updates the menu button to show the active behavior.
        """
        self.playback_loop = not self.playback_loop
        self.playback_repeat_button.setText(
            self._t("action.repeat_loop" if self.playback_loop else "action.repeat_once")
        )

    def _toggle_playback(self, checked=False):
        """Pause active playback or start it with the menu settings.

        Args:
            checked: Unused Qt button state.

        Return:
            None. Invalid text is rejected with a warning dialog.
        """
        if self.timer.isActive():
            self._pause_playback()
            return

        frame_count = len(self.coordinates)
        if frame_count == 0:
            return
        try:
            start_text = self.playback_range_start.text().strip()
            end_text = self.playback_range_end.text().strip()
            stride_text = self.playback_stride.text().strip()
            start = 0 if not start_text else int(start_text)
            end = frame_count - 1 if not end_text else int(end_text)
            stride = int(stride_text)
            if not 0 <= start <= end < frame_count:
                raise ValueError(
                    f"range must satisfy 0 <= start <= end <= {frame_count - 1}"
                )
            maximum_stride = max(frame_count - 1, 1)
            if stride < 1 or (frame_count > 1 and stride >= frame_count):
                raise ValueError(f"stride must be between 1 and {maximum_stride}")
            indices = list(range(start, end + 1, stride))
            self._start_playback(indices, self.playback_fps_slider.value())
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "Playback", str(error))

    def _set_play_button_state(self, playing):
        """Show the play or pause icon for the current timer state.

        Args:
            playing: Whether playback is currently active.

        Return:
            None. Updates only the toolbar button presentation.
        """
        icon = (
            QStyle.StandardPixmap.SP_MediaPause
            if playing
            else QStyle.StandardPixmap.SP_MediaPlay
        )
        self.play_button.setIcon(self.style().standardIcon(icon))
        self.play_button.setToolTip(
            self._t("action.pause" if playing else "action.play")
        )
        color = "#ef5350" if playing else "#4caf50"
        hover = "#e53935" if playing else "#43a047"
        self.play_button.setStyleSheet(
            "QToolButton {"
            f"background-color: {color};"
            "border: 1px solid #455a64; padding: 3px;"
            "}"
            f"QToolButton:hover {{ background-color: {hover}; }}"
        )

    def _pause_playback(self):
        """Stop playback with the displayed frame as the authoritative index.

        Args:
            None.

        Return:
            None. Dynamic geometry is restored once for the displayed frame.
        """
        self.timer.stop()
        if self.viewer is not None and 0 <= self.viewer.current_index < len(self.coordinates):
            self.current_index = int(self.viewer.current_index)
            active = self.window_manager.active_record()
            if active is not None:
                active["frame_index"] = self.current_index
        if self.viewer is not None:
            self.viewer.finish_playback()
        self._set_play_button_state(False)
        self._sync_frame_controls()

    def _on_zoom_changed(self, value):
        """Apply the zoom control to the active MDI viewer."""
        viewer = self._active_viewer()
        if viewer is None:
            return
        if viewer is not self.viewer:
            self._activate_viewer(viewer)
        viewer.set_zoom_percent(value)

    def _change_zoom(self, delta):
        """Change the zoom spin box by one toolbar step."""
        self.zoom_input.setValue(self.zoom_input.value() + delta)

    def _add_toolbar_group(self, key, title):
        """Create one visually bounded toolbar group in the top area."""
        bar = QToolBar(title, self)
        bar.setObjectName(f"toolbar_{key}")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setStyleSheet(
            "QToolBar { border: 1px solid rgba(128, 128, 128, 0.55); "
            "border-radius: 3px; padding: 1px; margin: 1px; spacing: 2px; }"
        )
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)
        self.toolbar_groups[key] = bar
        return bar

    def _create_menus(self):
        """Create the localized main-window menu bar and its controls.

        Args:
            None.

        Return:
            None. Menu objects are retained so a language switch can update
            their titles and action text in place.
        """
        menu_bar = self.menuBar()
        self.file_menu = menu_bar.addMenu("File")
        self.new_action = self.file_menu.addAction("New...")
        self.new_action.setShortcut("Ctrl+N")
        self.new_action.triggered.connect(self._new_frame)
        self.open_action = self.file_menu.addAction("Open...")
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self._open_dialog)
        self.save_action = self.file_menu.addAction("Save As...")
        self.save_action.setShortcut("Ctrl+Shift+S")
        self.save_action.triggered.connect(self._save_dialog)
        self.export_action = self.file_menu.addAction("Export Batch...")
        self.export_action.triggered.connect(self._export_dialog)

        self.settings_menu = menu_bar.addMenu("Settings")
        self.preferences_action = self.settings_menu.addAction("Preferences...")
        self.preferences_action.triggered.connect(self._show_settings)
        self.file_manager_action = self.settings_menu.addAction("File Manager...")
        self.file_manager_action.triggered.connect(self._show_file_manager_settings)
        self.clean_backend_action = self.settings_menu.addAction("Clean Backend...")
        self.clean_backend_action.triggered.connect(self._show_clean_backend)
        self.clear_cache_action = self.settings_menu.addAction("Clear File Cache")
        self.clear_cache_action.triggered.connect(self._clear_file_cache)
        self.language_menu = self.settings_menu.addMenu("Language")
        self.language_group = QActionGroup(self)
        self.language_group.setExclusive(True)
        for code, name in available_languages():
            action = QAction(name, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked=False, code=code: self._set_language(code)
            )
            self.language_group.addAction(action)
            self.language_menu.addAction(action)
            self.language_actions[code] = action

        self.appearance_action = QAction("Appearance", self)
        self.appearance_action.triggered.connect(self._show_appearance)
        menu_bar.addAction(self.appearance_action)

        self.windows_menu = menu_bar.addMenu("Windows")
        self.show_files_action = self.file_dock.toggleViewAction()
        self.show_files_action.setText("Show Files")
        self.windows_menu.addAction(self.show_files_action)
        self.show_frames_action = self.frame_dock.toggleViewAction()
        self.show_frames_action.setText("Show Frames")
        self.windows_menu.addAction(self.show_frames_action)
        self.delete_action = self.windows_menu.addAction("Delete...")
        self.delete_action.triggered.connect(self._show_delete_frames)
        self.close_all_action = self.windows_menu.addAction(
            "Close All", self._close_all_views
        )

        self.view_menu = menu_bar.addMenu("View")
        self.along_actions = [
            self.view_menu.addAction("Along A", lambda: self._set_view(0)),
            self.view_menu.addAction("Along B", lambda: self._set_view(1)),
            self.view_menu.addAction("Along C", lambda: self._set_view(2)),
        ]
        self.view_menu.addSeparator()
        self.perspective_action = self.view_menu.addAction("Perspective")
        self.perspective_action.setCheckable(True)
        self.perspective_action.setChecked(False)
        self.perspective_action.toggled.connect(self._set_perspective_enabled)
        self.view_menu.addSeparator()
        zoom_controls = QWidget(self.view_menu)
        zoom_layout = QHBoxLayout(zoom_controls)
        zoom_layout.setContentsMargins(8, 4, 8, 4)
        zoom_layout.addWidget(self.zoom_minus_button)
        zoom_layout.addWidget(self.zoom_input)
        zoom_layout.addWidget(self.zoom_plus_button)
        zoom_action = QWidgetAction(self.view_menu)
        zoom_action.setDefaultWidget(zoom_controls)
        self.view_menu.addAction(zoom_action)
        self._localize_main_page()

    def _t(self, key):
        """Translate a visible main-window key through the active catalog."""
        return self.catalog.text(key)

    def _sync_language_actions(self):
        """Reflect the active catalog in the Settings > Language menu."""
        for code, action in self.language_actions.items():
            action.setChecked(code == self.catalog.code)

    def _set_language(self, code):
        """Switch the main-window language and persist the choice."""
        if code not in language_codes():
            return
        self.settings["language"] = code
        self.catalog.set_language(code)
        self._localize_main_page()
        self._persist_settings()

    def _localize_main_page(self):
        """Apply the current catalog to visible main-window controls."""
        self.setWindowTitle(self._t("window.title"))
        self.file_menu.setTitle(self._t("menu.file"))
        self.settings_menu.setTitle(self._t("menu.settings"))
        self.windows_menu.setTitle(self._t("menu.windows"))
        self.view_menu.setTitle(self._t("menu.view"))
        self.appearance_action.setText(self._t("menu.appearance"))
        self.language_menu.setTitle(self._t("menu.language"))
        self.new_action.setText(self._t("action.new"))
        self.open_action.setText(self._t("action.open"))
        self.save_action.setText(self._t("action.save_as"))
        self.export_action.setText(self._t("action.export"))
        self.preferences_action.setText(self._t("action.preferences"))
        self.file_manager_action.setText(self._t("action.file_manager"))
        self.clean_backend_action.setText(self._t("action.clean_backend"))
        self.clear_cache_action.setText(self._t("action.clear_cache"))
        self.show_files_action.setText(self._t("action.show_files"))
        self.show_frames_action.setText(self._t("action.show_frames"))
        self.delete_action.setText(self._t("action.delete"))
        self.close_all_action.setText(self._t("action.close_all"))
        for action, key in zip(
            self.along_actions,
            ("action.along_a", "action.along_b", "action.along_c"),
        ):
            action.setText(self._t(key))
        self.perspective_action.setText(self._t("action.perspective"))
        self._sync_language_actions()

        for key, bar in self.toolbar_groups.items():
            bar.setWindowTitle(self._t(f"toolbar.{key}"))
        self.undo_action.setText(self._t("action.undo"))
        self.undo_action.setToolTip(self._t("action.undo_tooltip"))
        self.redo_action.setText(self._t("action.redo"))
        self.redo_action.setToolTip(self._t("action.redo_tooltip"))
        self.copy_action.setText(self._t("action.copy"))
        self.copy_action.setToolTip(self._t("action.copy_tooltip"))
        self.paste_action.setText(self._t("action.paste"))
        self.paste_action.setToolTip(self._t("action.paste_tooltip"))
        self.supercell_action.setText(self._t("action.supercell_build"))
        self.supercell_button.setText(self._t("action.supercell"))
        self.add_atom_action.setText(self._t("action.add_atom"))
        self.replace_atom_action.setText(self._t("action.replace_atom"))
        self.clean_action.setText(self._t("action.clean"))
        self.clean_action.setToolTip(self._t("action.clean_tooltip"))
        self.hydrogen_action.setText(self._t("action.add_h"))
        self.hydrogen_action.setToolTip(self._t("action.add_h_tooltip"))
        self.bond_button.setText(self._t("action.bonds"))
        self.add_bond_action.setText(self._t("action.add_bond"))
        self.check_bonds_action.setText(self._t("action.check_bonds"))
        self.bond_action.setText(self._t("action.show_bonds"))
        self.clear_bonds_action.setText(self._t("action.clear_bonds"))
        self.bond_frames_label.setText(self._t("label.frames"))
        self.bond_frames_edit.setPlaceholderText(
            self._t("placeholder.current_or_all")
        )
        self.tolerance_spin.setPrefix(self._t("label.tolerance") + " ")
        self.pbc_wrap_button.setText(self._t("action.pbc_wrap"))
        self.pbc_frames_edit.setPlaceholderText(
            self._t("placeholder.current_or_all")
        )
        self.metrics_button.setText(self._t("action.metrics"))
        self.length_measure_action.setText(self._t("action.length"))
        self.angle_measure_action.setText(self._t("action.angle"))
        self.metric_style_action.setText(self._t("action.metric_style"))
        self.update_button.setText(self._t("action.update"))
        self.update_button.setToolTip(self._t("action.update_tooltip"))
        self._rebuild_update_menu()
        self.previous_frame_button.setToolTip(self._t("action.previous_frame"))
        self.next_frame_button.setToolTip(self._t("action.next_frame"))
        self.frame_label.setText(self._t("label.current_frame"))
        self.fps_caption.setText(self._t("label.fps"))
        self.range_caption.setText(self._t("label.range"))
        self.stride_caption.setText(self._t("label.stride"))
        self.playback_repeat_button.setText(
            self._t(
                "action.repeat_loop" if self.playback_loop else "action.repeat_once"
            )
        )
        self.zoom_minus_button.setToolTip(self._t("action.zoom_out"))
        self.zoom_input.setToolTip(self._t("action.zoom_percentage"))
        self.zoom_plus_button.setToolTip(self._t("action.zoom_in"))
        self._set_play_button_state(self.timer.isActive())
        self.command.setPlaceholderText(self._t("placeholder.command"))
        self.frame_dock.setWindowTitle(self._t("dock.frames"))
        self.file_dock.setWindowTitle(self._t("dock.files"))
        self.file_placeholder.setText(self._t("placeholder.file_manager"))
        self.file_up_button.setToolTip(self._t("action.file_manager_parent"))
        self.file_back_button.setToolTip(self._t("action.file_manager_back"))
        self.file_forward_button.setToolTip(
            self._t("action.file_manager_forward")
        )
        self.file_recycle_button.setToolTip(
            self._t("action.file_manager_recycle")
        )
        if not self.file_recycle_button.icon().isNull():
            self.file_recycle_button.setText("")
        else:
            self.file_recycle_button.setText(
                self._t("action.file_manager_recycle")
            )
        self.file_proxy.set_header_labels(
            {
                0: self._t("header.name"),
                1: self._t("header.size"),
                2: self._t("header.type"),
                3: self._t("header.modified"),
            }
        )

    def set_data(self, elements, coordinates, lattices):
        """Set the three raw frame-ordered array lists.

        Args:
            elements: Per-frame, per-atom element-symbol lists.
            coordinates: Per-frame Cartesian position arrays.
            lattices: Per-frame lattice matrices.

        Return:
            None. Validates and attaches the lists without wrapping or copying.
        """
        validate_data(elements, coordinates, lattices)
        self._pause_playback()
        self.play_indices = []
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.elements = elements
        self.coordinates = coordinates
        self.lattices = lattices
        self.frame_names = []
        self._frame_name_serial = 0
        self._ensure_frame_names()
        self._initialize_supercell_state()
        self._initialize_frame_state()
        self.current_index = 0
        self._ensure_data_window()
        self.bond_frame_indices = []
        blocker = QSignalBlocker(self.bond_action)
        self.bond_action.setChecked(False)
        del blocker
        self.cli.bind_data(self.elements, self.coordinates, self.lattices)
        self.viewer.bond_frame_indices = set()
        self.window_manager.refresh_all(reset_camera=True)
        blocker = QSignalBlocker(self.zoom_input)
        self.zoom_input.setValue(100)
        del blocker
        self._sync_frame_controls()
        self._sync_frame_list()
        self._sync_history_actions()
        self._sync_file_manager()
        self._sync_cli_scope(force=True)
        self._log(f"Loaded {len(elements)} frame(s), {sum(map(len, elements))} atoms total.")

    def _rebuild_update_menu(self):
        """Rebuild the always-available Update drop-down menu.

        Args:
            None.

        Return:
            None. Loaded functions are followed by More and Help actions.
        """
        if not hasattr(self, "update_menu"):
            return
        self.update_menu.clear()
        for updater in self.updaters:
            label = (
                self._t("updater.clean")
                if updater["name"] == "clean"
                else updater["name"]
            )
            action = self.update_menu.addAction(label)
            action.setToolTip(str(updater["path"]))
            action.triggered.connect(
                lambda checked=False, record=updater: self._run_updater_dialog(record)
            )
        if self.updaters:
            self.update_menu.addSeparator()
        more_action = self.update_menu.addAction(self._t("action.more"))
        more_action.triggered.connect(self._show_updater_dialog)
        help_action = self.update_menu.addAction(self._t("action.help"))
        help_action.triggered.connect(self._show_updater_help)

    def set_updater(self, updater):
        """Install one embedded callable using the updater protocol.

        Args:
            updater: Callable receiving ``cell``, ``elem``, and ``coo`` lists.

        Return:
            None. The callable is added to both the menu and compatibility slot.
        """
        if not callable(updater):
            raise TypeError("updater must be callable")
        name = getattr(updater, "__name__", "update") or "update"
        self.updaters = [
            record for record in self.updaters if record.get("configured", False)
        ]
        self.updaters.insert(
            0,
            {
                "path": "<embedded>",
                "name": name,
                "callback": updater,
                "configured": False,
            },
        )
        self.updaters.extend(_preset_updater_records())
        self.updater = updater
        self._rebuild_update_menu()

    def _show_updater_help(self):
        """Show configuration and function-signature guidance for plugins.

        Args:
            None.

        Return:
            None. A read-only help window is displayed modally.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Updater Plugin Help")
        dialog.resize(720, 520)
        text = QPlainTextEdit(dialog)
        text.setReadOnly(True)
        text.setPlainText(
            "Configure independent module and function lists in settings.json:\n\n"
            "{\n"
            '  "updater_path": ["updaters/relax.py"],\n'
            '  "updater_function": ["relax_atoms", "relax_cell"]\n'
            "}\n\n"
            "Paths may be absolute or relative to settings.json. All modules are loaded "
            "first, then each function name is looked up across them. Lists need not "
            "have equal lengths. Functions are grouped by their defining module "
            "using __module__ metadata; reexports of the same function are deduplicated. "
            "Distinct functions with the same name in different modules are all listed. "
            "Legacy single-string values remain supported.\n\n"
            "Required function protocol:\n\n"
            "def update(cell, elem, coo, *args, **kwargs):\n"
            "    # cell: List[np.ndarray], each (3, 3)\n"
            "    # elem: List[np.ndarray], each (N,)\n"
            "    # coo:  List[np.ndarray], each (N, 3) Cartesian Å\n"
            "    return new_cell, new_elem, new_coo\n\n"
            "Inputs are detached copies. Return three equally sized lists of NumPy "
            "arrays. Shapes must be valid, values finite, and dtypes compatible with "
            "the input. Each selected source frame is passed separately as three "
            "singleton lists. Apply to uses one-based ranges (1,2-5), all, or empty "
            "for the focused frame; its default is that frame's one-based number. "
            "Indices are sorted and deduplicated. Apply appends results in this "
            "order without an Undo/Redo entry. Check in-place to replace selected "
            "frames (exactly one output per input), with one sparse undo step. "
            "Same atom count preserves bonds; changed atom count rebuilds an "
            "existing graph from the complete distance matrix. Uncomputed graphs "
            "remain uncomputed. All calls are validated before committing data. "
            "Dry Run uses the same scope and cardinality rules without changing "
            "data. args accepts a Python literal value, "
            "tuple, or list; kwargs accepts a dictionary literal."
        )
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
        buttons.rejected.connect(dialog.reject)
        layout = QVBoxLayout(dialog)
        layout.addWidget(text)
        layout.addWidget(buttons)
        dialog.exec()

    def _show_updater_dialog(self):
        """Open the detailed updater tree and operation controls.

        Args:
            None.

        Return:
            None. The modal dialog operates on the current plugin collection.
        """
        self._pause_playback()
        dialog = UpdaterDialog(self.updaters, self)
        dialog.apply_button.clicked.connect(
            lambda: self._apply_selected_updater(dialog)
        )
        dialog.help_button.clicked.connect(
            lambda: self._show_selected_updater_help(dialog)
        )
        dialog.dry_run_button.clicked.connect(
            lambda: self._dry_run_selected_updater(dialog)
        )
        dialog.refresh_button.clicked.connect(
            lambda: self._refresh_updater_dialog(dialog)
        )
        dialog.exec()

    def _updater_arguments_from_dialog(self, dialog):
        """Parse literal positional and keyword arguments from the dialog.

        Args:
            dialog: Active :class:`UpdaterDialog` instance.

        Return:
            A positional-argument tuple and keyword-argument dictionary.
        """
        args_text = dialog.args_edit.text().strip()
        kwargs_text = dialog.kwargs_edit.text().strip()
        if args_text:
            args_value = ast.literal_eval(args_text)
            if isinstance(args_value, (list, tuple)):
                args = tuple(args_value)
            else:
                args = (args_value,)
        else:
            args = ()
        if kwargs_text:
            kwargs = ast.literal_eval(kwargs_text)
            if not isinstance(kwargs, dict) or any(
                not isinstance(key, str) for key in kwargs
            ):
                raise ValueError("kwargs must be a dictionary with string keys")
        else:
            kwargs = {}
        return args, kwargs

    def _apply_selected_updater(self, dialog):
        """Apply the function selected in the detailed updater dialog.

        Args:
            dialog: Active :class:`UpdaterDialog` instance.

        Return:
            None. The inline scope and in-place controls determine the operation.
        """
        updater = dialog.selected_updater()
        if updater is None:
            return
        try:
            args, kwargs = self._updater_arguments_from_dialog(dialog)
        except (SyntaxError, TypeError, ValueError) as error:
            self._log(f"Updater argument error: {error}")
            return
        self._run_updater(
            updater, dialog.scope_edit.text(), args, kwargs,
            in_place=dialog.in_place_check.isChecked(),
        )

    def _dry_run_selected_updater(self, dialog):
        """Execute and validate a selected updater without appending frames.

        Args:
            dialog: Active :class:`UpdaterDialog` instance.

        Return:
            None. Warnings, errors, and captured output are sent to the console.
        """
        updater = dialog.selected_updater()
        if updater is None:
            return
        try:
            args, kwargs = self._updater_arguments_from_dialog(dialog)
            selected, outputs = self._evaluate_updater(
                updater, dialog.scope_edit.text(), args, kwargs,
                in_place=dialog.in_place_check.isChecked(),
            )
            returned = sum(len(result[0]) for result in outputs)
            self._log(
                f"Dry Run succeeded: {updater['name']} "
                f"({len(selected)} source frame(s), {returned} output frame(s))."
            )
        except Exception as error:
            self._log(f"Dry Run error [{updater['name']}]: {error}")
            self._log(traceback.format_exc().rstrip())

    def _evaluate_updater(self, updater, specification, args=(), kwargs=None, in_place=False):
        """Run an updater once per selected frame and detach each result.

        Args:
            updater: Loaded updater record.
            specification: Standard one-based frame selector text.
            args: Extra positional plugin arguments.
            kwargs: Extra keyword plugin arguments.
            in_place: Require exactly one returned frame per input frame.

        Return:
            ``(selected_indices, outputs)`` where each output is a detached
            ``(cell_list, elem_list, coo_list)`` from one source frame. The
            standard selector sorts and deduplicates indices before execution.
        """
        if not self.coordinates:
            raise ValueError("no structure data is loaded")
        selected = parse_frame_scope(
            specification or "", len(self.coordinates), self.current_index
        )
        outputs = []
        for frame in selected:
            try:
                result = self._invoke_updater(updater, str(frame + 1), args, kwargs)
                if in_place and len(result[0]) != 1:
                    raise ValueError("in-place update must return exactly one frame")
                # Detach immediately: a module may reuse the same output buffer
                # on its next call. Do not mutate GUI state until ALL calls pass.
                outputs.append(tuple(
                    [np.array(value, copy=True) for value in values]
                    for values in result
                ))
            except Exception as error:
                raise RuntimeError(f"source frame {frame + 1}: {error}") from error
        return selected, outputs

    def _show_selected_updater_help(self, dialog):
        """Print Python built-in help for the selected updater function.

        Args:
            dialog: Active :class:`UpdaterDialog` instance.

        Return:
            None. The generated help text is appended to the GUI console.
        """
        updater = dialog.selected_updater()
        if updater is None:
            return
        stream = io.StringIO()
        with redirect_stdout(stream):
            help(updater["callback"])
        help_text = stream.getvalue().rstrip()
        self._log(help_text or f"No help is available for {updater['name']}.")

    def _refresh_updater_dialog(self, dialog):
        """Reload updater configuration and refresh both menus and tree view.

        Args:
            dialog: Active :class:`UpdaterDialog` instance.

        Return:
            None. Valid configured plugins replace the previous configured set.
        """
        refreshed = self._load_settings()
        self.settings["updater_path"] = refreshed["updater_path"]
        self.settings["updater_function"] = refreshed["updater_function"]
        self._load_configured_updaters()
        dialog.set_updaters(self.updaters)
        if self.settings_warning is not None:
            self._log(self.settings_warning)
        if self.updater_warning is not None:
            self._log(self.updater_warning)
        self._log(f"Updater refresh loaded {len(self.updaters)} function(s).")

    def _run_updater_dialog(self, updater, args=(), kwargs=None):
        """Ask for a one-based frame scope and run one updater record.

        Args:
            updater: Loaded updater record containing its callback and name.
            args: Extra positional arguments appended after structure lists.
            kwargs: Extra keyword arguments appended after structure lists.

        Return:
            None. Accepted input is passed to the normal updater operation.
        """
        self._pause_playback()
        specification, accepted = QInputDialog.getText(
            self,
            f"Run Updater - {updater['name']}",
            "Frames (empty=current, all, or 1,2-5):",
            text=str(self.current_index + 1),
        )
        if accepted:
            self._run_updater(updater, specification, args, kwargs or {})

    def _run_default_updater(self, specification=None):
        """Run the first loaded updater for the legacy console command.

        Args:
            specification: Optional current/all/one-based frame scope.

        Return:
            None. Errors are written to the GUI console.
        """
        if not self.updaters:
            self._log("Updater error: no structure updater is configured")
            return
        self._run_updater(self.updaters[0], specification, (), {})

    def _run_updater(self, updater, specification=None, args=(), kwargs=None, in_place=False):
        """Invoke one updater record and report failures in the console.

        Args:
            updater: Loaded updater record containing its callback and name.
            specification: Optional current/all/one-based frame scope.
            args: Extra positional arguments for the plugin.
            kwargs: Extra keyword arguments for the plugin.
            in_place: Update selected frames instead of appending results.

        Return:
            None. Successful results are committed in the requested mode.
        """
        kwargs = {} if kwargs is None else dict(kwargs)
        if updater.get("preset") and updater.get("name") == "clean":
            kwargs = {**self._clean_option_kwargs(), **kwargs}
        try:
            self.apply_updater(
                specification,
                updater=updater["callback"],
                args=args,
                kwargs=kwargs,
                name=updater["name"],
                in_place=in_place,
            )
        except Exception as error:
            self._log(f"Updater error [{updater['name']}]: {error}")
            self._log(traceback.format_exc().rstrip())

    def _invoke_updater(self, updater, specification, args=(), kwargs=None):
        """Execute one updater against detached selected-frame inputs.

        Args:
            updater: Loaded updater record containing its callback and name.
            specification: Current/all/one-based frame scope text.
            args: Extra positional arguments for the plugin.
            kwargs: Extra keyword arguments for the plugin.

        Return:
            Validated ``(new_cell, new_elem, new_coo)`` lists.
        """
        if not self.coordinates:
            raise ValueError("no structure data is loaded")
        if not 0 <= self.current_index < len(self.coordinates):
            raise ValueError("the current frame index is invalid")
        selected = parse_frame_scope(
            specification or "",
            len(self.coordinates),
            self.current_index,
        )
        selected_cell = [self.lattices[index] for index in selected]
        selected_elem = [self.elements[index] for index in selected]
        selected_coo = [self.coordinates[index] for index in selected]
        stdout_stream = io.StringIO()
        stderr_stream = io.StringIO()
        captured_warnings = []
        try:
            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter("always")
                with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
                    result = apply_structure_updater(
                        updater["callback"],
                        selected_cell,
                        selected_elem,
                        selected_coo,
                        *tuple(args),
                        **dict(kwargs or {}),
                    )
        finally:
            standard_output = stdout_stream.getvalue().rstrip()
            standard_error = stderr_stream.getvalue().rstrip()
            if standard_output:
                self._log(standard_output)
            if standard_error:
                self._log(standard_error)
            for warning in captured_warnings:
                self._log(
                    f"Updater warning [{updater['name']}]: "
                    f"{warning.category.__name__}: {warning.message}"
                )
        return result

    def apply_updater(
        self,
        specification=None,
        updater=None,
        args=(),
        kwargs=None,
        name=None,
        in_place=False,
    ):
        """Run an updater per selected frame, then append or update in place.

        ``specification`` uses the one-based inclusive frame syntax accepted by
        :func:`parse_frame_scope`; an empty value selects the current frame and
        ``all`` selects every frame. The callback receives detached lists in
        ``(cell, elem, coo)`` order, with one frame in each input list. Append
        mode accepts any output count; in-place mode requires one per input.

        Args:
            specification: Optional current/all/one-based frame scope.
            updater: Optional callback overriding the default updater.
            args: Extra positional arguments forwarded after structure lists.
            kwargs: Extra keyword arguments forwarded to the callback.
            name: Optional display and history name for the callback.
            in_place: Replace selected frames with one sparse undo event.

        Return:
            Validated output lists, flattened in source-frame/output-frame order.
        """
        if callable(specification) and updater is None:
            updater = specification
            specification = None
        callback = self.updater if updater is None else updater
        if callback is None:
            raise ValueError("no structure updater is configured")
        if self.timer.isActive():
            self._pause_playback()
        updater_name = name or getattr(callback, "__name__", "update") or "update"
        record = {
            "path": "<direct>",
            "name": updater_name,
            "callback": callback,
            "configured": False,
        }
        selected, outputs = self._evaluate_updater(
            record, specification or "", args, kwargs, in_place=in_place,
        )
        new_cell = [value for result in outputs for value in result[0]]
        new_elem = [value for result in outputs for value in result[1]]
        new_coo = [value for result in outputs for value in result[2]]
        if in_place:
            # Section: Stage sparse fields and topology-dependent graphs before
            # committing anything. One callback failure/rebuild failure leaves
            # all selected frames and both history queues untouched.
            self._ensure_frame_state()
            self._ensure_supercell_state()
            snapshots = [[], [], [], []]
            replacements = [[], [], [], []]
            graph_changed, topology_frames, contexts = [], [], {}
            new_contexts = {}
            changed_frames = []
            for slot, frame in enumerate(selected):
                old_values = (
                    self.lattices[frame], self.elements[frame], self.coordinates[frame]
                )
                new_values = (new_cell[slot], new_elem[slot], new_coo[slot])
                frame_changed = False
                for field, (old, new) in enumerate(zip(old_values, new_values)):
                    changed = (
                        old.shape != new.shape or old.dtype != new.dtype
                        or not np.array_equal(old, new)
                    )
                    snapshots[field].append(np.array(old, copy=True) if changed else None)
                    replacements[field].append(new if changed else None)
                    frame_changed = frame_changed or changed
                topology_changed = len(old_values[1]) != len(new_values[1])
                rebuild = topology_changed and self.bond_adjacencies[frame] is not None
                graph_changed.append(rebuild)
                snapshots[3].append(
                    np.array(self.bond_adjacencies[frame], copy=True) if rebuild else None
                )
                try:
                    replacements[3].append(
                        build_bond_adjacency(
                            new_coo[slot], new_elem[slot],
                            getattr(self.viewer, "bond_tolerance", 1.2),
                        ) if rebuild else None
                    )
                except Exception as error:
                    raise RuntimeError(f"source frame {frame + 1}, bond rebuild: {error}") from error
                if frame_changed:
                    changed_frames.append(frame)
                    # Saved context objects are moved out of live state on
                    # commit. No unchanged structure arrays are copied into
                    # history, including for a cell-only updater.
                    contexts[frame] = {
                        "base": self.supercell_bases[frame],
                        "transform": self.supercell_transforms[frame],
                    }
                    # Rebase the updated frame before future supercell edits.
                    # Base arrays are read-only in this workflow (base edits
                    # replace the dictionary). Reuse identical components so
                    # a cell-only update does not duplicate unchanged atoms.
                    old_base = self.supercell_bases[frame]
                    new_base = {}
                    for key, value in zip(("lattice", "elements", "coordinates"), new_values):
                        previous = old_base[key]
                        identical = (
                            previous.shape == value.shape and previous.dtype == value.dtype
                            and np.array_equal(previous, value)
                        )
                        new_base[key] = previous if identical else np.array(value, copy=True)
                    new_contexts[frame] = {
                        "base": new_base,
                        "transform": (np.ones(3, dtype=np.int64), np.eye(3, dtype=np.int64)),
                    }
                if topology_changed:
                    topology_frames.append(frame)
                    contexts[frame]["styles"] = self.atom_frame_styles[frame]
                    contexts[frame]["metrics"] = self.frame_metrics[frame]
                    new_contexts[frame]["styles"] = [dict() for _ in new_elem[slot]]
                    new_contexts[frame]["metrics"] = {"length": {}, "angle": {}}
                if rebuild:
                    contexts[frame]["orders"] = self.bond_orders[frame]
                    new_contexts[frame]["orders"] = np.ones(
                        len(replacements[3][slot][0]), dtype=np.int8
                    )
            if changed_frames:
                # Four-field format: wholly unchanged fields collapse to None;
                # mixed fields keep aligned None slots, never dummy array copies.
                snapshot = tuple(
                    values if any(value is not None for value in values) else None
                    for values in snapshots
                )
                change = InplaceUpdaterChange(
                    f"Updater in-place: {updater_name}", tuple(selected), snapshot,
                    tuple(graph_changed), tuple(topology_frames), contexts,
                )
                for slot, frame in enumerate(selected):
                    for field, target in enumerate((
                        self.lattices, self.elements, self.coordinates, self.bond_adjacencies,
                    )):
                        if replacements[field][slot] is not None:
                            target[frame] = replacements[field][slot]
                    if frame in new_contexts:
                        context = new_contexts[frame]
                        self.supercell_bases[frame] = context["base"]
                        self.supercell_transforms[frame] = context["transform"]
                        if "styles" in context:
                            self.atom_frame_styles[frame] = context["styles"]
                            self.frame_metrics[frame] = context["metrics"]
                        if "orders" in context:
                            self.bond_orders[frame] = context["orders"]
                self._record_change(change, already_applied=True)
            self._log(f"Updater changed {len(changed_frames)} frame(s) in-place: {updater_name}.")
        else:
            # Section: Appended updater frames intentionally have NO history
            # entry. Copy only new data; never snapshot the existing batch.
            first_new = len(self.coordinates)
            for cell, elem, coo in zip(new_cell, new_elem, new_coo):
                self._append_frame(elem, coo, cell)
            if new_coo:
                self.redo_stack.clear()
                self._refresh_after_change(
                    changed_frame_indices=tuple(range(first_new, len(self.coordinates)))
                )
                self._sync_history_actions()
            self._log(f"Updater appended {len(new_coo)} frame(s): {updater_name}.")
        return new_cell, new_elem, new_coo

    # Explicit alias for embedding code that calls the operation an update.
    update_data = apply_updater

    def register_property_renderer(self, name, renderer):
        """Register a callable that adds one optional PyVista overlay.

        Args:
            name: Property name used as the registration key.
            renderer: Callable receiving ``(plotter, frame_index, positions)``.

        Return:
            None. Makes the renderer run whenever the viewer draws a frame.
        """
        self.viewer.property_renderers[name] = renderer
        if len(self.coordinates) > 0:
            self.viewer.draw_frame()

    def _open_dialog(self):
        """Open a structure file through a Qt file dialog.

        Args:
            None.

        Return:
            None. Loads selected data or does nothing when canceled.
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open structures",
            "",
            "Structures (*.vasp *.poscar POSCAR *.cif *.xyz *.extxyz OUTCAR *.outcar);;All files (*)",
        )
        if paths:
            try:
                self._open(paths)
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                self._log(f"Open error: {error}")
                QMessageBox.warning(self, "Open structure", self._open_error_message(error))

    @staticmethod
    def _open_error_message(error):
        """Choose the user-facing message for a failed external structure read.

        Args:
            error: Exception raised by the configured input handler.

        Returns:
            A concise diagnostic, including the required unknown-format text.
        """
        if isinstance(error, RuntimeError) and "BUCToolkit" in str(error):
            return str(error)
        return (
            "Unknown file format! Please check your input file. "
            "Support cif, xyz, vasp files."
        )

    def _open_dropped_files(self, paths):
        """Load local files dropped onto the viewer.

        Args:
            paths: Local file paths emitted by StructureViewer.

        Returns:
            None. Errors are logged and shown without exposing a Qt traceback.
        """
        try:
            self._open(paths)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            self._log(f"Open error: {error}")
            QMessageBox.warning(self, "Open structure", self._open_error_message(error))

    def dragEnterEvent(self, event):
        """Accept local files dropped on the main window frame.

        Args:
            event: Qt drag-enter event.

        Returns:
            None. Local URL drops are accepted for loading.
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event):
        """Load local files dropped outside the viewer child widget.

        Args:
            event: Qt drop event.

        Returns:
            None. Non-local drops are delegated to Qt.
        """
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self._open_dropped_files(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def _save_dialog(self):
        """Save the current raw lists through a Qt file dialog.

        Args:
            None.

        Return:
            None. Saves selected path or does nothing when canceled.
        """
        default_path = str(self.active_key) if self.active_key is not None else ""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save current frame",
            default_path,
            "POSCAR (*.vasp);;CIF (*.cif);;XYZ (*.xyz);;All files (*)",
        )
        if path:
            try:
                self._save(path)
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                self._log(f"Save error: {error}")

    def _export_dialog(self):
        """Open the batch frame-selection and text-format export dialog."""
        if self.save_handler is None or not hasattr(self.save_handler, "export_batch"):
            self._log("Batch export is unavailable for the configured save handler.")
            return
        if len(self.coordinates) == 0:
            self._log("Open structure data before exporting.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Export batch")
        form = QFormLayout(dialog)
        selection = QLineEdit(f"1-{len(self.coordinates)}", dialog)
        selection.setPlaceholderText("1-2, 4, 6,8,10-30")
        format_combo = QComboBox(dialog)
        format_combo.addItem("POSCAR", "POSCAR")
        format_combo.addItem("CIF", "cif")
        format_combo.addItem("XYZ", "xyz")
        output_button = QPushButton("Choose output directory", dialog)
        output_path = {"value": ""}
        output_button.clicked.connect(
            lambda: output_path.update(
                value=QFileDialog.getExistingDirectory(dialog, "Output directory")
            )
        )
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow("Frames (1-based)", selection)
        form.addRow("Format", format_combo)
        form.addRow(output_button)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted or not output_path["value"]:
            return
        try:
            indices = parse_frame_selection(selection.text(), len(self.coordinates))
            output = Path(output_path["value"])
            if output.is_dir():
                for existing in output.iterdir():
                    if existing.is_file():
                        self._backup_overwrite_target(existing)
            self.save_handler.export_batch(
                output_path["value"],
                indices,
                format_combo.currentData(),
                self.elements,
                self.coordinates,
                self.lattices,
            )
            self._log(f"Exported {len(indices)} frame(s) to {output_path['value']}")
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            self._log(f"Export error: {error}")

    def _open_file(self, path):
        """Open one file in a new window, or focus its existing window."""
        key = Path(path).expanduser().resolve()
        if key != self.active_key and self.timer.isActive():
            self._pause_playback()
        records = self.window_manager.records_for_file(key)
        if records:
            active = self.window_manager.active_record()
            target = active if active in records else records[-1]
            self._focus_record(target)
            return target
        state = self.files.get(key)
        previous = state
        # Section: reuse a cached state only while its file metadata matches.
        # The signature is captured before the read, so a file modified after
        # caching (or by another program) is reloaded instead of served stale.
        stale = (
            state is not None
            and not state.closed
            and state.window_refs == 0
            and state.source_signature != self._file_signature(key)
        )
        if state is None or state.closed or stale:
            signature = self._file_signature(key)
            elements, coordinates, lattices = self.open_handler([key])
            validate_data(elements, coordinates, lattices)
            state = self._prepare_file_state(
                key, elements, coordinates, lattices, signature
            )
            self.files[key] = state
            if stale:
                self._log(f"File cache: {key.name} changed on disk; reloaded.")
        record = self._create_viewer_for_state(state, key, 0, show=True)
        self.active_key = key
        self._active_state = state
        self.current_index = 0
        state.current_index = 0
        # Section: release the stale state only after the replacement owns the
        # file key and the active references, so no callback can observe a
        # closed state through the window manager.
        if previous is not None and previous is not state and not previous.closed:
            previous.close()
        self._trim_file_cache()
        self._activate_viewer(record["viewer"])
        self._log(
            f"Opened {key.name}: {len(state.elements)} frame(s), "
            f"{sum(map(len, state.elements))} atoms total."
        )
        return record

    def _open(self, paths):
        """Open selected files in separate windows and focus the last one."""
        if self.open_handler is None:
            raise RuntimeError("file opening is not configured; provide an open_handler")
        self._pause_playback()
        self.timer.stop()
        self.play_indices = []
        files = normalize_structure_files(paths)
        last_record = None
        for path in files:
            last_record = self._open_file(path) or last_record
        if last_record is not None:
            self._focus_record(last_record)
        self.cli.bind_data(self.elements, self.coordinates, self.lattices)

    def _backup_overwrite_target(self, path):
        """Copy an existing regular file into the recycle bin before writing."""
        path = Path(path)
        if not path.is_file():
            return None
        try:
            return self.recycler.backup_copy(path)
        finally:
            self._update_file_manager_buttons()

    def _save(self, path):
        """Save only the currently displayed frame to a file.

        Args:
            path: Destination passed to the configured saving handler.

        Return:
            None. Logs the saved path after writing one frame.
        """
        if len(self.coordinates) == 0:
            raise ValueError("there is no structure data to save")
        if self.save_handler is None:
            raise RuntimeError("file saving is not configured; provide a save_handler")
        frame_index = self.current_index
        if not 0 <= frame_index < len(self.coordinates):
            raise ValueError("the current frame index is invalid")
        self._backup_overwrite_target(path)
        self.save_handler(
            path,
            [self.elements[frame_index]],
            [self.coordinates[frame_index]],
            [self.lattices[frame_index]],
        )
        # Section: saving onto the active file rewrites its bytes, so refresh
        # the cache signature; otherwise the next reopen would drop the frames
        # that are held only in memory. The workspace itself is not split.
        target = Path(path).expanduser().resolve()
        if self.active_key is not None and target == self.active_key:
            self._active_state.source_signature = self._file_signature(self.active_key)
        self._log(f"Saved {path}")

    def _play(self, specification, fps):
        """Start timer-based playback over the requested frame slice.

        Args:
            specification: Optional Python-style ``start:stop[:step]`` slice.
            fps: Playback rate in frames per second.

        Return:
            None. Starts the timer at the requested rate.
        """
        if len(self.coordinates) == 0:
            raise ValueError("open structure data before playback")
        indices = (
            parse_frame_range(specification, len(self.coordinates))
            if specification is not None
            else list(range(len(self.coordinates)))
        )
        if not indices:
            raise ValueError("frame range is empty")
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._start_playback(indices, fps)

    def _start_playback(self, indices, fps):
        """Start playback over explicit valid frame indices.

        Args:
            indices: Ordered frame indices to display.
            fps: Requested timer rate in frames per second.

        Return:
            None. Playback begins at the current or next included frame.
        """
        if not indices:
            raise ValueError("frame range is empty")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if any(index < 0 or index >= len(self.coordinates) for index in indices):
            raise ValueError("playback contains an out-of-range frame")
        self.play_indices = list(indices)
        self.play_position = next(
            (
                position
                for position, index in enumerate(self.play_indices)
                if index >= self.current_index
            ),
            0,
        )
        self.viewer.begin_playback()
        self.timer.start(max(1, round(1000 / fps)))
        self._set_play_button_state(True)
        self._sync_frame_controls()
        self._advance_frame()

    def _advance_frame(self):
        """Advance to the next frame in the active playback sequence.

        Args:
            None.

        Return:
            None. Updates viewer and counter, then stops or loops at the end.
        """
        if not self.play_indices:
            self._pause_playback()
            return
        index = self.play_indices[self.play_position]
        if not self.viewer.update_frame_preview(index):
            self.viewer.finish_playback()
            self.current_index = index
            self.viewer.set_frame(index)
            self.viewer.begin_playback()
        else:
            self.current_index = index
        active = self.window_manager.active_record()
        if active is not None:
            active["frame_index"] = int(index)
            self._set_record_header(active)
        self._sync_frame_controls()
        self._sync_bond_action()
        if self.play_position == len(self.play_indices) - 1:
            if self.playback_loop:
                self.play_position = 0
            else:
                self._pause_playback()
        else:
            self.play_position += 1

    def _on_external_python_finished(self, returncode):
        """Report an external script result after its worker thread exits."""
        self._log(f"External Python exited with code {returncode}.")

    def _run_external_python(self, interpreter, path, args):
        """Run an external Python file and show a stop dialog after 0.2 s."""
        if self._python_runner.is_running():
            self._log("An external Python script is already running.")
            return
        try:
            started = self._python_runner.start(
                interpreter, path, args, cwd=Path.cwd()
            )
        except OSError as error:
            self._log(f"External Python error: {error}")
            return
        if not started:
            return
        self.setEnabled(False)
        try:
            finished = self._python_runner.wait(0.2)
        finally:
            self.setEnabled(True)
        if finished:
            return
        if not self._python_runner.is_running():
            return
        dialog = ExternalRunDialog(self._python_runner, self)
        dialog.exec()

    def _run_command(self):
        """Parse and execute one console command, reporting errors in the console.

        Args:
            None.

        Return:
            None. Clears the input after command execution.
        """
        source = self.command.text().strip()
        self.command.clear()
        if not source:
            return
        prompt = "..." if self.cli._continuation else ">>>"
        self._log(f"{prompt} {source}")
        continuation = False
        try:
            continuation = self.cli.execute(source)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError, IndexError) as error:
            self._log(f"Command error: {error}")
            continuation = self.cli._continuation
        self.command.setPlaceholderText(
            "... continue Python block" if continuation else ">>> Python or GUI command"
        )

    def _select_indices(self, indices):
        """Apply a CLI atom selection and redraw the current frame.

        Args:
            indices: Unique integer atom indices.

        Return:
            None. The viewer selection is updated in place.
        """
        self.viewer._update_selection(np.asarray(indices, dtype=int), additive=False)
        self._sync_atom_actions(self.viewer.selected_indices)

    def _clear_selection(self):
        """Clear the viewer atom selection.

        Args:
            None.

        Return:
            None. The current frame is redrawn without selected atoms.
        """
        if self.viewer is not None:
            self.viewer._update_selection(np.empty(0, dtype=int), additive=False)
            self._sync_atom_actions(self.viewer.selected_indices)

    def _reset_view(self):
        """Reset the camera and synchronize the zoom control.

        Args:
            None.

        Return:
            None. The viewer returns to its default camera view.
        """
        viewer = self._active_viewer()
        if viewer is None:
            return
        if viewer is not self.viewer:
            self._activate_viewer(viewer)
        viewer.reset_view()
        blocker = QSignalBlocker(self.zoom_input)
        self.zoom_input.setValue(100)
        del blocker

    def _on_cli_data_changed(
        self,
        elements,
        coordinates,
        lattices,
        description="CLI data edit",
        preserve_existing_context=False,
    ):
        """Refresh the GUI after a validated CLI list edit.

        Args:
            elements: Validated element arrays.
            coordinates: Validated Cartesian coordinate arrays.
            lattices: Validated lattice arrays.
            description: History label for the completed operation.
            preserve_existing_context: Preserve unchanged frame state when appending.

        Return:
            None. The current frame is redrawn without replacing list owners.
        """
        validate_data(elements, coordinates, lattices)
        before_context = self._all_supercell_context()
        before_elements = [np.array(value, copy=True) for value in self.elements]
        before_coordinates = [np.array(value, copy=True) for value in self.coordinates]
        before_lattices = [np.array(value, copy=True) for value in self.lattices]
        before_graphs = list(self.bond_adjacencies)
        before_orders = list(self.bond_orders)
        # Keep edges when node numbering is still valid (coordinate/species
        # edits or append-only updates). Unknown node-count replacements have
        # no reliable permutation, so their removed triples must be undoable.
        reset_graphs = {
            frame: graph for frame, graph in enumerate(before_graphs)
            if frame >= len(elements) or len(before_elements[frame]) != len(elements[frame])
        }
        reset_orders = {
            frame: before_orders[frame] for frame in reset_graphs
        }
        bonds = BondChange.between(
            "Reset updated frame bonds", reset_graphs,
            {frame: None for frame in reset_graphs},
            reset_orders, {frame: None for frame in reset_graphs},
        )
        topology_changed = len(before_elements) != len(elements) or any(
            not np.array_equal(previous, current)
            for previous, current in zip(before_elements, elements)
        )
        self._pause_playback()
        self.elements = elements
        self.coordinates = coordinates
        self.lattices = lattices
        if preserve_existing_context:
            self._ensure_supercell_state()
            self._ensure_frame_state()
        else:
            self._initialize_supercell_state()
        if topology_changed and not preserve_existing_context:
            self._initialize_frame_state()
        elif not preserve_existing_context:
            self._ensure_frame_state()
        self.bond_adjacencies[:] = [
            before_graphs[frame] if frame < len(before_graphs) and frame not in reset_graphs else None
            for frame in range(len(elements))
        ]
        self.bond_orders[:] = [
            before_orders[frame] if frame < len(before_orders) and frame not in reset_graphs else None
            for frame in range(len(elements))
        ]
        self._ensure_frame_names()
        after_context = self._all_supercell_context()
        change = StructureChange.for_batch(
            description,
            before_elements,
            before_coordinates,
            before_lattices,
            elements,
            coordinates,
            lattices,
            before_context=before_context,
            after_context=after_context,
        )
        if bonds.frame_indices:
            change = (bonds, change)
        self._record_change(change, already_applied=True)
        self._restore_all_supercell_context(after_context)

    def _refresh_after_cli_execution(self):
        """Refresh GUI state after one completed embedded CLI command or cell.

        Args:
            None.

        Returns:
            None. In-place NumPy mutations become visible immediately.
        """
        self._refresh_after_change(frame_index=self.current_index)

    def _log(self, message):
        """Append one line to the command output pane.

        Args:
            message: Text to append.

        Return:
            None. Scrolls the output view to the latest line.
        """
        self.output.appendPlainText(message)
        self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().maximum())
