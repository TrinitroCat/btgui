"""Main window, file actions, animation timer, and command console."""

from collections import deque
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np

from ..backend.lattice_transform import transform_structure
from PySide6.QtCore import QSignalBlocker, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QColor, QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSizePolicy,
    QSplitter,
    QStyle,
    QToolBar,
    QToolButton,
    QWidgetAction,
    QVBoxLayout,
    QWidget,
    QColorDialog,
    QGridLayout,
    QHBoxLayout,
    QMessageBox,
)

from btgui.backend.core import parse_frame_range, parse_frame_selection, validate_data
from btgui.backend.core import format_atom_selection, parse_atom_selection
from btgui.data.appearance import ELEMENT_COLORS, default_color
from btgui.frontend.cli import CLI
from btgui.frontend.changes import StructureChange
from btgui.frontend.viewer import StructureViewer


DEFAULT_SETTINGS = {
    "history_limit": 10,
    "default_atom_material": "default",
    "atom_lighting": True,
    "atom_edges": False,
    "atom_edge_color": "#202020",
    "atom_edge_width": 1.0,
    "buctoolkit_paths": [],
}


class AppearanceDialog(QDialog):
    """Edit per-element atom styles and global bond appearance."""

    def __init__(self, viewer, elements, selected_indices=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Appearance")
        self.resize(520, 430)
        self.styles = {key: dict(value) for key, value in viewer.atom_styles.items()}
        self.elements = list(dict.fromkeys(np.asarray(elements, dtype=str).tolist()))
        self.frame_elements = np.asarray(elements, dtype=str)
        self.accepted_values = None
        self.selector_edit = QLineEdit(self)
        self.selector_edit.setPlaceholderText("All atoms")
        self.selector_edit.setText(
            format_atom_selection(
                [] if selected_indices is None else selected_indices,
                self.frame_elements,
            )
        )
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
        self.material_combo.currentIndexChanged.connect(self._load_material_preset)
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
        self.lighting_check = QCheckBox("Lighting", self)
        self.lighting_check.setChecked(viewer.atom_lighting)
        self.edges_check = QCheckBox("Atom boundaries", self)
        self.edges_check.setChecked(viewer.atom_edges)
        self.axes_check = QCheckBox("Axes", self)
        self.axes_check.setChecked(viewer.axes_visible)
        self.grid_check = QCheckBox("Grid", self)
        self.grid_check.setChecked(viewer.grid_visible)
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
        self.background_button = QPushButton(self)
        self._set_color_button(viewer._background_color_value, self.background_button)
        self.background_button.clicked.connect(self._choose_background_color)
        form = QFormLayout()
        form.addRow("Atoms", self.selector_edit)
        form.addRow("Atom opacity", self.opacity_spin)
        form.addRow("Atom color", self.color_button)
        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        form.addRow(separator)
        form.addRow("Global atom display", self.global_style_combo)
        form.addRow("Global custom radius", self.global_radius_spin)
        form.addRow("Background color", self.background_button)
        form.addRow("Bond width", self.bond_width_spin)
        form.addRow("Bond opacity", self.bond_alpha_spin)
        form.addRow("Sphere material", self.material_combo)
        form.addRow("Ambient", self.ambient_spin)
        form.addRow("Diffuse", self.diffuse_spin)
        form.addRow("Specular", self.specular_spin)
        form.addRow("PBR roughness", self.roughness_spin)
        form.addRow("PBR metallic", self.metallic_spin)
        form.addRow(self.lighting_check)
        form.addRow(self.edges_check)
        form.addRow(self.axes_check)
        form.addRow(self.grid_check)
        form.addRow("Boundary width", self.edge_width_spin)
        form.addRow("Boundary color", self.edge_color_button)
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
        try:
            indices = parse_atom_selection(self.selector_edit.text(), self.frame_elements)
        except ValueError as error:
            self.selector_edit.clear()
            QMessageBox.warning(self, "Invalid atom selector", str(error))
            return
        self.accepted_values = {
            "indices": indices,
            "global_atoms": not bool(self.selector_edit.text().strip()),
            "styles": deepcopy(self.styles),
            "material": self.material_combo.currentData(),
            "lighting": self.lighting_check.isChecked(),
            "edges": self.edges_check.isChecked(),
            "edge_color": self.edge_color_button.property("color") or "#202020",
            "edge_width": self.edge_width_spin.value(),
            "atom_style": self.global_style_combo.currentData(),
            "custom_radius": self.global_radius_spin.value(),
            "background": self.background_button.property("color") or "#f5f7f7",
            "ambient": self.ambient_spin.value(),
            "diffuse": self.diffuse_spin.value(),
            "specular": self.specular_spin.value(),
            "roughness": self.roughness_spin.value(),
            "metallic": self.metallic_spin.value(),
            "axes": self.axes_check.isChecked(),
            "grid": self.grid_check.isChecked(),
            "bond_width": self.bond_width_spin.value(),
            "bond_alpha": self.bond_alpha_spin.value(),
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


class SettingsDialog(QDialog):
    """Edit persistent history and default rendering preferences.

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
        self.resize(420, 260)
        self.buctoolkit_paths = list(settings.get("buctoolkit_paths", []))
        self.history_limit_spin = QSpinBox(self)
        self.history_limit_spin.setRange(1, 1000)
        self.history_limit_spin.setValue(settings["history_limit"])
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

        form = QFormLayout()
        form.addRow("Undo history steps", self.history_limit_spin)
        form.addRow("Default material", self.material_combo)
        form.addRow(self.lighting_check)
        form.addRow(self.edges_check)
        form.addRow("Boundary width", self.edge_width_spin)
        form.addRow("Boundary color", self.edge_color_button)
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

    def __init__(self, open_handler=None, save_handler=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("btgui - Structure Viewer")
        self.resize(1100, 760)
        self.elements = []
        self.coordinates = []
        self.lattices = []
        self.atom_frame_styles = []
        self.frame_metrics = []
        self.open_handler = open_handler
        self.save_handler = save_handler
        self.current_index = 0
        self.play_indices = []
        self.play_position = 0
        self.settings_path = Path(__file__).resolve().parents[1] / "settings.json"
        self.settings_warning = None
        self.settings = self._load_settings()
        self.undo_stack = deque(maxlen=self.settings["history_limit"])
        self.redo_stack = deque()
        self.supercell_bases = {}
        self.supercell_transforms = {}
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_frame)

        self.toolbar = QToolBar("Tools", self)
        self.toolbar.setObjectName("mainToolbar")
        self.toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.viewer = StructureViewer(self)
        self.viewer.atom_material = self.settings["default_atom_material"]
        self.viewer._set_material_preset(self.viewer.atom_material)
        self.viewer.atom_lighting = self.settings["atom_lighting"]
        self.viewer.atom_edges = self.settings["atom_edges"]
        self.viewer.atom_edge_color = self.settings["atom_edge_color"]
        self.viewer.atom_edge_width = self.settings["atom_edge_width"]
        self.viewer.coordinatesEdited.connect(self._record_coordinate_edit)
        self.viewer.deleteRequested.connect(self._delete_selected_atoms)
        self.viewer.atomsAdded.connect(self._commit_reference_atoms)
        self.viewer.filesDropped.connect(self._open_dropped_files)
        self.viewer.measurementModeChanged.connect(self._sync_measurement_actions)
        self.viewer.selectionChanged.connect(self._sync_atom_actions)

        self.undo_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack),
            "Undo",
            self,
        )
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.setToolTip("Undo coordinate edit")
        self.undo_action.triggered.connect(self._undo)
        self.toolbar.addAction(self.undo_action)
        self.redo_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward),
            "Redo",
            self,
        )
        self.redo_action.setShortcut("Ctrl+Y")
        self.redo_action.setToolTip("Redo coordinate edit")
        self.redo_action.triggered.connect(self._redo)
        self.toolbar.addAction(self.redo_action)
        self.toolbar.addSeparator()

        self.supercell_action = QAction("Build / Transform...", self)
        self.supercell_action.triggered.connect(self._show_supercell)
        self.supercell_menu = QMenu(self)
        self.supercell_menu.addAction(self.supercell_action)
        self.supercell_button = QToolButton(self)
        self.supercell_button.setText("Supercell")
        self.supercell_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.supercell_button.setMenu(self.supercell_menu)
        self.toolbar.addWidget(self.supercell_button)
        self.add_atom_action = QAction("Add Atom", self)
        self.add_atom_action.triggered.connect(self._show_add_atom)
        self.toolbar.addAction(self.add_atom_action)
        self.replace_atom_action = QAction("Replace Atom", self)
        self.replace_atom_action.setEnabled(False)
        self.replace_atom_action.triggered.connect(self._show_replace_atom)
        self.toolbar.addAction(self.replace_atom_action)

        self.bond_action = QAction("Show Bonds", self)
        self.bond_action.setCheckable(True)
        self.bond_action.toggled.connect(self.viewer_set_bonds)
        self.bond_menu = QMenu(self)
        self.bond_menu.addAction(self.bond_action)
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
        self.bond_menu.addSeparator()
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
        self.bond_button = QToolButton(self)
        self.bond_button.setText("Check Bonds")
        self.bond_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.bond_button.setMenu(self.bond_menu)
        self.toolbar.addWidget(self.bond_button)

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
        self.toolbar.addWidget(self.metrics_button)

        self.toolbar.addSeparator()
        self.previous_frame_button = QToolButton(self)
        self.previous_frame_button.setText("↑")
        self.previous_frame_button.setToolTip("Previous frame")
        self.previous_frame_button.clicked.connect(lambda: self._step_frame(-1))
        self.toolbar.addWidget(self.previous_frame_button)
        self.frame_label = QLabel("Current Frame", self)
        self.toolbar.addWidget(self.frame_label)
        self.frame_input = QLineEdit("0", self)
        self.frame_input.setMaximumWidth(58)
        self.frame_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_input.editingFinished.connect(self._on_frame_input)
        self.toolbar.addWidget(self.frame_input)
        self.frame_total_label = QLabel("/ 0", self)
        self.toolbar.addWidget(self.frame_total_label)
        self.next_frame_button = QToolButton(self)
        self.next_frame_button.setText("↓")
        self.next_frame_button.setToolTip("Next frame")
        self.next_frame_button.clicked.connect(lambda: self._step_frame(1))
        self.toolbar.addWidget(self.next_frame_button)
        self.play_button = QToolButton(self)
        self.play_button.setText("Play")
        self.play_button.clicked.connect(lambda: self._start_default_playback())
        self.toolbar.addWidget(self.play_button)
        self.pause_button = QToolButton(self)
        self.pause_button.setText("Pause")
        self.pause_button.clicked.connect(self._pause_playback)
        self.toolbar.addWidget(self.pause_button)
        spacer = QWidget(self)
        spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.toolbar.addWidget(spacer)
        self.close_frame_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton),
            "Close Current Frame",
            self,
        )
        self.close_frame_action.setShortcut("Ctrl+W")
        self.close_frame_action.triggered.connect(self._close_current_frame)
        self.toolbar.addAction(self.close_frame_action)

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
        self.zoom_input.valueChanged.connect(self.viewer.set_zoom_percent)
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
            on_shortcut=self._create_shortcut,
            on_executed=self._refresh_after_cli_execution,
            buctoolkit_paths=self.settings["buctoolkit_paths"],
        )
        self.viewer.set_error_handler(self._log)
        self.setAcceptDrops(True)

        console = QWidget(self)
        console_layout = QVBoxLayout(console)
        console_layout.setContentsMargins(6, 4, 6, 6)
        console_layout.addWidget(self.output)
        console_layout.addWidget(self.command)
        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.addWidget(self.viewer)
        splitter.addWidget(console)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([590, 150])
        self.setCentralWidget(splitter)
        self.frame_dock = QDockWidget("Frames", self)
        self.frame_dock.setObjectName("framesDock")
        self.frame_list = FrameListWidget(self.frame_dock)
        self.frame_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.frame_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.frame_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.frame_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.frame_list.itemDoubleClicked.connect(self._activate_frame_item)
        self.frame_list.customContextMenuRequested.connect(self._show_frame_context_menu)
        self.frame_list.permutationChanged.connect(self._reorder_frames)
        self.frame_dock.setWidget(self.frame_list)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.frame_dock)
        self.frame_dock.hide()
        self._create_menus()
        self._sync_frame_controls()
        self._sync_history_actions()
        self._log("Ready. Supply structure lists directly or connect file I/O handlers.")
        if not self.cli.shortcut_available():
            QMessageBox.information(
                self,
                "Desktop shortcut",
                "pyshortcuts is not installed; no desktop shortcut will be created.",
            )
        if self.settings_warning is not None:
            self._log(self.settings_warning)

    def _load_settings(self):
        """Load and validate persistent settings from the package directory.

        Args:
            None.

        Return:
            A complete settings dictionary; invalid files fall back to defaults.
        """
        settings = dict(DEFAULT_SETTINGS)
        if not self.settings_path.exists():
            return settings
        try:
            loaded = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise TypeError("settings root must be a JSON object")
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
            buctoolkit_paths = loaded.get("buctoolkit_paths", settings["buctoolkit_paths"])
            if isinstance(buctoolkit_paths, str):
                buctoolkit_paths = buctoolkit_paths.split(os.pathsep)
            if not isinstance(buctoolkit_paths, list) or any(
                not isinstance(path, str) for path in buctoolkit_paths
            ):
                raise TypeError("buctoolkit_paths must be a string or a list of strings")
            settings_directory = self.settings_path.parent
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
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
            self.settings_warning = f"Settings warning: {error}; using defaults."
        return settings

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
        self.settings = dialog.values()
        self.undo_stack = deque(self.undo_stack, maxlen=self.settings["history_limit"])
        self.viewer.set_appearance(
            self.viewer.atom_styles,
            self.settings["default_atom_material"],
            self.settings["atom_lighting"],
            self.settings["atom_edges"],
            self.settings["atom_edge_color"],
            self.settings["atom_edge_width"],
        )
        self._sync_history_actions()
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
        del self.atom_frame_styles[len(self.elements):]
        del self.frame_metrics[len(self.elements):]
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
            "default_custom_radius", "_background_color_value", "material_ambient",
            "material_diffuse", "material_specular", "material_roughness",
            "material_metallic", "axes_visible", "grid_visible", "bond_width",
            "bond_alpha",
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

    def _refresh_after_change(self, frame_index=None, reset_camera=False):
        """Refresh viewer references and selection after a committed change.

        Args:
            frame_index: Preferred frame to display, or ``None`` for current.
            reset_camera: Whether the scene should fit a new camera.

        Returns:
            None. The viewer is redrawn with the current raw lists.
        """
        self._ensure_supercell_state()
        self._ensure_frame_state()
        self.viewer.elements = self.elements
        self.viewer.coordinates = self.coordinates
        self.viewer.lattices = self.lattices
        if self.coordinates:
            target = self.current_index if frame_index is None else int(frame_index)
            self.current_index = min(max(target, 0), len(self.coordinates) - 1)
            self.viewer.current_index = self.current_index
            self.viewer.set_frame_state(self.atom_frame_styles, self.frame_metrics)
            self.viewer.selected_indices = self.viewer.selected_indices[
                (self.viewer.selected_indices >= 0)
                & (self.viewer.selected_indices < len(self.elements[self.current_index]))
            ]
            self.viewer.draw_frame(reset_camera=reset_camera)
        else:
            self.viewer.current_index = 0
            self.viewer.set_frame_state(self.atom_frame_styles, self.frame_metrics)
            self.viewer.draw_frame(reset_camera=reset_camera)
        self._sync_frame_controls()
        self._sync_frame_list()
        self._sync_atom_actions(self.viewer.selected_indices)
        self.cli.bind_data(self.elements, self.coordinates, self.lattices)

    def _record_change(self, change, already_applied=False, frame_index=None):
        """Record one unified change and optionally apply its after snapshot.

        Args:
            change: `StructureChange` instance to retain.
            already_applied: Whether the caller already wrote the after state.
            frame_index: Frame to display after applying the change.

        Returns:
            None. Redo history is discarded after a new operation.
        """
        if not already_applied:
            context = change.apply(
                self.elements, self.coordinates, self.lattices, use_after=True
            )
            if change.frame_indices is not None and context is not None:
                self._restore_supercell_context(change.frame_indices[0], context)
            elif change.frame_indices is None and context is not None:
                self._restore_all_supercell_context(context)
        self.undo_stack.append(change)
        self.redo_stack.clear()
        self._refresh_after_change(frame_index=frame_index)
        self._sync_history_actions()

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
        """Restore the coordinate snapshot before the latest edit.

        Args:
            checked: Ignored QAction checked state.

        Return:
            None. An empty stack is a no-op.
        """
        if len(self.undo_stack) == 0:
            return
        change = self.undo_stack[-1]
        try:
            context = change.apply(
                self.elements, self.coordinates, self.lattices, use_after=False
            )
            if change.frame_indices is not None and context is not None:
                self._restore_supercell_context(change.frame_indices[0], context)
            elif change.frame_indices is None and context is not None:
                self._restore_all_supercell_context(context)
            self.undo_stack.pop()
            self.redo_stack.append(change)
            frame_index = change.frame_indices[0] if change.frame_indices else self.current_index
            self._refresh_after_change(frame_index=frame_index)
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            self._log(f"Undo error: {error}")
        self._sync_history_actions()

    def _redo(self, checked=False):
        """Restore the coordinate snapshot after the latest undone edit.

        Args:
            checked: Ignored QAction checked state.

        Return:
            None. An empty stack is a no-op.
        """
        if len(self.redo_stack) == 0:
            return
        change = self.redo_stack[-1]
        try:
            context = change.apply(
                self.elements, self.coordinates, self.lattices, use_after=True
            )
            if change.frame_indices is not None and context is not None:
                self._restore_supercell_context(change.frame_indices[0], context)
            elif change.frame_indices is None and context is not None:
                self._restore_all_supercell_context(context)
            self.redo_stack.pop()
            self.undo_stack.append(change)
            frame_index = change.frame_indices[0] if change.frame_indices else self.current_index
            self._refresh_after_change(frame_index=frame_index)
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

    def _set_view(self, direction):
        """Set a lattice-axis view and reset its relative zoom control.

        Args:
            direction: Lattice row index, from zero to two.

        Return:
            None. The viewer camera and toolbar zoom value are synchronized.
        """
        self.viewer.set_view(direction)
        blocker = QSignalBlocker(self.zoom_input)
        self.zoom_input.setValue(100)
        del blocker

    def _set_measurement_mode(self, mode):
        """Activate one special measurement selector or restore atom selection."""
        if mode == "length":
            self.angle_measure_action.setChecked(False)
        elif mode == "angle":
            self.length_measure_action.setChecked(False)
        self.viewer.set_measurement_mode(mode)

    def _sync_measurement_actions(self, mode):
        """Reflect the viewer's special measurement mode in toolbar buttons."""
        self.length_measure_action.blockSignals(True)
        self.angle_measure_action.blockSignals(True)
        self.length_measure_action.setChecked(mode == "length")
        self.angle_measure_action.setChecked(mode == "angle")
        self.length_measure_action.blockSignals(False)
        self.angle_measure_action.blockSignals(False)

    def viewer_set_bonds(self, enabled):
        """Toggle covalent-radius bond rendering."""
        self.viewer.set_bonds(enabled)

    def viewer_set_tolerance(self, tolerance):
        """Update bond cutoff tolerance."""
        self.viewer.set_bond_tolerance(tolerance)

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

    def _commit_added_atoms(self, frame_index, symbols, coordinates, description):
        """Append atoms to one frame through unified history.

        Args:
            frame_index: Frame receiving the new atoms.
            symbols: One-dimensional element symbols.
            coordinates: Cartesian positions shaped ``(N, 3)``.
            description: History description.

        Returns:
            None. The full scene is rebuilt after one atomic commit.
        """
        if not 0 <= int(frame_index) < len(self.coordinates):
            return
        frame_index = int(frame_index)
        symbols = np.asarray(symbols, dtype="<U3").reshape(-1)
        coordinates = np.asarray(coordinates, dtype=float).reshape((-1, 3))
        if len(symbols) != len(coordinates) or len(symbols) == 0:
            return
        if not self._confirm_metrics_topology_change():
            return
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
        self._record_change(change, frame_index=frame_index)

    def _commit_reference_atoms(self, frame_index, symbols, coordinates):
        """Commit a complete reference-placement chain as one change.

        Args:
            frame_index: Frame receiving the pending atoms.
            symbols: Pending element symbols.
            coordinates: Pending Cartesian coordinates.

        Returns:
            None. Delegates to the common atom-appending path.
        """
        self._commit_added_atoms(frame_index, symbols, coordinates, "Add atoms")

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
        self._record_change(change, frame_index=frame_index)

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
        )
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.accepted_values is None:
            return
        values = dialog.accepted_values
        before_context = (
            self._all_supercell_context()
            if values["global_atoms"]
            else self._supercell_context(frame_index)
        )
        targets = (
            (
                target_frame,
                range(len(self.elements[target_frame])),
            )
            for target_frame in (
                range(len(self.elements)) if values["global_atoms"] else [frame_index]
            )
        )
        for target_frame, atom_indices in targets:
            for atom_index in atom_indices:
                symbol = str(self.elements[target_frame][atom_index])
                element_style = values["styles"].get(
                    symbol, self.viewer.atom_styles.get(symbol, {})
                )
                self.atom_frame_styles[target_frame][atom_index] = {
                    "color": element_style.get("color", default_color(symbol)),
                    "alpha": float(element_style.get("alpha", 1.0)),
                    "style": values["atom_style"],
                    "radius": float(values["custom_radius"]),
                }
            if not values["global_atoms"]:
                break
        if values["global_atoms"]:
            self.viewer.atom_styles = deepcopy(values["styles"])
            self.viewer.default_atom_style = values["atom_style"]
            self.viewer.default_custom_radius = float(values["custom_radius"])
        for name in (
            "material", "lighting", "edges", "edge_color", "edge_width",
            "background", "ambient", "diffuse", "specular", "roughness",
            "metallic", "axes", "grid", "bond_width", "bond_alpha",
        ):
            target_name = {
                "material": "atom_material", "lighting": "atom_lighting",
                "edges": "atom_edges", "edge_color": "atom_edge_color",
                "edge_width": "atom_edge_width", "background": "_background_color_value",
                "ambient": "material_ambient", "diffuse": "material_diffuse",
                "specular": "material_specular", "roughness": "material_roughness",
                "metallic": "material_metallic", "axes": "axes_visible",
                "grid": "grid_visible", "bond_width": "bond_width",
                "bond_alpha": "bond_alpha",
            }[name]
            setattr(self.viewer, target_name, values[name])
        self.viewer.set_background(self.viewer._background_color_value)
        after_context = (
            self._all_supercell_context()
            if values["global_atoms"]
            else self._supercell_context(frame_index)
        )
        if values["global_atoms"]:
            change = StructureChange.for_batch(
                "Appearance",
                self.elements,
                self.coordinates,
                self.lattices,
                self.elements,
                self.coordinates,
                self.lattices,
                before_context=before_context,
                after_context=after_context,
            )
        else:
            change = StructureChange.for_frame(
                "Appearance",
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

    def _sync_atom_actions(self, indices=None):
        """Enable replacement only when the current frame has a valid selection."""
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
        has_frames = frame_count > 0
        self.previous_frame_button.setEnabled(has_frames)
        self.next_frame_button.setEnabled(has_frames)
        self.play_button.setEnabled(has_frames)
        self.pause_button.setEnabled(has_frames)
        self.close_frame_action.setEnabled(has_frames)
        self.add_atom_action.setEnabled(has_frames)
        self.metrics_button.setEnabled(has_frames)
        self._sync_atom_actions(self.viewer.selected_indices)

    def _sync_frame_list(self):
        """Rebuild the file-like frame list while retaining its highlight."""
        selected_row = self.frame_list.currentRow()
        blocker = QSignalBlocker(self.frame_list)
        self.frame_list.clear()
        for frame_index, frame_elements in enumerate(self.elements):
            item = QListWidgetItem(
                f"Frame {frame_index + 1}  ({len(frame_elements)} atoms)"
            )
            item.setData(Qt.ItemDataRole.UserRole, frame_index)
            self.frame_list.addItem(item)
        if self.frame_list.count():
            target_row = self.current_index if selected_row < 0 else selected_row
            self.frame_list.setCurrentRow(
                min(max(target_row, 0), self.frame_list.count() - 1)
            )
        del blocker

    def _activate_frame_item(self, item):
        """Navigate only when a highlighted frame is double-clicked."""
        self._change_frame(self.frame_list.row(item))

    def _show_frame_context_menu(self, position):
        """Show Copy and Delete actions for the highlighted frame."""
        item = self.frame_list.itemAt(position)
        if item is None:
            return
        self.frame_list.setCurrentItem(item)
        row = self.frame_list.row(item)
        menu = QMenu(self.frame_list)
        menu.addAction("Copy", lambda: self._duplicate_frame(row))
        menu.addAction("Delete", lambda: self._delete_frame(row))
        menu.exec(self.frame_list.mapToGlobal(position))

    def _append_frame(self, elements, coordinates, lattice):
        """Append one independent frame and all aligned auxiliary state."""
        self.elements.append(np.array(elements, copy=True))
        self.coordinates.append(np.array(coordinates, dtype=float, copy=True))
        self.lattices.append(np.array(lattice, dtype=float, copy=True))
        frame = len(self.elements) - 1
        self.atom_frame_styles.append([dict() for _ in self.elements[frame]])
        self.frame_metrics.append({"length": {}, "angle": {}})
        self.supercell_bases[frame] = {
            "elements": np.array(self.elements[frame], copy=True),
            "coordinates": np.array(self.coordinates[frame], copy=True),
            "lattice": np.array(self.lattices[frame], copy=True),
        }
        self.supercell_transforms[frame] = (
            np.ones(3, dtype=np.int64), np.eye(3, dtype=np.int64)
        )

    def _delete_frame(self, frame_index):
        """Delete one frame and every aligned container without history."""
        if not 0 <= int(frame_index) < len(self.elements):
            return
        frame_index = int(frame_index)
        old_bases = self.supercell_bases
        old_transforms = self.supercell_transforms
        for target in (
            self.elements, self.coordinates, self.lattices,
            self.atom_frame_styles, self.frame_metrics,
        ):
            target.pop(frame_index)
        retained_indices = [
            index for index in range(len(self.elements) + 1) if index != frame_index
        ]
        self.supercell_bases = {
            new: old_bases[old] for new, old in enumerate(retained_indices)
        }
        self.supercell_transforms = {
            new: old_transforms[old] for new, old in enumerate(retained_indices)
        }
        if frame_index < self.current_index:
            self.current_index -= 1
        elif frame_index == self.current_index:
            self.current_index = min(frame_index, max(len(self.elements) - 1, 0))
        self.viewer.selected_indices = np.empty(0, dtype=int)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._refresh_after_change()
        self._sync_history_actions()

    def _close_current_frame(self):
        """Close the displayed frame through the common deletion path."""
        self._delete_frame(self.current_index)

    def _duplicate_frame(self, frame_index):
        """Insert a deep independent copy directly below one selected frame."""
        if not 0 <= int(frame_index) < len(self.elements):
            return
        frame_index = int(frame_index)
        insert_index = frame_index + 1
        old_bases = self.supercell_bases
        old_transforms = self.supercell_transforms
        for target, value in (
            (self.elements, np.array(self.elements[frame_index], copy=True)),
            (self.coordinates, np.array(self.coordinates[frame_index], copy=True)),
            (self.lattices, np.array(self.lattices[frame_index], copy=True)),
            (self.atom_frame_styles, deepcopy(self.atom_frame_styles[frame_index])),
            (self.frame_metrics, deepcopy(self.frame_metrics[frame_index])),
        ):
            target.insert(insert_index, value)
        source_indices = list(range(insert_index)) + [frame_index] + list(
            range(insert_index, len(self.elements) - 1)
        )
        self.supercell_bases = {
            new: deepcopy(old_bases[old]) for new, old in enumerate(source_indices)
        }
        self.supercell_transforms = {
            new: deepcopy(old_transforms[old]) for new, old in enumerate(source_indices)
        }
        self.current_index = insert_index
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._refresh_after_change(frame_index=insert_index)
        self._sync_history_actions()

    def _apply_frame_permutation(self, permutation):
        """Apply new[k] = old[perm[k]] to every frame-aligned list."""
        permutation = np.asarray(permutation, dtype=int)
        if sorted(permutation.tolist()) != list(range(len(self.elements))):
            raise ValueError("frame permutation is invalid")
        inverse = np.argsort(permutation)
        for target in (self.elements, self.coordinates, self.lattices):
            old = list(target)
            target[:] = [old[index] for index in permutation]
        for target in (self.atom_frame_styles, self.frame_metrics):
            old = deepcopy(target)
            target[:] = [old[index] for index in permutation]
        old_bases = deepcopy(self.supercell_bases)
        old_transforms = deepcopy(self.supercell_transforms)
        self.supercell_bases = {new: old_bases[int(old)] for new, old in enumerate(permutation)}
        self.supercell_transforms = {
            new: old_transforms[int(old)] for new, old in enumerate(permutation)
        }
        if len(permutation):
            self.current_index = int(inverse[min(self.current_index, len(permutation) - 1)])
        return inverse

    def _reorder_frames(self, permutation):
        """Record one undoable frame permutation from an internal drag."""
        if list(permutation) == list(range(len(self.elements))):
            return
        before_elements = [np.array(value, copy=True) for value in self.elements]
        before_coordinates = [np.array(value, copy=True) for value in self.coordinates]
        before_lattices = [np.array(value, copy=True) for value in self.lattices]
        before_context = self._all_supercell_context()
        self._apply_frame_permutation(permutation)
        after_context = self._all_supercell_context()
        change = StructureChange.for_batch(
            "Reorder frames",
            before_elements, before_coordinates, before_lattices,
            self.elements, self.coordinates, self.lattices,
            before_context=before_context, after_context=after_context,
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
        self.current_index = index
        self.viewer.set_frame(index)
        self._sync_frame_controls()

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

    def _start_default_playback(self, checked=False):
        """Start playback over all retained frames at the default rate."""
        try:
            self._play(None, 10)
        except (RuntimeError, TypeError, ValueError) as error:
            self._log(f"Playback error: {error}")

    def _pause_playback(self):
        """Stop playback and rebuild dynamic geometry once."""
        self.timer.stop()
        self.viewer.finish_playback()

    def _change_zoom(self, delta):
        """Change the zoom spin box by one toolbar step."""
        self.zoom_input.setValue(self.zoom_input.value() + delta)

    def _create_menus(self):
        """Create File, Settings, Appearance, and View menus.

        Args:
            None.

        Return:
            None. Connects top-level actions and embeds zoom controls in View.
        """
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        new_action = file_menu.addAction("New...")
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_frame)
        open_action = file_menu.addAction("Open...")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_dialog)
        save_action = file_menu.addAction("Save As...")
        save_action.setShortcut("Ctrl+Shift+S")
        save_action.triggered.connect(self._save_dialog)
        shortcut_action = file_menu.addAction("Create Desktop Shortcut")
        shortcut_action.triggered.connect(self._create_shortcut)
        export_action = file_menu.addAction("Export Batch...")
        export_action.triggered.connect(self._export_dialog)
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self._show_settings)
        menu_bar.addAction(settings_action)
        appearance_action = QAction("Appearance", self)
        appearance_action.triggered.connect(self._show_appearance)
        menu_bar.addAction(appearance_action)

        windows_menu = menu_bar.addMenu("Windows")
        show_frames_action = self.frame_dock.toggleViewAction()
        show_frames_action.setText("Show Frames")
        windows_menu.addAction(show_frames_action)

        view_menu = menu_bar.addMenu("View")
        view_menu.addAction("Along A", lambda: self._set_view(0))
        view_menu.addAction("Along B", lambda: self._set_view(1))
        view_menu.addAction("Along C", lambda: self._set_view(2))
        view_menu.addSeparator()
        zoom_controls = QWidget(view_menu)
        zoom_layout = QHBoxLayout(zoom_controls)
        zoom_layout.setContentsMargins(8, 4, 8, 4)
        zoom_layout.addWidget(self.zoom_minus_button)
        zoom_layout.addWidget(self.zoom_input)
        zoom_layout.addWidget(self.zoom_plus_button)
        zoom_action = QWidgetAction(view_menu)
        zoom_action.setDefaultWidget(zoom_controls)
        view_menu.addAction(zoom_action)

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
        self._initialize_supercell_state()
        self._initialize_frame_state()
        self.cli.bind_data(self.elements, self.coordinates, self.lattices)
        self.current_index = 0
        self.viewer.set_data(elements, coordinates, lattices)
        self.viewer.set_frame_state(self.atom_frame_styles, self.frame_metrics)
        self.viewer.draw_frame(reset_camera=True)
        blocker = QSignalBlocker(self.zoom_input)
        self.zoom_input.setValue(100)
        del blocker
        self._sync_frame_controls()
        self._sync_frame_list()
        self._sync_history_actions()
        self._log(f"Loaded {len(elements)} frame(s), {sum(map(len, elements))} atoms total.")

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
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save current frame",
            "",
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

    def _open(self, paths):
        """Load selected files and append their frames to the current batch.

        Args:
            paths: One path or a list of paths passed to the configured handler.

        Return:
            None. Updates the viewer and logs the number of loaded frames.
        """
        if self.open_handler is None:
            raise RuntimeError("file opening is not configured; provide an open_handler")
        self._pause_playback()
        elements, coordinates, lattices = self.open_handler(paths)
        validate_data(elements, coordinates, lattices)
        previous_count = len(self.coordinates)
        self.timer.stop()
        self.play_indices = []
        returns_aggregate = bool(getattr(self.open_handler, "returns_aggregate", False))
        if previous_count == 0:
            self.set_data(elements, coordinates, lattices)
        elif returns_aggregate:
            self.elements = elements
            self.coordinates = coordinates
            self.lattices = lattices
            self._ensure_supercell_state()
            self._ensure_frame_state()
            self.viewer.elements = elements
            self.viewer.coordinates = coordinates
            self.viewer.lattices = lattices
            self.viewer.selected_indices = np.empty(0, dtype=int)
            self.viewer.set_frame_state(self.atom_frame_styles, self.frame_metrics)
            self._change_frame(min(previous_count, len(coordinates) - 1))
            self._log(f"Appended {len(coordinates) - previous_count} frame(s) from selected files")
        else:
            for frame_elements, frame_coordinates, frame_lattice in zip(
                elements, coordinates, lattices
            ):
                self._append_frame(frame_elements, frame_coordinates, frame_lattice)
            self.viewer.elements = self.elements
            self.viewer.coordinates = self.coordinates
            self.viewer.lattices = self.lattices
            self.viewer.set_frame_state(self.atom_frame_styles, self.frame_metrics)
            self.viewer.selected_indices = np.empty(0, dtype=int)
            self.viewer.current_index = min(previous_count, len(self.coordinates) - 1)
            self.viewer.draw_frame(reset_camera=False)
            self.current_index = self.viewer.current_index
            self._sync_frame_controls()
            self._sync_frame_list()
            self._log(f"Appended {len(elements)} frame(s) from selected files")
        self.cli.bind_data(self.elements, self.coordinates, self.lattices)

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
        self.save_handler(
            path,
            [self.elements[frame_index]],
            [self.coordinates[frame_index]],
            [self.lattices[frame_index]],
        )
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
        self.play_indices = (
            parse_frame_range(specification, len(self.coordinates))
            if specification is not None
            else list(range(len(self.coordinates)))
        )
        if not self.play_indices:
            raise ValueError("frame range is empty")
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.play_position = 0
        self.viewer.begin_playback()
        self.timer.start(max(1, round(1000 / fps)))
        self._sync_frame_controls()

    def _advance_frame(self):
        """Advance to the next frame in the active playback sequence.

        Args:
            None.

        Return:
            None. Updates the viewer and wraps playback at the end.
        """
        if not self.play_indices:
            self.timer.stop()
            return
        index = self.play_indices[self.play_position]
        if not self.viewer.update_frame_preview(index):
            self.viewer.finish_playback()
            self.current_index = index
            self.viewer.set_frame(index)
        else:
            self.current_index = index
        self._sync_frame_controls()
        self.play_position = (self.play_position + 1) % len(self.play_indices)

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
        try:
            continuation = self.cli.execute(source)
            self.command.setPlaceholderText(
                "... continue Python block" if continuation else ">>> Python or GUI command"
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError, IndexError) as error:
            self._log(f"Command error: {error}")

    def _select_indices(self, indices):
        """Apply a CLI atom selection and redraw the current frame.

        Args:
            indices: Unique integer atom indices.

        Return:
            None. The viewer selection is updated in place.
        """
        self.viewer.selected_indices = np.asarray(indices, dtype=int)
        self.viewer.refresh_selection_overlay()
        self._sync_atom_actions(self.viewer.selected_indices)

    def _clear_selection(self):
        """Clear the viewer atom selection.

        Args:
            None.

        Return:
            None. The current frame is redrawn without selected atoms.
        """
        self.viewer.selected_indices = np.empty(0, dtype=int)
        self.viewer.refresh_selection_overlay()
        self._sync_atom_actions(self.viewer.selected_indices)

    def _reset_view(self):
        """Reset the camera and synchronize the zoom control.

        Args:
            None.

        Return:
            None. The viewer returns to its default camera view.
        """
        self.viewer.reset_view()
        blocker = QSignalBlocker(self.zoom_input)
        self.zoom_input.setValue(100)
        del blocker

    def _on_cli_data_changed(self, elements, coordinates, lattices):
        """Refresh the GUI after a validated CLI list edit.

        Args:
            elements: Validated element arrays.
            coordinates: Validated Cartesian coordinate arrays.
            lattices: Validated lattice arrays.

        Return:
            None. The current frame is redrawn without replacing list owners.
        """
        validate_data(elements, coordinates, lattices)
        before_context = self._all_supercell_context()
        before_elements = [np.array(value, copy=True) for value in self.elements]
        before_coordinates = [np.array(value, copy=True) for value in self.coordinates]
        before_lattices = [np.array(value, copy=True) for value in self.lattices]
        topology_changed = len(before_elements) != len(elements) or any(
            not np.array_equal(previous, current)
            for previous, current in zip(before_elements, elements)
        )
        self._pause_playback()
        self.elements = elements
        self.coordinates = coordinates
        self.lattices = lattices
        self._initialize_supercell_state()
        if topology_changed:
            self._initialize_frame_state()
        else:
            self._ensure_frame_state()
        after_context = self._all_supercell_context()
        change = StructureChange.for_batch(
            "CLI data edit",
            before_elements,
            before_coordinates,
            before_lattices,
            elements,
            coordinates,
            lattices,
            before_context=before_context,
            after_context=after_context,
        )
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

    def _create_shortcut(self):
        """Create an optional desktop shortcut and report missing dependencies.

        Args:
            None.

        Return:
            None. A warning is shown when pyshortcuts is unavailable.
        """
        try:
            self.cli.create_shortcut()
        except RuntimeError as error:
            QMessageBox.information(self, "Desktop shortcut", str(error))
            self._log(str(error))
        except (OSError, TypeError, ValueError) as error:
            self._log(f"Desktop shortcut error: {error}")

    def _log(self, message):
        """Append one line to the command output pane.

        Args:
            message: Text to append.

        Return:
            None. Scrolls the output view to the latest line.
        """
        self.output.appendPlainText(message)
        self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().maximum())
