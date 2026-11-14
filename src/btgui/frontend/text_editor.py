"""Plain-text editor used for non-structure files and parse fallbacks."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)


class TextEditorDialog(QDialog):
    """Modeless UTF-8 text editor for a single file path."""

    def __init__(self, path, text, parent=None, before_write=None):
        super().__init__(parent)
        self.path = Path(path)
        self.before_write = before_write
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(800, 600)
        self.setWindowTitle(f"{self.path.name} - Text Editor")
        self.editor = QPlainTextEdit(self)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.editor.setFont(font)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setPlainText(text)
        self.editor.document().setModified(False)
        self.path_label = QLabel(str(self.path), self)
        self.path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.save_button = QPushButton("Save", self)
        self.save_as_button = QPushButton("Save As...", self)
        self.close_button = QPushButton("Close", self)
        self.save_button.clicked.connect(self.save)
        self.save_as_button.clicked.connect(self.save_as)
        self.close_button.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addWidget(self.path_label, 1)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.save_as_button)
        buttons.addWidget(self.close_button)
        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addWidget(self.editor, 1)
        self.save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        self.save_shortcut.activated.connect(self.save)
        self.save_as_shortcut = QShortcut(
            QKeySequence("Ctrl+Shift+S"), self
        )
        self.save_as_shortcut.activated.connect(self.save_as)

    def _write(self, path):
        """Write the editor contents and update the editor title."""
        path = Path(path)
        if self.before_write is not None and path.is_file():
            self.before_write(path)
        path.write_text(self.editor.toPlainText(), encoding="utf-8")
        self.path = path
        self.path_label.setText(str(path))
        self.setWindowTitle(f"{path.name} - Text Editor")
        self.editor.document().setModified(False)

    def save(self):
        """Save to the current path, reporting write failures."""
        try:
            self._write(self.path)
        except OSError as error:
            QMessageBox.warning(self, "Save", f"Could not save {self.path}:\n{error}")
            return False
        return True

    def save_as(self):
        """Choose a new path and write the editor contents there."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save As", str(self.path), "All files (*)"
        )
        if not path:
            return False
        try:
            self._write(path)
        except OSError as error:
            QMessageBox.warning(self, "Save As", f"Could not save {path}:\n{error}")
            return False
        return True

    def closeEvent(self, event):
        """Offer to save unsaved text before closing the editor."""
        if not self.editor.document().isModified():
            event.accept()
            return
        answer = QMessageBox.question(
            self,
            "Text Editor",
            "Save changes before closing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
        if answer == QMessageBox.StandardButton.Save and not self.save():
            event.ignore()
            return
        event.accept()
