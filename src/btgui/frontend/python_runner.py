"""Background external-Python runner and its modal stop dialog."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)


class ExternalPythonRunner(QObject):
    """Run one external Python file without blocking the Qt event loop."""

    output = Signal(str)
    finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None
        self._reader = None
        self._waiter = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def is_running(self):
        """Return whether the child process is still alive."""
        process = self._process
        return process is not None and process.poll() is None

    def start(self, interpreter, path, args=(), cwd=None):
        """Start the interpreter and its output/wait threads."""
        with self._lock:
            if self.is_running():
                return False
            self._stop_event.clear()
            command = [str(interpreter), str(path), *[str(arg) for arg in args]]
            creationflags = 0
            start_new_session = os.name != "nt"
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            self._process = subprocess.Popen(
                command,
                cwd=str(cwd) if cwd is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=start_new_session,
                creationflags=creationflags,
            )
            process = self._process
        self._reader = threading.Thread(
            target=self._read_output, args=(process,), daemon=True
        )
        self._waiter = threading.Thread(
            target=self._wait_process, args=(process,), daemon=True
        )
        self._reader.start()
        self._waiter.start()
        return True

    def wait(self, timeout):
        """Wait up to timeout seconds; return True when the child exited."""
        process = self._process
        if process is None:
            return True
        try:
            process.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def request_stop(self):
        """Request termination of the running child process."""
        self._stop_event.set()

    def _read_output(self, process):
        stream = process.stdout
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                self.output.emit(line.rstrip("\n"))
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def _terminate(self, process):
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                process.terminate()
            except OSError:
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass

    def _wait_process(self, process):
        while process.poll() is None:
            if self._stop_event.wait(0.1):
                self._terminate(process)
                break
        returncode = process.wait()
        with self._lock:
            self._process = None
        self.finished.emit(int(returncode))

    def shutdown(self):
        """Terminate a child process when the main window closes."""
        if self.is_running():
            self.request_stop()
            process = self._process
            if process is not None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._terminate(process)


class ExternalRunDialog(QDialog):
    """Application-modal status window with an explicit stop control."""

    def __init__(self, runner, parent=None):
        super().__init__(parent)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowTitle("Running Python")
        self.resize(640, 360)
        self.runner = runner
        self.status = QLabel("Running external Python script...", self)
        self.output = QPlainTextEdit(self)
        self.output.setReadOnly(True)
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self._request_stop)
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self._request_stop)
        buttons = QHBoxLayout()
        buttons.addWidget(self.status, 1)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(close_button)
        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addWidget(self.output, 1)
        self.runner.output.connect(self._append_output)
        self.runner.finished.connect(self._finished)
        self.stop_shortcut = QShortcut(QKeySequence("Ctrl+C"), self)
        self.stop_shortcut.activated.connect(self._request_stop)

    def _append_output(self, text):
        self.output.appendPlainText(text)

    def _request_stop(self):
        self.status.setText("Stopping...")
        self.stop_button.setEnabled(False)
        self.runner.request_stop()

    def _finished(self, returncode):
        if self.isVisible():
            self.accept()

    def closeEvent(self, event):
        if self.runner.is_running():
            self._request_stop()
            event.ignore()
            return
        event.accept()
