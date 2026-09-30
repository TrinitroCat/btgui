"""Persistent Python-style command console for raw structure lists."""

from __future__ import annotations

import code
from contextlib import redirect_stderr, redirect_stdout
import importlib
import importlib.util
import io
import shlex
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from btgui.backend.core import parse_frame_selection, validate_data


class _CapturedConsole(code.InteractiveConsole):
    """Track interactive execution failures while retaining Python behavior.

    Args:
        locals: Persistent interactive namespace.

    Returns:
        An interactive console whose errors can trigger protected-data rollback.
    """

    def __init__(self, locals: dict) -> None:
        super().__init__(locals=locals)
        self.had_error = False
        self.stream = io.StringIO()

    def write(self, data: str) -> None:
        """Capture traceback and syntax-error text.

        Args:
            data: Text emitted by the standard interactive console.

        Returns:
            None. Text is appended to the current command stream.
        """
        self.stream.write(data)

    def showsyntaxerror(self, filename: str | None = None) -> None:
        """Record and render a normal interactive syntax error.

        Args:
            filename: Optional filename displayed by Python.

        Returns:
            None. The error flag is set before normal formatting.
        """
        self.had_error = True
        super().showsyntaxerror(filename)

    def showtraceback(self) -> None:
        """Record and render a normal interactive runtime traceback.

        Args:
            None.

        Returns:
            None. The error flag is set before normal formatting.
        """
        self.had_error = True
        super().showtraceback()


class CLI:
    """Run built-in GUI commands and persistent interactive Python input.

    Args:
        log: Callback receiving console output.
        on_open: Callback loading structure files.
        on_save: Callback saving the current frame.
        on_frame: Callback changing the displayed frame.
        current_index: Callback returning the displayed frame index.
        on_play: Callback starting animation playback.
        on_pause: Callback stopping playback.
        on_select: Callback selecting atom indices.
        on_clear: Callback clearing atom selection.
        on_view_reset: Callback resetting the camera.
        on_data_changed: Callback atomically applying validated raw lists.
        on_shortcut: Callback requesting a desktop shortcut.

    Notes:
        ``coo``, ``cell``, and ``elements`` always reference detached working
        pages. The lists owned by the GUI are changed only after validation.
    """

    _IN_VAR_SET = frozenset({"coo", "cell", "elements"})

    def __init__(
        self,
        log: Callable[[str], None],
        on_open: Callable,
        on_save: Callable,
        on_frame: Callable,
        current_index: Callable[[], int],
        on_play: Callable,
        on_pause: Callable,
        on_select: Callable,
        on_clear: Callable,
        on_view_reset: Callable,
        on_data_changed: Callable,
        on_shortcut: Callable,
    ) -> None:
        self._log = log
        self._on_open = on_open
        self._on_save = on_save
        self._on_frame = on_frame
        self._current_index = current_index
        self._on_play = on_play
        self._on_pause = on_pause
        self._on_select = on_select
        self._on_clear = on_clear
        self._on_view_reset = on_view_reset
        self._on_data_changed = on_data_changed
        self._on_shortcut = on_shortcut
        self._targets = {"coo": [], "cell": [], "elements": []}
        self._backup = {name: [] for name in self._IN_VAR_SET}
        self._work = {name: [] for name in self._IN_VAR_SET}
        self.coo = self._work["coo"]
        self.cell = self._work["cell"]
        self.elements = self._work["elements"]
        self._buctoolkit = self._find_buctoolkit()
        self._namespace = {
            "__builtins__": __builtins__,
            "np": np,
            "to_bt": self.to_bt,
            "coo": self.coo,
            "cell": self.cell,
            "elements": self.elements,
        }
        self._console = _CapturedConsole(self._namespace)
        self._continuation = False

    @staticmethod
    def _find_buctoolkit():
        """Locate and preload optional BUCToolkit at console construction.

        Args:
            None.

        Returns:
            The imported BUCToolkit package, or ``None`` when unavailable.
        """
        if importlib.util.find_spec("BUCToolkit") is None:
            return None
        try:
            return importlib.import_module("BUCToolkit")
        except (ImportError, ModuleNotFoundError):
            return None

    @staticmethod
    def _clone_lists(values: dict[str, list]) -> dict[str, list]:
        """Copy the three protected list/array trees.

        Args:
            values: Mapping containing protected raw lists.

        Returns:
            Detached lists with copied NumPy arrays.
        """
        return {
            name: [np.array(item, copy=True) for item in values[name]]
            for name in CLI._IN_VAR_SET
        }

    @staticmethod
    def _variables_equal(left: dict[str, list], right: dict[str, list]) -> bool:
        """Compare protected variables without ambiguous NumPy truth values.

        Args:
            left: First protected-variable mapping.
            right: Second protected-variable mapping.

        Returns:
            ``True`` only when list structure, dtype, shape, and values match.
        """
        for name in CLI._IN_VAR_SET:
            if not isinstance(left.get(name), list) or not isinstance(right.get(name), list):
                return False
            if len(left[name]) != len(right[name]):
                return False
            for left_array, right_array in zip(left[name], right[name]):
                if not isinstance(left_array, np.ndarray) or not isinstance(right_array, np.ndarray):
                    return False
                if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
                    return False
                if not np.array_equal(left_array, right_array):
                    return False
        return True

    @staticmethod
    def _validate_variables(values: dict[str, list]) -> None:
        """Validate protected lists before they reach the GUI.

        Args:
            values: Candidate ``elements``, ``coo``, and ``cell`` lists.

        Returns:
            None when all data are safe for rendering.

        Raises:
            TypeError: If a protected value is not a list of NumPy arrays.
            ValueError: If frame counts, shapes, dtypes, or values are invalid.
        """
        for name in CLI._IN_VAR_SET:
            if not isinstance(values.get(name), list):
                raise TypeError(f"{name} must remain a List[np.ndarray]")
            if any(not isinstance(value, np.ndarray) for value in values[name]):
                raise TypeError(f"{name} must contain only np.ndarray objects")
        frame_counts = {len(values[name]) for name in CLI._IN_VAR_SET}
        if len(frame_counts) != 1:
            raise ValueError("elements, coo, and cell must have equal frame counts")
        if next(iter(frame_counts), 0) == 0:
            raise ValueError("at least one structure is required")
        for frame, (elements, coordinates, lattice) in enumerate(
            zip(values["elements"], values["coo"], values["cell"])
        ):
            if elements.ndim != 1 or elements.dtype.kind not in {"U", "S"}:
                raise ValueError(f"elements[{frame}] must be a one-dimensional string array")
            if coordinates.shape != (len(elements), 3):
                raise ValueError(f"coo[{frame}] must have shape ({len(elements)}, 3)")
            if coordinates.dtype.kind not in {"f", "i", "u"}:
                raise ValueError(f"coo[{frame}] must have a numeric dtype")
            if lattice.shape != (3, 3) or lattice.dtype.kind not in {"f", "i", "u"}:
                raise ValueError(f"cell[{frame}] must be a numeric (3, 3) array")
            if not np.isfinite(coordinates).all() or not np.isfinite(lattice).all():
                raise ValueError(f"frame {frame} contains non-finite values")

    def _install_work(self, values: dict[str, list]) -> None:
        """Install detached working pages in the persistent namespace.

        Args:
            values: Protected values used to create new working pages.

        Returns:
            None. Existing ordinary Python variables remain available.
        """
        if any(len(self._work[name]) != len(values[name]) for name in self._IN_VAR_SET):
            self._work = self._clone_lists(values)
        else:
            for name in self._IN_VAR_SET:
                for index, source in enumerate(values[name]):
                    target = self._work[name][index]
                    if (
                        isinstance(target, np.ndarray)
                        and target.shape == source.shape
                        and target.dtype == source.dtype
                    ):
                        np.copyto(target, source)
                    else:
                        self._work[name][index] = np.array(source, copy=True)
        self.coo = self._work["coo"]
        self.cell = self._work["cell"]
        self.elements = self._work["elements"]
        self._namespace["coo"] = self.coo
        self._namespace["cell"] = self.cell
        self._namespace["elements"] = self.elements

    def _refresh_backup_buffers(self) -> None:
        """Commit current GUI targets into backup and dirty-page buffers.

        Args:
            None.

        Returns:
            None. Compatible arrays are updated with `np.copyto`.
        """
        requires_allocation = any(
            len(self._backup[name]) != len(self._targets[name])
            for name in self._IN_VAR_SET
        )
        if requires_allocation:
            self._backup = self._clone_lists(self._targets)
        else:
            for name in self._IN_VAR_SET:
                for index, source in enumerate(self._targets[name]):
                    target = self._backup[name][index]
                    if (
                        isinstance(target, np.ndarray)
                        and target.shape == source.shape
                        and target.dtype == source.dtype
                    ):
                        np.copyto(target, source)
                    else:
                        self._backup[name][index] = np.array(source, copy=True)
        self._install_work(self._backup)

    def bind_data(self, elements: list, coordinates: list, lattices: list) -> None:
        """Bind committed GUI lists and refresh protected working pages.

        Args:
            elements: Per-frame element arrays.
            coordinates: Per-frame Cartesian coordinate arrays.
            lattices: Per-frame lattice arrays.

        Returns:
            None. The provided outer lists remain authoritative and unwrapped.
        """
        validate_data(elements, coordinates, lattices)
        for frame in range(len(elements)):
            if not isinstance(elements[frame], np.ndarray):
                elements[frame] = np.asarray(elements[frame], dtype=str)
            if not isinstance(coordinates[frame], np.ndarray):
                coordinates[frame] = np.asarray(coordinates[frame], dtype=float)
            if not isinstance(lattices[frame], np.ndarray):
                lattices[frame] = np.asarray(lattices[frame], dtype=float)
        self._targets = {"elements": elements, "coo": coordinates, "cell": lattices}
        self._backup = self._clone_lists(self._targets)
        self._work = self._clone_lists(self._backup)
        self._install_work(self._backup)
        self._console.resetbuffer()
        self._continuation = False

    def _candidate_variables(self) -> dict[str, list]:
        """Read protected names from the persistent Python namespace.

        Args:
            None.

        Returns:
            Current protected values, including invalid values for validation.
        """
        return {name: self._namespace.get(name) for name in self._IN_VAR_SET}

    def _emit_stream(self, stream: io.StringIO) -> None:
        """Append captured Python output to the GUI pane.

        Args:
            stream: Captured stdout, stderr, and traceback text.

        Returns:
            None. Empty output is ignored.
        """
        output = stream.getvalue().rstrip("\n")
        if len(output) > 0:
            self._log(output)

    def _push_python(self, source: str) -> bool:
        """Push one line into the persistent interactive interpreter.

        Args:
            source: One line entered by the user.

        Returns:
            ``True`` when Python requires another continuation line.
        """
        stream = io.StringIO()
        self._console.stream = stream
        if not self._continuation:
            self._console.had_error = False
        with redirect_stdout(stream), redirect_stderr(stream):
            requires_more = self._console.push(source)
        self._emit_stream(stream)
        self._continuation = requires_more
        if requires_more:
            return True

        candidate = self._candidate_variables()
        has_changed_protected_data = not self._variables_equal(candidate, self._backup)
        if self._console.had_error:
            self._install_work(self._backup)
            return False
        if not has_changed_protected_data:
            return False
        try:
            self._validate_variables(candidate)
            self._on_data_changed(
                candidate["elements"],
                candidate["coo"],
                candidate["cell"],
            )
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            self._log(f"Invalid change for inner variables: {error}")
            self._install_work(self._backup)
            return False
        self._refresh_backup_buffers()
        return False

    def _show_help(self) -> None:
        """Display detailed built-in and Python-console documentation.

        Args:
            None.

        Returns:
            None. Help text is appended to the GUI output pane.
        """
        self._log(
            "GUI commands:\n"
            "  open PATH                 append a supported structure file\n"
            "  save PATH                 save the current frame\n"
            "  frame INDEX               show a zero-based frame\n"
            "  play [A:B:S] [--fps N]   play a Python-style frame slice\n"
            "  pause | stop              pause, or pause and return to frame 0\n"
            "  select INDEX...           select zero-based atoms in the current frame\n"
            "  clear                      clear atom selection\n"
            "  view reset                 reset camera and zoom\n"
            "  to_bt [1,3-5]             store selected frames in Python variable bt\n"
            "  shortcut                   create an optional desktop shortcut\n"
            "\nPython console:\n"
            "  coo       List[np.ndarray], Cartesian coordinates shaped (N, 3)\n"
            "  cell      List[np.ndarray], row-vector cells shaped (3, 3)\n"
            "  elements  List[np.ndarray], element symbols shaped (N,)\n"
            "  np        NumPy module\n"
            "  to_bt(index=None) accepts None, one-based range text, or zero-based ints\n"
            "\nExamples:\n"
            "  coo[0]\n"
            "  np.linalg.det(cell[0])\n"
            "  coo[0][:, 2] += 0.5\n"
            "  elements[0] = np.array(['C', 'O'])\n"
            "  bt = to_bt('1, 3-4')\n"
            "\nExpressions print their repr automatically. print(), warnings, and Python "
            "tracebacks are shown here. Protected-list edits are committed only after "
            "shape, dtype, length, and finite-value validation."
        )

    def execute(self, source: str) -> bool:
        """Execute one GUI command or Python interpreter line.

        Args:
            source: Text entered in the command field.

        Returns:
            ``True`` when an incomplete Python block expects another line.

        Raises:
            ValueError: If a recognized GUI command has invalid arguments.
        """
        if self._continuation:
            return self._push_python(source)
        try:
            parts = shlex.split(source)
        except ValueError:
            return self._push_python(source)
        if len(parts) == 0:
            return False
        name, args = parts[0].lower(), parts[1:]
        if name == "help" and len(args) == 0:
            self._show_help()
            return False
        if name == "open":
            if len(args) != 1:
                raise ValueError("usage: open PATH")
            self._on_open(args[0])
            return False
        if name == "save":
            if len(args) != 1:
                raise ValueError("usage: save PATH")
            self._on_save(args[0])
            return False
        if name == "frame":
            if len(args) != 1:
                raise ValueError("usage: frame INDEX")
            self._on_frame(int(args[0]))
            return False
        if name == "play":
            specification = None
            fps = 10
            index = 0
            while index < len(args):
                if args[index] == "--fps" and index + 1 < len(args):
                    fps = int(args[index + 1])
                    index += 2
                elif specification is None:
                    specification = args[index]
                    index += 1
                else:
                    raise ValueError("usage: play [START:STOP:STEP] [--fps N]")
            self._on_play(specification, fps)
            return False
        if name == "pause" and len(args) == 0:
            self._on_pause()
            return False
        if name == "stop" and len(args) == 0:
            self._on_pause()
            if len(self._targets["coo"]) > 0:
                self._on_frame(0)
            return False
        if name == "select" and len(args) > 0:
            if len(self._targets["coo"]) == 0:
                raise ValueError("open a structure before selecting atoms")
            indices = np.asarray([int(value) for value in args], dtype=int)
            frame_index = int(self._current_index())
            if not 0 <= frame_index < len(self._targets["elements"]):
                raise ValueError("current frame is unavailable")
            if np.any(indices < 0) or np.any(
                indices >= len(self._targets["elements"][frame_index])
            ):
                raise ValueError("atom index is out of range")
            self._on_select(np.unique(indices))
            return False
        if name == "clear" and len(args) == 0:
            self._on_clear()
            return False
        if name == "view" and args == ["reset"]:
            self._on_view_reset()
            return False
        if name == "to_bt":
            if len(args) > 1:
                raise ValueError("usage: to_bt [1,3-4]")
            batch = self.to_bt(args[0] if len(args) == 1 else None)
            if batch is not None:
                self._namespace["bt"] = batch
                self._log("Created BUCToolkit BatchStructures as Python variable bt.")
            return False
        if name == "shortcut" and len(args) == 0:
            self._on_shortcut()
            return False
        return self._push_python(source)

    def to_bt(self, index: list[int] | str | None = None):
        """Copy selected display data into optional BUCToolkit BatchStructures.

        Args:
            index: Zero-based integer indices, one-based range text, or ``None``.

        Returns:
            A detached BUCToolkit ``BatchStructures``, or ``None`` when absent.

        Raises:
            ValueError: If the selection is empty, malformed, or out of range.
        """
        if self._buctoolkit is None:
            self._log("BUCToolkit is not installed; to_bt did nothing.")
            return None
        frame_count = len(self._targets["coo"])
        if index is None:
            if frame_count == 0:
                raise ValueError("there are no structures to convert")
            selected = list(range(frame_count))
        elif isinstance(index, str):
            selected = parse_frame_selection(index, frame_count)
        elif isinstance(index, list) and all(
            isinstance(value, int) and not isinstance(value, bool) for value in index
        ):
            selected = list(index)
            if len(selected) == 0:
                raise ValueError("to_bt index cannot be empty")
            if any(value < 0 or value >= frame_count for value in selected):
                raise ValueError("to_bt indices are out of range")
        else:
            raise ValueError("to_bt index must be None, range text, or List[int]")

        module = importlib.import_module("BUCToolkit.BatchStructures.BatchStructuresBase")
        batch = module.BatchStructures()
        selected_elements = [self._targets["elements"][frame] for frame in selected]
        batch.append_from_lists(
            add_Sample_ids=[f"btgui_{frame}" for frame in selected],
            add_Cells=[
                np.array(self._targets["cell"][frame], dtype=np.float32, copy=True)
                for frame in selected
            ],
            add_Elements=[
                [str(value) for value in frame_elements]
                for frame_elements in selected_elements
            ],
            add_Numbers=[[1] * len(frame_elements) for frame_elements in selected_elements],
            add_Coords_type=["C"] * len(selected),
            add_Coords=[
                np.array(self._targets["coo"][frame], dtype=np.float32, copy=True)
                for frame in selected
            ],
            add_Fixed=[
                np.ones(len(frame_elements), dtype=np.int8)
                for frame_elements in selected_elements
            ],
        )
        batch.generate_atom_list(force_update=True)
        return batch

    def create_shortcut(self) -> None:
        """Create a desktop shortcut through optional pyshortcuts.

        Args:
            None.

        Returns:
            None. No file is created when the dependency is unavailable.

        Raises:
            RuntimeError: If pyshortcuts is not installed.
        """
        try:
            pyshortcuts = importlib.import_module("pyshortcuts")
        except ImportError as error:
            raise RuntimeError(
                "pyshortcuts is not installed; desktop shortcut was not created"
            ) from error
        pyshortcuts.make_shortcut(str(Path(sys.argv[0]).resolve()), name="btgui")

    @staticmethod
    def shortcut_available() -> bool:
        """Report whether optional pyshortcuts can be imported.

        Args:
            None.

        Returns:
            ``True`` when the desktop-shortcut dependency is installed.
        """
        return importlib.util.find_spec("pyshortcuts") is not None
