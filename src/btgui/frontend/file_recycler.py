"""Session recycle bin used by the file manager and overwrite safeguards."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


class FileRecycler:
    """Move or copy files into a temporary, session-scoped recycle bin."""

    def __init__(self):
        self.root = Path(tempfile.gettempdir()) / "btgui" / "recycle_bin"

    def ensure(self):
        """Create the recycle directory when possible."""
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def _unique_path(self, source):
        candidate = self.root / source.name
        if not candidate.exists():
            return candidate
        stem = source.stem
        suffix = source.suffix
        index = 1
        while True:
            candidate = self.root / f"{stem} ({index}){suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    def has_items(self):
        """Return whether the recycle directory currently contains anything."""
        try:
            return self.root.is_dir() and any(self.root.iterdir())
        except OSError:
            return False

    def move(self, path):
        """Move one file or directory into the recycle bin."""
        path = Path(path)
        self.ensure()
        destination = self._unique_path(path)
        try:
            shutil.move(str(path), str(destination))
        except OSError:
            if path.is_file():
                shutil.copy2(path, destination)
                path.unlink()
            else:
                shutil.copytree(path, destination)
                shutil.rmtree(path)
        return destination

    def backup_copy(self, path):
        """Copy an existing file into the recycle bin before overwriting it."""
        path = Path(path)
        if not path.is_file():
            return None
        self.ensure()
        destination = self._unique_path(path)
        shutil.copy2(path, destination)
        return destination

    def clear(self):
        """Delete only the validated recycle directory itself."""
        expected = Path(tempfile.gettempdir()).resolve() / "btgui" / "recycle_bin"
        root = self.root
        if root.is_symlink():
            raise OSError("refusing to remove a symlinked recycle bin")
        resolved = root.resolve()
        if resolved != expected:
            raise OSError(f"refusing to remove unexpected recycle path: {resolved}")
        if root.exists():
            shutil.rmtree(root)
