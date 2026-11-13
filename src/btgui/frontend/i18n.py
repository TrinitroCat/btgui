"""Small JSON-backed catalog for the main window's visible controls."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_LANGUAGE = "en"
LOCALE_DIR = Path(__file__).with_name("locales")
_LANGUAGE_CACHE: list[tuple[str, str]] | None = None


def _read_locale(path):
    """Return one locale mapping, or ``None`` when the file is unusable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    metadata = data.get("_meta")
    if not isinstance(metadata, dict):
        return None
    code = metadata.get("code")
    name = metadata.get("name")
    if not isinstance(code, str) or not isinstance(name, str):
        return None
    strings = {
        key: value
        for key, value in data.items()
        if not key.startswith("_") and isinstance(value, str)
    }
    return code, name, strings


def available_languages():
    """Return ``(code, native_name)`` pairs found in the locale directory."""
    global _LANGUAGE_CACHE
    if _LANGUAGE_CACHE is not None:
        return list(_LANGUAGE_CACHE)
    languages = []
    if LOCALE_DIR.is_dir():
        for path in sorted(LOCALE_DIR.glob("*.json")):
            parsed = _read_locale(path)
            if parsed is None:
                continue
            code, name, _strings = parsed
            languages.append((code, name))
    if not languages:
        languages = [(DEFAULT_LANGUAGE, "English")]
    languages.sort(key=lambda item: item[1].casefold())
    _LANGUAGE_CACHE = languages
    return list(languages)


def language_codes():
    """Return the available language codes."""
    return [code for code, _name in available_languages()]


class Catalog:
    """Resolve keys with an English fallback for a selected language."""

    def __init__(self, code=DEFAULT_LANGUAGE):
        self.code = code if code in language_codes() else DEFAULT_LANGUAGE
        self._strings = self._load(self.code)
        self._fallback = (
            self._strings
            if self.code == DEFAULT_LANGUAGE
            else self._load(DEFAULT_LANGUAGE)
        )

    @staticmethod
    def _load(code):
        path = LOCALE_DIR / f"{code}.json"
        parsed = _read_locale(path)
        if parsed is None:
            return {}
        return parsed[2]

    def set_language(self, code):
        """Switch the catalog to another available language."""
        if code not in language_codes():
            code = DEFAULT_LANGUAGE
        self.code = code
        self._strings = self._load(code)
        self._fallback = (
            self._strings
            if code == DEFAULT_LANGUAGE
            else self._load(DEFAULT_LANGUAGE)
        )

    def text(self, key):
        """Return the selected string, falling back to English then the key."""
        return self._strings.get(key) or self._fallback.get(key) or key
