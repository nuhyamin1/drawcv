"""Deterministic font resolution boundary for CSS/SVG typography."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path
import sys

from drawcv.core.exceptions import ValidationError
from drawcv.typography.assets import FontAsset


@dataclass(frozen=True)
class ResolvedFontDescriptor:
    """Descriptor capturing resolved font assets along with semantic identity and substitution state."""
    semantic_family: str
    fonts: tuple[FontAsset, ...]
    is_substituted: bool = False

    def __post_init__(self):
        if not isinstance(self.semantic_family, str) or not self.semantic_family.strip():
            raise ValidationError("ResolvedFontDescriptor semantic_family must be a nonempty string")
        if not isinstance(self.fonts, tuple) or not self.fonts or not all(isinstance(f, FontAsset) for f in self.fonts):
            raise ValidationError("ResolvedFontDescriptor fonts must be a nonempty tuple of FontAsset instances")


class FontResolver(ABC):
    """Abstract font resolver converting semantic CSS font specifications to resolved FontAsset tuples."""

    @abstractmethod
    def resolve(
        self,
        family: str | None,
        weight: str | int | None = None,
        style: str | None = None,
    ) -> ResolvedFontDescriptor | None:
        """Resolve semantic CSS font descriptor to concrete FontAsset descriptor."""
        pass


def _normalize_family_name(family: str) -> str:
    cleaned = family.strip().strip("'\"").strip()
    return " ".join(cleaned.split()).lower()


def _normalize_weight(weight: str | int | None) -> str:
    if weight is None:
        return "normal"
    w_str = str(weight).strip().lower()
    if w_str in ("normal", "400"):
        return "normal"
    if w_str in ("bold", "700"):
        return "bold"
    return w_str


def _normalize_style(style: str | None) -> str:
    if style is None:
        return "normal"
    s_str = str(style).strip().lower()
    if s_str in ("normal", "italic", "oblique"):
        return s_str
    return s_str


class DictFontResolver(FontResolver):
    """Deterministic, dictionary-backed font resolver with explicit registration.
    
    Performs zero host OS font discovery or network access.
    """

    def __init__(
        self,
        default_fonts: tuple[FontAsset, ...] | FontAsset | None = None,
        default_family: str = "Noto Sans",
    ):
        self._registry: dict[tuple[str, str, str], tuple[str, tuple[FontAsset, ...]]] = {}
        if default_fonts is not None:
            if isinstance(default_fonts, FontAsset):
                self.default_fonts: tuple[FontAsset, ...] | None = (default_fonts,)
            else:
                self.default_fonts = tuple(default_fonts)
        else:
            self.default_fonts = None
        self.default_family = default_family

    def register(
        self,
        family: str,
        fonts: tuple[FontAsset, ...] | FontAsset,
        weight: str | int | None = None,
        style: str | None = None,
    ) -> None:
        """Explicitly register a font family with its resolved FontAsset instances."""
        if not isinstance(family, str) or not family.strip():
            raise ValidationError("Font family name must be a nonempty string")
        norm_fam = _normalize_family_name(family)
        norm_w = _normalize_weight(weight)
        norm_s = _normalize_style(style)

        font_tuple = (fonts,) if isinstance(fonts, FontAsset) else tuple(fonts)
        if not font_tuple or not all(isinstance(f, FontAsset) for f in font_tuple):
            raise ValidationError("Fonts must be a nonempty sequence of FontAsset instances")

        self._registry[(norm_fam, norm_w, norm_s)] = (family.strip().strip("'\""), font_tuple)

    def resolve(
        self,
        family: str | None,
        weight: str | int | None = None,
        style: str | None = None,
    ) -> ResolvedFontDescriptor | None:
        norm_w = _normalize_weight(weight)
        norm_s = _normalize_style(style)

        if family is not None and family.strip():
            # Support comma-separated font fallback list in CSS style: "Noto Sans, Arial, sans-serif"
            candidates = [c.strip().strip("'\"") for c in family.split(",") if c.strip().strip("'\"")]
            for cand in candidates:
                norm_cand = _normalize_family_name(cand)
                # Try exact weight and style
                if (norm_cand, norm_w, norm_s) in self._registry:
                    canonical, fonts = self._registry[(norm_cand, norm_w, norm_s)]
                    return ResolvedFontDescriptor(semantic_family=canonical, fonts=fonts, is_substituted=False)
                # Fallback to normal weight/style: this is a substitution if requested weight/style wasn't normal
                if (norm_cand, "normal", "normal") in self._registry:
                    canonical, fonts = self._registry[(norm_cand, "normal", "normal")]
                    is_sub = (norm_w != "normal" or norm_s != "normal")
                    return ResolvedFontDescriptor(semantic_family=canonical, fonts=fonts, is_substituted=is_sub)
                # Any other registered variant in the same family
                for (rfam, rw, rs), (canonical, fonts) in self._registry.items():
                    if rfam == norm_cand:
                        return ResolvedFontDescriptor(semantic_family=canonical, fonts=fonts, is_substituted=True)

            # Check generic family aliases
            for cand in candidates:
                norm_cand = _normalize_family_name(cand)
                if norm_cand in ("sans-serif", "serif", "monospace") and self.default_fonts is not None:
                    is_sub = (norm_w != "normal" or norm_s != "normal")
                    return ResolvedFontDescriptor(
                        semantic_family=self.default_family,
                        fonts=self.default_fonts,
                        is_substituted=is_sub,
                    )

            # Family specified but not registered: substitute with default if available
            if self.default_fonts is not None:
                return ResolvedFontDescriptor(
                    semantic_family=candidates[0],
                    fonts=self.default_fonts,
                    is_substituted=True,
                )
            return None

        # No family specified: return default font
        if self.default_fonts is not None:
            is_sub = (norm_w != "normal" or norm_s != "normal")
            return ResolvedFontDescriptor(
                semantic_family=self.default_family,
                fonts=self.default_fonts,
                is_substituted=is_sub,
            )

        return None

def _system_font_dirs():
    home = Path.home()
    if sys.platform == "win32":
        return [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
                Path(os.environ.get("LOCALAPPDATA", home / "AppData/Local")) / "Microsoft/Windows/Fonts"]
    if sys.platform == "darwin":
        return [home / "Library/Fonts", Path("/Library/Fonts"), Path("/System/Library/Fonts")]
    data_home = Path(os.environ.get("XDG_DATA_HOME", home / ".local/share"))
    data_dirs = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(os.pathsep)
    return [home / ".fonts", data_home / "fonts", *(Path(d) / "fonts" for d in data_dirs if d)]


class SystemFontResolver(FontResolver):
    """Discover local OpenType families on first use. Custom directories override OS defaults."""

    _generic = {
        "sans-serif": ("Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans", "Liberation Sans", "Noto Sans"),
        "serif": ("Times New Roman", "Times", "DejaVu Serif", "Liberation Serif", "Noto Serif"),
        "cursive": ("Segoe Print", "Apple Chancery", "Comic Sans MS", "URW Chancery L", "Z003"),
        "monospace": ("Consolas", "Menlo", "DejaVu Sans Mono", "Liberation Mono", "Courier New"),
    }

    def __init__(self, directories=None):
        self.directories = tuple(map(Path, directories)) if directories is not None else tuple(_system_font_dirs())
        self._index = None

    def _discover(self):
        try:
            from fontTools.ttLib import TTCollection, TTFont, TTLibError
        except ImportError as exc:
            raise ValidationError('SystemFontResolver requires pydrawcv[typography]') from exc
        index = {}
        for directory in self.directories:
            for path in sorted(directory.rglob("*")):
                suffix = path.suffix.lower()
                if suffix not in (".ttf", ".otf", ".ttc", ".otc"):
                    continue
                try:
                    collection = suffix in (".ttc", ".otc")
                    source = TTCollection(path, lazy=True) if collection else TTFont(path, lazy=True)
                    with source:
                        for i, face in enumerate(source.fonts if collection else [source]):
                            names = face["name"]
                            family = names.getBestFamilyName()
                            if not family:
                                continue
                            flags = face["head"].macStyle
                            weight = "bold" if flags & 1 else "normal"
                            style = "italic" if flags & 2 else "normal"
                            for alias in (family, names.getDebugName(1)):
                                if alias:
                                    index.setdefault((_normalize_family_name(alias), weight, style),
                                                     (family, path, i if collection else None))
                except (OSError, TTLibError, KeyError):
                    continue
        return index

    @staticmethod
    @lru_cache(maxsize=32)
    def _asset(path, face_index):
        if face_index is None:
            return FontAsset.from_file(path)
        from fontTools.ttLib import TTFont
        with TTFont(path, fontNumber=face_index) as face:
            data = BytesIO()
            face.save(data)
            return FontAsset(f"{path.stem}-{face_index}.ttf", data.getvalue())

    def resolve(self, family, weight=None, style=None):
        if self._index is None:
            self._index = self._discover()
        requested = [_normalize_family_name(f) for f in (family or "sans-serif").split(",")]
        w, s = _normalize_weight(weight), _normalize_style(style)
        if w.isdecimal():
            w = "bold" if int(w) >= 600 else "normal"
        if s == "oblique":
            s = "italic"
        assets = []
        primary = None
        for candidate in requested:
            for alias in self._generic.get(candidate, (candidate,)):
                name = _normalize_family_name(alias)
                match = self._index.get((name, w, s))
                substituted = match is None
                if match is None:
                    match = self._index.get((name, "normal", "normal"))
                if match is None:
                    continue
                canonical, path, face_index = match
                asset = self._asset(path, face_index)
                if primary is None:
                    primary = canonical, substituted
                if asset not in assets:
                    assets.append(asset)
                break
        if primary is None:
            return None
        return ResolvedFontDescriptor(primary[0], tuple(assets), primary[1])
