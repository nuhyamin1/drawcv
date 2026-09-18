"""Deterministic font resolution boundary for CSS/SVG typography."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

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
