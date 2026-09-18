"""Optional font typography. Importing descriptors does not load native engines."""
from .assets import FontAsset
from .metrics import TextMetrics, TextLineMetrics
from .resolver import FontResolver, DictFontResolver, ResolvedFontDescriptor

__all__ = [
    "FontAsset",
    "TextMetrics",
    "TextLineMetrics",
    "FontResolver",
    "DictFontResolver",
    "ResolvedFontDescriptor",
]
