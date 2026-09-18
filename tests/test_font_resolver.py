import pytest
from drawcv.core.exceptions import ValidationError
from drawcv.typography.assets import FontAsset
from drawcv.typography.resolver import DictFontResolver, ResolvedFontDescriptor


def test_resolved_font_descriptor_validation():
    fa = FontAsset("mock", b"\x00\x01\x00\x00mockfontbytes")
    desc = ResolvedFontDescriptor(semantic_family="Noto Sans", fonts=(fa,), is_substituted=False)
    assert desc.semantic_family == "Noto Sans"
    assert desc.fonts == (fa,)
    assert not desc.is_substituted

    with pytest.raises(ValidationError):
        ResolvedFontDescriptor(semantic_family="", fonts=(fa,))
    with pytest.raises(ValidationError):
        ResolvedFontDescriptor(semantic_family="Test", fonts=())
    with pytest.raises(ValidationError):
        ResolvedFontDescriptor(semantic_family="Test", fonts=("not a font asset",))  # type: ignore


def test_dict_font_resolver_registration_and_exact_lookup():
    fa_noto = FontAsset("noto", b"\x00\x01\x00\x00notofontbytes")
    fa_roboto = FontAsset("roboto", b"\x00\x01\x00\x00robotobytes")
    fa_roboto_bold = FontAsset("roboto_bold", b"\x00\x01\x00\x00robotoboldbytes")

    resolver = DictFontResolver(default_fonts=(fa_noto,), default_family="Noto Sans")
    resolver.register("Roboto", fa_roboto)
    resolver.register("Roboto", (fa_roboto_bold,), weight="bold")

    # Exact normal match
    res_normal = resolver.resolve("Roboto")
    assert res_normal is not None
    assert res_normal.semantic_family == "Roboto"
    assert res_normal.fonts == (fa_roboto,)
    assert not res_normal.is_substituted

    # Exact bold match
    res_bold = resolver.resolve("Roboto", weight="bold")
    assert res_bold is not None
    assert res_bold.semantic_family == "Roboto"
    assert res_bold.fonts == (fa_roboto_bold,)
    assert not res_bold.is_substituted

    # Fallback to normal if requested style is not registered must be marked as substituted
    res_italic = resolver.resolve("Roboto", style="italic")
    assert res_italic is not None
    assert res_italic.semantic_family == "Roboto"
    assert res_italic.fonts == (fa_roboto,)
    assert res_italic.is_substituted is True

    # Requested weight not registered (e.g. 900) falls back to normal and is substituted
    res_black = resolver.resolve("Roboto", weight="900")
    assert res_black is not None
    assert res_black.semantic_family == "Roboto"
    assert res_black.is_substituted is True


def test_dict_font_resolver_fallback_chain():
    fa_noto = FontAsset("noto", b"\x00\x01\x00\x00notofontbytes")
    fa_source = FontAsset("source", b"OTTOsourcebytes")

    resolver = DictFontResolver(default_fonts=fa_noto, default_family="Noto Sans")
    resolver.register("Source Sans 3", fa_source)

    # First candidate not registered, second candidate registered
    res = resolver.resolve("'Custom Font', 'Source Sans 3', sans-serif")
    assert res is not None
    assert res.semantic_family == "Source Sans 3"
    assert res.fonts == (fa_source,)
    assert not res.is_substituted

    # Generic alias
    res_generic = resolver.resolve("sans-serif")
    assert res_generic is not None
    assert res_generic.semantic_family == "Noto Sans"
    assert res_generic.fonts == (fa_noto,)
    assert not res_generic.is_substituted

    # Unregistered family substitutes with default and marks is_substituted=True
    res_unregistered = resolver.resolve("UnknownFont")
    assert res_unregistered is not None
    assert res_unregistered.semantic_family == "UnknownFont"
    assert res_unregistered.fonts == (fa_noto,)
    assert res_unregistered.is_substituted is True

    # None family resolves to default font without substitution
    res_none = resolver.resolve(None)
    assert res_none is not None
    assert res_none.semantic_family == "Noto Sans"
    assert res_none.fonts == (fa_noto,)
    assert not res_none.is_substituted


def test_dict_font_resolver_without_default():
    resolver = DictFontResolver()
    assert resolver.resolve("Unknown") is None
    assert resolver.resolve(None) is None
