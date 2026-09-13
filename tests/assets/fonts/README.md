# Typography evaluation fonts

These unmodified fonts are test assets, not installed system fonts or a new default
font for pydrawcv. They are excluded from the wheel by the existing package rules.

| File | Source project | Outlines |
| --- | --- | --- |
| notosans.ttf | Noto Sans / Google Fonts | Variable TrueType |
| notosansthai.ttf | Noto Sans Thai / Google Fonts | Variable TrueType |
| notosansarabic.ttf | Noto Sans Arabic / Google Fonts | Variable TrueType |
| sourcesans3.otf | Adobe Source Sans 3 | CFF OpenType |

Each font has its original copyright and SIL Open Font License in the adjacent
`*-OFL.txt` file. `manifest.json` records immutable upstream commit URLs and SHA-256
hashes for both fonts and licenses. Tests verify all those hashes. Only the default
variation instances are evaluated; axis animation is not implemented.
