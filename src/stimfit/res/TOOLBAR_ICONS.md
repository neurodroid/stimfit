# Toolbar icon provenance

Toolbar icons are provided by Tabler Icons and stored as generated PNG assets.

## Source set

- Upstream: Tabler Icons (`https://github.com/tabler/tabler-icons`)
- License: MIT
- Local license copy: `TABLER_LICENSE.txt`

## Asset layout

- SVG masters: `src/stimfit/res/toolbar/svg/*.svg`
- Raster sizes: `src/stimfit/res/toolbar/20/*.png`, `40/*.png`, `60/*.png`

## Runtime loading

Toolbar icon loading is implemented in `src/stimfit/gui/parentframe.cpp`:

- chooses DIP-aware target size from a logical 20x20 icon size,
- selects the nearest raster bucket (20/40/60),
- resolves resource directories for source-tree, build-tree, and install-tree layouts,
- falls back to `wxArtProvider` if a toolbar PNG is not found.

## Visual style

Generated icons use a blue/green accent palette with semantic highlights:

- navigation emphasis in green,
- general tools in blue,
- channel 2 accent in red,
- event icon in amber,
- measurement cursor uses a crosshair glyph,
- average/aligned average use a sum (sigma-style) glyph,
- x/y zoom tool variants include a small axis marker label.

## Notes

- Legacy toolbar XPM includes/usages were removed from `src/stimfit/gui/parentframe.cpp`.
- App launcher icons (`.ico`/`.icns`) are intentionally unchanged.
