# Toolbar icon themes and provenance

Choose **View > Toolbar icons > Modern / Legacy** to change the toolbar
artwork. The choice is saved between sessions. Modern is the default.

## Modern

- Upstream: Tabler Icons (`https://github.com/tabler/tabler-icons`)
- License: MIT
- Local license copy: `toolbar/TABLER_LICENSE.txt`

- SVG masters: `src/stimfit/res/toolbar/svg/*.svg`
- Raster sizes: `src/stimfit/res/toolbar/20/*.png`, `40/*.png`, `60/*.png`

The modern theme continues to use the shipped PNG artwork; some of its SVG
masters contain text, which is outside NanoSVG's supported subset.

## Legacy (0.16.x)

The legacy SVGs are faithful manual redraws of the original 16x16 XPM resources
in `src/stimfit/res`, which are unchanged from Stimfit 0.16.10. They preserve the
original silhouettes, colors, gradients and action mapping using smooth paths
and geometric shapes. They do not embed the old rasters or trace individual
pixel boundaries. The artwork has a transparent 20x20 logical canvas, with the
original 16-unit geometry centered inside it.

- SVG masters: `src/stimfit/res/toolbar/legacy/svg/*.svg`
- PNG fallbacks: `src/stimfit/res/toolbar/legacy/{20,40,60,80}/*.png`
- Original resources and their existing provenance remain in `src/stimfit/res`.
- Open, Save and Print use native `wxArtProvider` icons, as they did in 0.16.x.
- The x/y zoom variants share their original zoom-in or zoom-out artwork.

Legacy SVGs use only paths, geometric shapes, linear gradients and transforms,
so they can be rasterized by wxWidgets' NanoSVG implementation. They do not
require fonts, filters, masks or external references.

To regenerate the transparent PNG fallbacks directly from the SVG masters:

```sh
python3 src/stimfit/res/generate_legacy_toolbar.py
```

The maintenance script requires the librsvg, Cairo and GObject shared libraries
(Debian/Ubuntu: `librsvg2-2`, `libcairo2`, `libglib2.0-0`). It uses Python's
standard library; no Python imaging package is needed. These generation tools
are not additional Stimfit runtime dependencies. Each PNG is rendered from
vectors at its target resolution, rather than enlarging a small bitmap.

## Runtime loading

Toolbar icon loading is implemented in `src/stimfit/gui/toolbaricons.cpp`:

- uses bitmap bundles with a logical 20x20 icon size on supported wxWidgets,
- prefers legacy SVGs when SVG loading is available (`wxHAS_SVG`),
- uses the theme's PNG resolutions when SVG loading is unavailable,
- resolves resource directories for source-tree, build-tree, and install-tree layouts,
- falls back to `wxArtProvider` if the selected theme's artwork is unavailable.

The bundle lets wxWidgets choose artwork for the display's physical pixel
density, including Retina displays. SVG masters and PNG fallbacks are packaged
for Windows, Linux and macOS.

## Modern visual style

Generated icons use a blue/green accent palette with semantic highlights:

- navigation emphasis in green,
- general tools in blue,
- channel 2 accent in red,
- event icon in amber,
- measurement cursor uses a crosshair glyph,
- average/aligned average use a sum (sigma-style) glyph,
- x/y zoom tool variants include a small axis marker label.

## Notes

- App launcher icons (`.ico`/`.icns`) are intentionally unchanged.
