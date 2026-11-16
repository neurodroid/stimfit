#!/usr/bin/env python3
"""Render the hand-drawn legacy SVG masters into the shipped PNG fallbacks.

Requires the librsvg, Cairo and GObject shared libraries. This is an
asset-maintenance tool; none of these are added as Stimfit runtime dependencies.
"""

from pathlib import Path
import ctypes
from ctypes.util import find_library


def library(name: str) -> ctypes.CDLL:
    path = find_library(name)
    if path is None:
        raise SystemExit(f"Install the {name} shared library to regenerate the PNGs")
    return ctypes.CDLL(path)


def function(lib, name, arguments, result):
    method = getattr(lib, name)
    method.argtypes, method.restype = arguments, result
    return method


def main() -> None:
    rsvg, cairo, gobject = library("rsvg-2"), library("cairo"), library("gobject-2.0")
    pointer, integer, real = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    load = function(rsvg, "rsvg_handle_new_from_file", [ctypes.c_char_p, pointer], pointer)
    render = function(rsvg, "rsvg_handle_render_cairo", [pointer, pointer], integer)
    release = function(gobject, "g_object_unref", [pointer], None)
    surface_new = function(cairo, "cairo_image_surface_create", [integer, integer, integer], pointer)
    context_new = function(cairo, "cairo_create", [pointer], pointer)
    scale = function(cairo, "cairo_scale", [pointer, real, real], None)
    save = function(cairo, "cairo_surface_write_to_png", [pointer, ctypes.c_char_p], integer)
    context_free = function(cairo, "cairo_destroy", [pointer], None)
    surface_free = function(cairo, "cairo_surface_destroy", [pointer], None)

    theme = Path(__file__).resolve().parent / "toolbar" / "legacy"
    masters = sorted((theme / "svg").glob("*.svg"))
    if not masters:
        raise SystemExit(f"No SVG masters found in {theme / 'svg'}")

    for size in (20, 40, 60, 80):
        output = theme / str(size)
        output.mkdir(exist_ok=True)
        for master in masters:
            handle = load(bytes(master), None)
            if not handle:
                raise RuntimeError(f"Could not load {master}")
            # Cairo starts with transparent ARGB32 pixels. Rasterize vector
            # geometry at each target resolution, without enlarging a bitmap.
            surface = surface_new(0, size, size)
            context = context_new(surface)
            try:
                scale(context, size / 20.0, size / 20.0)
                if not render(handle, context):
                    raise RuntimeError(f"Could not render {master}")
                target = output / (master.stem + ".png")
                if save(surface, bytes(target)) != 0:
                    raise RuntimeError(f"Could not write {target}")
            finally:
                context_free(context)
                surface_free(surface)
                release(handle)
    print(f"Rendered {len(masters)} SVG masters at 20, 40, 60 and 80 pixels")


if __name__ == "__main__":
    main()
