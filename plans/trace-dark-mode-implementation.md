# Trace dark mode implementation notes

## UI entry point

- Added a new checkable View menu item: `Dark trace display`.
- Event id: `ID_VIEW_DARK_TRACE`.
- Handler: `wxStfParentFrame::OnDarkTraceDisplay`.

## Persisted setting

- Profile key: `/Settings/ViewDarkTraceDisplay`
- Values: `1` (dark on), `0` (dark off)
- Default when missing: `1` (dark mode ON)

## Rendering scope

- Affects on-screen graph rendering only (`wxStfGraph`).
- Print output pens/brushes remain unchanged.

## Palette approach

- Implemented explicit light and dark palettes for graph elements.
- No direct RGB inversion is used.
- Updated items include:
  - trace pens (active/reference/background)
  - cursor and measurement overlays
  - scale bars and text colors
  - event/annotation indicators
  - integral brushes
  - zoom rectangle

## Regression checklist

- [ ] Startup with no existing profile key defaults to dark mode.
- [ ] Toggling View → Dark trace display updates active graph immediately.
- [ ] Scale-bar labels remain readable in dark mode.
- [ ] Secondary channel labels are readable and distinct.
- [ ] Event arrows, annotation lines, and cursor overlays are visible in dark mode.
- [ ] Printing retains legacy light print palette.
