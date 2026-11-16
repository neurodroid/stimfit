// Toolbar resources shared by the application and icon rendering checks.
// Copyright (C) Stimfit contributors. Licensed under GPL-2.0-or-later.
#ifndef STF_TOOLBARICONS_H
#define STF_TOOLBARICONS_H

#include <wx/bitmap.h>
#include <wx/version.h>
#if wxCHECK_VERSION(3, 1, 6)
#include <wx/bmpbndl.h>
#endif

namespace stf {

#if wxCHECK_VERSION(3, 1, 6)
typedef wxBitmapBundle ToolbarBitmap;
#else
typedef wxBitmap ToolbarBitmap;
#endif

wxString ResolveToolbarIconBaseDir();

// Returns an empty bitmap when no asset is available, allowing the caller to
// use native wxArtProvider icons (as 0.16.x did for Open, Save and Print).
// size is the display size for old wxWidgets; bundles always use 20 DIP.
ToolbarBitmap LoadToolbarIcon(const wxString& baseDir,
                             const wxString& iconName, bool legacy,
                             const wxSize& size = wxSize(20, 20));

} // namespace stf

#endif
