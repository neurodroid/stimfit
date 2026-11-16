// Copyright (C) Stimfit contributors. Licensed under GPL-2.0-or-later.
#include "toolbaricons.h"

#include <wx/filename.h>
#include <wx/image.h>
#include <wx/log.h>
#include <wx/stdpaths.h>
#include <wx/utils.h>

namespace {

wxFileName IconPath(const wxString& baseDir, const wxString& subDir,
                    const wxString& iconName, const wxString& extension)
{
    wxFileName path(baseDir, wxEmptyString);
    path.AppendDir(subDir);
    path.SetFullName(iconName + extension);
    return path;
}

} // namespace

wxString stf::ResolveToolbarIconBaseDir()
{
    static wxString cached;
    if (!cached.IsEmpty()) {
        return cached;
    }

    wxFileName exePath(wxStandardPaths::Get().GetExecutablePath());
    wxFileName climb(exePath.GetPath(), wxEmptyString);
    wxArrayString candidates;

    auto addCandidate = [&candidates](const wxFileName& anchor, const wxArrayString& dirs) {
        wxFileName path(anchor);
        for (size_t i = 0; i < dirs.GetCount(); ++i) {
            path.AppendDir(dirs[i]);
        }
        path.Normalize(wxPATH_NORM_DOTS);
        candidates.Add(path.GetPath());
    };

    // Prefer installed resources, including the bundle's Contents/Resources.
    // A source checkout used to build the executable may still exist, but must
    // not shadow the resources shipped with an installed application.
    for (int depth = 0; depth <= 6; ++depth) {
        wxArrayString installDirs;
        installDirs.Add(wxT("share"));
        installDirs.Add(wxT("stimfit"));
        installDirs.Add(wxT("toolbar"));
        addCandidate(climb, installDirs);
#ifdef __WXMAC__
        wxArrayString bundleDirs;
        bundleDirs.Add(wxT("Resources"));
        bundleDirs.Add(wxT("toolbar"));
        addCandidate(climb, bundleDirs);
#endif
        if (climb.GetDirCount() == 0) {
            break;
        }
        climb.RemoveLastDir();
    }

#ifdef STF_TOOLBAR_INSTALL_DIR
    candidates.Add(wxString::FromUTF8(STF_TOOLBAR_INSTALL_DIR));
#endif
#ifdef STF_TOOLBAR_SOURCE_DIR
    candidates.Add(wxString::FromUTF8(STF_TOOLBAR_SOURCE_DIR));
#endif

    climb = wxFileName(exePath.GetPath(), wxEmptyString);
    for (int depth = 0; depth <= 6; ++depth) {
        wxArrayString sourceDirs;
        sourceDirs.Add(wxT("src"));
        sourceDirs.Add(wxT("stimfit"));
        sourceDirs.Add(wxT("res"));
        sourceDirs.Add(wxT("toolbar"));
        addCandidate(climb, sourceDirs);
        if (climb.GetDirCount() == 0) {
            break;
        }
        climb.RemoveLastDir();
    }

    wxFileName cwdSource(wxGetCwd(), wxEmptyString);
    wxArrayString sourceDirs;
    sourceDirs.Add(wxT("src"));
    sourceDirs.Add(wxT("stimfit"));
    sourceDirs.Add(wxT("res"));
    sourceDirs.Add(wxT("toolbar"));
    addCandidate(cwdSource, sourceDirs);

    for (size_t i = 0; i < candidates.GetCount(); ++i) {
        if (IconPath(candidates[i], wxT("20"), wxT("open"), wxT(".png")).FileExists()) {
            cached = candidates[i];
            return cached;
        }
    }
    return wxEmptyString;
}

stf::ToolbarBitmap stf::LoadToolbarIcon(const wxString& baseDir,
                                      const wxString& iconName, bool legacy,
                                      const wxSize& size)
{
    if (baseDir.IsEmpty()) {
        return ToolbarBitmap();
    }

    wxFileName themeDir(baseDir, wxEmptyString);
    if (legacy) {
        themeDir.AppendDir(wxT("legacy"));
    }
    const wxString themePath = themeDir.GetPath();

    // Keep modern PNGs: some of their SVG masters contain text, which NanoSVG
    // doesn't render. Legacy masters use only supported vector geometry.
#if wxCHECK_VERSION(3, 1, 6) && defined(wxHAS_SVG) && !defined(STF_TOOLBAR_DISABLE_SVG)
    if (legacy) {
        const wxFileName svg = IconPath(themePath, wxT("svg"), iconName, wxT(".svg"));
        if (svg.FileExists()) {
            wxLogNull silenceInvalidAsset;
            const wxBitmapBundle bundle = wxBitmapBundle::FromSVGFile(svg.GetFullPath(), wxSize(20, 20));
            if (bundle.IsOk()) {
                return bundle;
            }
        }
    }
#endif

    const int buckets[] = {20, 40, 60, 80};
#if wxCHECK_VERSION(3, 1, 6)
    wxUnusedVar(size);
    wxVector<wxBitmap> bitmaps;
    for (size_t i = 0; i < WXSIZEOF(buckets); ++i) {
        const wxFileName png = IconPath(themePath, wxString::Format(wxT("%d"), buckets[i]),
                                        iconName, wxT(".png"));
        if (!png.FileExists()) {
            continue;
        }
        wxLogNull silenceInvalidAsset;
        wxImage image(png.GetFullPath(), wxBITMAP_TYPE_PNG);
        if (image.IsOk() && image.GetWidth() == buckets[i] && image.GetHeight() == buckets[i]) {
            wxBitmap bitmap(image);
            // Keep the default size at 20 DIP even if a smaller asset is absent.
            bitmap.SetScaleFactor(buckets[i] / 20.0);
            bitmaps.push_back(bitmap);
        }
    }
    return wxBitmapBundle::FromBitmaps(bitmaps);
#else
    // Compatibility for older wxWidgets without SVG/bitmap bundle support.
    wxImage best;
    const int target = wxMax(size.GetWidth(), size.GetHeight());
    for (size_t i = 0; i < WXSIZEOF(buckets); ++i) {
        const wxFileName png = IconPath(themePath, wxString::Format(wxT("%d"), buckets[i]),
                                        iconName, wxT(".png"));
        if (!png.FileExists()) {
            continue;
        }
        wxLogNull silenceInvalidAsset;
        wxImage image(png.GetFullPath(), wxBITMAP_TYPE_PNG);
        if (image.IsOk()) {
            best = image;
            if (buckets[i] >= target) {
                break;
            }
        }
    }
    return best.IsOk() ? wxBitmap(best.Scale(size.x, size.y, wxIMAGE_QUALITY_HIGH)) : wxBitmap();
#endif
}
