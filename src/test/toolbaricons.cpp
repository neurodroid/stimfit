// Headless checks for the toolbar's vector and raster loading paths.
// Copyright (C) Stimfit contributors. Licensed under GPL-2.0-or-later.
#include "toolbaricons.h"

#include <wx/file.h>
#include <wx/filefn.h>
#include <wx/filename.h>
#include <wx/image.h>
#include <wx/init.h>
#include <wx/log.h>

#include <iostream>
#include <stdexcept>
#include <string>

#if !wxCHECK_VERSION(3, 1, 6)
int main()
{
    std::cout << "Bitmap bundle checks require wxWidgets 3.1.6 or newer\n";
    return 77;
}
#else

namespace {

void Require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

wxString Path(const wxString& base, const wxString& relative)
{
    return wxFileName(base + wxFILE_SEP_PATH + relative).GetFullPath();
}

void MakeDir(const wxString& directory)
{
    Require(wxFileName::DirExists(directory) ||
            wxFileName::Mkdir(directory, wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL),
            "Could not create test fixture directory");
}

// Only remove the unique directory allocated for this invocation. Both test
// executables can run concurrently using the same scratch root.
class TestDirectory {
public:
    explicit TestDirectory(const wxString& scratchRoot)
    {
        MakeDir(scratchRoot);
        m_path = wxFileName::CreateTempFileName(Path(scratchRoot, wxT("toolbaricons-")));
        Require(!m_path.IsEmpty(), "Could not allocate fixture path");
        Require(wxRemoveFile(m_path), "Could not prepare fixture directory");
        MakeDir(m_path);
    }

    ~TestDirectory()
    {
        wxFileName::Rmdir(m_path, wxPATH_RMDIR_RECURSIVE);
    }

    const wxString& GetPath() const { return m_path; }

private:
    wxString m_path;
};

void CopyFixture(const wxString& source, const wxString& destination)
{
    MakeDir(wxFileName(destination).GetPath());
    Require(wxCopyFile(source, destination), "Could not copy test resource");
}

void WriteFixture(const wxString& destination, const char* content)
{
    MakeDir(wxFileName(destination).GetPath());
    wxFile file(destination, wxFile::write);
    Require(file.IsOpened() && file.Write(wxString::FromUTF8(content)),
            "Could not write malformed fixture");
}

void CheckRender(const wxBitmapBundle& bundle, int pixels, const std::string& name)
{
    Require(bundle.IsOk(), name + ": missing bitmap bundle");
    Require(bundle.GetDefaultSize() == wxSize(20, 20),
            name + ": logical icon size changed from 20 DIP");
    const wxBitmap bitmap = bundle.GetBitmap(wxSize(pixels, pixels));
    Require(bitmap.IsOk(), name + ": could not render bitmap");
    Require(bitmap.GetWidth() == pixels && bitmap.GetHeight() == pixels,
            name + ": wrong rendered pixel dimensions");
    const wxImage image = bitmap.ConvertToImage();
    Require(image.IsOk() && image.HasAlpha(), name + ": transparency missing");
    bool visible = false;
    bool transparent = false;
    const unsigned char* alpha = image.GetAlpha();
    for (int i = 0; i < image.GetWidth() * image.GetHeight(); ++i) {
        visible = visible || alpha[i] > 0;
        transparent = transparent || alpha[i] == 0;
    }
    Require(visible && transparent, name + ": empty or opaque icon");
}

void CheckInstalledAssets(const wxString& resources)
{
    const char* legacyNames[] = {
        "first", "previous", "next", "last", "left", "right", "up", "down",
        "fit", "zoom_in_x", "zoom_out_x", "zoom_in_y", "zoom_out_y",
        "ch1", "ch2", "mean", "aligned", "fitdecay", "table", "select",
        "snapshot", "wmf_snapshot", "measure", "peak", "base", "decay",
        "latency", "pslope", "zoom", "event", "annotation"
    };
    const int pixelSizes[] = {20, 25, 30, 40, 60, 80, 100};
    for (const char* name : legacyNames) {
        const wxString iconName = wxString::FromUTF8(name);
        const wxBitmapBundle legacy = stf::LoadToolbarIcon(resources, iconName, true);
        const wxBitmapBundle modern = stf::LoadToolbarIcon(resources, iconName, false);
        for (int pixels : pixelSizes) {
            CheckRender(legacy, pixels, std::string("legacy/") + name);
            CheckRender(modern, pixels, std::string("modern/") + name);
        }
    }
    const char* nativeLegacyNames[] = {"open", "save", "print"};
    for (const char* name : nativeLegacyNames) {
        const wxString iconName = wxString::FromUTF8(name);
        Require(!stf::LoadToolbarIcon(resources, iconName, true).IsOk(),
                std::string("Legacy native-art override unexpectedly supplied: ") + name);
        CheckRender(stf::LoadToolbarIcon(resources, iconName, false), 40, name);
    }
    for (const char* name : {"zoom_in", "zoom_out"}) {
        CheckRender(stf::LoadToolbarIcon(resources, wxString::FromUTF8(name), false), 40, name);
    }
}

void CheckResourceFailures(const wxString& resources, const wxString& fixture)
{
    Require(!stf::LoadToolbarIcon(wxEmptyString, wxT("select"), true).IsOk(),
            "An empty resource directory must defer to native art");
    Require(!stf::LoadToolbarIcon(resources, wxT("does-not-exist"), true).IsOk(),
            "Missing legacy resources must defer to native art");
    Require(!stf::LoadToolbarIcon(resources, wxT("does-not-exist"), false).IsOk(),
            "Missing modern resources must defer to native art");
    Require(!stf::LoadToolbarIcon(Path(fixture, wxT("absent")), wxT("select"), true).IsOk(),
            "A missing resource directory must defer to native art");

    // A vector-only resource distinguishes the SVG implementation from a build
    // configured without it; no PNG can accidentally satisfy this assertion.
    CopyFixture(Path(resources, wxT("legacy/svg/select.svg")),
                Path(fixture, wxT("legacy/svg/vector-only.svg")));
    const wxBitmapBundle vectorOnly = stf::LoadToolbarIcon(fixture, wxT("vector-only"), true);
#if defined(wxHAS_SVG) && !defined(STF_TOOLBAR_DISABLE_SVG)
    CheckRender(vectorOnly, 100, "vector-only");
#else
    Require(!vectorOnly.IsOk(), "SVG-disabled loader unexpectedly loaded vector-only asset");
#endif

    // Broken SVGs must use the PNG fallback. The 20px PNG is deliberately
    // absent, so a larger fallback must still advertise a 20 DIP logical size.
    WriteFixture(Path(fixture, wxT("legacy/svg/fallback.svg")), "this is not SVG");
    CopyFixture(Path(resources, wxT("legacy/40/select.png")),
                Path(fixture, wxT("legacy/40/fallback.png")));
    CheckRender(stf::LoadToolbarIcon(fixture, wxT("fallback"), true), 40, "SVG fallback");
    CheckRender(stf::LoadToolbarIcon(fixture, wxT("fallback"), true), 25, "SVG fallback scaled");

    // A corrupt or incorrectly sized smaller PNG must not hide a valid larger
    // resource. Without the larger resource, it must return an empty bundle.
    WriteFixture(Path(fixture, wxT("20/corrupt.png")), "this is not PNG");
    Require(!stf::LoadToolbarIcon(fixture, wxT("corrupt"), false).IsOk(),
            "Corrupt PNG accepted as a valid resource");
    CopyFixture(Path(resources, wxT("40/select.png")), Path(fixture, wxT("40/corrupt.png")));
    CheckRender(stf::LoadToolbarIcon(fixture, wxT("corrupt"), false), 40, "corrupt PNG fallback");

    CopyFixture(Path(resources, wxT("40/select.png")), Path(fixture, wxT("20/wrong-size.png")));
    Require(!stf::LoadToolbarIcon(fixture, wxT("wrong-size"), false).IsOk(),
            "Incorrectly sized PNG accepted as a valid bucket");
    CopyFixture(Path(resources, wxT("60/select.png")), Path(fixture, wxT("60/wrong-size.png")));
    CheckRender(stf::LoadToolbarIcon(fixture, wxT("wrong-size"), false), 60, "wrong-size PNG fallback");
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: toolbaricons_test RESOURCE_DIRECTORY [SCRATCH_DIRECTORY]\n";
        return 2;
    }
    // GTK bitmap rendering works without a display. Initialize native modules
    // on Windows/Cocoa, where bitmap bundles use platform image objects.
#if defined(__WXMSW__) || defined(__WXOSX__)
    wxInitializer initialization;
    if (!initialization.IsOk()) {
        std::cerr << "Could not initialize wxWidgets for bitmap checks\n";
        return 1;
    }
#endif
    wxInitAllImageHandlers();
    wxLogNull silenceExpectedInvalidResources;
    try {
        const wxString resources = wxString::FromUTF8(argv[1]);
        const wxString scratch = argc == 3 ? wxString::FromUTF8(argv[2]) : wxFileName::GetTempDir();
        TestDirectory fixtures(scratch);
        CheckInstalledAssets(resources);
        CheckResourceFailures(resources, fixtures.GetPath());
        std::cout << "Toolbar resource checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
