// For compilers that support precompilation, includes "wx/wx.h".
#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include "./app.h"

wxAppConsole *wxCreateApp() {
    wxAppConsole::CheckBuildOptions(WX_BUILD_OPTIONS_SIGNATURE, "your program"); 
    return new wxStfApp; 
}

wxAppInitializer wxTheAppInitializer((wxAppInitializerFunction) wxCreateApp);

#if defined(WITH_PYTHON) && defined(__WXMSW__) && wxCHECK_VERSION(3, 3, 0)
extern "C" int WINAPI WinMain(wxIN HINSTANCE hInstance,
                             wxIN_OPT HINSTANCE hPrevInstance,
                             wxIN wxCmdLineArgType lpCmdLine,
                             wxIN int nCmdShow)
{
    wxDISABLE_DEBUG_SUPPORT();
    const int result = wxEntry(hInstance, hPrevInstance, lpCmdLine, nCmdShow);
    // wxEntry has now destroyed native event handlers and their Python
    // callbacks. Exit_wxPython restored the main thread's GIL before cleanup.
    if (wxStfApp::FinalizePython() < 0 && result == 0)
        return 120;
    return result;
}
#else
IMPLEMENT_WXWIN_MAIN
#endif
IMPLEMENT_WX_THEME_SUPPORT
