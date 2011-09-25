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

IMPLEMENT_WXWIN_MAIN
IMPLEMENT_WX_THEME_SUPPORT
