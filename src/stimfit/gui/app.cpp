// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

// app.cpp
// The application, derived from wxApp
// 2007-12-27, Christoph Schmidt-Hieber, University of Freiburg

#include <sstream>

// For compilers that support precompilation, includes "wx/wx.h".
#include <wx/wxprec.h>

#include <wx/memory.h>
#include <wx/progdlg.h>
#include <wx/cmdline.h>
#include <wx/thread.h>
#include <wx/evtloop.h>
#include <wx/init.h>
#include <wx/datetime.h>
#include <wx/filename.h>
#include <wx/stockitem.h>

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#if !wxUSE_DOC_VIEW_ARCHITECTURE
#error You must set wxUSE_DOC_VIEW_ARCHITECTURE to 1 in setup.h!
#endif

#if !wxUSE_MDI_ARCHITECTURE
#error You must set wxUSE_MDI_ARCHITECTURE to 1 in setup.h!
#endif

#ifdef _WINDOWS
#include "../../stfconf.h"
#else
#include "stfconf.h"
#endif
#include "./app.h"
#include "./doc.h"
#include "./view.h"
#include "./parentframe.h"
#include "./childframe.h"
#include "./graph.h"
#include "./dlgs/cursorsdlg.h"
#include "./dlgs/smalldlgs.h"
#include "./../math/funclib.h"

#if defined(__LINUX__) || defined(__WXMAC__)
#include "./../../libstfio/abf/axon/Common/axodefn.h"
#include "./../../libstfio/abf/axon/AxAbfFio32/abffiles.h"
#endif
#include "./../math/fit.h"

#ifdef __WXMAC__
#include <ApplicationServices/ApplicationServices.h>
#endif

// #if defined(_WINDOWS) || defined (STF_TEST)
extern wxStfApp& wxGetApp();
wxStfApp& wxGetApp() { return *static_cast<wxStfApp*>(wxApp::GetInstance()); }
// #endif
// #if !defined(_WINDOWS) && !defined(STF_TEST)
// IMPLEMENT_APP(wxStfApp)
// #endif

wxStfParentFrame *frame = (wxStfParentFrame *) NULL;

BEGIN_EVENT_TABLE( wxStfApp, wxApp )
EVT_KEY_DOWN( wxStfApp::OnKeyDown )

EVT_MENU( ID_CURSORS, wxStfApp::OnCursorSettings )
EVT_MENU( ID_NEWFROMSELECTED, wxStfApp::OnNewfromselected )
EVT_MENU( ID_NEWFROMALL, wxStfApp::OnNewfromall )
EVT_MENU( ID_APPLYTOALL, wxStfApp::OnApplytoall )

#ifdef WITH_PYTHON
EVT_MENU( ID_IMPORTPYTHON, wxStfApp::OnPythonImport )
EVT_MENU_RANGE(ID_USERDEF, ID_USERDEF+32, wxStfApp::OnUserdef)
#endif // WITH_PYTHON
END_EVENT_TABLE()

wxStfApp::wxStfApp(void) : directTxtImport(false), isBars(true), isHires(false), txtImport(), funcLib(),
    extensionLib(), CursorsDialog(NULL), storedLinFunc( stf::initLinFunc() ), /*m_file_menu(0),*/ m_fileToLoad(wxEmptyString)/*, activeDoc(0)*/ {}

void wxStfApp::OnInitCmdLine(wxCmdLineParser& parser)
{
    wxApp::OnInitCmdLine(parser);

    parser.AddOption(wxT("d"), wxT("dir"),
                     wxT("Working directory to change to"), wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL );
    parser.AddParam(wxT("File to open"),
                    wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL );
}

bool wxStfApp::OnCmdLineParsed(wxCmdLineParser& parser)
{
    // Check if we should change the working directory:
    wxString new_cwd( wxT("\0") );
    if ( parser.Found( wxT("dir"), &new_cwd ) ) {

        // Check whether the directory exists:
        if ( !wxDirExists( new_cwd ) ) {
            wxString msg;
            msg << wxT("New working directory ") << new_cwd << wxT(" doesn't exist.");
            ErrorMsg( msg );
            return false;
        }
        // Change to the new wd:
        if ( !wxSetWorkingDirectory( new_cwd ) ) {
            wxString msg;
            msg << wxT("Couldn't change working directory to ") << new_cwd;
            ErrorMsg( msg );
            return false;
        }
    }
    
    // Get file to load
    if ( parser.GetParamCount() > 0 ) {
        m_fileToLoad = parser.GetParam();
    }

    return wxApp::OnCmdLineParsed(parser);
}

bool wxStfApp::OnInit(void)
{
    if (!wxApp::OnInit()) {
        return false;
    }


#ifdef WITH_PYTHON
    if ( !Init_wxPython() ) {
        // don't start the app if we can't initialize wxPython.
        return false;
    }
#endif
    
    // Config:
    config.reset(new wxFileConfig(wxT("Stimfit")));


    //// Create a document manager
    wxDocManager* docManager = new wxDocManager;
    //// Create a template relating drawing documents to their views
    m_cfsTemplate=new wxDocTemplate( docManager,
                                     wxT("CED filing system"), wxT("*.dat;*.cfs"), wxT(""), wxT("dat;cfs"),
                                     wxT("CFS Document"), wxT("CFS View"), CLASSINFO(wxStfDoc),
                                     CLASSINFO(wxStfView) );

    m_hdf5Template=new wxDocTemplate( docManager,
                                      wxT("hdf5 file"), wxT("*.h5"), wxT(""), wxT("h5"),
                                      wxT("HDF5 Document"), wxT("HDF5 View"), CLASSINFO(wxStfDoc),
                                      CLASSINFO(wxStfView) );

    m_abfTemplate=new wxDocTemplate( docManager,
                                     wxT("Axon binary file"), wxT("*.abf"), wxT(""), wxT("abf"),
                                     wxT("ABF Document"), wxT("ABF View"), CLASSINFO(wxStfDoc),
                                     CLASSINFO(wxStfView) );
#if defined(__LINUX__) || defined(__WXMAC__)
    ABF_Initialize();
#endif
    m_atfTemplate=new wxDocTemplate( docManager,
                                     wxT("Axon text file"), wxT("*.atf"), wxT(""), wxT("atf"),
                                     wxT("ATF Document"), wxT("ATF View"), CLASSINFO(wxStfDoc),
                                     CLASSINFO(wxStfView) );
    m_axgTemplate=new wxDocTemplate( docManager,
                                     wxT("Axograph binary file"), wxT("*.axgd;*.axgx"), wxT(""), wxT("axgd;axgx"),
                                     wxT("AXG Document"), wxT("AXG View"), CLASSINFO(wxStfDoc),
                                     CLASSINFO(wxStfView) );

    m_hekaTemplate=new wxDocTemplate( docManager,
                                     wxT("HEKA file"), wxT("*.dat"), wxT(""), wxT("dat"),
                                     wxT("HEKA Document"), wxT("HEKA View"), CLASSINFO(wxStfDoc),
                                     CLASSINFO(wxStfView) );
#ifdef WITH_BIOSIG
    m_biosigTemplate=new wxDocTemplate( docManager,
                                     wxT("Biosig file"), wxT("*.bs"), wxT(""), wxT("bs"),
                                     wxT("Biosig Document"), wxT("Biosig View"), CLASSINFO(wxStfDoc),
                                     CLASSINFO(wxStfView) );
#endif
#if 0
    m_sonTemplate=new wxDocTemplate( docManager,
                                     wxT("CED Spike 2 (SON) file"), wxT("*.smr"), wxT(""), wxT(""),
                                     wxT("SON Document"), wxT("SON View"), CLASSINFO(wxStfDoc),
                                     CLASSINFO(wxStfView) );
#endif
    m_txtTemplate=new wxDocTemplate( docManager,
                                     wxT("General text file import"), wxT("*.*"), wxT(""), wxT(""),
                                     wxT("Text Document"), wxT("Text View"), CLASSINFO(wxStfDoc),
                                     CLASSINFO(wxStfView) );

    // read last directory from config:
    wxString lastDir = wxGetProfileString( wxT("Settings"), wxT("Last directory"), wxT("") );
    if (lastDir == wxT("") || !wxFileName::DirExists( lastDir )) {
        lastDir = wxFileName::GetCwd();
    }
    docManager->SetLastDirectory( lastDir );

    //// Create the main frame window
    frame = new wxStfParentFrame(docManager, (wxFrame *)NULL,
                                 wxT("Stimfit"), wxDefaultPosition,
#ifndef __WXMAC__
                                 wxSize(1024, 768),
#else
                                 wxSize(640, 480),
#endif
                                 wxDEFAULT_FRAME_STYLE | wxFULL_REPAINT_ON_RESIZE | wxMAXIMIZE);
                                 
#if 0
    frame->SetIcon( wxICON(sample) );
#endif
    // frame->SetIcon(wxIcon(wxT("doc.xbm")));

#ifndef __WXGTK__
    //// Make a menubar
    wxMenu* m_file_menu = new wxMenu;
    //	wxMenu *edit_menu = (wxMenu *) NULL;

    m_file_menu->Append(wxID_OPEN);
    
//#ifdef _WINDOWS
    m_file_menu->AppendSeparator();
    m_file_menu->Append(ID_CONVERT, wxT("Convert file series..."));
//#endif
#ifdef WITH_PYTHON
    m_file_menu->AppendSeparator();
    m_file_menu->Append(
                        ID_IMPORTPYTHON,
                        wxT("&Import Python module...\tCtrl+I"),
                        wxT("Import or reload user-defined Python modules")
                        );
#endif // WITH_PYTHON
    m_file_menu->AppendSeparator();
    m_file_menu->Append(wxID_EXIT);

    // A nice touch: a history of files visited. Use this menu.
    GetDocManager()->FileHistoryLoad( *config );
    GetDocManager()->FileHistoryUseMenu(m_file_menu);
    GetDocManager()->FileHistoryAddFilesToMenu();

    wxMenu *help_menu = new wxMenu;
    help_menu->Append(wxID_HELP);
    help_menu->Append(ID_UPDATE, wxT("&Check for updates"));
    help_menu->Append(wxID_ABOUT);

    wxMenu *m_view_menu = new wxMenu;
#ifdef WITH_PYTHON
    m_file_menu->AppendSeparator();
    m_view_menu->Append(ID_VIEW_SHELL, wxT("&Toggle Python shell"),
                        wxT("Shows or hides the Python shell"));
#endif // WITH_PYTHON

    wxMenuBar *menu_bar = new wxMenuBar;

    menu_bar->Append(m_file_menu, wxT("&File"));
    /*	if (edit_menu)
        menu_bar->Append(edit_menu, wxT("&Edit"));
    */
    menu_bar->Append(m_view_menu, wxT("&View"));

    menu_bar->Append(help_menu, wxT("&Help"));

#ifdef __WXMAC__
    // wxApp::SetExitOnFrameDelete(false);
    wxMenuBar::MacSetCommonMenuBar(menu_bar);
#endif //def __WXMAC__
    //// Associate the menu bar with the frame
#else // __WXGTK__
    wxMenuBar* menu_bar = CreateUnifiedMenuBar();
#endif //__WXGTK__
    frame->SetMenuBar(menu_bar);
    
    frame->Centre(wxBOTH);

    /*    pStatusBar = new wxStatusBar(frame);
          frame->SetStatusBar(pStatusBar);
    */
#if 1 //ndef __WXMAC__
    frame->Show(true);
#endif //ndef __WXMAC__
    
    // check for updates in background:
#ifndef __WXMAC__
    frame->CheckUpdate();
#endif
    // load user-defined plugins:
    // pluginLib = stf::GetPluginLib();
    extensionLib = LoadExtensions();
    
    // load fit function library:
    funcLib = stf::GetFuncLib();

    SetTopWindow(frame);

    if (!m_fileToLoad.empty()) {
        wxDocTemplate* templ=GetDocManager()->FindTemplateForPath(m_fileToLoad);
        wxStfDoc* NewDoc=(wxStfDoc*)templ->CreateDocument(m_fileToLoad,wxDOC_NEW);
        NewDoc->SetDocumentTemplate(templ);
        if (!NewDoc->OnOpenDocument(m_fileToLoad)) {
            ErrorMsg(wxT("Couldn't open file, aborting file import"));
            GetDocManager()->CloseDocument(NewDoc);
            return false;
        }
    }

    return true;
}

int wxStfApp::OnExit()
{
#ifdef WITH_PYTHON
    Exit_wxPython();
#endif

    GetDocManager()->FileHistorySave( *config );

    delete wxDocManager::GetDocumentManager();
    return wxApp::OnExit();
}

// "Fake" registry
void wxStfApp::wxWriteProfileInt(const wxString& main, const wxString& sub, int value) const {
    // create a wxConfig-compatible path:
    wxString path=wxT("/")+main+wxT("/")+sub;
    if (!config->Write(path,(long)value)) {
        ErrorMsg(wxT("Couldn't write application settings"));
        return;
    }
    config->Flush();
}

int wxStfApp::wxGetProfileInt(const wxString& main,const wxString& sub, int default_) const {
    wxString path=wxT("/")+main+wxT("/")+sub;
    return config->Read(path,default_);
}

void wxStfApp::wxWriteProfileString( const wxString& main, const wxString& sub, const wxString& value ) const {
    // create a wxConfig-compatible path:
    wxString path=wxT("/")+main+wxT("/")+sub;
    if (!config->Write(path,value)) {
        ErrorMsg(wxT("Couldn't write application settings"));
        return;
    }
    config->Flush();
}

wxString wxStfApp::wxGetProfileString( const wxString& main, const wxString& sub, const wxString& default_) const {
    wxString path=wxT("/")+main+wxT("/")+sub;
    return config->Read(path,default_);
}


void wxStfApp::OnPeakcalcexecMsg(wxStfDoc* actDoc) {
    if (actDoc==0) {
        actDoc = GetActiveDoc();
        if (!actDoc)
            return;
    }
    
    // wxStfView* actView = (wxStfView*)actDoc->GetFirstView();
    wxStfView* actView = GetActiveView();
    if (actView!=NULL) {
        wxStfGraph* pGraph = actView->GetGraph();
        if (pGraph != NULL)
            pGraph->Refresh();
        else
            return;
    }

    if (CursorsDialog != NULL &&
        CursorsDialog->IsShown() &&
        actView!=NULL &&
        actDoc!=NULL &&
        actDoc->IsInitialized())
    {
        CursorsDialog->SetActiveDoc(actDoc);
        switch (CursorsDialog->CurrentCursor()) {
         case stf::measure_cursor: 
             actDoc->SetMeasCursor(CursorsDialog->GetCursorM());// * GetDocument()->GetSR()));
             break;
         
             //Get limits for peak calculation from the dialog box:
         case stf::peak_cursor: 
             actDoc->SetPeakBeg(CursorsDialog->GetCursor1P());// * GetDocument()->GetSR()));
             actDoc->SetPeakEnd(CursorsDialog->GetCursor2P());// * GetDocument()->GetSR()));
             actDoc->CheckBoundaries();
             break;
         
         case stf::base_cursor: 
             actDoc->SetBaseBeg(CursorsDialog->GetCursor1B());
             actDoc->SetBaseEnd(CursorsDialog->GetCursor2B());
             break;
         
         case stf::decay_cursor: 
             actDoc->SetFitBeg(CursorsDialog->GetCursor1D());
             actDoc->SetFitEnd(CursorsDialog->GetCursor2D());
             break;

         case stf::latency_cursor:
             // Latency start mode
             actDoc->SetLatencyBeg(CursorsDialog->GetCursor1L());
             // set latency mode in wxStfDoc
             actDoc->SetLatencyStartMode(CursorsDialog->GetLatencyStartMode() );
             // write latency start mode in Stimfit Profile
             wxWriteProfileInt(wxT("Settings"), wxT("LatencyStartMode"), CursorsDialog->GetLatencyStartMode() );
            
             // Latency end mode
             actDoc->SetLatencyEnd(CursorsDialog->GetCursor2L());
             actDoc->SetLatencyEndMode(CursorsDialog->GetLatencyEndMode() );
             wxWriteProfileInt(wxT("Settings"), wxT("LatencyEndMode"), CursorsDialog->GetLatencyEndMode() );
             break;
            
         
#ifdef WITH_PSLOPE
            // Get cursor location from the dialog box:
         case stf::pslope_cursor: 

             // first PSlope cursor
             actDoc->SetPSlopeBegMode(CursorsDialog->GetPSlopeBegMode());
             if (actDoc->GetPSlopeBegMode() == stfio::psBeg_manualMode)
                actDoc->SetPSlopeBeg(CursorsDialog->GetCursor1PS());

             // second PSlope cursor
             actDoc->SetPSlopeEndMode(CursorsDialog->GetPSlopeEndMode());
             if (actDoc->GetPSlopeEndMode() == stfio::psEnd_manualMode)
                actDoc->SetPSlopeEnd(CursorsDialog->GetCursor2PS());
             // we take data from CursorsDialog only if we need the DeltaT
             //else if (actDoc->GetPSlopeEndMode() == stfio::psEnd_DeltaTMode){
                actDoc->SetDeltaT(CursorsDialog->GetDeltaT());
             
             break;
#endif 
         case stf::undefined_cursor:
             
                 ErrorMsg(wxT("Undefined cursor in wxStfApp::OnPeakcalcexecMsg()"));
                 return;
             
         default:
             break;
        }


        //Update edit peak limits in the peak calculation dialog box
        if (CursorsDialog->GetPeakAtEnd())
        {	//If 'Upper limit at end of trace' is selected in the dialog box
            //Set upper limit to end of trace
            actDoc->SetPeakEnd((int)actDoc->cur().size()-1);
            try {
                CursorsDialog->UpdateCursors();
            }
            catch (const std::runtime_error& e) {
                ExceptMsg(wxString(e.what(), wxConvLocal));
                return;
            }
            actDoc->SetPeakAtEnd(true);
        }
        // Get number of peak points from the dialog box...
        actDoc->SetPM(CursorsDialog->GetPeakPoints());
        wxWriteProfileInt(wxT("Settings"),wxT("PeakMean"),(int)actDoc->GetPM());
        
        // Get direction from the dialog box
        actDoc->SetDirection(CursorsDialog->GetDirection());
        wxWriteProfileInt(wxT("Settings"),wxT("Direction"), CursorsDialog->GetDirection());
        
        // Get reference for AP kinetics from the dialog box
        actDoc->SetFromBase(CursorsDialog->GetFromBase());
        wxWriteProfileInt(wxT("Settings"),wxT("FromBase"), CursorsDialog->GetFromBase());
        
        // Get slope for threshold:
        actDoc->SetSlopeForThreshold( CursorsDialog->GetSlope() );
        wxString wxsSlope;
        wxsSlope << CursorsDialog->GetSlope();
        wxWriteProfileString(wxT("Settings"), wxT("Slope"), wxsSlope);


    }

    // Calculate peak, base, 20/80 rise time, half duration,
    // ratio of rise/slope, maximum slope and geometrical slope (PSlope).
    try {
        if (actDoc != NULL) {
            actDoc->Measure( );
        }
    }
    catch (const std::out_of_range& e) {
        ExceptMsg(wxString( e.what(), wxConvLocal ));
    }

    // Set fit start cursor to new peak if necessary.
    if (actDoc != NULL && CursorsDialog != NULL && CursorsDialog->GetStartFitAtPeak())
    {
        actDoc->SetFitBeg(actDoc->GetMaxT());
        try {
            CursorsDialog->UpdateCursors();
        }
        catch (const std::runtime_error& e) {
            ExceptMsg(wxString( e.what(), wxConvLocal ));
            return;
        }
    }

    // Updates strings in the result box
    if (actView != NULL) {
        wxStfChildFrame* pChild=(wxStfChildFrame*)actView->GetFrame();
        if (pChild != NULL)
            pChild->UpdateResults();
        wxStfGraph* pGraph = actView->GetGraph();
		if (pGraph != NULL) 
			pGraph->SetFocus();
    }
}

wxMenuBar *wxStfApp::CreateUnifiedMenuBar(wxStfDoc* doc) {
    //// Make a menubar
    wxMenu *file_menu = new wxMenu;

    file_menu->Append(wxID_OPEN);
    file_menu->Append(wxID_CLOSE);
    //	file_menu->Append(wxID_SAVE, wxT("&Save"));
    file_menu->Append(wxID_SAVEAS);

//#ifdef _WINDOWS
    file_menu->AppendSeparator();
    file_menu->Append(ID_CONVERT, wxT("&Convert file series..."));
//#endif
    file_menu->AppendSeparator();
    file_menu->Append(ID_FILEINFO, wxT("File information..."));

    file_menu->AppendSeparator();
    file_menu->Append(ID_MPL, wxT("Create &figure..."));
#ifdef _WINDOWS
    file_menu->Append(ID_PRINT_PRINT, wxT("&Print..."));
    file_menu->Append(ID_PRINT_PAGE_SETUP, wxT("Print &Setup..."));
#endif
    
#ifdef WITH_PYTHON
    file_menu->AppendSeparator();
    file_menu->Append(
                        ID_IMPORTPYTHON,
                        wxT("&Import Python module...\tCtrl+I"),
                        wxT("Import or reload user-defined Python modules")
                        );
#endif // WITH_PYTHON

    file_menu->AppendSeparator();
    file_menu->Append(wxID_EXIT);

#ifndef __WXGTK__
    ((wxStfDoc*)doc)->SetFileMenu( file_menu );
#else
    GetDocManager()->FileHistoryLoad( *config );
#endif
    
    GetDocManager()->FileHistoryUseMenu(file_menu);
    GetDocManager()->FileHistoryAddFilesToMenu();

    wxMenu* m_edit_menu=new wxMenu;
    m_edit_menu->Append(
                        ID_CURSORS,
                        wxT("&Cursor settings..."),
                        wxT("Set cursor position, direction, etc.")
                        );
    m_edit_menu->AppendSeparator();
    m_edit_menu->Append(
                        ID_MYSELECTALL,
                        wxT("&Select all traces"),
                        wxT("Select all traces in this file")
                        );
    m_edit_menu->Append(
                        ID_SELECTSOME,
                        wxT("S&elect some traces..."),
                        wxT("Select every n-th trace in this file")
                        );
    m_edit_menu->Append(
                        ID_UNSELECTALL,
                        wxT("&Unselect all traces"),
                        wxT("Unselect all traces in this file")
                        );
    m_edit_menu->Append(
                        ID_UNSELECTSOME,
                        wxT("U&nselect some traces"),
                        wxT("Unselect some traces in this file")
                        );
    wxMenu *editSub=new wxMenu;
    editSub->Append(
                    ID_NEWFROMSELECTEDTHIS,
                    wxT("&selected traces from this file"),
                    wxT("Create a new window showing all selected traces from this file")
                    );
    editSub->Append(
                    ID_NEWFROMSELECTED,
                    wxT("&selected traces from all files"),
                    wxT("Create a new window showing all selected traces from all files")
                    );
    editSub->Append(ID_NEWFROMALL,
                    wxT("&all traces from all files"),
                    wxT("Create a new window showing all traces from all files")
                    );
    m_edit_menu->AppendSeparator();
    m_edit_menu->AppendSubMenu(editSub,wxT("New window with..."));
    m_edit_menu->Append(
                        ID_CONCATENATE,
                        wxT("&Concatenate selected traces"),
                        wxT("Create one large trace by merging selected traces in this file")
                        );
#if 0
    wxMenu *latencyStartSub=new wxMenu;
    latencyStartSub->AppendCheckItem(ID_LATENCYSTART_MAXSLOPE, wxT("max. slope of second channel"));
    latencyStartSub->AppendCheckItem(ID_LATENCYSTART_HALFRISE, wxT("half-maximal amplitude of second channel"));
    latencyStartSub->AppendCheckItem(ID_LATENCYSTART_PEAK, wxT("peak of second channel"));
    latencyStartSub->AppendCheckItem(ID_LATENCYSTART_MANUAL, wxT("Manual"));
    wxMenu *latencyEndSub=new wxMenu;
    latencyEndSub->AppendCheckItem(ID_LATENCYEND_FOOT, wxT("beginning of event in active channel"));
    latencyEndSub->AppendCheckItem(ID_LATENCYEND_MAXSLOPE, wxT("max. slope of active channel"));
    latencyEndSub->AppendCheckItem(ID_LATENCYEND_HALFRISE, wxT("half-maximal amplitude of active channel"));
    latencyEndSub->AppendCheckItem(ID_LATENCYEND_PEAK, wxT("peak of active channel"));
    latencyEndSub->AppendCheckItem(ID_LATENCYEND_MANUAL, wxT("Manual"));
    m_edit_menu->AppendSeparator();
    m_edit_menu->AppendSubMenu(
                               latencyStartSub,
                               wxT("Measure latency from..."),
                               wxT("Choose starting point of latency measurement")
                               );
    m_edit_menu->AppendSubMenu(
                               latencyEndSub,
                               wxT("Measure latency to..."),
                               wxT("Choose ending point of latency measurement")
                               );
#endif
    m_edit_menu->AppendCheckItem(
                                 ID_LATENCYWINDOW,
                                 wxT("Use peak window for latency cursor"),
                                 wxT("Uses the current peak window to measure the peak in the reference channel")
                                 );
    wxMenu* m_view_menu = new wxMenu;
    m_view_menu->Append(
                        ID_VIEW_RESULTS,
                        wxT("&Results..."),
                        wxT("Select analysis results to be shown in the results table")
                        );
    m_view_menu->Append(
                        ID_APPLYTOALL,
                        wxT("&Apply scaling to all windows"),
                        wxT("Apply this trace's scaling to all other windows")
                        );
    m_view_menu->AppendCheckItem(
                                 ID_SCALE,
                                 wxT("&View scale bars"),
                                 wxT("If checked, use scale bars rather than coordinates")
                                 );
    m_view_menu->AppendCheckItem(
                                 ID_HIRES,
                                 wxT("View &full resolution"),
                                 wxT("If checked, plot large traces at high resolution")
                                 );
    m_view_menu->AppendSeparator();
    m_view_menu->Append(ID_SAVEPERSPECTIVE,wxT("&Save window positions"));
    m_view_menu->Append(ID_LOADPERSPECTIVE,wxT("&Load window positions"));
    m_view_menu->Append(ID_RESTOREPERSPECTIVE,wxT("&Restore default window positions"));
#ifdef WITH_PYTHON
    m_view_menu->AppendSeparator();
    m_view_menu->Append(ID_VIEW_SHELL, wxT("&Toggle Python shell"),
                        wxT("Shows or hides the Python shell"));
#endif // WITH_PYTHON
    wxMenu* ch2Sub=new wxMenu;
    ch2Sub->Append(ID_CH2BASE, wxT("Match &baseline"));
    ch2Sub->Append(ID_CH2POS, wxT("Match &abs. position"));
    ch2Sub->Append(ID_CH2ZOOM, wxT("Match &y-scale"));
    ch2Sub->Append(ID_CH2BASEZOOM, wxT("Match baseline a&nd y-scale"));

    m_view_menu->AppendSeparator();
    m_view_menu->AppendSubMenu(ch2Sub, wxT("&Channel 2 scaling"));
    m_view_menu->Append(ID_SWAPCHANNELS, wxT("&Swap channels"));
    
    wxMenu *analysis_menu = new wxMenu;
    wxMenu *fitSub = new wxMenu;
    fitSub->Append(
                   ID_FIT,
                   wxT("&Nonlinear regression..."),
                   wxT("Fit a function to this trace between fit cursors")
                   );
    fitSub->Append(
                   ID_LFIT,
                   wxT("&Linear fit..."),
                   wxT("Fit a linear function to this trace between fit cursors")
                   );
    analysis_menu->AppendSubMenu(fitSub, wxT("&Fit"));
    wxMenu *transformSub = new wxMenu;
    transformSub->Append(
                         ID_LOG,
                         wxT("&Logarithmic (base e)..."),
                         wxT("Transform selected traces logarithmically")
                         );
    analysis_menu->AppendSubMenu(transformSub, wxT("&Transform"));
    analysis_menu->Append(
                          ID_MULTIPLY,
                          wxT("&Multiply..."),
                          wxT("Multiply selected traces")
                          );
    analysis_menu->Append(
                          ID_INTEGRATE,
                          wxT("&Integrate"),
                          wxT("Integrate this trace between fit cursors")
                          );
    analysis_menu->Append(
                          ID_DIFFERENTIATE,
                          wxT("&Differentiate"),
                          wxT("Differentiate selected traces")
                          );
    analysis_menu->Append(
                          ID_SUBTRACTBASE,
                          wxT("&Subtract baseline"),
                          wxT("Subtract baseline from selected traces")
                          );
    analysis_menu->Append(
                          ID_FILTER,
                          wxT("F&ilter..."),
                          wxT("Filter selected traces")
                          );
    analysis_menu->Append(
                          ID_SPECTRUM,
                          wxT("&Power spectrum..."),
                          wxT("Compute an estimate of the power spectrum of the selected traces")
                          );
    analysis_menu->Append(
                          ID_POVERN,
                          wxT("P over &N correction..."),
                          wxT("Apply P over N correction to all traces of this file")
                          );
    wxMenu* eventPlotSub = new wxMenu;
    eventPlotSub->Append(ID_PLOTCRITERION, wxT("&Detection criterion..."));
    eventPlotSub->Append(ID_PLOTCORRELATION, wxT("&Correlation coefficient..."));
    wxMenu* eventSub = new wxMenu;
    eventSub->AppendSubMenu(eventPlotSub,wxT("Plot"));
    eventSub->Append(ID_EXTRACT,wxT("&Template matching..."));
    eventSub->Append(ID_THRESHOLD,wxT("Threshold &crossing..."));
    analysis_menu->AppendSubMenu(eventSub,wxT("Event detection"));
    analysis_menu->Append(
                          ID_BATCH,
                          wxT("&Batch analysis..."),
                          wxT("Analyze selected traces and show results in a table")
                          );

#if 0
    wxMenu* userdefSub=new wxMenu;
    for (std::size_t n=0;n<GetPluginLib().size();++n) {
        userdefSub->Append(
                           ID_USERDEF1+(int)n,
                           GetPluginLib()[n].menuEntry
                           );
    }
    analysis_menu->AppendSubMenu(userdefSub,wxT("User-defined functions"));
#endif
    wxMenu *extensions_menu = new wxMenu;
    for (std::size_t n=0;n<GetExtensionLib().size();++n) {
        extensions_menu->Append(ID_USERDEF+(int)n,
                                stf::std2wx(GetExtensionLib()[n].menuEntry));
    }
    
    wxMenu *help_menu = new wxMenu;
    help_menu->Append(wxID_HELP, wxT("Online &help\tF1"));
    help_menu->Append(wxID_ABOUT, wxT("&About"));
    help_menu->Append(ID_UPDATE, wxT("&Check for updates"));

    wxMenuBar *menu_bar = new wxMenuBar;

    menu_bar->Append(file_menu, wxT("&File"));
    menu_bar->Append(m_edit_menu, wxT("&Edit"));
    menu_bar->Append(m_view_menu, wxT("&View"));
    menu_bar->Append(analysis_menu, wxT("&Analysis"));
    menu_bar->Append(extensions_menu, wxT("E&xtensions"));
    menu_bar->Append(help_menu, wxT("&Help"));

    return menu_bar;
}

/*
 * Centralised code for creating a document frame.
 * Called from view.cpp when a view is created.
 */
wxStfChildFrame *wxStfApp::CreateChildFrame(wxDocument *doc, wxView *view)
{
    //// Make a child frame
#ifdef __WXMAC__
    int xpos = (GetDocCount()-1) * 16 + 64;
    int ypos = (GetDocCount()-1) * 16 + 80;
#endif
    wxStfChildFrame *subframe = new wxStfChildFrame(
                                                    doc, view, 
#ifdef __WXMAC__
                                                    GetMainFrame(), wxID_ANY, doc->GetTitle(),
                                                    wxPoint(xpos,ypos), wxSize(800,600),
                                                    wxDEFAULT_FRAME_STYLE |
                                                    // wxNO_FULL_REPAINT_ON_RESIZE |
                                                    wxWANTS_CHARS | wxMAXIMIZE
#else
                                                    GetMainFrame(), wxID_ANY, doc->GetTitle(),
                                                    wxDefaultPosition, wxDefaultSize,
                                                    wxDEFAULT_FRAME_STYLE |
                                                    // wxNO_FULL_REPAINT_ON_RESIZE |
                                                    wxWANTS_CHARS | wxMAXIMIZE
#endif
                                                    );

#ifdef __WXMSW__
    subframe->SetIcon(wxString(wxT("chart")));
#endif
#ifdef __X__
    // subframe->SetIcon(wxIcon(wxT("doc.xbm")));
#endif

#ifndef __WXGTK__
    wxMenuBar* menu_bar = CreateUnifiedMenuBar((wxStfDoc*)doc);
    //// Associate the menu bar with the frame
    subframe->SetMenuBar(menu_bar);

#endif // __WXGTK__
    
    return subframe;
}

wxStfDoc* wxStfApp::NewChild(const Recording& NewData, const wxStfDoc* Sender,
                             const wxString& title)
{
    wxStfDoc* NewDoc=(wxStfDoc*)m_cfsTemplate->CreateDocument(title,wxDOC_NEW);
    NewDoc->SetDocumentName(title);
    NewDoc->SetTitle(title);
    NewDoc->SetDocumentTemplate(m_cfsTemplate);
    if (!NewDoc->OnNewDocument()) return NULL;
    try {
        NewDoc->SetData(NewData,Sender,title);
    }
    catch (const std::out_of_range& e) {
        wxString msg;
        msg << wxT("Error while creating new document:\n")
            << wxString( e.what(), wxConvLocal );
        ExceptMsg( msg );
        // Close file:
        NewDoc->OnCloseDocument();
        return NULL;
    }
    catch (const std::runtime_error& e) {
        wxString msg;
        msg << wxT("Runtime error while creating new document:\n")
            << wxString( e.what(), wxConvLocal );
        ExceptMsg( msg );
        // Close file:
        if (!NewDoc->OnCloseDocument())
            ErrorMsg(wxT("Could not close file; please close manually"));
        return NULL;
    }
    return NewDoc;
}

wxStfView* wxStfApp::GetActiveView() const {
    if ( GetDocManager() == 0) {
        ErrorMsg( wxT("Couldn't access the document manager"));
        return NULL;
    }
    return (wxStfView*)GetDocManager()->GetCurrentView();
}

wxStfDoc* wxStfApp::GetActiveDoc() const {
    if ( GetDocManager() == 0) {
        ErrorMsg( wxT("Couldn't access the document manager"));
        return NULL;
    }
    if (GetDocManager()->GetDocuments().empty())
        return NULL;
    wxStfDoc* pDoc = (wxStfDoc*)GetDocManager()->GetCurrentDocument();
    
    return pDoc;
}
/*
void wxStfApp::SetActiveDoc(wxStfDoc* pDoc) {
    if (pDoc != activeDoc.back()) {
        activeDoc.push_back( pDoc );
    }
}
*/
void wxStfApp::OnKeyDown( wxKeyEvent& event ) {
    event.Skip();
    wxStfDoc* actDoc = GetActiveDoc();
    if (!actDoc)
        return;
    
    // wxStfView* actView = (wxStfView*)actDoc->GetFirstView();
    wxStfView* actView = GetActiveView();
    if (actView) {
        wxStfGraph* pGraph = actView->GetGraph();
        wxStfChildFrame* pChild=(wxStfChildFrame*)actView->GetFrame();
        if (pGraph && pChild && pChild->IsActive())
            pGraph->OnKeyDown(event);
    }
}

void wxStfApp::OnCursorSettings( wxCommandEvent& WXUNUSED(event) ) {
    wxStfDoc* actDoc=GetActiveDoc();
    if (CursorsDialog==NULL && actDoc!=NULL) {
        CursorsDialog=new wxStfCursorsDlg(frame, actDoc);
        CursorsDialog->Show();
        CursorsDialog->SetActiveDoc(actDoc);
        //set CEdit controls to given values
        try {
            CursorsDialog->UpdateCursors();
        }
        catch (const std::runtime_error& e) {
            ExceptMsg(wxString( e.what(), wxConvLocal ));
            return;
        }
        //set CButton to given direction
        CursorsDialog->SetDirection(actDoc->GetDirection());
        CursorsDialog->SetPeakPoints((int)actDoc->GetPM());
        CursorsDialog->SetFromBase(actDoc->GetFromBase());
        CursorsDialog->SetSlope( actDoc->GetSlopeForThreshold() );
        return;
    }

    if(CursorsDialog!=NULL && !CursorsDialog->IsShown() && actDoc!=NULL) {
        CursorsDialog->Show();
        CursorsDialog->SetActiveDoc(actDoc);
        //set CEdit controls to given values
        try {
            CursorsDialog->UpdateCursors();
        }
        catch (const std::runtime_error& e) {
            ExceptMsg(wxString( e.what(), wxConvLocal ));
            return;
        }
        //set CButton to given direction
        CursorsDialog->SetDirection(actDoc->GetDirection());
        CursorsDialog->SetPeakPoints((int)actDoc->GetPM());
        CursorsDialog->SetFromBase(actDoc->GetFromBase());
        CursorsDialog->SetSlope( actDoc->GetSlopeForThreshold() );
    }

}

void wxStfApp::OnNewfromselected( wxCommandEvent& WXUNUSED(event) ) {

    // number of selected traces across all open documents:
    std::size_t nwxT=0;
    // Search the document's template list for open documents:
    wxList docList=GetDocManager()->GetDocuments();
    if (docList.IsEmpty()) {
        ErrorMsg(wxT("No traces were found"));
        return;
    }
    // Since random access is expensive, go through the list node by node:
    // Get first node:
    wxObjectList::compatibility_iterator curNode=docList.GetFirst();
    std::size_t n_channels=((wxStfDoc*)curNode->GetData())->size();

    while (curNode!=NULL) {

        wxStfDoc* pDoc=(wxStfDoc*)curNode->GetData();
        if (pDoc->size()!=n_channels) {
            ErrorMsg(wxT("Can't combine files: different numbers of channels"));
            return;
        }
        try {
            nwxT+=pDoc->GetSelectedSections().size();
        }
        catch (const std::out_of_range& e) {
            ExceptMsg(wxString( e.what(), wxConvLocal ));
            return;
        }
        curNode=curNode->GetNext();
    }
    if (nwxT==0) {
        ErrorMsg(wxT("No selected traces were found"));
        return;
    }
    Recording Selected(n_channels,nwxT);
    // Do the same iteration once again filling the channel with data:
    curNode=docList.GetFirst();
    wxStfDoc* pDoc=NULL;

    nwxT=0;
    std::vector<std::vector<std::string> > channel_names(n_channels);
    while (curNode!=NULL) {
        pDoc=(wxStfDoc*)curNode->GetData();
        if (pDoc->GetSelectedSections().size() > 0) {
            for (std::size_t n_c=0;n_c<pDoc->size();++n_c) {
                channel_names[n_c].push_back(pDoc->get()[n_c].GetChannelName());
                for (std::size_t n=0; n<pDoc->GetSelectedSections().size(); ++n) {
                    try {
                        Selected[n_c].InsertSection(
                                                    pDoc->get()[n_c][pDoc->GetSelectedSections()[n]],
                                                    n+nwxT
                                                    );
                    }
                    catch (const std::out_of_range& e) {
                        ExceptMsg(wxString( e.what(), wxConvLocal ));
                        return;
                    }
                }
            }
        }
        nwxT+=pDoc->GetSelectedSections().size();
        curNode=curNode->GetNext();
    }

    // Set channel names:
    for (std::size_t n_c=0;n_c<n_channels;++n_c) {
        std::ostringstream channel_name;
        channel_name << channel_names[n_c][0];
        for (std::size_t n_n=1;n_n<channel_names[n_c].size();++n_n) {
            // add channel name if it hasn't been used yet:
            bool used=false;
            for (int n_used=(int)n_n-1;n_used>=0 && !used;--n_used) {
                // can't use size_t here because
                // n_used might be negative when checking loop condition
                used = ( channel_names[n_c][n_n].compare( channel_names[n_c][n_used] ) == 0 );
            }
            if (!used) {
                channel_name << wxT(", ") << channel_names[n_c][n_n];
            }
        }
        Selected.get()[n_c].SetChannelName(channel_name.str());
    }
    // Copy some variables from the last document's recording
    // to the new recording:
    Selected.CopyAttributes(*pDoc);

    // Create a new document in a new child window, using the settings
    // of the last open document:
    NewChild(Selected,pDoc,wxT("New from selected traces"));
}

void wxStfApp::OnNewfromall( wxCommandEvent& WXUNUSED(event) ) {
    // number of traces in all open documents:
    std::size_t nwxT=0;
    // minimal number of channels:
    // Search the document's template list for open documents:
    wxList docList=GetDocManager()->GetDocuments();
    if (docList.IsEmpty()) {
        ErrorMsg(wxT("No traces were found"));
        return;
    }
    // Since random access is expensive, go through the list node by node:
    // Get first node:
    wxObjectList::compatibility_iterator curNode=docList.GetFirst();
    std::size_t n_channels=((wxStfDoc*)curNode->GetData())->size();
    while (curNode!=NULL) {
        wxStfDoc* pDoc=(wxStfDoc*)curNode->GetData();
        if (pDoc->size()!=n_channels) {
            ErrorMsg(wxT("Can't combine files: different numbers of channels"));
            return;
        }
        try {
            nwxT+=pDoc->get().at(pDoc->GetCurCh()).size();
        }
        catch (const std::out_of_range& e) {
            ExceptMsg(wxString( e.what(), wxConvLocal ));
            return;
        }
        curNode=curNode->GetNext();
    }
    Recording Selected(n_channels,nwxT);
    //Do the same iteration once again filling the channel with data:
    curNode=docList.GetFirst();
    nwxT=0;
    wxStfDoc* pDoc=NULL;
    std::vector<std::vector<std::string> > channel_names(n_channels);
    while (curNode!=NULL) {
        pDoc=(wxStfDoc*)curNode->GetData();
        if (pDoc->get()[pDoc->GetCurCh()].size() > 0) {
            for (std::size_t n_c=0;n_c<n_channels;++n_c) {
                channel_names[n_c].push_back(pDoc->get()[n_c].GetChannelName());
                for (std::size_t n=0; n<pDoc->get()[n_c].size(); ++n) {
                    try {
                        Selected[n_c].InsertSection(pDoc->get()[n_c][n],n+nwxT);
                    }
                    catch (const std::out_of_range& e) {
                        ExceptMsg(wxString( e.what(), wxConvLocal ));
                        return;
                    }
                }
            }
        }
        nwxT+=pDoc->get()[pDoc->GetCurCh()].size();
        curNode=curNode->GetNext();
    }

    // Set channel names:
    for (std::size_t n_c=0;n_c<n_channels;++n_c) {
        std::ostringstream channel_name;
        channel_name << channel_names[n_c][0];
        for (std::size_t n_n=1;n_n<channel_names[n_c].size();++n_n) {
            // add channel name if it hasn't been used yet:
            bool used=false;
            for (int n_used=(int)n_n-1;n_used>=0 && !used;--n_used) {
                // can't use size_t here because
                // n_used might be negative when checking loop condition
                used = ( channel_names[n_c][n_n].compare( channel_names[n_c][n_used] ) == 0 );
            }
            if (!used) {
                channel_name << wxT(", ") << channel_names[n_c][n_n];
            }
        }
        Selected.get()[n_c].SetChannelName(channel_name.str());
    }

    // Copy some variables from the last document's recording
    // to the new recording:
    Selected.CopyAttributes(*pDoc);

    // Create a new document in a new child window, using the settings
    // of the last open document:
    NewChild(Selected,pDoc,wxT("New from all traces"));
}

void wxStfApp::OnApplytoall( wxCommandEvent& WXUNUSED(event) ) {
    // toggle through open documents to find out
    // which one is active:

    // Search the document's template list for open documents:
    wxList docList=GetDocManager()->GetDocuments();
    if (docList.IsEmpty()) {
        ErrorMsg(wxT("No traces were found"));
        return;
    }
    wxStfDoc* pDoc=GetActiveDoc();
    wxStfView* pView=GetActiveView();
    if (pDoc==NULL || pView==NULL) {
        ErrorMsg(wxT("Couldn't find an active window"));
        return;
    }
    std::size_t llbToApply=pDoc->GetBaseBeg();
    std::size_t ulbToApply=pDoc->GetBaseEnd();
    std::size_t llpToApply=pDoc->GetPeakBeg();
    std::size_t ulpToApply=pDoc->GetPeakEnd();
    std::size_t lldToApply=pDoc->GetFitBeg();
    std::size_t uldToApply=pDoc->GetFitEnd();
    double latencyStartCursorToApply=pDoc->GetLatencyBeg();
    double latencyEndCursorToApply=pDoc->GetLatencyEnd();

    // Since random access is expensive, go through the list node by node:
    // Get first node:
    wxObjectList::compatibility_iterator curNode=docList.GetFirst();
    while (curNode!=NULL) {
        wxStfDoc* OpenDoc=(wxStfDoc*)curNode->GetData();
        if (OpenDoc==NULL)
            return;
        wxStfView* curView = (wxStfView*)OpenDoc->GetFirstView(); //(GetActiveView());
        if (curView!=pView && curView!=NULL) {
            OpenDoc->GetXZoomW() = pDoc->GetXZoom();
            for ( std::size_t n_c=0; n_c < OpenDoc->size(); ++n_c ) {
                if ( n_c < pDoc->size() ) {
                    OpenDoc->GetYZoomW(n_c) = pDoc->GetYZoom(n_c);
                }
            }
            OpenDoc->SetBaseBeg((int)llbToApply);
            OpenDoc->SetBaseEnd((int)ulbToApply);
            OpenDoc->SetPeakBeg((int)llpToApply);
            OpenDoc->SetPeakEnd((int)ulpToApply);
            OpenDoc->SetFitBeg((int)lldToApply);
            OpenDoc->SetFitEnd((int)uldToApply);
            OpenDoc->SetLatencyBeg(latencyStartCursorToApply);
            OpenDoc->SetLatencyEnd(latencyEndCursorToApply);
            wxStfChildFrame* pChild=(wxStfChildFrame*)curView->GetFrame();
            pChild->UpdateResults();
            if (curView->GetGraph() != NULL)
                curView->GetGraph()->Refresh();
        }
        curNode=curNode->GetNext();
    }
}

bool wxStfApp::OpenFileSeries(const wxArrayString& fNameArray) {
    int nFiles=(int)fNameArray.GetCount();
    if (nFiles==0) return false;
    bool singleWindow=false;
    if (nFiles!=1) {
        // Ask whether to put files into a single window:
        singleWindow=(wxMessageDialog(
                                      frame,
                                      wxT("Put files into a single window?"),
                                      wxT("File series import"),
                                      wxYES_NO
                                      ).ShowModal() == wxID_YES);
    }
    wxProgressDialog progDlg(
                             wxT("Importing file series"),
                             wxT("Starting file import"),
                             100,
                             frame,
                             wxPD_SMOOTH | wxPD_AUTO_HIDE
                             );
    int n_opened=0;
    Recording seriesRec;
    while (n_opened!=nFiles) {
        wxString progStr;
        progStr << wxT("Reading file #") << n_opened + 1 << wxT(" of ") << nFiles;
        progDlg.Update(
                       (int)((double)n_opened/(double)nFiles*100.0),
                       progStr
                       );
        if (!singleWindow) {
            wxDocTemplate* templ=GetDocManager()->FindTemplateForPath(fNameArray[n_opened]);
            wxStfDoc* NewDoc=(wxStfDoc*)templ->CreateDocument(fNameArray[n_opened],wxDOC_NEW);
            NewDoc->SetDocumentTemplate(templ);
            if (!NewDoc->OnOpenDocument(fNameArray[n_opened++])) {
                ErrorMsg(wxT("Couldn't open file, aborting file import"));
                GetDocManager()->CloseDocument(NewDoc);
                return false;
            }
        } else {
            // Add to recording first:
            // Find a template:
            wxDocTemplate* templ=GetDocManager()->FindTemplateForPath(fNameArray[n_opened]);
            // Use this template only for type recognition:
            wxString filter(templ->GetFileFilter());
            stfio::filetype type=
                stfio::findType(stf::wx2std(templ->GetFileFilter()));
#if 0 // TODO: re-implement ascii
            if (type==stfio::ascii) {
                if (!get_directTxtImport()) {
                    wxStfTextImportDlg ImportDlg(NULL, stf::CreatePreview(fNameArray[n_opened]), 1,
                                                 true);
                    if (ImportDlg.ShowModal()!=wxID_OK) {
                        return false;
                    }
                    // store settings in application:
                    set_txtImportSettings(ImportDlg.GetTxtImport());
                    set_directTxtImport(ImportDlg.ApplyToAll());
                }
            }
#endif
            // add this file to the series recording:
            Recording singleRec;
            try {
                stf::wxProgressInfo progDlg("Reading file", "Opening file", 100);
                stfio::importFile(stf::wx2std(fNameArray[n_opened++]),type,singleRec,txtImport, progDlg);
                if (n_opened==1) {
                    seriesRec.resize(singleRec.size());
                    // reserve memory to avoid allocations:
                    for (std::size_t n_c=0;n_c<singleRec.size();++n_c) {
                        seriesRec[n_c].reserve(singleRec[n_c].size()*nFiles);
                    }
                    seriesRec.SetXScale(singleRec.GetXScale());
                }
                seriesRec.AddRec(singleRec);
            }
            catch (const std::runtime_error& e) {
                wxString errorMsg;
                errorMsg << wxT("Couldn't open file, aborting file import:\n")
                         << stf::std2wx(e.what());
                ErrorMsg(errorMsg);
                return false;
            }
            catch (const std::out_of_range& e) {
                wxString errorMsg;
                errorMsg << wxT("Couldn't open file, aborting file import:\n")
                         << wxString( e.what(), wxConvLocal );
                ErrorMsg(errorMsg);
                return false;
            }
            // check whether this was the last file in the queue:
            if (n_opened==nFiles) {
                NewChild(seriesRec,NULL,wxT("File series"));
            }
        }
    }
    // reset direct import:
    directTxtImport=false;
    return true;
}

#ifdef WITH_PYTHON
bool wxStfApp::OpenFilePy(const wxString& filename) {
    wxDocTemplate* templ = GetDocManager()->FindTemplateForPath( filename );
    if ( templ == NULL ) {
        ErrorMsg(wxT("Couldn't open file, aborting file import"));
        return false;
    }
    wxStfDoc* NewDoc = (wxStfDoc*)templ->CreateDocument( filename, wxDOC_NEW );
    if ( NewDoc == NULL ) {
        ErrorMsg(wxT("Couldn't open file, aborting file import"));
        return false;
    }
    NewDoc->SetDocumentTemplate(templ);
    if (!NewDoc->OnOpenPyDocument(filename)) {
        ErrorMsg(wxT("Couldn't open file, aborting file import"));
        GetDocManager()->CloseDocument(NewDoc);
        return false;
    }
    return true;
}
#endif //WITH_PYTHON

void wxStfApp::CleanupDocument(wxStfDoc* pDoc) {
    // count open docs:
    if (GetDocManager() && GetDocManager()->GetDocuments().GetCount()==1) {
        // Clean up if this was the last document:
        if (CursorsDialog!=NULL) {
            CursorsDialog->Destroy();
            CursorsDialog=NULL;
        }
    }
    // Update active document:
    /* activeDoc.remove(pDoc); */
    // Remove menu from file history menu list:
    // GetDocManager()->FileHistoryUseMenu(m_file_menu);
    // GetDocManager()->FileHistoryAddFilesToMenu();
}

std::vector<Section*> wxStfApp::GetSectionsWithFits() const {
    // Search the document's template list for open documents:
    wxList docList=GetDocManager()->GetDocuments();
    if (docList.IsEmpty()) {
        return std::vector<Section*>(0);
    }
    std::vector<Section*> sectionList;
    // Since random access is expensive, go through the list node by node:
    // Get first node:
    wxObjectList::compatibility_iterator curNode=docList.GetFirst();
    while (curNode!=NULL) {
        wxStfDoc* pDoc=(wxStfDoc*)curNode->GetData();
        try {
            for (std::size_t n_sec=0;n_sec<pDoc->get().at(pDoc->GetCurCh()).size();++n_sec) {
                if (pDoc->get().at(pDoc->GetCurCh()).at(n_sec).IsFitted()) {
                    sectionList.push_back(&pDoc->get()[pDoc->GetCurCh()][n_sec]);
                }
            }
        }
        catch (const std::out_of_range& e) {
            ExceptMsg( wxString( e.what(), wxConvLocal ) );
            return std::vector<Section*>(0);
        }
        curNode=curNode->GetNext();
    }
    return sectionList;
}

wxString wxStfApp::GetVersionString() const {
    wxString verString;
    verString << wxT("Stimfit ")
              << wxString(VERSION, wxConvLocal)
#ifdef _STFDEBUG
              << wxT(", debug build, ");
#else
    << wxT(", release build, ");
#endif

    verString << wxT(__DATE__) << wxT(", ") << wxT(__TIME__);

    return verString;
}

wxStfParentFrame *GetMainFrame(void)
{
    return frame;
}

stf::wxProgressInfo::wxProgressInfo(const std::string& title, const std::string& message, int maximum, bool verbose)
    : ProgressInfo(title, message, maximum, verbose),
      pd(stf::std2wx(title), stf::std2wx(message), maximum, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL )
{
    
}

bool stf::wxProgressInfo::Update(int value, const std::string& newmsg, bool* skip) {
    return pd.Update(value, stf::std2wx(newmsg), skip);
}

std::string stf::wx2std(const wxString& wxs) {
    return std::string(wxs.mb_str());
}
wxString stf::std2wx(const std::string& sst) {
    return wxString(sst.c_str(), wxConvUTF8);
}

//  LocalWords:  wxStfView

#ifdef WITH_PYTHON
void wxStfApp::OnPythonImport(wxCommandEvent& WXUNUSED(event)) {

    // show a file selection dialog menu.
    wxString pyFilter; // file filter only show *.py
    pyFilter = wxT("Python file (*.py)|*.py");
    wxFileDialog LoadModuleDialog (frame,
                wxT("Import/reload Python module"),
                wxT(""),
                wxT(""),
                pyFilter,
                wxFD_OPEN | wxFD_PREVIEW );

    if (LoadModuleDialog.ShowModal() == wxID_OK) {
        wxString modulelocation = LoadModuleDialog.GetPath();
        ImportPython(modulelocation); // see in /src/app/unopt.cpp L196
    }

    else {
        return;
    }
}

#endif // WITH_PYTHON
