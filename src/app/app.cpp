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

// For compilers that support precompilation, includes "wx/wx.h".

#include <wx/wxprec.h>

#include <wx/memory.h>
#include <wx/progdlg.h>
#include <wx/cmdline.h>
#include <wx/thread.h>
#include <wx/evtloop.h>
#include <wx/init.h>
#include <wx/datetime.h>

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#if !wxUSE_DOC_VIEW_ARCHITECTURE
#error You must set wxUSE_DOC_VIEW_ARCHITECTURE to 1 in setup.h!
#endif

#if !wxUSE_MDI_ARCHITECTURE
#error You must set wxUSE_MDI_ARCHITECTURE to 1 in setup.h!
#endif

#ifdef WITH_PYTHON
    #include <Python.h>
    #include <wx/wxPython/wxPython.h>
#endif

#include "./app.h"
#include "./doc.h"
#include "./view.h"
#include "./frame.h"
#include "./graph.h"
#include "./dlgs/cursorsdlg.h"
#include "./dlgs/smalldlgs.h"
#include "./funclib/funclib.h"
#include "./plugins/plugins.h"
#ifdef __UNIX__
#include "./../core/filelib/axon/Common/axodefn.h"
#include "./../core/filelib/axon/AxAbfFio32/abffiles.h"
#endif
#include "./../core/fitlib.h"

#ifdef _WINDOWS
	extern wxStfApp& wxGetApp();
	wxStfApp& wxGetApp() {
		return *wx_static_cast( wxStfApp*, wxApp::GetInstance() );
	}
#else
IMPLEMENT_APP(wxStfApp)
#endif

wxStfParentFrame *frame = (wxStfParentFrame *) NULL;

BEGIN_EVENT_TABLE( wxStfApp, wxApp )
EVT_MENU( wxID_CURSORS, wxStfApp::OnCursorSettings )
EVT_MENU( wxID_NEWFROMSELECTED, wxStfApp::OnNewfromselected )
EVT_MENU( wxID_NEWFROMALL, wxStfApp::OnNewfromall )
EVT_MENU( wxID_APPLYTOALL, wxStfApp::OnApplytoall )
END_EVENT_TABLE()


wxStfApp::wxStfApp(void) :
directTxtImport(false), isBars(true), isHires(false), txtImport(), funcLib(), pluginLib(), CursorsDialog(NULL), storedLinFunc( stf::initLinFunc() ) {}

bool wxStfApp::OnInit(void)
{
#if wxCHECK_VERSION(2, 9, 0)
    static const wxCmdLineEntryDesc s_cmdLineDesc[] =
    {
        { wxCMD_LINE_SWITCH, "h", "help", "Show this help message.", wxCMD_LINE_VAL_STRING, wxCMD_LINE_OPTION_HELP    },
        { wxCMD_LINE_OPTION, "d", "dir", "Working directory to change to." },
        { wxCMD_LINE_PARAM, NULL, NULL, "File to open.", wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL },
        { wxCMD_LINE_NONE }
    };
#else
    static const wxCmdLineEntryDesc s_cmdLineDesc[] =
    {
        { wxCMD_LINE_SWITCH, wxT("h"), wxT("help"), wxT("Show this help message."), wxCMD_LINE_VAL_STRING, wxCMD_LINE_OPTION_HELP  },
        { wxCMD_LINE_PARAM, wxT(""), wxT(""), wxT("File to open."), wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL },
        { wxCMD_LINE_NONE }
    };
#endif
    // Parse command line
    wxCmdLineParser parser( s_cmdLineDesc, argc, argv );
    if ( parser.Parse() != 0 )
        return false;

#ifdef _WINDOWS
    // Check if we should change the working directory:
    wxString new_cwd( wxT("\0") );
    if ( parser.Found( wxT("dir"), &new_cwd ) ) {

        // Check whether the directory exists:
        if ( !wxDirExists( new_cwd ) ) {
            wxString msg;
            msg << "New working directory " << new_cwd << " doesn't exist.";
            ErrorMsg( msg );
            return false;
        }
        // Change to the new wd:
        if ( !wxSetWorkingDirectory( new_cwd ) ) {
            wxString msg;
            msg << "Couldn't change working directory to " << new_cwd;
            ErrorMsg( msg );
            return false;
        }
    }
#endif

#ifdef WITH_PYTHON
    if ( !Init_wxPython() ) {
        // don't start the app if we can't initialize wxPython.
        return false;
    }
#endif

    // Config:
    config.reset(new wxConfig(wxT("Stimfit")));

    //// Create a document manager
    m_docManager.reset(new wxDocManager);
    //// Create a template relating drawing documents to their views
    m_cfsTemplate=new wxDocTemplate( (wxDocManager *)m_docManager.get(),
            wxT("CED filing system"), wxT("*.dat"), wxT(""), wxT("dat"),
            wxT("CFS Document"), wxT("CFS View"), CLASSINFO(wxStfDoc),
            CLASSINFO(wxStfView) );

    m_hdf5Template=new wxDocTemplate( (wxDocManager *)m_docManager.get(),
            wxT("hdf5 file"), wxT("*.h5"), wxT(""), wxT("h5"),
            wxT("HDF5 Document"), wxT("HDF5 View"), CLASSINFO(wxStfDoc),
            CLASSINFO(wxStfView) );

    m_abfTemplate=new wxDocTemplate( (wxDocManager *)m_docManager.get(),
            wxT("Axon binary file"), wxT("*.abf"), wxT(""), wxT("abf"),
            wxT("ABF Document"), wxT("ABF View"), CLASSINFO(wxStfDoc),
            CLASSINFO(wxStfView) );
#ifdef __UNIX__
    ABF_Initialize();
#endif
    m_atfTemplate=new wxDocTemplate( (wxDocManager *)m_docManager.get(),
            wxT("Axon text file"), wxT("*.atf"), wxT(""), wxT("atf"),
            wxT("ATF Document"), wxT("ATF View"), CLASSINFO(wxStfDoc),
            CLASSINFO(wxStfView) );
    m_axgTemplate=new wxDocTemplate( (wxDocManager *)m_docManager.get(),
            wxT("Axograph binary file"), wxT("*.axgd"), wxT(""), wxT("axgd"),
            wxT("AXG Document"), wxT("AXG View"), CLASSINFO(wxStfDoc),
            CLASSINFO(wxStfView) );
#if 0
    m_sonTemplate=new wxDocTemplate( (wxDocManager *)m_docManager.get(),
            wxT("CED Spike 2 (SON) file"), wxT("*.smr"), wxT(""), wxT(""),
            wxT("SON Document"), wxT("SON View"), CLASSINFO(wxStfDoc),
            CLASSINFO(wxStfView) );
#endif
    m_txtTemplate=new wxDocTemplate( (wxDocManager *)m_docManager.get(),
            wxT("General text file import"), wxT("*.*"), wxT(""), wxT(""),
            wxT("Text Document"), wxT("Text View"), CLASSINFO(wxStfDoc),
            CLASSINFO(wxStfView) );

    //// Create the main frame window
    frame = new wxStfParentFrame((wxDocManager *) m_docManager.get(), (wxFrame *) NULL,
            wxT("Stimfit"), /*wxPoint(0, 0)*/wxDefaultPosition, wxSize(1024, 768),
            wxDEFAULT_FRAME_STYLE | wxFULL_REPAINT_ON_RESIZE | wxMAXIMIZE);

#ifdef _WINDOWS
    frame->SetIcon( wxICON(sample) );
#endif
#ifdef __X__
    frame->SetIcon(wxIcon(wxT("doc.xbm")));
#endif

    //// Make a menubar
    wxMenu *file_menu = new wxMenu;
    //	wxMenu *edit_menu = (wxMenu *) NULL;

    file_menu->Append(wxID_OPEN, wxT("&Open...\tCtrl-X"));

    file_menu->AppendSeparator();
    file_menu->Append(ID_CONVERT, wxT("&Convert file series..."));
    file_menu->AppendSeparator();
    file_menu->Append(wxID_EXIT, wxT("E&xit\tAlt-X"));

    // A nice touch: a history of files visited. Use this menu.
    m_docManager->FileHistoryUseMenu(file_menu);

    wxMenu *help_menu = new wxMenu;
    help_menu->Append(wxID_ABOUT, wxT("&About\tF1"));

    wxMenu *m_view_menu = new wxMenu;
    m_view_menu->Append(wxID_VIEW_SHELL, wxT("&Toggle Python shell"),
            wxT("Shows or hides the Python shell"));

    wxMenuBar *menu_bar = new wxMenuBar;

    menu_bar->Append(file_menu, wxT("&File"));
    /*	if (edit_menu)
		menu_bar->Append(edit_menu, wxT("&Edit"));
     */
    menu_bar->Append(m_view_menu, wxT("&View"));

    menu_bar->Append(help_menu, wxT("&Help"));

#ifdef __WXMAC__
     wxMenuBar::MacSetCommonMenuBar(menu_bar);
#endif //def __WXMAC__
     //// Associate the menu bar with the frame
     frame->SetMenuBar(menu_bar);

     frame->Centre(wxBOTH);

     /*    pStatusBar = new wxStatusBar(frame);
    frame->SetStatusBar(pStatusBar);
      */
#ifndef __WXMAC__
     frame->Show(true);
#endif //ndef __WXMAC__

     // load user-defined plugins:
     pluginLib = stf::GetPluginLib();
     // load fit function library:
     funcLib = stf::GetFuncLib();

     SetTopWindow(frame);

     // Get file to load
     wxString fileToLoad = wxEmptyString;
     if ( parser.GetParamCount() > 0 ) {
         fileToLoad = parser.GetParam();
     }

     if (!fileToLoad.empty()) {
         wxDocTemplate* templ=m_docManager->FindTemplateForPath(fileToLoad);
         wxStfDoc* NewDoc=(wxStfDoc*)templ->CreateDocument(fileToLoad,wxDOC_NEW);
         NewDoc->SetDocumentTemplate(templ);
         if (!NewDoc->OnOpenDocument(fileToLoad)) {
             ErrorMsg(wxT("Couldn't open file, aborting file import"));
             m_docManager->CloseDocument(NewDoc);
             return false;
         }
     }

/*    if (!wxApp::OnInit()) {
       return false;
    }
*/
     return true;
}

int wxStfApp::OnExit()
{
#ifdef WITH_PYTHON
    Exit_wxPython();
#endif
    return wxApp::OnExit();
}

// "Fake" registry
void wxStfApp::wxWriteProfileInt(const wxString& main,const wxString& sub, int value) const {
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


void wxStfApp::OnPeakcalcexecMsg() {
    wxStfDoc* actDoc = GetActiveDoc();
    wxStfView* actView = GetActiveView();
    if (actView!=NULL) actView->GetGraph()->Refresh();
    bool baseToSlope=false;
    double slope=0.0;
    if (CursorsDialog != NULL &&
            CursorsDialog->IsShown() &&
            actView!=NULL &&
        actDoc!=NULL &&
        actDoc->IsInitialized())
    {
        CursorsDialog->SetActiveDoc(actDoc);
        switch (CursorsDialog->CurrentCursor()) {
        case stf::measure_cursor: {
            actDoc->SetMeasCursor(CursorsDialog->GetCursorM());// * GetDocument()->GetSR()));
            break;
        }
        //Get limits for peak calculation from the dialog box:
        case stf::peak_cursor: {
            actDoc->SetPeakBeg(CursorsDialog->GetCursor1P());// * GetDocument()->GetSR()));
            actDoc->SetPeakEnd(CursorsDialog->GetCursor2P());// * GetDocument()->GetSR()));
            actDoc->CheckBoundaries();
            break;
        }
        case stf::base_cursor: {
            actDoc->SetBaseBeg(CursorsDialog->GetCursor1B());
            actDoc->SetBaseEnd(CursorsDialog->GetCursor2B());
            break;
        }
        case stf::decay_cursor: {
            actDoc->SetFitBeg(CursorsDialog->GetCursor1D());
            actDoc->SetFitEnd(CursorsDialog->GetCursor2D());
            break;
        }
        case stf::undefined_cursor:
        {
            ErrorMsg(wxT("Undefined cursor in MyApp::OnPeakcalcexecMsg()"));
            return;
        }
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
        //Get number of peak points from the dialog box...
        actDoc->SetPM(CursorsDialog->GetPeakPoints());
        //.. and write to reg:
        wxWriteProfileInt(wxT("Settings"),wxT("PeakMean"),(int)actDoc->GetPM());
        //Get direction from the dialog box
        actDoc->SetDirection(CursorsDialog->GetDirection());
        wxWriteProfileInt(wxT("Settings"),wxT("Direction"),CursorsDialog->GetDirection());
        baseToSlope=CursorsDialog->GetBaseToSlope();
        if (baseToSlope) {
            slope=CursorsDialog->GetSlope();
        }
    }

    // Calculate peak, base, 20/80 rise time, half duration,
    // ratio of rise/slope and maximum slope.
    try {
        if (actDoc != NULL)
            actDoc->Measure(baseToSlope,slope);
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
        actView->GetGraph()->SetFocus();
    }
}

/*
 * Centralised code for creating a document frame.
 * Called from view.cpp, when a view is created.
 */

wxStfChildFrame *wxStfApp::CreateChildFrame(wxDocument *doc, wxView *view)
{
    //// Make a child frame
    wxStfChildFrame *subframe = new wxStfChildFrame(
            doc, view, GetMainFrame(), wxID_ANY, doc->GetTitle(),
            wxDefaultPosition, wxDefaultSize,
            wxDEFAULT_FRAME_STYLE |
            wxNO_FULL_REPAINT_ON_RESIZE |
            wxWANTS_CHARS | wxMAXIMIZE
    );

#ifdef __WXMSW__
    subframe->SetIcon(wxString(wxT("chart")));
#endif
#ifdef __X__
    subframe->SetIcon(wxIcon(wxT("doc.xbm")));
#endif

    //// Make a menubar
    wxMenu *file_menu = new wxMenu;

    file_menu->Append(wxID_OPEN, wxT("&Open..."));
    file_menu->Append(wxID_CLOSE, wxT("&Close"));
    //	file_menu->Append(wxID_SAVE, wxT("&Save"));
    file_menu->Append(wxID_SAVEAS, wxT("Save &As..."));
    file_menu->AppendSeparator();
    wxMenu* exportSub=new wxMenu;
    exportSub->Append(wxID_EXPORTFILE, wxT("as text file series..."));
    exportSub->Append(wxID_EXPORTATF, wxT("as ATF file..."));
    exportSub->Append(wxID_EXPORTHDF5, wxT("as HDF5 file..."));
    exportSub->Append(wxID_EXPORTIGOR, wxT("as Igor binary waves..."));

    file_menu->AppendSubMenu(exportSub, wxT("Export..."));
    file_menu->Append(wxID_EXPORTIMAGE, wxT("Export &image..."));

    wxMenu* vectorSub=new wxMenu;
    vectorSub->Append(wxID_EXPORTPS, wxT("Export &postscript..."));
    vectorSub->Append(wxID_EXPORTLATEX, wxT("Export &latex..."));
    vectorSub->Append(wxID_EXPORTSVG, wxT("Export &svg..."));
    file_menu->AppendSubMenu(vectorSub, wxT("Export &vector graphics"));


    file_menu->Append(ID_CONVERT, wxT("&Convert file series..."));
    file_menu->AppendSeparator();
    file_menu->Append(wxID_FILEINFO, wxT("File information..."));

    file_menu->AppendSeparator();
    file_menu->Append(WXPRINT_PRINT, wxT("&Print..."));
    file_menu->Append(WXPRINT_PAGE_SETUP, wxT("Print &Setup..."));
#if 0
    file_menu->Append(WXPRINT_PREVIEW, wxT("Print Pre&view"));
#endif
    /*
    file_menu->Append(wxID_PRINT, wxT("&Print..."));
    file_menu->Append(wxID_PRINT_SETUP, wxT("Print &Setup..."));
    file_menu->Append(wxID_PREVIEW, wxT("Print Pre&view"));
     */
    file_menu->AppendSeparator();
    file_menu->Append(wxID_EXIT, wxT("E&xit"));
    wxMenu* m_edit_menu=new wxMenu;
    m_edit_menu->Append(
            wxID_CURSORS,
            wxT("&Cursor settings..."),
            wxT("Set cursor position, direction, etc.")
    );
    m_edit_menu->AppendSeparator();
    m_edit_menu->Append(
            wxID_MYSELECTALL,
            wxT("&Select all traces"),
            wxT("Select all traces in this file")
    );
    m_edit_menu->Append(
            wxID_SELECTSOME,
            wxT("S&elect some traces..."),
            wxT("Select every n-th trace in this file")
    );
    m_edit_menu->Append(
            wxID_UNSELECTALL,
            wxT("&Unselect all traces"),
            wxT("Unselect all traces in this file")
    );
    m_edit_menu->Append(
            wxID_UNSELECTSOME,
            wxT("U&nselect some traces"),
            wxT("Unselect some traces in this file")
    );
    wxMenu *editSub=new wxMenu;
    editSub->Append(
            wxID_NEWFROMSELECTEDTHIS,
            wxT("&selected traces from this file"),
            wxT("Create a new window showing all selected traces from this file")
    );
    editSub->Append(
            wxID_NEWFROMSELECTED,
            wxT("&selected traces from all files"),
            wxT("Create a new window showing all selected traces from all files")
    );
    editSub->Append(wxID_NEWFROMALL,
            wxT("&all traces from all files"),
            wxT("Create a new window showing all traces from all files")
    );
    m_edit_menu->AppendSeparator();
    m_edit_menu->AppendSubMenu(editSub,wxT("New window with..."));
    m_edit_menu->Append(
            wxID_CONCATENATE,
            wxT("&Concatenate selected traces"),
            wxT("Create one large trace by merging selected traces in this file")
    );
    wxMenu *latencyStartSub=new wxMenu;
    latencyStartSub->AppendCheckItem(wxID_LATENCYSTART_MAXSLOPE, wxT("max. slope of second channel"));
    latencyStartSub->AppendCheckItem(wxID_LATENCYSTART_HALFRISE, wxT("half-maximal amplitude of second channel"));
    latencyStartSub->AppendCheckItem(wxID_LATENCYSTART_PEAK, wxT("peak of second channel"));
    latencyStartSub->AppendCheckItem(wxID_LATENCYSTART_MANUAL, wxT("Manual"));
    wxMenu *latencyEndSub=new wxMenu;
    latencyEndSub->AppendCheckItem(wxID_LATENCYEND_FOOT, wxT("beginning of event in active channel"));
    latencyEndSub->AppendCheckItem(wxID_LATENCYEND_MAXSLOPE, wxT("max. slope of active channel"));
    latencyEndSub->AppendCheckItem(wxID_LATENCYEND_HALFRISE, wxT("half-maximal amplitude of active channel"));
    latencyEndSub->AppendCheckItem(wxID_LATENCYEND_PEAK, wxT("peak of active channel"));
    latencyEndSub->AppendCheckItem(wxID_LATENCYEND_MANUAL, wxT("Manual"));
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
    m_edit_menu->AppendCheckItem(
            wxID_LATENCYWINDOW,
            wxT("Use peak window for latency cursor"),
            wxT("Uses the current peak window to measure the peak in the inactive channel")
    );
    wxMenu* m_view_menu = new wxMenu;
    m_view_menu->Append(
            wxID_VIEW_RESULTS,
            wxT("&Results..."),
            wxT("Select analysis results to be shown in the results table")
    );
    m_view_menu->Append(
            wxID_APPLYTOALL,
            wxT("&Apply scaling to all windows"),
            wxT("Apply this trace's scaling to all other windows")
    );
    m_view_menu->AppendCheckItem(
            wxID_SCALE,
            wxT("&View scale bars"),
            wxT("If checked, use scale bars rather than coordinates")
    );
    m_view_menu->AppendCheckItem(
            wxID_HIRES,
            wxT("View &full resolution"),
            wxT("If checked, plot large traces at high resolution")
    );
    m_view_menu->AppendSeparator();
    m_view_menu->Append(wxID_SAVEPERSPECTIVE,wxT("&Save window positions"));
    m_view_menu->Append(wxID_LOADPERSPECTIVE,wxT("&Load window positions"));
    m_view_menu->Append(wxID_RESTOREPERSPECTIVE,wxT("&Restore default window positions"));
    m_view_menu->AppendSeparator();
    m_view_menu->Append(wxID_VIEW_SHELL, wxT("&Toggle Python shell"),
            wxT("Shows or hides the Python shell"));

    wxMenu *analysis_menu = new wxMenu;
    wxMenu *fitSub = new wxMenu;
    fitSub->Append(
            wxID_FIT,
            wxT("&Nonlinear regression..."),
            wxT("Fit a function to this trace between fit cursors")
    );
    fitSub->Append(
            wxID_LFIT,
            wxT("&Linear fit..."),
            wxT("Fit a linear function to this trace between fit cursors")
    );
    analysis_menu->AppendSubMenu(fitSub, wxT("&Fit"));
    wxMenu *transformSub = new wxMenu;
    transformSub->Append(
            wxID_LOG,
            wxT("&Logarithmic (base e)..."),
            wxT("Transform selected traces logarithmically")
    );
    analysis_menu->AppendSubMenu(transformSub, wxT("&Transform"));
    analysis_menu->Append(
            wxID_MULTIPLY,
            wxT("&Multiply..."),
            wxT("Multiply selected traces")
    );
    analysis_menu->Append(
            wxID_INTEGRATE,
            wxT("&Integrate"),
            wxT("Integrate this trace between fit cursors")
    );
    analysis_menu->Append(
            wxID_DIFFERENTIATE,
            wxT("&Differentiate"),
            wxT("Differentiate selected traces")
    );
    analysis_menu->Append(
            wxID_SUBTRACTBASE,
            wxT("&Subtract baseline"),
            wxT("Subtract baseline from selected traces")
    );
    analysis_menu->Append(
            wxID_FILTER,
            wxT("F&ilter..."),
            wxT("Filter selected traces")
    );
    analysis_menu->Append(
            wxID_SPECTRUM,
            wxT("&Power spectrum..."),
            wxT("Compute an estimate of the power spectrum of the selected traces")
    );
    analysis_menu->Append(
            wxID_POVERN,
            wxT("P over &N correction..."),
            wxT("Apply P over N correction to all traces of this file")
    );
    wxMenu* eventPlotSub = new wxMenu;
    eventPlotSub->Append(wxID_PLOTCRITERION, wxT("&Detection criterion..."));
    eventPlotSub->Append(wxID_PLOTCORRELATION, wxT("&Correlation coefficient..."));
    wxMenu* eventSub = new wxMenu;
    eventSub->AppendSubMenu(eventPlotSub,wxT("Plot"));
    eventSub->Append(wxID_EXTRACT,wxT("&Template matching..."));
    eventSub->Append(wxID_THRESHOLD,wxT("Threshold &crossing..."));
    analysis_menu->AppendSubMenu(eventSub,wxT("Event detection"));
    analysis_menu->Append(
            wxID_BATCH,
            wxT("&Batch analysis..."),
            wxT("Analyze selected traces and show results in a table")
    );
    wxMenu* userdefSub=new wxMenu;

    for (std::size_t n=0;n<GetPluginLib().size();++n) {
        userdefSub->Append(
                wxID_USERDEF1+(int)n,
                GetPluginLib()[n].menuEntry
        );
    }
    analysis_menu->AppendSubMenu(userdefSub,wxT("User-defined functions"));

    wxMenu *help_menu = new wxMenu;
    help_menu->Append(wxID_ABOUT, wxT("&About"));

    wxMenuBar *menu_bar = new wxMenuBar;

    menu_bar->Append(file_menu, wxT("&File"));
    menu_bar->Append(m_edit_menu, wxT("&Edit"));
    menu_bar->Append(m_view_menu, wxT("&View"));
    menu_bar->Append(analysis_menu, wxT("&Analysis"));
    menu_bar->Append(help_menu, wxT("&Help"));

    //// Associate the menu bar with the frame
    subframe->SetMenuBar(menu_bar);

    return subframe;
}

wxStfDoc* wxStfApp::NewChild(
        const Recording& NewData,
        const wxStfDoc* Sender,
        const wxString& title
) {
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
    return (wxStfView*)m_docManager->GetCurrentView();
}

wxStfDoc* wxStfApp::GetActiveDoc() const {
    if ( m_docManager == 0) {
        ErrorMsg( wxT("Couldn't access the document manager"));
        return NULL;
    }
    return (wxStfDoc*)m_docManager->GetCurrentDocument();
}

void wxStfApp::OnCursorSettings( wxCommandEvent& WXUNUSED(event) ) {
    wxStfDoc* actDoc=GetActiveDoc();
    if (CursorsDialog==NULL && actDoc!=NULL) {
        CursorsDialog=new wxStfCursorsDlg(frame);
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
    }

}

void wxStfApp::OnNewfromselected( wxCommandEvent& WXUNUSED(event) ) {
    // number of selected traces across all open documents:
    std::size_t nwxT=0;
    // Search the document's template list for open documents:
    wxList docList=m_docManager->GetDocuments();
    if (docList.IsEmpty()) {
        ErrorMsg(wxT("No traces were found"));
        return;
    }
    // Since random access is expensive, go through the list node by node:
    // Get first node:
    wxNode *curNode=docList.GetFirst();
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
    nwxT=0;
    wxStfDoc* pDoc=NULL;
    std::vector<std::vector<wxString> > channel_names(n_channels);
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
        wxString channel_name;
        channel_name << channel_names[n_c][0];
        for (std::size_t n_n=1;n_n<channel_names[n_c].size();++n_n) {
            // add channel name if it hasn't been used yet:
            bool used=false;
            for (int n_used=(int)n_n-1;n_used>=0 && !used;--n_used) {
                // can't use size_t here because
                // n_used might be negative when checking loop condition
                used = ( channel_names[n_c][n_n].Cmp( channel_names[n_c][n_used] ) == 0 );
            }
            if (!used) {
                channel_name << wxT(", ") << channel_names[n_c][n_n];
            }
        }
        Selected.get()[n_c].SetChannelName(channel_name);
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
    wxList docList=m_docManager->GetDocuments();
    if (docList.IsEmpty()) {
        ErrorMsg(wxT("No traces were found"));
        return;
    }
    // Since random access is expensive, go through the list node by node:
    // Get first node:
    wxNode *curNode=docList.GetFirst();
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
    std::vector<std::vector<wxString> > channel_names(n_channels);
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
        wxString channel_name;
        channel_name << channel_names[n_c][0];
        for (std::size_t n_n=1;n_n<channel_names[n_c].size();++n_n) {
            // add channel name if it hasn't been used yet:
            bool used=false;
            for (int n_used=(int)n_n-1;n_used>=0 && !used;--n_used) {
                // can't use size_t here because
                // n_used might be negative when checking loop condition
                used = ( channel_names[n_c][n_n].Cmp( channel_names[n_c][n_used] ) == 0 );
            }
            if (!used) {
                channel_name << wxT(", ") << channel_names[n_c][n_n];
            }
        }
        Selected.get()[n_c].SetChannelName(channel_name);
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
    wxList docList=m_docManager->GetDocuments();
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
    wxNode *curNode=docList.GetFirst();
    while (curNode!=NULL) {
        wxStfDoc* OpenDoc=(wxStfDoc*)curNode->GetData();
        if (OpenDoc==NULL)
            return;
        wxStfView* curView((wxStfView*)OpenDoc->GetFirstView());
        if (curView!=pView && curView!=NULL) {
            OpenDoc->GetXZoomW() = pDoc->GetXZoom();
            for ( std::size_t n_c=0; n_c < OpenDoc->size(); ++n_c ) {
                if ( n_c < pDoc->size() ) {
                    OpenDoc->at(n_c).GetYZoomW() = pDoc->at(n_c).GetYZoom();
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
            wxDocTemplate* templ=m_docManager->FindTemplateForPath(fNameArray[n_opened]);
            wxStfDoc* NewDoc=(wxStfDoc*)templ->CreateDocument(fNameArray[n_opened],wxDOC_NEW);
            NewDoc->SetDocumentTemplate(templ);
            if (!NewDoc->OnOpenDocument(fNameArray[n_opened++])) {
                ErrorMsg(wxT("Couldn't open file, aborting file import"));
                m_docManager->CloseDocument(NewDoc);
                return false;
            }
        } else {
            // Add to recording first:
            // Find a template:
            wxDocTemplate* templ=m_docManager->FindTemplateForPath(fNameArray[n_opened]);
            // Use this template only for type recognition:
            wxString filter(templ->GetFileFilter());
            stf::filetype type=
                stf::findType(templ->GetFileFilter());
            if (type==stf::ascii) {
                if (!get_directTxtImport()) {
                    wxStfTextImportDlg ImportDlg(
                            NULL,
                            stf::CreatePreview(fNameArray[n_opened]),
                            1,
                            true
                    );
                    if (ImportDlg.ShowModal()!=wxID_OK) {
                        return false;
                    }
                    // store settings in application:
                    set_txtImportSettings(ImportDlg.GetTxtImport());
                    set_directTxtImport(ImportDlg.ApplyToAll());
                }
            }
            // add this file to the series recording:
            Recording singleRec;
            try {
                stf::importFile(fNameArray[n_opened++],type,singleRec,txtImport);
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
                         << wxString( e.what(), wxConvLocal );
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

bool wxStfApp::OpenFilePy(const wxString& filename) {
    wxDocTemplate* templ = m_docManager->FindTemplateForPath( filename );
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
        m_docManager->CloseDocument(NewDoc);
        return false;
    }
    return true;
}

void wxStfApp::OnCloseDocument() {
    // count open docs:
    if (m_docManager->GetDocuments().GetCount()==1) {
        if (CursorsDialog!=NULL) {
            CursorsDialog->Destroy();
            CursorsDialog=NULL;
        }
    }
}

std::vector<Section*> wxStfApp::GetSectionsWithFits() const {
    // Search the document's template list for open documents:
    wxList docList=m_docManager->GetDocuments();
    if (docList.IsEmpty()) {
        return std::vector<Section*>(0);
    }
    std::vector<Section*> sectionList;
    // Since random access is expensive, go through the list node by node:
    // Get first node:
    wxNode *curNode=docList.GetFirst();
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
              << wxT(STFVERSION)
#ifdef _STFDEBUG
              << wxT(", debug build, ");
#else
              << wxT(", release build, ");
#endif

    verString << wxT(STFDATE);

    return verString;
}

wxStfParentFrame *GetMainFrame(void)
{
    return frame;
}


//  LocalWords:  wxStfView
