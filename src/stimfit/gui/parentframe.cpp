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

// parentframe.cpp
// These are the top-level and child windows of the application.
// 2007-12-27, Christoph Schmidt-Hieber, University of Freiburg

#ifdef _STFDEBUG
#include <iostream>
#endif
// For compilers that support precompilation, includes "wx/wx.h".
#include <wx/wxprec.h>
#include <wx/grid.h>
#include <wx/artprov.h>
#include <wx/printdlg.h>
#include <wx/file.h>
#include <wx/filename.h>
#include <wx/progdlg.h>
#include <wx/splitter.h>
#include <wx/choicdlg.h>
#include <wx/aboutdlg.h>
#include <wx/protocol/http.h>
#include <wx/sstream.h>
#include <wx/progdlg.h>

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
#include "./graph.h"
#include "./table.h"
#include "./printout.h"
#include "./dlgs/smalldlgs.h"
#include "./copygrid.h"
#include "./../../libstfio/atf/atflib.h"
#include "./../../libstfio/igor/igorlib.h"

#include "./childframe.h"
#include "./parentframe.h"

#include "./../res/16-em-down.xpm"
#include "./../res/16-em-open.xpm"
#include "./../res/accept.xpm"
#include "./../res/arrow_down.xpm"
#include "./../res/arrow_left.xpm"
#include "./../res/arrow_out.xpm"
#include "./../res/arrow_right.xpm"
#include "./../res/arrow_up.xpm"
//#include "./../res/bin.xpm"
#include "./../res/camera.xpm"
#ifdef _WINDOWS
#include "./../res/camera_ps.xpm"
#endif
#include "./../res/ch1.xpm"
#include "./../res/ch2.xpm"
#include "./../res/cursor.xpm"
#include "./../res/event.xpm"
#include "./../res/fit.xpm"
#include "./../res/fit_lim.xpm"
#include "./../res/latency_lim.xpm"
#include "./../res/resultset_first.xpm"
#include "./../res/resultset_last.xpm"
#include "./../res/resultset_next.xpm"
#include "./../res/resultset_previous.xpm"
#include "./../res/sum_new.xpm"
#include "./../res/sum_new_aligned.xpm"
#include "./../res/table.xpm"
#include "./../res/zoom.xpm"
#include "./../res/zoom_in.xpm"
#include "./../res/zoom_out.xpm"

#ifdef WITH_PSLOPE 
#include "./../res/slope.xpm"
#endif

IMPLEMENT_CLASS(wxStfParentFrame, wxStfParentType)
BEGIN_EVENT_TABLE(wxStfParentFrame, wxStfParentType)
EVT_MENU(wxID_HELP, wxStfParentFrame::OnHelp)
EVT_MENU(ID_UPDATE, wxStfParentFrame::OnCheckUpdate)
EVT_MENU(wxID_ABOUT, wxStfParentFrame::OnAbout)

EVT_TOOL(ID_TOOL_SELECT,wxStfParentFrame::OnToggleSelect)
EVT_TOOL(ID_TOOL_FIRST, wxStfParentFrame::OnToolFirst)
EVT_TOOL(ID_TOOL_NEXT, wxStfParentFrame::OnToolNext)
EVT_TOOL(ID_TOOL_PREVIOUS, wxStfParentFrame::OnToolPrevious)
EVT_TOOL(ID_TOOL_LAST, wxStfParentFrame::OnToolLast)
EVT_TOOL(ID_TOOL_XENL, wxStfParentFrame::OnToolXenl)
EVT_TOOL(ID_TOOL_XSHRINK, wxStfParentFrame::OnToolXshrink)
EVT_TOOL(ID_TOOL_YENL, wxStfParentFrame::OnToolYenl)
EVT_TOOL(ID_TOOL_YSHRINK, wxStfParentFrame::OnToolYshrink)
EVT_TOOL(ID_TOOL_UP, wxStfParentFrame::OnToolUp)
EVT_TOOL(ID_TOOL_DOWN, wxStfParentFrame::OnToolDown)
EVT_TOOL(ID_TOOL_FIT, wxStfParentFrame::OnToolFit)
EVT_TOOL(ID_TOOL_LEFT, wxStfParentFrame::OnToolLeft)
EVT_TOOL(ID_TOOL_RIGHT, wxStfParentFrame::OnToolRight)
#ifdef _WINDOWS
EVT_TOOL(ID_TOOL_SNAPSHOT_WMF, wxStfParentFrame::OnToolSnapshotwmf)
#endif
EVT_TOOL(ID_TOOL_CH1, wxStfParentFrame::OnToolCh1)
EVT_TOOL(ID_TOOL_CH2, wxStfParentFrame::OnToolCh2)

EVT_TOOL(ID_TOOL_MEASURE, wxStfParentFrame::OnToolMeasure)
EVT_TOOL(ID_TOOL_PEAK,wxStfParentFrame::OnToolPeak)
EVT_TOOL(ID_TOOL_BASE,wxStfParentFrame::OnToolBase)
EVT_TOOL(ID_TOOL_DECAY,wxStfParentFrame::OnToolDecay)
#ifdef WITH_PSLOPE
EVT_TOOL(ID_TOOL_PSLOPE,wxStfParentFrame::OnToolPSlope)
#endif
EVT_TOOL(ID_TOOL_LATENCY,wxStfParentFrame::OnToolLatency)
EVT_TOOL(ID_TOOL_ZOOM,wxStfParentFrame::OnToolZoom)
EVT_TOOL(ID_TOOL_EVENT,wxStfParentFrame::OnToolEvent)

//#ifdef _WINDOWS
EVT_MENU(ID_CONVERT, wxStfParentFrame::OnConvert)
//#endif
EVT_MENU(ID_AVERAGE, wxStfParentFrame::OnAverage)
EVT_MENU(ID_ALIGNEDAVERAGE, wxStfParentFrame::OnAlignedAverage)
EVT_MENU( ID_VIEW_RESULTS, wxStfParentFrame::OnViewResults)
EVT_MENU( ID_CH2BASE, wxStfParentFrame::OnCh2base )
EVT_MENU( ID_CH2POS, wxStfParentFrame::OnCh2pos )
EVT_MENU( ID_CH2ZOOM, wxStfParentFrame::OnCh2zoom )
EVT_MENU( ID_CH2BASEZOOM, wxStfParentFrame::OnCh2basezoom )
EVT_MENU( ID_SCALE, wxStfParentFrame::OnScale )
EVT_MENU( ID_HIRES, wxStfParentFrame::OnHires )
#ifdef _WINDOWS
EVT_MENU( ID_PRINT_PRINT, wxStfParentFrame::OnPrint)
#endif
EVT_MENU( ID_MPL, wxStfParentFrame::OnMpl)
EVT_MENU( ID_PRINT_PAGE_SETUP, wxStfParentFrame::OnPageSetup)
EVT_MENU( ID_SAVEPERSPECTIVE, wxStfParentFrame::OnSaveperspective )
EVT_MENU( ID_LOADPERSPECTIVE, wxStfParentFrame::OnLoadperspective )
EVT_MENU( ID_RESTOREPERSPECTIVE, wxStfParentFrame::OnRestoreperspective )
#ifdef WITH_PYTHON
EVT_MENU( ID_VIEW_SHELL, wxStfParentFrame::OnViewshell )
#endif
#if 0
EVT_MENU( ID_LATENCYSTART_MAXSLOPE, wxStfParentFrame::OnLStartMaxslope )
EVT_MENU( ID_LATENCYSTART_HALFRISE, wxStfParentFrame::OnLStartHalfrise )
EVT_MENU( ID_LATENCYSTART_PEAK, wxStfParentFrame::OnLStartPeak )
EVT_MENU( ID_LATENCYSTART_MANUAL, wxStfParentFrame::OnLStartManual )
EVT_MENU( ID_LATENCYEND_FOOT, wxStfParentFrame::OnLEndFoot )
EVT_MENU( ID_LATENCYEND_MAXSLOPE, wxStfParentFrame::OnLEndMaxslope )
EVT_MENU( ID_LATENCYEND_PEAK, wxStfParentFrame::OnLEndPeak )
EVT_MENU( ID_LATENCYEND_HALFRISE, wxStfParentFrame::OnLEndHalfrise )
EVT_MENU( ID_LATENCYEND_MANUAL, wxStfParentFrame::OnLEndManual )
#endif
EVT_MENU( ID_LATENCYWINDOW, wxStfParentFrame::OnLWindow )
END_EVENT_TABLE()

wxStfParentFrame::wxStfParentFrame(wxDocManager *manager, wxFrame *frame, const wxString& title,
                 const wxPoint& pos, const wxSize& size, long type):
wxStfParentType(manager, frame, wxID_ANY, title, pos, size, type, _T("myFrame"))
{
    // ::wxInitAllImageHandlers();

    m_mgr.SetManagedWindow(this);
    m_mgr.SetFlags(
        wxAUI_MGR_ALLOW_FLOATING |
        wxAUI_MGR_TRANSPARENT_DRAG |
        wxAUI_MGR_VENETIAN_BLINDS_HINT |
        wxAUI_MGR_ALLOW_ACTIVE_PANE
                   );

#if wxUSE_DRAG_AND_DROP
    m_drop = new wxStfFileDrop; // obviously gets deleted when the frame is destructed
    SetDropTarget(m_drop);
#endif
    m_printData.reset(new wxPrintData);

    // initial paper size
    //	m_printData->SetQuality(wxPRINT_QUALITY_HIGH);
    //	int ppi = m_printData->GetQuality();
    m_printData->SetPaperId(wxPAPER_A4);
    // initial orientation
    m_printData->SetOrientation(wxLANDSCAPE);
    m_pageSetupData.reset(new wxPageSetupDialogData);
    // copy over initial paper size from print record
    m_pageSetupData->SetPrintData(*m_printData);
    // Set some initial page margins in mm.
    m_pageSetupData->SetMarginTopLeft(wxPoint(15, 15));
    m_pageSetupData->SetMarginBottomRight(wxPoint(15, 15));

    // create some toolbars

    wxStfToolBar* tb1 = CreateStdTb();
    tb1->Realize();

    m_scaleToolBar=CreateScaleTb();
    m_scaleToolBar->Realize();

    wxStfToolBar* tb4=CreateEditTb();
    tb4->Realize();

    m_cursorToolBar=CreateCursorTb();
    m_cursorToolBar->Realize();

    // add the toolbars to the manager
    m_mgr.AddPane( tb1, wxAuiPaneInfo().Name(wxT("tb1")).Caption(wxT("Std Toolbar")).ToolbarPane().Resizable(false).
                   Position(0).Top().Gripper().RightDockable(false) );

#ifdef __WXMAC__
    int xpos = 64, ypos = 32;
#endif
    m_mgr.AddPane( m_cursorToolBar, wxAuiPaneInfo().Name(wxT("tb2")).Caption(wxT("Edit Toolbar")).
                   ToolbarPane().Resizable(false).
#ifndef __WXMAC__
                   Position(1).Top().Gripper().RightDockable(false) );
#else
                   Dockable(false).Float().FloatingPosition(xpos, ypos) );
    xpos += m_cursorToolBar->GetSize().GetWidth()+8;
#endif
    m_mgr.AddPane( tb4, wxAuiPaneInfo().Name(wxT("tb4")).Caption(wxT("Analysis Toolbar")).
                   ToolbarPane().Resizable(false).
#ifndef __WXMAC__
                   Position(2).Top().Gripper().RightDockable(false) );
#else
                   Dockable(false).Float().FloatingPosition(xpos,ypos) );
    xpos += tb4->GetSize().GetWidth()+8;
#endif
    m_mgr.AddPane( m_scaleToolBar, wxAuiPaneInfo().Name(wxT("m_scaleToolBar")).Caption(wxT("Navigation Toolbar")).
                   ToolbarPane().Resizable(false).
#ifndef __WXMAC__
                   Position(3).Top().Gripper().RightDockable(false) );
#else
                   Dockable(false).Float().FloatingPosition(xpos,ypos) );
#endif

    SetMouseQual( stf::measure_cursor );

#ifdef WITH_PYTHON
    python_code2 << wxT("import sys\n")
                 << wxT("sys.path.append('.')\n")
#ifdef IPYTHON
                 << wxT("import embedded_ipython\n")
#else
                 << wxT("import embedded_stf\n")
#endif
                 << wxT("import embedded_mpl\n")
                 << wxT("\n")
                 << wxT("def makeWindow(parent):\n")
#ifdef IPYTHON
                 << wxT("    win = embedded_ipython.MyPanel(parent)\n")
#else
                 << wxT("    win = embedded_stf.MyPanel(parent)\n")
#endif
                 << wxT("    return win\n")
                 << wxT("\n")
                 << wxT("def makeWindowMpl(parent):\n")
                 << wxT("    win = embedded_mpl.MplPanel(parent)\n")
                 << wxT("    win.plot_screen()\n")
                 << wxT("    return win\n");

    /*  The window remains open after the main application has been closed; deactivated for the time being.
     *  RedirectStdio();
     */
    wxWindow* pPython = DoPythonStuff(this, false);
    if ( pPython == 0 ) {
        wxGetApp().ErrorMsg(wxT("Can't create a window for the python shell\nPointer is zero"));
    } else {
#ifndef __WXMAC__
        bool show = wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewShell"), 1);
#endif
        m_mgr.AddPane( pPython, wxAuiPaneInfo().Name(wxT("pythonShell")).
#ifndef __WXMAC__
                       CloseButton(true).
                       Show(show).Caption(wxT("Python Shell")).Dockable(true).Bottom().
                       BestSize(GetClientSize().GetWidth(),GetClientSize().GetHeight()/5) );
#else
                       Floatable(false).CaptionVisible(false).
                       BestSize(GetClientSize().GetWidth(),GetClientSize().GetHeight()).Fixed() );
#endif
    }

#ifdef _STFDEBUG
#ifdef _WINDOWS
    wxGetApp().InfoMsg( python_code2 );
#else
    std::cout << "python startup script:\n" << std::string( python_code2.char_str() );
#endif // _WINDOWS
#endif // _STFDEBUG
#endif // WITH_PYTHON
    m_mgr.Update();
    wxStatusBar* pStatusBar = new wxStatusBar(this, wxID_ANY, wxST_SIZEGRIP);
    SetStatusBar(pStatusBar);
    //int widths[] = { 60, 60, -1 };
    //pStatusBar->SetFieldWidths(WXSIZEOF(widths), widths);
    //pStatusBar->SetStatusText(wxT("Test"), 0);
}

wxStfParentFrame::~wxStfParentFrame() {
    // deinitialize the frame manager
#ifdef WITH_PYTHON
    // write visibility of the shell to config:
    bool shell_state = m_mgr.GetPane(wxT("pythonShell")).IsShown();
    wxGetApp().wxWriteProfileInt( wxT("Settings"),wxT("ViewShell"), int(shell_state) );
#endif
    m_mgr.UnInit();
}

wxStfToolBar* wxStfParentFrame::CreateStdTb() {
    wxStfToolBar* tb1=new wxStfToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize,
                                        wxAUI_TB_DEFAULT_STYLE );
    tb1->SetToolBitmapSize(wxSize(20,20));
    tb1->AddTool( wxID_OPEN,
                  wxT("Open"),
                  wxArtProvider::GetBitmap( wxART_FILE_OPEN, wxART_TOOLBAR, wxSize(16,16) ),
                  wxT("Open file"),
                  wxITEM_NORMAL );
    tb1->AddTool( wxID_SAVEAS,
                  wxT("Save"),
                  wxArtProvider::GetBitmap( wxART_FILE_SAVE_AS, wxART_TOOLBAR, wxSize(16,16) ),
                  wxT("Save traces"),
                  wxITEM_NORMAL );
    tb1->AddTool( ID_PRINT_PRINT,
                  wxT("Print"),
                  wxArtProvider::GetBitmap( wxART_PRINT, wxART_TOOLBAR, wxSize(16,16) ),
                  wxT("Print traces"),
                  wxITEM_NORMAL );
    return tb1;
}

wxStfToolBar* wxStfParentFrame::CreateScaleTb() {
    wxStfToolBar* scaleToolBar =
        new wxStfToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxAUI_TB_DEFAULT_STYLE );
    scaleToolBar->SetToolBitmapSize(wxSize(20,20));
    scaleToolBar->AddTool( ID_TOOL_FIRST,
                           wxT("First"),
                           wxBitmap(resultset_first),
                           wxT("Go to first trace"),
                           wxITEM_NORMAL );
    scaleToolBar->AddTool( ID_TOOL_PREVIOUS,
                           wxT("Prev."),
                           wxBitmap(resultset_previous),
                           wxT("Go to previous trace (left cursor)"),
                           wxITEM_NORMAL );
    scaleToolBar->AddTool( ID_TOOL_NEXT,
                           wxT("Next"),
                           wxBitmap(resultset_next),
                           wxT("Go to next trace (right cursor)"),
                           wxITEM_NORMAL );
    scaleToolBar->AddTool( ID_TOOL_LAST,
                           wxT("Last"),
                           wxBitmap(resultset_last),
                           wxT("Go to last trace"),
                           wxITEM_NORMAL );
    scaleToolBar->AddSeparator();
    scaleToolBar->AddTool( ID_TOOL_LEFT,
                           wxT("Left"),
                           wxBitmap(arrow_left),
                           wxT("Move traces left (CTRL+left cursor)"),
                           wxITEM_NORMAL );
    scaleToolBar->AddTool( ID_TOOL_RIGHT,
                           wxT("Right"),
                           wxBitmap(arrow_right),
                           wxT("Move traces right (CTRL+right cursor)"),
                           wxITEM_NORMAL );
    scaleToolBar->AddTool( ID_TOOL_FIT,
                           wxT("Fit"),
                           wxBitmap(arrow_out),
                           wxT("Fit traces to window (\"F\")"),
                           wxITEM_NORMAL );
    scaleToolBar->AddTool( ID_TOOL_UP,
                           wxT("Up"),
                           wxBitmap(arrow_up),
                           wxT("Move traces up (up cursor)"),
                           wxITEM_NORMAL );
    scaleToolBar->AddTool( ID_TOOL_DOWN,
                           wxT("Down"),
                           wxBitmap(arrow_down),
                           wxT("Move traces down (down cursor)"),
                           wxITEM_NORMAL );
    scaleToolBar->AddTool( ID_TOOL_XENL,
                           wxT("Zoom X"),
                           wxBitmap(zoom_in),
                           wxT("Enlarge x-scale (CTRL + \"+\")"),
                           wxITEM_NORMAL );
    scaleToolBar->AddTool( ID_TOOL_XSHRINK,
                           wxT("Shrink X"),
                           wxBitmap(zoom_out),
                           wxT("Shrink x-scale (CTRL + \"-\")"),
                           wxITEM_NORMAL );
    scaleToolBar->AddTool( ID_TOOL_YENL,
                           wxT("Zoom Y"),
                           wxBitmap(zoom_in),
                           wxT("Enlarge y-scale (\"+\")"),
                           wxITEM_NORMAL );
    scaleToolBar->AddTool( ID_TOOL_YSHRINK,
                           wxT("Shrink Y"),
                           wxBitmap(zoom_out),
                           wxT("Shrink y-scale (\"-\")"),
                           wxITEM_NORMAL );
    scaleToolBar->AddSeparator();
    scaleToolBar->AddTool( ID_TOOL_CH1,
                           wxT("Ch 1"),
                           wxBitmap(ch_),
                           wxT("Scaling applies to active (black) channel (\"1\")"),
                           wxITEM_CHECK );
    scaleToolBar->AddTool( ID_TOOL_CH2,
                           wxT("Ch 2"),
                           wxBitmap(ch2_),
                           wxT("Scaling applies to reference (red) channel (\"2\")"),
                           wxITEM_CHECK );
    return scaleToolBar;
}

wxStfToolBar* wxStfParentFrame::CreateEditTb() {
    wxStfToolBar* tb4= new wxStfToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize,
                                         wxAUI_TB_DEFAULT_STYLE );
    tb4->SetToolBitmapSize(wxSize(20,20));
    tb4->AddTool( ID_AVERAGE,
                  wxT("Mean"),
                  wxBitmap(sum_new),
                  wxT("Average of selected traces"),
                  wxITEM_NORMAL );
    tb4->AddTool( ID_ALIGNEDAVERAGE,
                  wxT("Aligned"),
                  wxBitmap(sum_new_aligned),
                  wxT("Aligned average of selected traces"),
                  wxITEM_NORMAL );
    tb4->AddTool( ID_FIT,
                  wxT("Fit"),
                  wxBitmap(fit),//chart_line),
                  wxT("Fit function to data"),
                  wxITEM_NORMAL );
    tb4->AddTool( ID_VIEWTABLE,
                  wxT("Table"),
                  wxBitmap(table),
                  wxT("View current trace as a table"),
                  wxITEM_NORMAL );
    return tb4;
}

wxStfToolBar* wxStfParentFrame::CreateCursorTb() {
    wxStfToolBar* cursorToolBar = new wxStfToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize,
                                                    wxAUI_TB_DEFAULT_STYLE );
    cursorToolBar->SetToolBitmapSize(wxSize(20,20));
    cursorToolBar->AddTool( ID_TOOL_SELECT,
                            wxT("Select"),
                            wxBitmap( acceptbmp ),
                            wxT("Select or unselect this trace (\"S\" / \"R\")"),
                            wxITEM_CHECK );
    // cursorToolBar->AddTool( ID_TOOL_REMOVE,
    //                         wxT("Unselect"),
    //                         wxBitmap( bin ),
    //                         wxT("Unselect this trace (\"R\")"),
    //                         wxITEM_NORMAL );
    cursorToolBar->AddSeparator();
    cursorToolBar->AddTool( ID_MPL,
                            wxT("Snapshot"),
                            wxBitmap(camera),
                            wxT("Create snapshot with matplotlib"),
                            wxITEM_NORMAL );
#ifdef _WINDOWS
    cursorToolBar->AddTool( ID_TOOL_SNAPSHOT_WMF,
                            wxT("WMF Snapshot"),
                            wxBitmap(camera_ps),
                            wxT("Copy vectorized image to clipboard"),
                            wxITEM_NORMAL );
#endif
    cursorToolBar->AddSeparator();
    cursorToolBar->AddTool( ID_TOOL_MEASURE,
                            _T("Measure"),
                            wxBitmap(cursor),
                            wxT("Mouse selects measurement (crosshair) cursor (\"M\")"),
                            wxITEM_CHECK );
    cursorToolBar->AddTool( ID_TOOL_PEAK,
                            _T("Peak"),
                            wxBitmap(___em_open),
                            wxT("Mouse selects peak cursors (\"P\")"),
                            wxITEM_CHECK );
    cursorToolBar->AddTool( ID_TOOL_BASE,
                            _T("Base"),
                            wxBitmap(___em_down),
                            wxT("Mouse selects base cursors (\"B\")"),
                            wxITEM_CHECK );
    cursorToolBar->AddTool( ID_TOOL_DECAY,
                            _T("Fit"),
                            wxBitmap(fit_lim),//chart_curve),
                            wxT("Mouse selects fit cursors (\"D\")"),
                            wxITEM_CHECK );
    cursorToolBar->AddTool( ID_TOOL_LATENCY,
                            _T("Latency"),
                            wxBitmap(latency_lim),//chart_curve),
                            wxT("Mouse selects latency cursors (\"L\")"),
                            wxITEM_CHECK );
#ifdef WITH_PSLOPE
    cursorToolBar->AddTool( ID_TOOL_PSLOPE,
                            _T("Slope"),
                            wxBitmap(slope),
                            wxT("Mouse selects slope cursors (\"O\")"),
                            wxITEM_CHECK );
#endif
    cursorToolBar->AddTool( ID_TOOL_ZOOM,
                            _T("Zoom"),
                            wxBitmap(zoom),
                            wxT("Draw a zoom window with left mouse button (\"Z\")"),
                            wxITEM_CHECK );
    cursorToolBar->AddTool( ID_TOOL_EVENT,
                            _T("Events"),
                            wxBitmap(event),
                            wxT( "Add, erase or extract events manually with right mouse button (\"E\")" ),
                            wxITEM_CHECK );
    return cursorToolBar;
}

void wxStfParentFrame::OnAbout(wxCommandEvent& WXUNUSED(event) )
{
	wxAboutDialogInfo info;
	info.SetName(wxT("stimfit"));
	info.SetVersion(wxString(VERSION, wxConvLocal));
	info.SetWebSite(wxT("http://www.stimfit.org"));
	wxString about(wxT("Credits:\n\nOriginal idea (Stimfit for DOS):\n\
Peter Jonas, Physiology Department, University of Freiburg\n\n\
Fourier transform:\nFFTW, http://www.fftw.org\n\n\
Levenberg-Marquardt non-linear regression:\n\
Manolis Lourakis, http://www.ics.forth.gr/~lourakis/levmar/ \n\n\
Documentation:\n\
Jose Guzman\n\n\
Event detection by template matching:\n\
Jonas, P., Major, G. & Sakmann B. (1993) J Physiol 472:615-63\n\
Clements, J. D. & Bekkers, J. M. (1997) Biophys J 73:220-229\n\n\
Thanks to Bill Anderson (www.winltp.com) for helpful suggestions"));
	info.SetDescription(about);
	info.SetCopyright(wxT("(C) 2001-2011 Christoph Schmidt-Hieber <christsc@gmx.de>\n\
Christoph Schmidt-Hieber, University College London\n\
Published under the GNU general public license (http://www.gnu.org/licenses/gpl.html)"));

	wxAboutBox(info);
}

void wxStfParentFrame::OnHelp(wxCommandEvent& WXUNUSED(event) )
{
    wxLaunchDefaultBrowser( wxT("http://www.stimfit.org/doc/sphinx/index.html") );
}

std::vector<int> ParseVersionString( const wxString& VersionString ) {
    std::vector<int> VersionInt(5);
    
    const char pt = '.';
    
    // Major version:
    long major=0;
    wxString sMajor = VersionString.BeforeFirst(pt);
    if ( sMajor.length() == VersionString.length() ) {
        major = 0;
    } else {
        sMajor.ToLong( &major );
    }
    VersionInt[0] = major;

    // Minor version:
    long minor=0;
    wxString sMinor1 = VersionString.AfterFirst(pt);
    if ( sMinor1.empty() ) {
        minor = 0;
    } else {
        wxString sMinor = sMinor1.BeforeFirst(pt);
        if ( sMinor1.length() == sMinor.length() ) {
            minor = 0;
        } else {
            sMinor.ToLong( &minor );
        }
    }
    VersionInt[1] = minor;

    // Build version:
    long build=0;
    wxString sBuild = VersionString.AfterLast(pt);
    if ( sBuild.empty() ) {
        build = 0;
    } else {
        sBuild.ToLong( &build );
    }
    VersionInt[2] = build;
    return VersionInt;
}

bool CompVersion( const std::vector<int>& version ) {
    // Get current version:
    wxString currentString(VERSION, wxConvLocal);
    std::vector<int> current = ParseVersionString(currentString);
    if (version[0] > current[0]) {
        return true;
    } else {
        if (version[0] == current[0]) {
            if (version[1] > current[1]) {
                return true;
            } else {
                if (version[1] == current[1]) {
                    if (version[2] > current[2]) {
                        return true;
                    } else {
                        return false;
                    }
                } else {
                    // version[0] == current[0] && version[1] < current[1]
                    return false;
                }
            }
        } else {
            // version[0] < current[0]
            return false;
        }
    }
}

void wxStfParentFrame::CheckUpdate( wxProgressDialog* progDlg ) const {
    
#ifdef __LINUX__
    wxString address(wxT("/latest_linux"));
#elif defined (_WINDOWS)
    wxString address(wxT("/latest_windows"));
#elif defined (__APPLE__)
    wxString address(wxT("/latest_mac"));
#else
    return;
#endif
    
    wxHTTP http;
    http.SetHeader( wxT("Accept") , wxT("text/*") );
    http.SetHeader( wxT("User-Agent"), wxT("Mozilla") );
    http.SetTimeout( 1 ); // seconds

    // Note that Connect() wants a host address, not an URL. 80 is the server's port.
    wxString server( wxT("www.stimfit.org") );
    if( http.Connect(server) )  {
        if(wxInputStream* in_stream = http.GetInputStream (address)) {
            wxString verS;
            int c_int = in_stream->GetC();
            while ( c_int != wxEOF ) {
                if (progDlg != NULL) {
                    progDlg->Pulse( wxT("Reading version information...") );
                }
                verS << wxChar(c_int);
                c_int = in_stream->GetC();
            }
            wxDELETE(in_stream);
            std::vector<int> version = ParseVersionString( verS );
            if ( CompVersion(version) ) {
                wxString msg;
                msg << wxT("A newer version of Stimfit (")
                    << verS << wxT(") is available. ")
                    << wxT("Would you like to download it now?");
                wxMessageDialog newversion( NULL, msg, wxT("New version available"), wxYES_NO );
                if ( newversion.ShowModal() == wxID_YES ) {
                    wxLaunchDefaultBrowser( wxT("http://code.google.com/p/stimfit/downloads/list") );
                }
            } else {
                if (progDlg != NULL) {
                    wxMessageDialog newversion( NULL, wxT("You already have the newest version"), wxT("No new version available"), wxOK );
                    newversion.ShowModal();
                }
            }
        } else {
            if (progDlg != NULL) {
                wxGetApp().ErrorMsg( wxT("Couldn't retrieve update information. Are you connected to the internet?") );
            }
        }
    } else {
        if (progDlg != NULL) {
            wxGetApp().ErrorMsg( wxT("Couldn't connect to server. Are you connected to the internet?") );
        }
    }
}

void wxStfParentFrame::OnCheckUpdate(wxCommandEvent& WXUNUSED(event) )
{
    wxProgressDialog progDlg( wxT("Checking for updates"), wxT("Connecting to server..."),
                              100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE );
    
    CheckUpdate( &progDlg );
}

//#ifdef _WINDOWS
void wxStfParentFrame::OnConvert(wxCommandEvent& WXUNUSED(event) ) {
    // Choose export file type:
    /*
    std::vector< wxString > choices(2);
    choices[0] = wxT("Axon text file (*.atf)");
    choices[1] = wxT("Igor binary wave (*.ibw)");
    wxSingleChoiceDialog typeDlg(this, wxT("Please specify the export file type:"),
                                 wxT("Choose file type"), 2, &choices[0]);
    if (typeDlg.ShowModal() != wxID_OK)
        return;
    stfio::filetype eft = stfio::atf;
    switch ( typeDlg.GetSelection() ) {
     case 0:
         eft = stfio::atf;
         break;
     case 1:
         eft = stfio::igor;
         break;
     default:
         eft = stfio::atf;
    }
    */
    int nfiles; // files to convert
    wxString src_ext; // extension of the source file
    wxString dest_ext; // extesion of the destiny file

    // "Convert files" Dialog (see wxStfConvertDlg in smalldlgs.cpp)
    wxStfConvertDlg myDlg(this);
    if(myDlg.ShowModal()==wxID_OK) {
        //std::cout << myDlg.GetSrcFileExt() << std::endl;
		stfio::filetype ift = myDlg.GetSrcFileExt();
		stfio::filetype eft = myDlg.GetDestFileExt();
        src_ext = myDlg.GetSrcFilter();

        // wxProgressDialog
        wxProgressDialog progDlg( wxT("CFS conversion utility"), wxT("Starting file conversion"),
            100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );

        std::vector<wxString> srcFilenames(myDlg.GetSrcFileNames());
        nfiles = srcFilenames.size(); // number of files to convert
        wxString myDestDir = myDlg.GetDestDir();
        std::cout << myDestDir.c_str() << std::endl;

        for (std::size_t nFile=0; nFile<srcFilenames.size(); ++nFile) {
            wxString progStr;

            // construct new filename:
            wxFileName srcWxFilename(srcFilenames[nFile]);
            wxString destFilename(
                                  myDlg.GetDestDir()+
#ifdef __LINUX__
                                  wxT("/")+
#else
                                  wxT("\\")+
#endif
                                  srcWxFilename.GetName()  // returns file name without path and extension
                                  );
            if ( eft == stfio::atf ) {
                destFilename += wxT(".atf");
            }
            // Update progress bar:
            progStr << wxT("Converting file #") << (int)nFile + 1
                << wxT(" of ") << (int)srcFilenames.size() << wxT("\n")
                << srcFilenames[nFile] << wxT(" -> ") << destFilename;
            progDlg.Update(
                (int)(((double)nFile/(double)srcFilenames.size())*100.0),
                progStr
                );

            // Open source file and convert:
            Recording sourceFile;
            try {
#if 0 //TODO
                if (ift==stfio::ascii) {
                    if (!wxGetApp().get_directTxtImport()) {
                        wxStfTextImportDlg ImportDlg( this,
                                                      stfio::CreatePreview(srcFilenames[nFile]), 1, false );
                        if (ImportDlg.ShowModal()!=wxID_OK) {
                            return;
                        }
                        // store settings in application:
                        wxGetApp().set_txtImportSettings(ImportDlg.GetTxtImport());
                    }
                }
#endif
                stf::wxProgressInfo progDlgIn("Reading file", "Opening file", 100);
                stfio::importFile(stf::wx2std(srcFilenames[nFile]), ift, sourceFile, wxGetApp().GetTxtImport(), progDlgIn);

                stf::wxProgressInfo progDlgOut("Writing file", "Opening file", 100);
                switch ( eft ) {
                 case stfio::atf:
                     stfio::exportATFFile(stf::wx2std(destFilename), sourceFile);
                     dest_ext = wxT("Axon textfile [*.atf]");
                     break;

                 case stfio::igor:
                     stfio::exportIGORFile(stf::wx2std(destFilename), sourceFile, progDlgOut);
                     dest_ext = wxT("Igor binary file [*.ibw]");
                     break;

                 default:
                     wxString errorMsg(wxT("Unknown export file type\n"));
                     wxGetApp().ErrorMsg(errorMsg);
                     return;
                }
            }
           catch (const std::runtime_error& e) {
                wxString errorMsg(wxT("Error opening file\n"));
                errorMsg += wxT("Runtime Error\n");
                errorMsg += wxString( e.what(), wxConvLocal );
                wxGetApp().ExceptMsg(errorMsg);
                return;
            }

            catch (const std::exception& e) {
                wxString errorMsg(wxT("Error opening file\n"));
                errorMsg += wxT("Exception\n");
                errorMsg += wxString( e.what(), wxConvLocal );
                wxGetApp().ExceptMsg(errorMsg);
                return;
            }
        }
    // Show now a smal information dialog
    //std::count << srcFilter.c_str() << std::endl;
    wxString msg;
    msg = wxString::Format(wxT("%i"), nfiles);
    msg << src_ext;
    msg << wxT(" files \nwere converted to ");
    msg << dest_ext;
	wxMessageDialog Simple(this, msg);
	Simple.ShowModal();
    } // end of wxStfConvertDlg

}
//#endif

// Creates a graph. Called from view.cpp when a new drawing
// view is created.
wxStfGraph *wxStfParentFrame::CreateGraph(wxView *view, wxStfChildFrame *parent)
{
    int width=800, height=600;
    parent->GetClientSize(&width, &height);

    // Non-retained graph
    wxStfGraph *graph = new wxStfGraph(
        view,
        parent,
#ifndef __APPLE__
        wxPoint(0, 0),
#else
        wxDefaultPosition,
#endif
        wxSize(width, height),
        wxFULL_REPAINT_ON_RESIZE | wxWANTS_CHARS
        );

    return graph;
}

#ifdef _WINDOWS
void wxStfParentFrame::OnPrint(wxCommandEvent& WXUNUSED(event))
{
    if (wxGetApp().GetActiveDoc()==NULL) return;

    wxPrintDialogData printDialogData(* m_printData);

    wxPrinter printer(& printDialogData);

    wxStfPreprintDlg myDlg(this);
    if (myDlg.ShowModal()!=wxID_OK) return;
    wxStfView* pView=wxGetApp().GetActiveView();
    pView->GetGraph()->set_downsampling(myDlg.GetDownSampling());
    pView->GetGraph()->set_noGimmicks(!myDlg.GetGimmicks());

    wxStfPrintout printout(_T("Trace printout"));

    if (!printer.Print(this, &printout, true /*prompt*/))
    {
        if (wxPrinter::GetLastError() == wxPRINTER_ERROR)
            wxMessageBox(
            _T("There was a problem printing.\nPerhaps your current printer is not set correctly?"),
            _T("Printing"),
            wxOK
            );
        else
            wxMessageBox(_T("You canceled printing"), _T("Printing"), wxOK);
    } else {
        (*m_printData) = printer.GetPrintDialogData().GetPrintData();
    }
}
#endif

void wxStfParentFrame::OnMpl(wxCommandEvent& WXUNUSED(event))
{
    if (wxGetApp().GetActiveDoc()==NULL) return;

    wxWindow* pPython = DoPythonStuff(this, true);
    if ( pPython == 0 ) {
        wxGetApp().ErrorMsg(wxT("Can't create a window for matplotlib\nPointer is zero"));
    } else {
        m_mgr.AddPane( pPython, wxAuiPaneInfo().Name(wxT("mpl")).
                       CloseButton(true).
                       Show(true).Caption(wxT("Matplotlib")).Float().BestSize(800,600));
    }
    m_mgr.Update();
}

void wxStfParentFrame::OnPageSetup(wxCommandEvent& WXUNUSED(event))
{
    (*m_pageSetupData) = *m_printData;

    wxPageSetupDialog pageSetupDialog(this, m_pageSetupData.get());
    pageSetupDialog.ShowModal();

    (*m_printData) = pageSetupDialog.GetPageSetupDialogData().GetPrintData();
    (*m_pageSetupData) = pageSetupDialog.GetPageSetupDialogData();
}

void wxStfParentFrame::OnToggleSelect(wxCommandEvent& WXUNUSED(event)) {
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pDoc!=NULL) {
        pDoc->ToggleSelect();
    }
}

void wxStfParentFrame::OnToolFirst(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->OnFirst();
    }
}

void wxStfParentFrame::OnToolNext(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->OnNext();
    }
}

void wxStfParentFrame::OnToolPrevious(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->OnPrevious();
    }
}

void wxStfParentFrame::OnToolLast(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->OnLast();
    }
}

void wxStfParentFrame::OnToolXenl(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->OnXenllo();
    }
}

void wxStfParentFrame::OnToolXshrink(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->OnXshrinklo();
    }
}

void wxStfParentFrame::OnToolYenl(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->OnYenllo();
    }
}

void wxStfParentFrame::OnToolYshrink(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->OnYshrinklo();
    }
}

void wxStfParentFrame::OnToolUp(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->OnUp();
    }
}

void wxStfParentFrame::OnToolDown(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->OnDown();
    }
}

void wxStfParentFrame::OnToolFit(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->Fittowindow(true);
    }
}

void wxStfParentFrame::OnToolLeft(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->OnLeft();
    }
}

void wxStfParentFrame::OnToolRight(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->OnRight();
    }
}

void wxStfParentFrame::OnToolCh1(wxCommandEvent& WXUNUSED(event)) {
    // activate channel 1 if no channel is active:
    if (!m_scaleToolBar->GetToolToggled(ID_TOOL_CH1) &&
        !m_scaleToolBar->GetToolToggled(ID_TOOL_CH2)) {
            m_scaleToolBar->ToggleTool(ID_TOOL_CH1,true);
    }
    m_scaleToolBar->Refresh();
    
}

void wxStfParentFrame::OnToolCh2(wxCommandEvent& WXUNUSED(event)) {
    // activate channel 1 if no channel is active:
    if (!m_scaleToolBar->GetToolToggled(ID_TOOL_CH1) &&
        !m_scaleToolBar->GetToolToggled(ID_TOOL_CH2)) {
            m_scaleToolBar->ToggleTool(ID_TOOL_CH1,true);
    }
    m_scaleToolBar->Refresh();
}

void wxStfParentFrame::SetSingleChannel(bool value) {
    if (!m_scaleToolBar) return;
    if (value) {
        if (!m_scaleToolBar->GetToolEnabled(ID_TOOL_CH1))
            m_scaleToolBar->EnableTool(ID_TOOL_CH1,true);
        if (m_scaleToolBar->GetToolEnabled(ID_TOOL_CH2))
            m_scaleToolBar->EnableTool(ID_TOOL_CH2,false);
    } else {
        if (!m_scaleToolBar->GetToolEnabled(ID_TOOL_CH1))
            m_scaleToolBar->EnableTool(ID_TOOL_CH1,true);
        if (!m_scaleToolBar->GetToolEnabled(ID_TOOL_CH2))
            m_scaleToolBar->EnableTool(ID_TOOL_CH2,true);
    }

    // Make sure at least one value is selected:
    if (!m_scaleToolBar->GetToolToggled(ID_TOOL_CH1) &&
        (value || !m_scaleToolBar->GetToolToggled(ID_TOOL_CH2))) {
        m_scaleToolBar->ToggleTool(ID_TOOL_CH1, true);
    }
    m_scaleToolBar->Refresh();
}

#ifdef _WINDOWS
void wxStfParentFrame::OnToolSnapshotwmf(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->Snapshotwmf();
    }
}
#endif

void wxStfParentFrame::OnToolMeasure(wxCommandEvent& WXUNUSED(event)) {
    SetMouseQual( stf::measure_cursor );
}

void wxStfParentFrame::OnToolPeak(wxCommandEvent& WXUNUSED(event)) {
    SetMouseQual( stf::peak_cursor );
}

void wxStfParentFrame::OnToolBase(wxCommandEvent& WXUNUSED(event)) {
    SetMouseQual( stf::base_cursor );
}

void wxStfParentFrame::OnToolDecay(wxCommandEvent& WXUNUSED(event)) {
    SetMouseQual( stf::decay_cursor );
}

#ifdef WITH_PSLOPE
void wxStfParentFrame::OnToolPSlope(wxCommandEvent& WXUNUSED(event)) {
    SetMouseQual( stf::pslope_cursor );
}
#endif

void wxStfParentFrame::OnToolLatency(wxCommandEvent& WXUNUSED(event)) {
    SetMouseQual( stf::latency_cursor );
}

void wxStfParentFrame::OnToolZoom(wxCommandEvent& WXUNUSED(event)) {
    SetMouseQual( stf::zoom_cursor );
}

void wxStfParentFrame::OnToolEvent(wxCommandEvent& WXUNUSED(event)) {
    SetMouseQual( stf::event_cursor );
}

void wxStfParentFrame::OnCh2zoom(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->Ch2zoom();
    }
}

void wxStfParentFrame::OnCh2base(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->Ch2base();
    }
}

void wxStfParentFrame::OnCh2pos(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->Ch2pos();
    }
}

void wxStfParentFrame::OnCh2basezoom(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->Ch2basezoom();
    }
}

void wxStfParentFrame::OnViewResults(wxCommandEvent& WXUNUSED(event)) {
    wxStfChildFrame* pChild=(wxStfChildFrame*)GetActiveChild();
    if (pChild!=NULL) {
        pChild->GetCopyGrid()->ViewResults();
    }
}

void wxStfParentFrame::OnScale(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        if (GetActiveChild()->GetMenuBar() && GetActiveChild()->GetMenuBar()->GetMenu(2)->IsChecked(ID_SCALE)) {
            wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("ViewScaleBars"),1);
            wxGetApp().set_isBars(true);
        } else {
            wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("ViewScaleBars"),0);
            wxGetApp().set_isBars(false);
        }
        if (pView->GetGraph() != NULL)
            pView->GetGraph()->Refresh();
    }
}

void wxStfParentFrame::OnHires(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        if (GetActiveChild()->GetMenuBar() && GetActiveChild()->GetMenuBar()->GetMenu(2)->IsChecked(ID_HIRES)) {
            wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("ViewHiRes"),1);
#ifndef __APPLE__
            wxGetApp().set_isHires(true);
#else
            wxGetApp().set_isHires(false);
#endif
        } else {
            wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("ViewHiRes"),0);
            wxGetApp().set_isHires(false);
        }
        if (pView->GetGraph()!=NULL)
            pView->GetGraph()->Refresh();
    }
}

void wxStfParentFrame::OnAverage(wxCommandEvent& WXUNUSED(event)) {
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pDoc!=NULL) {
        pDoc->CreateAverage(false,false);
    }
}

void wxStfParentFrame::OnAlignedAverage(wxCommandEvent& WXUNUSED(event)) {
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pDoc!=NULL) {
        pDoc->CreateAverage(false,true);
    }
}

#if 0
void wxStfParentFrame::OnUserdef(wxCommandEvent& event) {
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pDoc!=NULL) {
        pDoc->Userdef(event.GetId()-ID_USERDEF1);
    }
}
#endif

void wxStfParentFrame::OnSaveperspective(wxCommandEvent& WXUNUSED(event)) {
    wxStfChildFrame* pChild=(wxStfChildFrame*)GetActiveChild();
    if (pChild!=NULL) {
        pChild->Saveperspective();
    }
}

void wxStfParentFrame::OnLoadperspective(wxCommandEvent& WXUNUSED(event)) {
    wxStfChildFrame* pChild=(wxStfChildFrame*)GetActiveChild();
    if (pChild!=NULL) {
        pChild->Loadperspective();
    }
}

void wxStfParentFrame::OnRestoreperspective(wxCommandEvent& WXUNUSED(event)) {
    wxStfChildFrame* pChild=(wxStfChildFrame*)GetActiveChild();
    if (pChild!=NULL) {
        pChild->Restoreperspective();
    }
}

#ifdef WITH_PYTHON
void wxStfParentFrame::OnViewshell(wxCommandEvent& WXUNUSED(event)) {
    // Save the current visibility state:
    bool old_state = m_mgr.GetPane(wxT("pythonShell")).IsShown();
    // Toggle python shell visibility:
    m_mgr.GetPane(wxT("pythonShell")).Show( !old_state );
    wxGetApp().wxWriteProfileInt( wxT("Settings"),wxT("ViewShell"), int(!old_state) );
    m_mgr.Update();
}
#endif

void wxStfParentFrame::OnLStartMaxslope(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        // get previous mode:
        //		bool prevMode=pDoc->GetLatencyStartMode()==stfio::riseMode;
        // toggle on if it wasn't the previous mode:
        //		if (!prevMode) {
        pDoc->SetLatencyStartMode(stf::riseMode);
        wxGetApp().wxWriteProfileInt(wxT("Settings"),
                                     wxT("LatencyStartMode"),
                                     pDoc->GetLatencyStartMode());
        if (pView->GetGraph()!=NULL)
            pView->GetGraph()->Refresh();
    }
}

void wxStfParentFrame::OnLStartHalfrise(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        pDoc->SetLatencyStartMode(stf::halfMode);
        wxGetApp().wxWriteProfileInt(
                                     wxT("Settings"),
                                     wxT("LatencyStartMode"),
                                     pDoc->GetLatencyStartMode()
                                     );
        if (pView->GetGraph()!=NULL)
            pView->GetGraph()->Refresh();
    }

}

void wxStfParentFrame::OnLStartPeak(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        pDoc->SetLatencyStartMode(stf::peakMode);
        wxGetApp().wxWriteProfileInt( wxT("Settings"), wxT("LatencyStartMode"), pDoc->GetLatencyStartMode() );
        if (pView->GetGraph()!=NULL)
            pView->GetGraph()->Refresh();
    }
}

void wxStfParentFrame::OnLStartManual(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        // Always keep manual mode as a default, even if attempted to uncheck:
        pDoc->SetLatencyStartMode(stf::manualMode);
        wxGetApp().wxWriteProfileInt(
            wxT("Settings"),
            wxT("LatencyStartMode"),
            pDoc->GetLatencyStartMode()
            );
        if (pView->GetGraph()!=NULL)
            pView->GetGraph()->Refresh();
    }
}

void wxStfParentFrame::OnLEndFoot(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        pDoc->SetLatencyEndMode(stf::footMode);
	wxGetApp().wxWriteProfileInt(
                                     wxT("Settings"),
                                     wxT("LatencyEndMode"),
                                     pDoc->GetLatencyEndMode()
                                     );
        if (pView->GetGraph()!=NULL)
            pView->GetGraph()->Refresh();
    }
}

void wxStfParentFrame::OnLEndMaxslope(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        pDoc->SetLatencyEndMode(stf::riseMode);
        wxGetApp().wxWriteProfileInt(
                                     wxT("Settings"),
                                     wxT("LatencyEndMode"),
                                     pDoc->GetLatencyEndMode()
                                     );
        if (pView->GetGraph()!=NULL)
            pView->GetGraph()->Refresh();
    }
}

void wxStfParentFrame::OnLEndHalfrise(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        pDoc->SetLatencyEndMode(stf::halfMode);
        wxGetApp().wxWriteProfileInt( wxT("Settings"), wxT("LatencyEndMode"), pDoc->GetLatencyEndMode() );
        if (pView->GetGraph()!=NULL)
            pView->GetGraph()->Refresh();
    }
}

void wxStfParentFrame::OnLEndPeak(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        pDoc->SetLatencyEndMode(stf::peakMode);
        wxGetApp().wxWriteProfileInt( wxT("Settings"), wxT("LatencyEndMode"), pDoc->GetLatencyEndMode() );
        if (pView->GetGraph()!=NULL)
            pView->GetGraph()->Refresh();
    }

}

void wxStfParentFrame::OnLEndManual(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        pDoc->SetLatencyEndMode(stf::manualMode);
        wxGetApp().wxWriteProfileInt( wxT("Settings"), wxT("LatencyEndMode"), pDoc->GetLatencyEndMode() );
        if (pView->GetGraph()!=NULL)
            pView->GetGraph()->Refresh();
    }

}

void wxStfParentFrame::OnLWindow(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL) {
        // Select
        if (GetActiveChild()->GetMenuBar() && GetActiveChild()->GetMenuBar()->GetMenu(1)->IsChecked(ID_LATENCYWINDOW)) {
            wxGetApp().wxWriteProfileInt( wxT("Settings"), wxT("LatencyWindowMode"), stf::windowMode );
            pDoc->SetLatencyWindowMode(stf::windowMode);
        } else {
            wxGetApp().wxWriteProfileInt( wxT("Settings"), wxT("LatencyWindowMode"), stf::defaultMode );
            pDoc->SetLatencyWindowMode(stf::defaultMode);
        }
        if (pView->GetGraph()!=NULL)
            pView->GetGraph()->Refresh();
    }
}

stf::cursor_type wxStfParentFrame::GetMouseQual() const {
    if (m_cursorToolBar->GetToolToggled(ID_TOOL_MEASURE))
        return stf::measure_cursor;
    if (m_cursorToolBar->GetToolToggled(ID_TOOL_PEAK))
        return stf::peak_cursor;
    if (m_cursorToolBar->GetToolToggled(ID_TOOL_BASE))
        return stf::base_cursor;
    if (m_cursorToolBar->GetToolToggled(ID_TOOL_DECAY))
        return stf::decay_cursor;
    if (m_cursorToolBar->GetToolToggled(ID_TOOL_LATENCY))
        return stf::latency_cursor;
    if (m_cursorToolBar->GetToolToggled(ID_TOOL_ZOOM))
        return stf::zoom_cursor;
    if (m_cursorToolBar->GetToolToggled(ID_TOOL_EVENT))
        return stf::event_cursor;
#ifdef WITH_PSLOPE
    if (m_cursorToolBar->GetToolToggled(ID_TOOL_PSLOPE))
        return stf::pslope_cursor;
#endif
    return stf::undefined_cursor;
}

void wxStfParentFrame::SetMouseQual(stf::cursor_type value) {

    if (m_cursorToolBar == NULL)
        return;

    // Need to set everything to false explicitly first:
    m_cursorToolBar->ToggleTool(ID_TOOL_MEASURE,false);
    m_cursorToolBar->ToggleTool(ID_TOOL_PEAK,false);
    m_cursorToolBar->ToggleTool(ID_TOOL_BASE,false);
    m_cursorToolBar->ToggleTool(ID_TOOL_DECAY,false);
    m_cursorToolBar->ToggleTool(ID_TOOL_LATENCY,false);
    m_cursorToolBar->ToggleTool(ID_TOOL_ZOOM,false);
    m_cursorToolBar->ToggleTool(ID_TOOL_EVENT,false);
#ifdef WITH_PSLOPE
    m_cursorToolBar->ToggleTool(ID_TOOL_PSLOPE,false);
#endif

    // Then set the state of the selected button:
    if (value==stf::measure_cursor)
        m_cursorToolBar->ToggleTool(ID_TOOL_MEASURE,true);
    if (value==stf::peak_cursor)
        m_cursorToolBar->ToggleTool(ID_TOOL_PEAK,true);
    if (value==stf::base_cursor)
        m_cursorToolBar->ToggleTool(ID_TOOL_BASE,true);
    if (value==stf::decay_cursor)
        m_cursorToolBar->ToggleTool(ID_TOOL_DECAY,true);
    if (value==stf::latency_cursor)
        m_cursorToolBar->ToggleTool(ID_TOOL_LATENCY,true);
#ifdef WITH_PSLOPE
    if (value==stf::pslope_cursor)
        m_cursorToolBar->ToggleTool(ID_TOOL_PSLOPE,true);
#endif
    if (value==stf::zoom_cursor)
        m_cursorToolBar->ToggleTool(ID_TOOL_ZOOM,true);
    if (value==stf::event_cursor)
        m_cursorToolBar->ToggleTool(ID_TOOL_EVENT,true);

    m_cursorToolBar->Refresh();
}

void wxStfParentFrame::SetSelectedButton(bool selected) {
    if (m_cursorToolBar==NULL)
        return;

    m_cursorToolBar->ToggleTool(ID_TOOL_SELECT, selected);
    m_cursorToolBar->Refresh();
}

stf::zoom_channels wxStfParentFrame::GetZoomQual() const {
    if (m_scaleToolBar->GetToolToggled(ID_TOOL_CH1)) {
        if (m_scaleToolBar->GetToolToggled(ID_TOOL_CH2)) {
            return stf::zoomboth;
        } else {
            return stf::zoomch1;
        }
    }
    return stf::zoomch2;
}

void wxStfParentFrame::SetZoomQual(stf::zoom_channels value) {
    if (m_scaleToolBar==NULL)
        return;

    if (value==stf::zoomch1) {
        m_scaleToolBar->ToggleTool(ID_TOOL_CH1,true);
        m_scaleToolBar->ToggleTool(ID_TOOL_CH2,false);
    }
    if (value==stf::zoomch2) {
        m_scaleToolBar->ToggleTool(ID_TOOL_CH1,false);
        m_scaleToolBar->ToggleTool(ID_TOOL_CH2,true);
    }
    if (value==stf::zoomboth) {
        m_scaleToolBar->ToggleTool(ID_TOOL_CH1,true);
        m_scaleToolBar->ToggleTool(ID_TOOL_CH2,true);
    }
    m_scaleToolBar->Refresh();
}
