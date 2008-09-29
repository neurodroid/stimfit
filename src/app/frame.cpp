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

// frame.cpp
// These are the top-level and child windows of the application.
// 2007-12-27, Christoph Schmidt-Hieber, University of Freiburg

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

#ifdef WITH_PYTHON
    #include <Python.h>
    #include <wx/wxPython/wxPython.h>
#endif

#include "./app.h"
#include "./doc.h"
#include "./view.h"
#include "./graph.h"
#include "./table.h"
#include "./printout.h"
#include "./dlgs/smalldlgs.h"
#include "./copygrid.h"
#include "./../core/filelib/cfslib.h"
#include "./../core/filelib/atflib.h"
#include "./../core/filelib/asciilib.h"
#ifdef _WINDOWS
#include "./../core/filelib/igorlib.h"
#endif
#include "./frame.h"

#include "./../icons/16-em-down.xpm"
#include "./../icons/16-em-open.xpm"
#include "./../icons/accept.xpm"
#include "./../icons/arrow_down.xpm"
#include "./../icons/arrow_left.xpm"
#include "./../icons/arrow_out.xpm"
#include "./../icons/arrow_right.xpm"
#include "./../icons/arrow_up.xpm"
#include "./../icons/bin.xpm"
#include "./../icons/camera.xpm"
#ifdef _WINDOWS
#include "./../icons/camera_ps.xpm"
#endif
#include "./../icons/ch1.xpm"
#include "./../icons/ch2.xpm"
#include "./../icons/cursor.xpm"
#include "./../icons/event.xpm"
#include "./../icons/fit.xpm"
#include "./../icons/fit_lim.xpm"
#include "./../icons/latency_lim.xpm"
#include "./../icons/resultset_first.xpm"
#include "./../icons/resultset_last.xpm"
#include "./../icons/resultset_next.xpm"
#include "./../icons/resultset_previous.xpm"
#include "./../icons/sum_new.xpm"
#include "./../icons/sum_new_aligned.xpm"
#include "./../icons/table.xpm"
#include "./../icons/zoom.xpm"
#include "./../icons/zoom_in.xpm"
#include "./../icons/zoom_out.xpm"

IMPLEMENT_CLASS(wxStfParentFrame, wxDocMDIParentFrame)
BEGIN_EVENT_TABLE(wxStfParentFrame, wxDocMDIParentFrame)
EVT_MENU(wxID_ABOUT, wxStfParentFrame::OnAbout)
EVT_MENU(wxID_TOOL_FIRST, wxStfParentFrame::OnToolFirst)
EVT_MENU(wxID_TOOL_NEXT, wxStfParentFrame::OnToolNext)
EVT_MENU(wxID_TOOL_PREVIOUS, wxStfParentFrame::OnToolPrevious)
EVT_MENU(wxID_TOOL_LAST, wxStfParentFrame::OnToolLast)
EVT_MENU(wxID_TOOL_XENL, wxStfParentFrame::OnToolXenl)
EVT_MENU(wxID_TOOL_XSHRINK, wxStfParentFrame::OnToolXshrink)
EVT_MENU(wxID_TOOL_YENL, wxStfParentFrame::OnToolYenl)
EVT_MENU(wxID_TOOL_YSHRINK, wxStfParentFrame::OnToolYshrink)
EVT_MENU(wxID_TOOL_UP, wxStfParentFrame::OnToolUp)
EVT_MENU(wxID_TOOL_DOWN, wxStfParentFrame::OnToolDown)
EVT_MENU(wxID_TOOL_FIT, wxStfParentFrame::OnToolFit)
EVT_MENU(wxID_TOOL_LEFT, wxStfParentFrame::OnToolLeft)
EVT_MENU(wxID_TOOL_RIGHT, wxStfParentFrame::OnToolRight)
EVT_MENU(wxID_TOOL_SNAPSHOT, wxStfParentFrame::OnToolSnapshot)
#ifdef _WINDOWS
EVT_MENU(wxID_TOOL_SNAPSHOT_WMF, wxStfParentFrame::OnToolSnapshotwmf)
#endif
EVT_MENU(wxID_TOOL_CH1, wxStfParentFrame::OnToolCh1)
EVT_MENU(wxID_TOOL_CH2, wxStfParentFrame::OnToolCh2)
EVT_MENU(wxID_EXPORTFILE, wxStfParentFrame::OnExportfile)
EVT_MENU(wxID_EXPORTATF, wxStfParentFrame::OnExportatf)
EVT_MENU(wxID_EXPORTIGOR, wxStfParentFrame::OnExportigor)
EVT_MENU(wxID_EXPORTIMAGE, wxStfParentFrame::OnExportimage)
EVT_MENU(wxID_EXPORTPS, wxStfParentFrame::OnExportps)
#if wxCHECK_VERSION(2, 9, 0)
EVT_MENU(wxID_EXPORTSVG, wxStfParentFrame::OnExportsvg)
#endif
EVT_MENU(wxID_EXPORTLATEX, wxStfParentFrame::OnExportlatex)
EVT_MENU(ID_CONVERT, wxStfParentFrame::OnConvert)
EVT_MENU(wxID_AVERAGE, wxStfParentFrame::OnAverage)
EVT_MENU(wxID_ALIGNEDAVERAGE, wxStfParentFrame::OnAlignedAverage)
EVT_MENU_RANGE(wxID_USERDEF1,wxID_USERDEF21,wxStfParentFrame::OnUserdef)
EVT_MENU( wxID_VIEW_RESULTS, wxStfParentFrame::OnViewResults)
EVT_MENU( wxID_CH2BASE, wxStfParentFrame::OnCh2base )
EVT_MENU( wxID_CH2POS, wxStfParentFrame::OnCh2pos )
EVT_MENU( wxID_CH2ZOOM, wxStfParentFrame::OnCh2zoom )
EVT_MENU( wxID_CH2BASEZOOM, wxStfParentFrame::OnCh2basezoom )
EVT_MENU( wxID_SCALE, wxStfParentFrame::OnScale )
EVT_MENU( wxID_HIRES, wxStfParentFrame::OnHires )
EVT_MENU( WXPRINT_PRINT, wxStfParentFrame::OnPrint)
#if 0
EVT_MENU( WXPRINT_PREVIEW, wxStfParentFrame::OnPrintPreview)
#endif
EVT_MENU( WXPRINT_PAGE_SETUP, wxStfParentFrame::OnPageSetup)
EVT_MENU( wxID_SAVEPERSPECTIVE, wxStfParentFrame::OnSaveperspective )
EVT_MENU( wxID_LOADPERSPECTIVE, wxStfParentFrame::OnLoadperspective )
EVT_MENU( wxID_RESTOREPERSPECTIVE, wxStfParentFrame::OnRestoreperspective )
EVT_MENU( wxID_VIEW_SHELL, wxStfParentFrame::OnViewshell )
EVT_MENU( wxID_LATENCYSTART_MAXSLOPE, wxStfParentFrame::OnLStartMaxslope )
EVT_MENU( wxID_LATENCYSTART_HALFRISE, wxStfParentFrame::OnLStartHalfrise )
EVT_MENU( wxID_LATENCYSTART_PEAK, wxStfParentFrame::OnLStartPeak )
EVT_MENU( wxID_LATENCYSTART_MANUAL, wxStfParentFrame::OnLStartManual )
EVT_MENU( wxID_LATENCYEND_FOOT, wxStfParentFrame::OnLEndFoot )
EVT_MENU( wxID_LATENCYEND_MAXSLOPE, wxStfParentFrame::OnLEndMaxslope )
EVT_MENU( wxID_LATENCYEND_PEAK, wxStfParentFrame::OnLEndPeak )
EVT_MENU( wxID_LATENCYEND_HALFRISE, wxStfParentFrame::OnLEndHalfrise )
EVT_MENU( wxID_LATENCYEND_MANUAL, wxStfParentFrame::OnLEndManual )
EVT_MENU( wxID_LATENCYWINDOW, wxStfParentFrame::OnLWindow )

END_EVENT_TABLE()

wxStfParentFrame::wxStfParentFrame(wxDocManager *manager, wxFrame *frame, const wxString& title,
                 const wxPoint& pos, const wxSize& size, long type):
wxDocMDIParentFrame(manager, frame, wxID_ANY, title, pos, size, type, _T("myFrame"))
{
    ::wxInitAllImageHandlers();
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

    wxToolBar* tb1 = CreateStdTb();
    tb1->Realize();

    m_scaleToolBar=CreateScaleTb();
    m_scaleToolBar->Realize();

    wxToolBar* tb4=CreateEditTb();
    tb4->Realize();

    m_cursorToolBar=CreateCursorTb();
    m_cursorToolBar->Realize();

    // add the toolbars to the manager
    m_mgr.AddPane( tb1, wxAuiPaneInfo().Name(wxT("tb1")).Caption(wxT("Std Toolbar")).
        ToolbarPane().Position(0).Top().Gripper().LeftDockable(false).RightDockable(false) );
    m_mgr.AddPane( m_cursorToolBar, wxAuiPaneInfo().Name(wxT("tb2")).Caption(wxT("Edit Toolbar")).
        ToolbarPane().Position(1).Top().Gripper().LeftDockable(false).RightDockable(false) );
    m_mgr.AddPane( tb4, wxAuiPaneInfo().Name(wxT("tb4")).Caption(wxT("Analysis Toolbar")).
        ToolbarPane().Position(2).Top().Gripper().LeftDockable(false).RightDockable(false) );
    m_mgr.AddPane( m_scaleToolBar, wxAuiPaneInfo().Name(wxT("m_scaleToolBar")).Caption(wxT("Navigation Toolbar")).
        ToolbarPane().Position(3).Top().Gripper().LeftDockable(false).RightDockable(false) );

#ifdef WITH_PYTHON
    python_code2 << wxT("import sys\n")
                 << wxT("sys.path.append('.')\n")
                 << wxT("import wx\n")
                 << wxT("from wx.py import shell, version\n")
#ifndef TEST_MINIMAL
                 << wxT("import stf\n")
                 << wxT("import numpy\n")
#endif
				 << wxT("\n")
                 << wxT("class MyPanel(wx.Panel):\n")
                 << wxT("    def __init__(self, parent):\n")
                 << wxT("        wx.Panel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER)\n")
                 << wxT("\n")
                 << wxT("        version_s = \'NumPy \%s, wxPython \%s\' \% (numpy.version.version, wx.version())\n")
                 << wxT("        intro = '") << wxGetApp().GetVersionString() << wxT(", using \%s' \% version_s \n")
                 << wxT("        pycrust = shell.Shell(self, -1, introText=intro)\n")
#ifndef TEST_MINIMAL
                 << wxT("        pycrust.push('import numpy as N', silent=True)\n")
                 << wxT("        pycrust.push('import stf', silent=True)\n")
                 << wxT("        pycrust.push('from stf import *', silent=True)\n")
#endif
                 << wxT("        sizer = wx.BoxSizer(wx.VERTICAL)\n")
                 << wxT("        sizer.Add(pycrust, 1, wx.EXPAND|wx.BOTTOM|wx.LEFT|wx.RIGHT, 10)\n")
                 << wxT("\n")
                 << wxT("        self.SetSizer(sizer)\n")
                 << wxT("\n")
                 << wxT("def makeWindow(parent):\n")
                 << wxT("    win = MyPanel(parent)\n")
                 << wxT("    return win\n");

    RedirectStdio();
    wxWindow* pPython = DoPythonStuff(this);
    if ( pPython == 0 ) {
        wxGetApp().ErrorMsg(wxT("Can't create a window for the python shell\nPointer is zero"));
    } else {
        bool show = wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewShell"), 1);
        m_mgr.AddPane( pPython, wxAuiPaneInfo().Name(wxT("pythonShell")).
            Caption(wxT("Python Shell")).Bottom().CloseButton(true).Dockable(true).Show(show).
            BestSize(GetClientSize().GetWidth(),GetClientSize().GetHeight()/5) );
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
    wxStatusBar* pStatusBar = new wxStatusBar(this);
    SetStatusBar(pStatusBar);
}

wxStfParentFrame::~wxStfParentFrame() {
    // deinitialize the frame manager
    m_mgr.UnInit();
}

wxToolBar* wxStfParentFrame::CreateStdTb() {
    wxToolBar* tb1=new wxToolBar(
        this,
        wxID_ANY,
        wxDefaultPosition,
        wxDefaultSize,
        wxTB_FLAT | wxTB_NODIVIDER
        );
    tb1->SetToolBitmapSize(wxSize(20,20));
    tb1->AddTool( wxID_OPEN,
        wxT("Open file"),
        wxArtProvider::GetBitmap(
        wxART_FILE_OPEN,
        wxART_OTHER,
        wxSize(16,16)
        ),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Open file")
        );
    tb1->AddTool(
        wxID_SAVEAS,
        wxT("Save as"),
        wxArtProvider::GetBitmap(
        wxART_FILE_SAVE_AS,
        wxART_OTHER,
        wxSize(16,16)
        ),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Save traces")
        );
    tb1->AddTool(
        WXPRINT_PRINT,
        wxT("Print"),
        wxArtProvider::GetBitmap(
        wxART_PRINT,
        wxART_OTHER,
        wxSize(16,16)
        ),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Print traces")
        );
    return tb1;
}

wxToolBar* wxStfParentFrame::CreateScaleTb() {
    wxToolBar* scaleToolBar = new wxToolBar(
        this,
        wxID_ANY,
        wxDefaultPosition,
        wxDefaultSize,
        wxTB_FLAT | wxTB_NODIVIDER //| wxTB_TEXT
        );
    scaleToolBar->SetToolBitmapSize(wxSize(20,20));
    scaleToolBar->AddTool(
        wxID_TOOL_FIRST,
        wxT("First"),
        wxBitmap(resultset_first),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Go to first trace")
        );
    scaleToolBar->AddTool(
        wxID_TOOL_PREVIOUS,
        wxT("Previous"),
        wxBitmap(resultset_previous),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Go to previous trace (left cursor)")
        );
    scaleToolBar->AddTool(
        wxID_TOOL_NEXT,
        wxT("Next"),
        wxBitmap(resultset_next),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Go to next trace (right cursor)")
        );
    scaleToolBar->AddTool(
        wxID_TOOL_LAST,
        wxT("Last"),
        wxBitmap(resultset_last),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Go to last trace")
        );
    scaleToolBar->AddSeparator();
    scaleToolBar->AddTool(
        wxID_TOOL_LEFT,
        wxT("Left"),
        wxBitmap(arrow_left),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Move traces left (CTRL+left cursor)")
        );
    scaleToolBar->AddTool(
        wxID_TOOL_RIGHT,
        wxT("Right"),
        wxBitmap(arrow_right),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Move traces right (CTRL+right cursor)")
        );
    scaleToolBar->AddTool(
        wxID_TOOL_FIT,
        wxT("Fit"),
        wxBitmap(arrow_out),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Fit traces to window (\"F\")")
        );
    scaleToolBar->AddTool(
        wxID_TOOL_UP,
        wxT("Up"),
        wxBitmap(arrow_up),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Move traces up (up cursor)")
        );
    scaleToolBar->AddTool(
        wxID_TOOL_DOWN,
        wxT("Down"),
        wxBitmap(arrow_down),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Move traces down (down cursor)")
        );
    scaleToolBar->AddTool(
        wxID_TOOL_XENL,
        wxT("Enlarge X"),
        wxBitmap(zoom_in),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Enlarge x-scale (CTRL + \"+\")")
        );
    scaleToolBar->AddTool(
        wxID_TOOL_XSHRINK,
        wxT("Shrink X"),
        wxBitmap(zoom_out),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Shrink x-scale (CTRL + \"-\")")
        );
    scaleToolBar->AddTool(
        wxID_TOOL_YENL,
        wxT("Enlarge Y"),
        wxBitmap(zoom_in),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Enlarge y-scale (\"+\")")
        );
    scaleToolBar->AddTool(
        wxID_TOOL_YSHRINK,
        wxT("Shrink Y"),
        wxBitmap(zoom_out),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Shrink y-scale (\"-\")")
        );
    scaleToolBar->AddSeparator();
    scaleToolBar->AddCheckTool(
        wxID_TOOL_CH1,
        wxT("Channel 1"),
        wxBitmap(ch_),
        wxNullBitmap,
        wxT("Scaling applies to active (black) channel (\"1\")")
        );
    scaleToolBar->AddCheckTool(
        wxID_TOOL_CH2,
        wxT("Channel 2"),
        wxBitmap(ch2_),
        wxNullBitmap,
        wxT("Scaling applies to inactive (red) channel (\"2\")")
        );
    return scaleToolBar;
}

wxToolBar* wxStfParentFrame::CreateEditTb() {
    wxToolBar* tb4= new wxToolBar(
        this,
        wxID_ANY,
        wxDefaultPosition,
        wxDefaultSize,
        wxTB_FLAT | wxTB_NODIVIDER
        );
    tb4->SetToolBitmapSize(wxSize(20,20));
    tb4->AddTool(
        wxID_AVERAGE,
        wxT("Average"),
        wxBitmap(sum_new),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Average of selected traces")
        );
    tb4->AddTool(
        wxID_ALIGNEDAVERAGE,
        wxT("Aligned average"),
        wxBitmap(sum_new_aligned),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Aligned average of selected traces")
        );
    tb4->AddTool(
        wxID_FIT,
        wxT("Fit"),
        wxBitmap(fit),//chart_line),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Fit function to data")
        );
    tb4->AddTool(
        wxID_VIEWTABLE,
        wxT("Table"),
        wxBitmap(table),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("View current trace as a table")
        );
    return tb4;
}

wxToolBar* wxStfParentFrame::CreateCursorTb() {
    wxToolBar* cursorToolBar = new wxToolBar(
        this,
        wxID_ANY,
        wxDefaultPosition,
        wxDefaultSize,
        wxTB_FLAT | wxTB_NODIVIDER
        );
    cursorToolBar->SetToolBitmapSize(wxSize(20,20));
    cursorToolBar->AddTool(
        wxID_TOOL_SELECT,
        wxT("Select trace"),
        wxBitmap(acceptbmp),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Select this trace (\"S\")")
        );
    cursorToolBar->AddTool(
        wxID_TOOL_REMOVE,
        wxT("Unselect trace"),
        wxBitmap(bin),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Unselect this trace (\"R\")")
        );
    cursorToolBar->AddTool(
        wxID_TOOL_SNAPSHOT,
        wxT("Snapshot"),
        wxBitmap(camera),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Copy bitmap image to clipboard")
        );
#ifdef _WINDOWS
    cursorToolBar->AddTool(
        wxID_TOOL_SNAPSHOT_WMF,
        wxT("WMF Snapshot"),
        wxBitmap(camera_ps),
        wxNullBitmap,
        wxITEM_NORMAL,
        wxT("Copy vectorized image to clipboard")
        );
#endif
    cursorToolBar->AddSeparator();
    cursorToolBar->AddRadioTool(
        wxID_TOOL_MEASURE,
        _T("Measure"),
        wxBitmap(cursor),
        wxNullBitmap,
        wxT("Mouse selects measurement (crosshair) cursor (\"M\")")
        );
    cursorToolBar->AddRadioTool(
        wxID_TOOL_PEAK,
        _T("Peak"),
        wxBitmap(___em_open),
        wxNullBitmap,
        wxT("Mouse selects peak cursors (\"P\")")
        );
    cursorToolBar->AddRadioTool(
        wxID_TOOL_BASE,
        _T("Base"),
        wxBitmap(___em_down),
        wxNullBitmap,
        wxT("Mouse selects base cursors (\"B\")")
        );
    cursorToolBar->AddRadioTool(
        wxID_TOOL_DECAY,
        _T("Fit function"),
        wxBitmap(fit_lim),//chart_curve),
        wxNullBitmap,
        wxT("Mouse selects fit cursors (\"D\")")
        );
    cursorToolBar->AddRadioTool(
        wxID_TOOL_LATENCY,
        _T("Latency"),
        wxBitmap(latency_lim),//chart_curve),
        wxNullBitmap,
        wxT("Mouse selects latency cursors (\"L\")")
        );
    cursorToolBar->AddRadioTool(
        wxID_TOOL_ZOOM,
        _T("Zoom"),
        wxBitmap(zoom),
        wxNullBitmap,
        wxT("Draw a zoom window with left mouse button (\"Z\")")
        );
    cursorToolBar->AddRadioTool(
        wxID_TOOL_EVENT,
        _T("Edit events"),
        wxBitmap(event),
        wxNullBitmap,
        wxT( "Add, erase or extract events manually with right mouse button (\"E\")" )
        );
    return cursorToolBar;
}

void wxStfParentFrame::OnAbout(wxCommandEvent& WXUNUSED(event) )
{
	wxAboutDialogInfo info;
	info.SetName(wxT("stimfit"));
	info.SetVersion(wxT(STFVERSION));
	info.SetWebSite(wxT("http://www.stimfit.org"));
	wxString about(wxT("Credits:\n\nOriginal idea (Stimfit for DOS):\n\
Peter Jonas, Physiology Department, University of Freiburg\n\n\
Fourier transform:\nFFTW, http://www.fftw.org\n\n\
Levenberg-Marquardt non-linear regression:\n\
Manolis Lourakis, http://www.ics.forth.gr/~lourakis/levmar/ \n\n\
Cubic spline interpolation:\n\
John Burkardt, http://www.scs.fsu.edu/~burkardt/index.html \n\n\
Event detection by template matching:\n\
Jonas, P., Major, G. & Sakmann B. (1993) J Physiol 472:615-63\n\
Clements, J. D. & Bekkers, J. M. (1997) Biophys J 73:220-229\n\n\
Thanks to Bill Anderson (www.winltp.com) for helpful suggestions"));
	info.SetDescription(about);
	info.SetCopyright(wxT("(C) 2001-2008 Christoph Schmidt-Hieber <christsc@gmx.de>\n\
Christoph Schmidt-Hieber, Physiology Department, University of Freiburg\n\
Published under the GNU general public license (http://www.gnu.org/licenses/gpl.html)"));

	wxAboutBox(info);
}

void wxStfParentFrame::OnExportfile(wxCommandEvent& WXUNUSED(event) ) {
    wxFileDialog exportDialog(
        this,
        wxT("Save channel as file series"),
        wxT(""),
        wxT(""),
        wxT("File series (*.*)|*.*"),
        wxFD_SAVE
        );
    if (exportDialog.ShowModal()!=wxID_OK) return;
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pDoc!=NULL) {
        wxString traceFilename(exportDialog.GetPath());
        stf::exportASCIIFile(traceFilename,pDoc->get()[pDoc->GetCurCh()]);
    }
}

void wxStfParentFrame::OnExportatf(wxCommandEvent& WXUNUSED(event) ) {
    wxFileDialog SelectFileDialog(
        this,
        wxT("Export channel as ATF file"),
        wxT(""),
        wxT(""),
        wxT("*.atf"),
        wxFD_SAVE
        );
    if(SelectFileDialog.ShowModal()==wxID_OK) {
        //Get path of the current file
        wxString traceFilename(SelectFileDialog.GetPath());
        wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
        if (pDoc!=NULL) {
            try {
                stf::exportATFFile(traceFilename,*pDoc);
            }
            catch (const std::runtime_error& e) {
                wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
            }
        }
    }
}

void wxStfParentFrame::OnExportigor(wxCommandEvent& WXUNUSED(event) ) {
#ifdef _WINDOWS
    wxFileDialog SelectFileDialog( this, wxT("Set file base for Igor binary waves"),
        wxT(""), wxT(""), wxT("*.*"), wxFD_SAVE );
    if(SelectFileDialog.ShowModal()==wxID_OK) {
        //Get path of the current file
        wxString traceFilename(SelectFileDialog.GetPath());
        wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
        if (pDoc!=NULL) {
            try {
                stf::exportIGORFile(traceFilename,*pDoc);
            }
            catch (const std::runtime_error& e) {
                wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
            }
        }
    }
#else
    wxGetApp().ErrorMsg( wxT("Igor file export only available in Windows version") );
#endif
}

void wxStfParentFrame::OnExportimage(wxCommandEvent& WXUNUSED(event) ) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        wxStfGraph* graph=pView->GetGraph();
        if (graph!=NULL) {
            graph->Exportimage();
        }
    }
}

void wxStfParentFrame::OnExportps(wxCommandEvent& WXUNUSED(event) ) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        wxStfGraph* graph=pView->GetGraph();
        if (graph!=NULL) {
            graph->Exportps();
        }
    }
}

void wxStfParentFrame::OnExportlatex(wxCommandEvent& WXUNUSED(event) ) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        wxStfGraph* graph=pView->GetGraph();
        if (graph!=NULL) {
            graph->Exportlatex();
        }
    }
}
#if wxCHECK_VERSION(2, 9, 0)
void wxStfParentFrame::OnExportsvg(wxCommandEvent& WXUNUSED(event) ) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        wxStfGraph* graph=pView->GetGraph();
        if (graph!=NULL) {
            graph->Exportsvg();
        }
    }
}
#endif

void wxStfParentFrame::OnConvert(wxCommandEvent& WXUNUSED(event) ) {
#ifdef _WINDOWS
    // Choose file type:
	std::vector< wxString > choices(2);
	choices[0] = wxT("Axon text file (*.atf)");
	choices[1] = wxT("Igor binary wave (*.ibw)");
	wxSingleChoiceDialog typeDlg(this, wxT("Please specify the export file type:"),
		wxT("Choose file type"), 2, &choices[0]);
	if (typeDlg.ShowModal() != wxID_OK)
		return;
	stf::filetype eft = stf::atf;
	switch ( typeDlg.GetSelection() ) {
	case 0:
		eft = stf::atf;
		break;
	case 1:
		eft = stf::igor;
		break;
	default:
		eft = stf::atf;
	}
	wxStfConvertDlg myDlg(this);
    if(myDlg.ShowModal()==wxID_OK) {
		stf::filetype ift = stf::findType( myDlg.GetSrcFilter() );
		wxMessageDialog Simple(this, myDlg.GetSrcFilter());
		Simple.ShowModal();
        wxProgressDialog progDlg( wxT("CFS conversion utility"), wxT("Starting file conversion"),
            100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );
        std::vector<wxString> srcFilenames(myDlg.GetSrcFileNames());
        for (std::size_t nFile=0; nFile<srcFilenames.size(); ++nFile) {
            wxString progStr;

            // construct new filename:
            wxFileName srcWxFilename(srcFilenames[nFile]);
            wxString destFilename(
                myDlg.GetDestDir()+
#ifdef __UNIX__
                wxT("/")+
#else
                wxT("\\")+
#endif
                srcWxFilename.GetName()  // returns file name without path and extension
                );
			if ( eft == stf::atf ) {
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
				if (ift==stf::ascii) {
					if (!wxGetApp().get_directTxtImport()) {
						wxStfTextImportDlg ImportDlg( this,
							stf::CreatePreview(srcFilenames[nFile]), 1, false );
						if (ImportDlg.ShowModal()!=wxID_OK) {
							return;
						}
						// store settings in application:
						wxGetApp().set_txtImportSettings(ImportDlg.GetTxtImport());
					}
				}
				stf::importFile(srcFilenames[nFile], ift, sourceFile, wxGetApp().GetTxtImport());
				switch ( eft ) {
				case stf::atf:
					stf::exportATFFile( destFilename, sourceFile );
					break;
				case stf::igor:
					stf::exportIGORFile( destFilename, sourceFile );
					break;
				default:
					wxString errorMsg(wxT("Unknown export file type\n"));
					wxGetApp().ErrorMsg(errorMsg);
					return;
				}
            }
			catch (const std::runtime_error& e) {
				wxString errorMsg(wxT("Error opening file\n"));
				errorMsg += wxString( e.what(),wxConvLocal );
				wxGetApp().ExceptMsg(errorMsg);
				return;
			}
			catch (const std::exception& e) {
				wxString errorMsg(wxT("Error opening file\n"));
				errorMsg += wxString( e.what(), wxConvLocal );
				wxGetApp().ExceptMsg(errorMsg);
				return;
			}
        }
    }
#endif
}

// Creates a graph. Called from view.cpp when a new drawing
// view is created.
wxStfGraph *wxStfParentFrame::CreateGraph(wxView *view, wxStfChildFrame *parent)
{
    int width, height;
    parent->GetClientSize(&width, &height);

    // Non-retained graph
    wxStfGraph *graph = new wxStfGraph(
        view,
        parent,
        wxPoint(0, 0),
        wxSize(width, height),
        wxFULL_REPAINT_ON_RESIZE | wxWANTS_CHARS
        );

    return graph;
}

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

#if 0
void wxStfParentFrame::OnPrintPreview(wxCommandEvent& WXUNUSED(event))
{
    // Pass two printout objects: for preview, and possible printing.
    wxPrintDialogData printDialogData(* m_printData);
    wxStfPreprintDlg myDlg(this);
    if (myDlg.ShowModal()!=wxID_OK) return;
    wxStfView* pView=wxGetApp().GetActiveView();
    pView->GetGraph()->set_downsampling(myDlg.GetDownSampling());
    pView->GetGraph()->set_noGimmicks(!myDlg.GetGimmicks());
    wxPrintPreview *preview = new wxPrintPreview(new wxStfPrintout, new wxStfPrintout, & printDialogData);
    if (!preview->Ok())
    {
        delete preview;
        wxMessageBox(
            _T("There was a problem previewing.\nPerhaps your current printer is not set correctly?"),
            _T("Previewing"), wxOK
            );
        return;
    }

    wxPreviewFrame *pFrame = new wxPreviewFrame(
        preview,
        this,
        _T("Demo Print Preview"),
        wxPoint(100, 100),
        wxSize(600, 650)
        );
    pFrame->Centre(wxBOTH);
    pFrame->Initialize();
    pFrame->Show();
}
#endif

void wxStfParentFrame::OnPageSetup(wxCommandEvent& WXUNUSED(event))
{
    (*m_pageSetupData) = *m_printData;

    wxPageSetupDialog pageSetupDialog(this, m_pageSetupData.get());
    pageSetupDialog.ShowModal();

    (*m_printData) = pageSetupDialog.GetPageSetupDialogData().GetPrintData();
    (*m_pageSetupData) = pageSetupDialog.GetPageSetupDialogData();
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
    if (!m_scaleToolBar->GetToolState(wxID_TOOL_CH1) &&
        !m_scaleToolBar->GetToolState(wxID_TOOL_CH2)) {
            m_scaleToolBar->ToggleTool(wxID_TOOL_CH1,true);
    }
}

void wxStfParentFrame::OnToolCh2(wxCommandEvent& WXUNUSED(event)) {
    // activate channel 1 if no channel is active:
    if (!m_scaleToolBar->GetToolState(wxID_TOOL_CH1) &&
        !m_scaleToolBar->GetToolState(wxID_TOOL_CH2)) {
            m_scaleToolBar->ToggleTool(wxID_TOOL_CH1,true);
    }
}

void wxStfParentFrame::SetSingleChannel(bool value) {
    if (!m_scaleToolBar) return;
    if (value) {
        m_scaleToolBar->ToggleTool(wxID_TOOL_CH1,true);
        m_scaleToolBar->ToggleTool(wxID_TOOL_CH2,false);
        m_scaleToolBar->EnableTool(wxID_TOOL_CH1,false);
        m_scaleToolBar->EnableTool(wxID_TOOL_CH2,false);
    } else {
        m_scaleToolBar->ToggleTool(wxID_TOOL_CH1,true);
        m_scaleToolBar->ToggleTool(wxID_TOOL_CH2,true);
        m_scaleToolBar->EnableTool(wxID_TOOL_CH1,true);
        m_scaleToolBar->EnableTool(wxID_TOOL_CH2,true);
    }
}

void wxStfParentFrame::OnToolSnapshot(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->Snapshot();
    }
}
#ifdef _WINDOWS
void wxStfParentFrame::OnToolSnapshotwmf(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        pView->GetGraph()->Snapshotwmf();
    }
}
#endif

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
        if (GetActiveChild()->GetMenuBar()->GetMenu(2)->IsChecked(wxID_SCALE)) {
            wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("ViewScaleBars"),1);
            wxGetApp().set_isBars(true);
        } else {
            wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("ViewScaleBars"),0);
            wxGetApp().set_isBars(false);
        }
        pView->GetGraph()->Refresh();
    }
}

void wxStfParentFrame::OnHires(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    if (pView!=NULL) {
        if (GetActiveChild()->GetMenuBar()->GetMenu(2)->IsChecked(wxID_HIRES)) {
            wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("ViewHiRes"),1);
            wxGetApp().set_isHires(true);
        } else {
            wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("ViewHiRes"),0);
            wxGetApp().set_isHires(false);
        }
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

void wxStfParentFrame::OnUserdef(wxCommandEvent& event) {
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pDoc!=NULL) {
        pDoc->Userdef(event.GetId()-wxID_USERDEF1);
    }
}

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

void wxStfParentFrame::OnViewshell(wxCommandEvent& WXUNUSED(event)) {
    // Toggle python shell visibility:
    m_mgr.GetPane(wxT("pythonShell")).Show( !(m_mgr.GetPane(wxT("pythonShell")).IsShown()) );
    wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("ViewShell"),
            int(m_mgr.GetPane(wxT("pythonShell")).IsShown()));
    m_mgr.Update();
}

void wxStfParentFrame::OnLStartMaxslope(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        // get previous mode:
        //		bool prevMode=pDoc->GetLatencyStartMode()==stf::riseMode;
        // toggle on if it wasn't the previous mode:
        //		if (!prevMode) {
        pDoc->SetLatencyStartMode(stf::riseMode);
        // Uncheck the other choices:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_MAXSLOPE,true);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_HALFRISE,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_PEAK,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_MANUAL,false);
        /*		} else {
        // else, toggle to manual mode (default)
        pDoc->SetLatencyStartMode(stf::manualMode);
        // Check manual mode:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_MANUAL,true);
        // Uncheck (sometimes isn't done automatically):
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_MAXSLOPE,false);
        }
        */		wxGetApp().wxWriteProfileInt(
        wxT("Settings"),
        wxT("LatencyStartMode"),
        pDoc->GetLatencyStartMode()
        );
        pView->GetGraph()->Refresh();
    }
}

void wxStfParentFrame::OnLStartHalfrise(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        //		if (GetActiveChild()->GetMenuBar()->GetMenu(1)->IsChecked(wxID_LATENCYSTART_HALFRISE)) {
        pDoc->SetLatencyStartMode(stf::halfMode);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_HALFRISE,true);
        // Uncheck the other choices:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_MAXSLOPE,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_PEAK,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_MANUAL,false);
        /*		} else {
        pDoc->SetLatencyStartMode(stf::manualMode);
        // Check manual mode:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_MANUAL,true);
        }
        */		wxGetApp().wxWriteProfileInt(
        wxT("Settings"),
        wxT("LatencyStartMode"),
        pDoc->GetLatencyStartMode()
        );
        pView->GetGraph()->Refresh();
    }

}

void wxStfParentFrame::OnLStartPeak(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        //		if (GetActiveChild()->GetMenuBar()->GetMenu(1)->IsChecked(wxID_LATENCYSTART_PEAK)) {
        pDoc->SetLatencyStartMode(stf::peakMode);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_PEAK,true);
        // Uncheck the other choices:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_MAXSLOPE,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_HALFRISE,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_MANUAL,false);
        /*		} else {
        pDoc->SetLatencyStartMode(stf::manualMode);
        // Check manual mode:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_MANUAL,true);
        }
        */		wxGetApp().wxWriteProfileInt(
        wxT("Settings"),
        wxT("LatencyStartMode"),
        pDoc->GetLatencyStartMode()
        );
        pView->GetGraph()->Refresh();
    }
}

void wxStfParentFrame::OnLStartManual(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        // Always keep manual mode as a default, even if attempted to uncheck:
        pDoc->SetLatencyStartMode(stf::manualMode);
        // Uncheck the other choices:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_MANUAL,true);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_MAXSLOPE,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_HALFRISE,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYSTART_PEAK,false);
        wxGetApp().wxWriteProfileInt(
            wxT("Settings"),
            wxT("LatencyStartMode"),
            pDoc->GetLatencyStartMode()
            );
        pView->GetGraph()->Refresh();
    }
}

void wxStfParentFrame::OnLEndFoot(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        //		if (GetActiveChild()->GetMenuBar()->GetMenu(1)->IsChecked(wxID_LATENCYEND_FOOT)) {
        pDoc->SetLatencyEndMode(stf::footMode);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_FOOT,true);
        // Uncheck the other choices:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_MAXSLOPE,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_MANUAL,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_HALFRISE,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_PEAK,false);
        /*		} else {
        pDoc->SetLatencyEndMode(stf::manualMode);
        // Check manual mode:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_MANUAL,true);
        }
        */		wxGetApp().wxWriteProfileInt(
        wxT("Settings"),
        wxT("LatencyEndMode"),
        pDoc->GetLatencyEndMode()
        );
        pView->GetGraph()->Refresh();
    }
}

void wxStfParentFrame::OnLEndMaxslope(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        //		if (GetActiveChild()->GetMenuBar()->GetMenu(1)->IsChecked(wxID_LATENCYEND_HALFRISE)) {
        pDoc->SetLatencyEndMode(stf::riseMode);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_MAXSLOPE,true);
        // Uncheck the other choices:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_HALFRISE,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_MANUAL,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_FOOT,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_PEAK,false);
        /*		} else {
        pDoc->SetLatencyEndMode(stf::manualMode);
        // Check manual mode:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_MANUAL,true);
        }
        */		wxGetApp().wxWriteProfileInt(
        wxT("Settings"),
        wxT("LatencyEndMode"),
        pDoc->GetLatencyEndMode()
        );
        pView->GetGraph()->Refresh();
    }
}

void wxStfParentFrame::OnLEndHalfrise(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        //		if (GetActiveChild()->GetMenuBar()->GetMenu(1)->IsChecked(wxID_LATENCYEND_HALFRISE)) {
        pDoc->SetLatencyEndMode(stf::halfMode);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_HALFRISE,true);
        // Uncheck the other choices:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_MAXSLOPE,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_MANUAL,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_FOOT,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_PEAK,false);
        /*		} else {
        pDoc->SetLatencyEndMode(stf::manualMode);
        // Check manual mode:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_MANUAL,true);
        }
        */		wxGetApp().wxWriteProfileInt(
        wxT("Settings"),
        wxT("LatencyEndMode"),
        pDoc->GetLatencyEndMode()
        );
        pView->GetGraph()->Refresh();
    }
}

void wxStfParentFrame::OnLEndPeak(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        //		if (GetActiveChild()->GetMenuBar()->GetMenu(1)->IsChecked(wxID_LATENCYEND_PEAK)) {
        pDoc->SetLatencyEndMode(stf::peakMode);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_PEAK,true);
        // Uncheck the other choices:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_MAXSLOPE,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_MANUAL,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_FOOT,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_HALFRISE,false);
        /*		} else {
        pDoc->SetLatencyEndMode(stf::manualMode);
        // Check manual mode:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_MANUAL,true);
        }
        */		wxGetApp().wxWriteProfileInt(
        wxT("Settings"),
        wxT("LatencyEndMode"),
        pDoc->GetLatencyEndMode()
        );
        pView->GetGraph()->Refresh();
    }

}

void wxStfParentFrame::OnLEndManual(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL && pDoc!=NULL) {
        //		if (GetActiveChild()->GetMenuBar()->GetMenu(1)->IsChecked(wxID_LATENCYEND_MANUAL)) {
        pDoc->SetLatencyEndMode(stf::manualMode);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_MANUAL,true);
        // Uncheck the other choices:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_MAXSLOPE,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_PEAK,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_FOOT,false);
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_HALFRISE,false);
        /*		} else {
        pDoc->SetLatencyEndMode(stf::manualMode);
        // Check manual mode:
        GetActiveChild()->GetMenuBar()->GetMenu(1)->Check(wxID_LATENCYEND_MANUAL,true);
        }
        */		wxGetApp().wxWriteProfileInt(
        wxT("Settings"),
        wxT("LatencyEndMode"),
        pDoc->GetLatencyEndMode()
        );
        pView->GetGraph()->Refresh();
    }

}

void wxStfParentFrame::OnLWindow(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=wxGetApp().GetActiveView();
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pView!=NULL) {
        // Select
        if (GetActiveChild()->GetMenuBar()->GetMenu(1)->IsChecked(wxID_LATENCYWINDOW)) {
            wxGetApp().wxWriteProfileInt( wxT("Settings"), wxT("LatencyWindowMode"), stf::windowMode );
            pDoc->SetLatencyWindowMode(stf::windowMode);
        } else {
            wxGetApp().wxWriteProfileInt( wxT("Settings"), wxT("LatencyWindowMode"), stf::defaultMode );
            pDoc->SetLatencyWindowMode(stf::defaultMode);
        }
        pView->GetGraph()->Refresh();
    }
}

stf::cursor_type wxStfParentFrame::GetMouseQual() const {
    if (m_cursorToolBar->GetToolState(wxID_TOOL_MEASURE))
        return stf::measure_cursor;
    if (m_cursorToolBar->GetToolState(wxID_TOOL_PEAK))
        return stf::peak_cursor;
    if (m_cursorToolBar->GetToolState(wxID_TOOL_BASE))
        return stf::base_cursor;
    if (m_cursorToolBar->GetToolState(wxID_TOOL_DECAY))
        return stf::decay_cursor;
    if (m_cursorToolBar->GetToolState(wxID_TOOL_LATENCY))
        return stf::latency_cursor;
    if (m_cursorToolBar->GetToolState(wxID_TOOL_ZOOM))
        return stf::zoom_cursor;
    if (m_cursorToolBar->GetToolState(wxID_TOOL_EVENT))
        return stf::event_cursor;
    return stf::undefined_cursor;
}

void wxStfParentFrame::SetMouseQual(stf::cursor_type value) {

    // Need to set everything to false explicitly first:
    m_cursorToolBar->ToggleTool(wxID_TOOL_MEASURE,false);
    m_cursorToolBar->ToggleTool(wxID_TOOL_PEAK,false);
    m_cursorToolBar->ToggleTool(wxID_TOOL_BASE,false);
    m_cursorToolBar->ToggleTool(wxID_TOOL_DECAY,false);
    m_cursorToolBar->ToggleTool(wxID_TOOL_LATENCY,false);
    m_cursorToolBar->ToggleTool(wxID_TOOL_ZOOM,false);
    m_cursorToolBar->ToggleTool(wxID_TOOL_EVENT,false);

    // Then set the state of the selected button:
    if (value==stf::measure_cursor)
        m_cursorToolBar->ToggleTool(wxID_TOOL_MEASURE,true);
    if (value==stf::peak_cursor)
        m_cursorToolBar->ToggleTool(wxID_TOOL_PEAK,true);
    if (value==stf::base_cursor)
        m_cursorToolBar->ToggleTool(wxID_TOOL_BASE,true);
    if (value==stf::decay_cursor)
        m_cursorToolBar->ToggleTool(wxID_TOOL_DECAY,true);
    if (value==stf::latency_cursor)
        m_cursorToolBar->ToggleTool(wxID_TOOL_LATENCY,true);
    if (value==stf::zoom_cursor)
        m_cursorToolBar->ToggleTool(wxID_TOOL_ZOOM,true);
    if (value==stf::event_cursor)
        m_cursorToolBar->ToggleTool(wxID_TOOL_EVENT,true);
}

stf::zoom_channels wxStfParentFrame::GetZoomQual() const {
    if (m_scaleToolBar->GetToolState(wxID_TOOL_CH1)) {
        if (m_scaleToolBar->GetToolState(wxID_TOOL_CH2)) {
            return stf::zoomboth;
        } else {
            return stf::zoomch1;
        }
    }
    return stf::zoomch2;
}

void wxStfParentFrame::SetZoomQual(stf::zoom_channels value) {
    if (value==stf::zoomch1) {
        m_scaleToolBar->ToggleTool(wxID_TOOL_CH1,true);
        m_scaleToolBar->ToggleTool(wxID_TOOL_CH2,false);
    }
    if (value==stf::zoomch2) {
        m_scaleToolBar->ToggleTool(wxID_TOOL_CH1,false);
        m_scaleToolBar->ToggleTool(wxID_TOOL_CH2,true);
    }
    if (value==stf::zoomboth) {
        m_scaleToolBar->ToggleTool(wxID_TOOL_CH1,true);
        m_scaleToolBar->ToggleTool(wxID_TOOL_CH2,true);
    }
}

wxStfAppAboutDialog::wxStfAppAboutDialog( wxWindow* parent, int id, wxString title, wxPoint pos,
										 wxSize size, int style )
										 : wxDialog( parent, id, title, pos, size, style )
{
    this->SetSize(464,464);
    this->SetTitle(wxT("About stimfit"));
    wxBoxSizer* bSizer;
    bSizer = new wxBoxSizer( wxVERTICAL );
    wxStaticText* titleStatic;
    titleStatic = new wxStaticText(this,wxID_ANY,wxT(""),wxDefaultPosition,wxDefaultSize,0);
    titleStatic->SetFont( wxFont(
        10, wxFONTFAMILY_ROMAN, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false ) );
    titleStatic->SetLabel(wxGetApp().GetVersionString());
    bSizer->Add( titleStatic, 0, wxALIGN_CENTER | wxALL, 2 );

    wxString about(wxT("\nChristoph Schmidt-Hieber, Physiology Department, University of Freiburg\n\
Copyright (C) 2001-2008\n\
Published under the GNU general public license (http://www.gnu.org/licenses/gpl.html)\n\n\
Credits:\n\nOriginal idea (Stimfit for DOS):\n\
Peter Jonas, Physiology Department, University of Freiburg\n\n\
Fourier transform:\nFFTW, http://www.fftw.org\n\n\
Levenberg-Marquardt non-linear regression:\n\
Manolis Lourakis, http://www.ics.forth.gr/~lourakis/levmar/ \n\n\
Cubic spline interpolation:\n\
John Burkardt, http://www.scs.fsu.edu/~burkardt/index.html \n\n\
Event detection by template matching:\n\
Jonas, P., Major, G. & Sakmann B. (1993) J Physiol 472:615-63\n\
Clements, J. D. & Bekkers, J. M. (1997) Biophys J 73:220-229\n\n\
Thanks to Bill Anderson (www.winltp.com) for helpful suggestions"));

    m_textCtrl1 =
        new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition,
                wxSize( 432,320 ), wxTE_MULTILINE|wxTE_READONLY );
    m_textCtrl1->SetFont( wxFont( 10, 74, 90, 90, false ) );
    m_textCtrl1->AppendText(about);

    bSizer->Add( m_textCtrl1, 0, wxALIGN_CENTER | wxALL, 2 );

    m_sdbSizer = new wxStdDialogButtonSizer();
    m_sdbSizer->AddButton( new wxButton( this, wxID_OK ) );
    m_sdbSizer->Realize();
    bSizer->Add( m_sdbSizer, 0, wxALIGN_CENTER | wxALL, 2 );
    bSizer->SetSizeHints(this);
    this->SetSizer( bSizer );

    this->Layout();

}

IMPLEMENT_CLASS(wxStfChildFrame, wxDocMDIChildFrame)

BEGIN_EVENT_TABLE(wxStfChildFrame, wxDocMDIChildFrame)
EVT_COMBOBOX( wxCOMBOTRACES, wxStfChildFrame::OnComboTraces )
EVT_COMBOBOX( wxCOMBOACTCHANNEL, wxStfChildFrame::OnComboActChannel )
EVT_COMBOBOX( wxCOMBOINACTCHANNEL, wxStfChildFrame::OnComboInactChannel )
EVT_CHECKBOX( wxID_PLOTSELECTED, wxStfChildFrame::OnPlotselected )
// workaround for status bar:
EVT_MENU_HIGHLIGHT_ALL( wxStfChildFrame::OnMenuHighlight )
END_EVENT_TABLE()

wxStfChildFrame::wxStfChildFrame(wxDocument* doc, wxView* view, wxDocMDIParentFrame* parent,
        wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size,
        long style, const wxString& name) :
            wxDocMDIChildFrame(doc,view,parent,id,title,pos,size,style,name),
            m_notebook(NULL)
{
    m_mgr.SetManagedWindow(this);
    m_mgr.SetFlags( wxAUI_MGR_ALLOW_FLOATING | wxAUI_MGR_TRANSPARENT_DRAG |
        wxAUI_MGR_VENETIAN_BLINDS_HINT | wxAUI_MGR_ALLOW_ACTIVE_PANE );

    m_table=CreateTable();
    m_mgr.AddPane( m_table, wxAuiPaneInfo().Caption(wxT("Results")).Position(2).
            CloseButton(false).Floatable().Dock().Top().Name(wxT("Results")) );

    // m_mgr.Update() is done when a graph is created.

}

wxStfChildFrame::~wxStfChildFrame() {
    // deinitialize the frame manager
    m_mgr.UnInit();
}

wxStfGrid* wxStfChildFrame::CreateTable() {
    // create the notebook off-window to avoid flicker
    wxSize client_size = GetClientSize();

    wxStfGrid* ctrl = new wxStfGrid( this, wxID_ANY,
            wxDefaultPosition, wxDefaultSize,
            wxVSCROLL | wxHSCROLL );
    wxFont font( 8, wxFONTFAMILY_MODERN, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL );
    ctrl->SetDefaultCellFont(font);
    ctrl->SetDefaultColSize(108);
    ctrl->SetColLabelSize(20);
    ctrl->SetDefaultCellAlignment(wxALIGN_RIGHT,wxALIGN_CENTRE);
    ctrl->CreateGrid(3,10);

    ctrl->EnableEditing(false);
    return ctrl;
}

wxAuiNotebook* wxStfChildFrame::CreateNotebook() {
    // create the notebook off-window to avoid flicker
    wxSize client_size = GetClientSize();
    m_notebook_style =
        wxAUI_NB_SCROLL_BUTTONS |
        wxAUI_NB_CLOSE_ON_ACTIVE_TAB |/*wxAUI_NB_DEFAULT_STYLE | */
        wxNO_BORDER;

    wxAuiNotebook* ctrl = new wxAuiNotebook( this, wxID_ANY,
            wxPoint(client_size.x, client_size.y), wxSize(200,200),
            m_notebook_style );

    return ctrl;
}

wxPanel* wxStfChildFrame::CreateTraceCounter() {
    wxSize client_size = GetClientSize();

    wxPanel* ctrl = new wxPanel( this, wxID_ANY, wxDefaultPosition,
        wxSize(160,88), 0 );

    pTraceSizer = new wxFlexGridSizer( 0, 1, 2, 2 );

    return ctrl;
}

wxPanel* wxStfChildFrame::CreateChannelCounter() {
    wxPanel* ctrl = new wxPanel( this, wxID_ANY, wxDefaultPosition,
            wxSize(224,88), 0 );

    pChannelSizer = new wxFlexGridSizer( 2, 2, 2, 2 );

    return ctrl;
}

void wxStfChildFrame::CreateComboTraces(const std::size_t value) {

    m_traceCounter = CreateTraceCounter();
    pTraceNumberSizer = new wxFlexGridSizer( 0, 3, 2, 2 );

    pSelected = new wxStaticText( m_traceCounter, wxID_ANY, wxT("Selected traces: 0") );
    pTraceSizer->Add( pSelected );

    wxStaticText* pTraceStaticText = new wxStaticText( m_traceCounter, wxID_ANY, wxT("Trace") );
    pTraceNumberSizer->Add( pTraceStaticText, wxALIGN_BOTTOM );

    pSize=new wxStaticText( m_traceCounter, wxID_ANY, wxT("of 0") );

    wxArrayString szTraces;
    szTraces.Alloc(value);
    for (std::size_t n=0;n<value;++n) {
		wxString number;
        number << (int)n+1;
        szTraces.Add(number);
    }
    pTraces=new wxComboBox(	m_traceCounter, wxCOMBOTRACES, wxT("1"), wxDefaultPosition,
            wxSize(64,24), szTraces, wxCB_DROPDOWN | wxCB_READONLY );
    pTraceNumberSizer->Add( pTraces, wxALIGN_CENTRE );

    wxString sizeStr;
    sizeStr << wxT("of ") << (int)value;
    pSize->SetLabel(sizeStr);
    pTraceNumberSizer->Add( pSize, wxALIGN_BOTTOM );

    pTraceSizer->Add( pTraceNumberSizer, wxALIGN_BOTTOM );

    pPlotSelected=new wxCheckBox( m_traceCounter, wxID_PLOTSELECTED, wxT("Plot selected traces") );
    pPlotSelected->SetValue(false);
    pTraceSizer->Add( pPlotSelected );

    pTraceIndex = new wxStaticText( m_traceCounter, wxID_ANY, wxT("Current trace index: 0") );
    pTraceSizer->Add( pTraceIndex );

    m_traceCounter->SetSizer( pTraceSizer );
    m_traceCounter->Layout();

    m_mgr.AddPane( m_traceCounter, wxAuiPaneInfo().Caption(wxT("Trace selection")).Fixed().
            Position(1).CloseButton(false).Floatable().Dock().Top().Name(wxT("SelectionT")) );
}

void wxStfChildFrame::CreateComboChannels(const wxArrayString& channelStrings) {
    m_channelCounter = CreateChannelCounter();

    wxStaticText* pActIndex  = new wxStaticText( m_channelCounter, wxID_ANY, wxT("Active channel index: ") );
    pChannelSizer->Add( pActIndex );

    pActChannel = new wxComboBox( m_channelCounter, wxCOMBOACTCHANNEL, wxT("0"),
        wxDefaultPosition, wxSize(64,24), channelStrings, wxCB_DROPDOWN | wxCB_READONLY );
    pChannelSizer->Add( pActChannel );

    wxStaticText* pInactIndex = new wxStaticText( m_channelCounter, wxID_ANY, wxT("Inactive channel index: ") );
    pChannelSizer->Add( pInactIndex );

    pInactChannel = new wxComboBox( m_channelCounter, wxCOMBOINACTCHANNEL, wxT("1"),
            wxDefaultPosition, wxSize(64,24), channelStrings, wxCB_DROPDOWN | wxCB_READONLY );
    pChannelSizer->Add( pInactChannel );

    m_channelCounter->SetSizer( pChannelSizer );
    m_channelCounter->Layout();

    m_mgr.AddPane( m_channelCounter, wxAuiPaneInfo().Caption(wxT("Channel selection")).Fixed().
            Position(0).CloseButton(false).Floatable().Dock().Top().Name(wxT("SelectionC")) );

}

void wxStfChildFrame::SetSelected(std::size_t value) {
    wxString selStr;
    selStr << wxT("Selected traces: ") << (int)value;
    pSelected->SetLabel(selStr);
}

void wxStfChildFrame::SetChannels( std::size_t act, std::size_t inact ) {
    pActChannel->SetSelection( act );
    pInactChannel->SetSelection( inact );
}

std::size_t wxStfChildFrame::GetCurTrace() const {
    return pTraces->GetCurrentSelection();
}

void wxStfChildFrame::SetCurTrace(std::size_t n) {
    pTraces->SetSelection((int)n);
    wxString indStr;
    indStr << wxT("Zero-based index: ") << (int)n;
    pTraceIndex->SetLabel( indStr );
}

void wxStfChildFrame::OnComboTraces(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=(wxStfView*)GetView();
    wxStfDoc* pDoc=(wxStfDoc*)GetDocument();
    pDoc->SetSection(GetCurTrace());
    wxString indStr;
    indStr << wxT("Zero-based index: ") << GetCurTrace();
    pTraceIndex->SetLabel( indStr );
    wxGetApp().OnPeakcalcexecMsg();
    pView->GetGraph()->Refresh();
    pView->GetGraph()->SetFocus();
}

void wxStfChildFrame::OnComboActChannel(wxCommandEvent& WXUNUSED(event)) {
    if ( pActChannel->GetCurrentSelection() == pInactChannel->GetCurrentSelection()) {
        // correct selection:
        for (int n_c=0;n_c<(int)pActChannel->GetCount();++n_c) {
            if (n_c!=pActChannel->GetCurrentSelection()) {
                pInactChannel->SetSelection(n_c);
                break;
            }
        }
    }

    UpdateChannels();
}

void wxStfChildFrame::OnComboInactChannel(wxCommandEvent& WXUNUSED(event)) {
    if (pInactChannel->GetCurrentSelection()==pActChannel->GetCurrentSelection()) {
        // correct selection:
        for (int n_c=0;n_c<(int)pInactChannel->GetCount();++n_c) {
            if (n_c!=pInactChannel->GetCurrentSelection()) {
                pActChannel->SetSelection(n_c);
                break;
            }
        }
    }

    UpdateChannels();
}

void wxStfChildFrame::UpdateChannels( ) {

    wxStfDoc* pDoc=(wxStfDoc*)GetDocument();

    if ( pDoc != NULL && pDoc->size() > 1) {
        try {
            pDoc->SetCurCh( pActChannel->GetCurrentSelection() );
            pDoc->SetSecCh( pInactChannel->GetCurrentSelection() );
        }
        catch (const std::out_of_range& e) {
            wxString msg(wxT("Error while changing channels\nPlease close file\n"));
            msg += wxString( e.what(), wxConvLocal );
            wxGetApp().ExceptMsg(msg);
            return;
        }

        // Update measurements:
        wxGetApp().OnPeakcalcexecMsg();
        UpdateResults();
        wxStfView* pView=(wxStfView*)GetView();
        if ( pView == NULL ) {
            wxGetApp().ErrorMsg( wxT("View is zero in wxStfDoc::SwapChannels"));
            return;
        }
        pView->GetGraph()->Refresh();
        pView->GetGraph()->SetFocus();
    }
}

void wxStfChildFrame::OnPlotselected(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=(wxStfView*)GetView();
    pView->GetGraph()->Refresh();
    pView->GetGraph()->SetFocus();
}

void wxStfChildFrame::ActivateGraph() {
       wxStfView* pView=(wxStfView*)GetView();
       // Set the focus somewhere else:
       m_traceCounter->SetFocus();
       pView->GetGraph()->SetFocus();
}

void wxStfChildFrame::ShowTable(const stf::Table &table,const wxString& caption) {

    // Create and show notebook if necessary:
    if (m_notebook==NULL && !m_mgr.GetPane(m_notebook).IsOk()) {
        m_notebook=CreateNotebook();
        m_mgr.AddPane( m_notebook, wxAuiPaneInfo().Caption(wxT("Analysis results")).
            Floatable().Dock().Left().Name( wxT("Notebook") ) );
    } else {
        // Re-open notebook if it has been closed:
        if (!m_mgr.GetPane(m_notebook).IsShown()) {
            m_mgr.GetPane(m_notebook).Show();
        }
    }
    wxStfGrid* pGrid = new wxStfGrid( m_notebook, wxID_ANY, wxPoint(0,20), wxDefaultSize );
    wxStfTable* pTable(new wxStfTable(table));
    pGrid->SetTable(pTable,true); // the grid will take care of the deletion
    pGrid->SetEditable(false);
    pGrid->SetDefaultCellAlignment(wxALIGN_RIGHT,wxALIGN_CENTRE);
    for (std::size_t n_row=0; n_row<=table.nRows()+1; ++n_row) {
        pGrid->SetCellAlignment(wxALIGN_LEFT,(int)n_row,0);
    }
    m_notebook->AddPage( pGrid, caption, true );

    // "commit" all changes made to wxAuiManager
    m_mgr.Update();
    wxStfView* pView=(wxStfView*)GetView();
    pView->GetGraph()->SetFocus();
}

void wxStfChildFrame::UpdateResults() {
    wxStfDoc* pDoc=(wxStfDoc*)GetDocument();
    stf::Table table(pDoc->CurResultsTable());
    // Delete or append columns:
    if (m_table->GetNumberCols()<(int)table.nCols()) {
        m_table->AppendCols((int)table.nCols()-(int)m_table->GetNumberCols());
    } else {
        if (m_table->GetNumberCols()>(int)table.nCols()) {
            m_table->DeleteCols(0,(int)m_table->GetNumberCols()-(int)table.nCols());
        }
    }

    // Delete or append row:
    if (m_table->GetNumberRows()<(int)table.nRows()) {
        m_table->AppendRows((int)table.nRows()-(int)m_table->GetNumberRows());
    } else {
        if (m_table->GetNumberRows()>(int)table.nRows()) {
            m_table->DeleteRows(0,(int)m_table->GetNumberRows()-(int)table.nRows());
        }
    }

    for (std::size_t nRow=0;nRow<table.nRows();++nRow) {
        // set row label:
        m_table->SetRowLabelValue((int)nRow,table.GetRowLabel(nRow));
        for (std::size_t nCol=0;nCol<table.nCols();++nCol) {
            if (nRow==0) m_table->SetColLabelValue((int)nCol,table.GetColLabel(nCol));
            if (!table.IsEmpty(nRow,nCol)) {
                wxString entry; entry << table.at(nRow,nCol);
                m_table->SetCellValue((int)nRow,(int)nCol,entry);
            } else {
                m_table->SetCellValue((int)nRow,(int)nCol,wxT("n.a."));
            }
        }
    }
}

void wxStfChildFrame::Saveperspective() {
    wxString perspective = m_mgr.SavePerspective();
    // Save to wxConfig:
    wxGetApp().wxWriteProfileString(wxT("Settings"),wxT("Windows"),perspective);
#ifdef _STFDEBUG
    wxFile persp(wxT("perspective.txt"), wxFile::write);
    persp.Write(perspective);
    persp.Close();
#endif
}

void wxStfChildFrame::Loadperspective() {
    wxString perspective = wxGetApp().wxGetProfileString(wxT("Settings"),wxT("Windows"),wxT(""));
    if (perspective!=wxT("")) {
        m_mgr.LoadPerspective(perspective);
    } else {
        wxGetApp().ErrorMsg(wxT("Couldn't find saved windows settings"));
    }
}


void wxStfChildFrame::Restoreperspective() {
    m_mgr.LoadPerspective(defaultPersp);
    m_mgr.Update();
}

void wxStfChildFrame::OnMenuHighlight(wxMenuEvent& event) {
    wxMenuItem *item = this->GetMenuBar()->FindItem(event.GetId());
    if(item)
        wxLogStatus(item->GetHelp());

}

#if wxUSE_DRAG_AND_DROP
bool wxStfFileDrop::OnDropFiles(wxCoord WXUNUSED(x), wxCoord WXUNUSED(y), const wxArrayString& filenames) {
    int nFiles=(int)filenames.GetCount();
    if (nFiles>0) {
        return wxGetApp().OpenFileSeries(filenames);
    } else {
        return false;
    }
}
#endif
