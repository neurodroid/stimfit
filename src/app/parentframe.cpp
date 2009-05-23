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

#include "stfconf.h"
#include "./app.h"
#include "./doc.h"
#include "./view.h"
#include "./graph.h"
#include "./table.h"
#include "./printout.h"
#include "./dlgs/smalldlgs.h"
#include "./copygrid.h"
#ifdef _WINDOWS
#include "./../core/filelib/atflib.h"
#include "./../core/filelib/igorlib.h"
#endif

#include "./childframe.h"
#include "./parentframe.h"

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

IMPLEMENT_CLASS(wxStfParentFrame, wxStfParentType)
BEGIN_EVENT_TABLE(wxStfParentFrame, wxStfParentType)
EVT_MENU(wxID_HELP, wxStfParentFrame::OnHelp)
EVT_MENU(wxID_ABOUT, wxStfParentFrame::OnAbout)

EVT_TOOL(wxID_TOOL_FIRST, wxStfParentFrame::OnToolFirst)
EVT_TOOL(wxID_TOOL_NEXT, wxStfParentFrame::OnToolNext)
EVT_TOOL(wxID_TOOL_PREVIOUS, wxStfParentFrame::OnToolPrevious)
EVT_TOOL(wxID_TOOL_LAST, wxStfParentFrame::OnToolLast)
EVT_TOOL(wxID_TOOL_XENL, wxStfParentFrame::OnToolXenl)
EVT_TOOL(wxID_TOOL_XSHRINK, wxStfParentFrame::OnToolXshrink)
EVT_TOOL(wxID_TOOL_YENL, wxStfParentFrame::OnToolYenl)
EVT_TOOL(wxID_TOOL_YSHRINK, wxStfParentFrame::OnToolYshrink)
EVT_TOOL(wxID_TOOL_UP, wxStfParentFrame::OnToolUp)
EVT_TOOL(wxID_TOOL_DOWN, wxStfParentFrame::OnToolDown)
EVT_TOOL(wxID_TOOL_FIT, wxStfParentFrame::OnToolFit)
EVT_TOOL(wxID_TOOL_LEFT, wxStfParentFrame::OnToolLeft)
EVT_TOOL(wxID_TOOL_RIGHT, wxStfParentFrame::OnToolRight)
EVT_TOOL(wxID_TOOL_SNAPSHOT, wxStfParentFrame::OnToolSnapshot)
#ifdef _WINDOWS
EVT_TOOL(wxID_TOOL_SNAPSHOT_WMF, wxStfParentFrame::OnToolSnapshotwmf)
#endif
EVT_TOOL(wxID_TOOL_CH1, wxStfParentFrame::OnToolCh1)
EVT_TOOL(wxID_TOOL_CH2, wxStfParentFrame::OnToolCh2)

EVT_TOOL(wxID_TOOL_MEASURE, wxStfParentFrame::OnToolMeasure)
EVT_TOOL(wxID_TOOL_PEAK,wxStfParentFrame::OnToolPeak)
EVT_TOOL(wxID_TOOL_BASE,wxStfParentFrame::OnToolBase)
EVT_TOOL(wxID_TOOL_DECAY,wxStfParentFrame::OnToolDecay)
EVT_TOOL(wxID_TOOL_LATENCY,wxStfParentFrame::OnToolLatency)
EVT_TOOL(wxID_TOOL_ZOOM,wxStfParentFrame::OnToolZoom)
EVT_TOOL(wxID_TOOL_EVENT,wxStfParentFrame::OnToolEvent)

EVT_MENU(wxID_EXPORTIMAGE, wxStfParentFrame::OnExportimage)
EVT_MENU(wxID_EXPORTPS, wxStfParentFrame::OnExportps)
#if wxCHECK_VERSION(2, 9, 0)
EVT_MENU(wxID_EXPORTSVG, wxStfParentFrame::OnExportsvg)
#endif
EVT_MENU(wxID_EXPORTLATEX, wxStfParentFrame::OnExportlatex)
EVT_MENU(ID_CONVERT, wxStfParentFrame::OnConvert)
EVT_MENU(wxID_AVERAGE, wxStfParentFrame::OnAverage)
EVT_MENU(wxID_ALIGNEDAVERAGE, wxStfParentFrame::OnAlignedAverage)
// EVT_MENU_RANGE(wxID_USERDEF1,wxID_USERDEF2,wxStfParentFrame::OnUserdef)
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
wxStfParentType(manager, frame, wxID_ANY, title, pos, size, type, _T("myFrame"))
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

    wxAuiToolBar* tb1 = CreateStdTb();
    tb1->Realize();

    m_scaleToolBar=CreateScaleTb();
    m_scaleToolBar->Realize();

    wxAuiToolBar* tb4=CreateEditTb();
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
                 << wxT("import wx\n")
                 << wxT("from wx.py import shell, version\n")
                 << wxT("import stf\n")
                 << wxT("import numpy\n")
                 << wxT("try:\n")
                 << wxT("    import stf_init\n")
                 << wxT("except ImportError:\n")
                 << wxT("    loaded = \"\"\n")
                 << wxT("except SyntaxError:\n")
                 << wxT("    loaded = \"\\nSyntax error in custom initialization script stf_init.py\"\n")
                 << wxT("else:\n")
                 << wxT("    loaded = \"\\nSuccessfully loaded custom initialization script stf_init.py\"\n")
		 << wxT("\n")
                 << wxT("class MyPanel(wx.Panel):\n")
                 << wxT("    def __init__(self, parent):\n")
                 << wxT("        wx.Panel.__init__(self, parent, -1, style=wx.BORDER_NONE | wx.MAXIMIZE)\n")
                 << wxT("\n")
                 << wxT("        version_s = \'NumPy \%s, wxPython \%s\' \% (numpy.version.version, wx.version())\n")
                 << wxT("        intro = '") << wxGetApp().GetVersionString() << wxT(", using \%s' \% version_s \n")
                 << wxT("        pycrust = shell.Shell(self, -1, introText=intro + loaded)\n")
                 << wxT("        pycrust.push('import numpy as N', silent=True)\n")
                 << wxT("        pycrust.push('import stf', silent=True)\n")
                 << wxT("        pycrust.push('from stf import *', silent=True)\n")
                 << wxT("        pycrust.push('try:', silent=True)\n")
                 << wxT("        pycrust.push('    from stf_init import *', silent=True)\n")
                 << wxT("        pycrust.push('except ImportError:', silent=True)\n")
                 << wxT("        pycrust.push('    pass', silent=True)\n")
                 << wxT("        pycrust.push('except SyntaxError:', silent=True)\n")
                 << wxT("        pycrust.push('    pass', silent=True)\n")
                 << wxT("        pycrust.push('else:', silent=True)\n")
                 << wxT("        pycrust.push('    pass', silent=True)\n")
                 << wxT("        pycrust.push('', silent=True)\n")
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
                       CloseButton(true).Show(show).
#ifndef __WXMAC__
                       Caption(wxT("Python Shell")).Dockable(true).Bottom().BestSize(GetClientSize().GetWidth(),GetClientSize().GetHeight()/5) );
#else
                       CenterPane().Floatable(false).CaptionVisible(false).MinSize(GetClientSize().GetWidth(),GetClientSize().GetHeight()) );
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
    wxStatusBar* pStatusBar = new wxStatusBar(this);
    SetStatusBar(pStatusBar);
}

wxStfParentFrame::~wxStfParentFrame() {
    // deinitialize the frame manager
    // write visiblity of the shell to config:
    bool shell_state = m_mgr.GetPane(wxT("pythonShell")).IsShown();
    wxGetApp().wxWriteProfileInt( wxT("Settings"),wxT("ViewShell"), int(shell_state) );
    m_mgr.UnInit();
}

wxAuiToolBar* wxStfParentFrame::CreateStdTb() {
    wxAuiToolBar* tb1=new wxAuiToolBar(
        this,
        wxID_ANY,
        wxDefaultPosition,
        wxDefaultSize,
        wxAUI_TB_DEFAULT_STYLE
        );
    tb1->SetToolBitmapSize(wxSize(20,20));
    tb1->AddTool(
                 wxID_OPEN,
                 wxT("Open"),
                 wxArtProvider::GetBitmap(
                                          wxART_FILE_OPEN,
                                          wxART_TOOLBAR,
                                          wxSize(16,16)
                                          ),
                 wxT("Open file"),
                 wxITEM_NORMAL
                 );
    tb1->AddTool(
        wxID_SAVEAS,
        wxT("Save"),
        wxArtProvider::GetBitmap(
        wxART_FILE_SAVE_AS,
        wxART_TOOLBAR,
        wxSize(16,16)
        ),
        wxT("Save traces"),
        wxITEM_NORMAL
        );
    tb1->AddTool(
        WXPRINT_PRINT,
        wxT("Print"),
        wxArtProvider::GetBitmap(
        wxART_PRINT,
        wxART_TOOLBAR,
        wxSize(16,16)
        ),
        wxT("Print traces"),
        wxITEM_NORMAL
        );
    return tb1;
}

wxAuiToolBar* wxStfParentFrame::CreateScaleTb() {
    wxAuiToolBar* scaleToolBar = new wxAuiToolBar(
        this,
        wxID_ANY,
        wxDefaultPosition,
        wxDefaultSize,
        wxAUI_TB_DEFAULT_STYLE
        );
    scaleToolBar->SetToolBitmapSize(wxSize(20,20));
    scaleToolBar->AddTool(
        wxID_TOOL_FIRST,
        wxT("First"),
        wxBitmap(resultset_first),
        wxT("Go to first trace"),
        wxITEM_NORMAL
        );
    scaleToolBar->AddTool(
        wxID_TOOL_PREVIOUS,
        wxT("Prev."),
        wxBitmap(resultset_previous),
        wxT("Go to previous trace (left cursor)"),
        wxITEM_NORMAL
        );
    scaleToolBar->AddTool(
        wxID_TOOL_NEXT,
        wxT("Next"),
        wxBitmap(resultset_next),
        wxT("Go to next trace (right cursor)"),
        wxITEM_NORMAL
        );
    scaleToolBar->AddTool(
        wxID_TOOL_LAST,
        wxT("Last"),
        wxBitmap(resultset_last),
        wxT("Go to last trace"),
        wxITEM_NORMAL
        );
    scaleToolBar->AddSeparator();
    scaleToolBar->AddTool(
        wxID_TOOL_LEFT,
        wxT("Left"),
        wxBitmap(arrow_left),
        wxT("Move traces left (CTRL+left cursor)"),
        wxITEM_NORMAL
        );
    scaleToolBar->AddTool(
        wxID_TOOL_RIGHT,
        wxT("Right"),
        wxBitmap(arrow_right),
        wxT("Move traces right (CTRL+right cursor)"),
        wxITEM_NORMAL
        );
    scaleToolBar->AddTool(
        wxID_TOOL_FIT,
        wxT("Fit"),
        wxBitmap(arrow_out),
        wxT("Fit traces to window (\"F\")"),
        wxITEM_NORMAL
        );
    scaleToolBar->AddTool(
        wxID_TOOL_UP,
        wxT("Up"),
        wxBitmap(arrow_up),
        wxT("Move traces up (up cursor)"),
        wxITEM_NORMAL
        );
    scaleToolBar->AddTool(
        wxID_TOOL_DOWN,
        wxT("Down"),
        wxBitmap(arrow_down),
        wxT("Move traces down (down cursor)"),
        wxITEM_NORMAL
        );
    scaleToolBar->AddTool(
        wxID_TOOL_XENL,
        wxT("Zoom X"),
        wxBitmap(zoom_in),
        wxT("Enlarge x-scale (CTRL + \"+\")"),
        wxITEM_NORMAL
        );
    scaleToolBar->AddTool(
        wxID_TOOL_XSHRINK,
        wxT("Shrink X"),
        wxBitmap(zoom_out),
        wxT("Shrink x-scale (CTRL + \"-\")"),
        wxITEM_NORMAL
        );
    scaleToolBar->AddTool(
        wxID_TOOL_YENL,
        wxT("Zoom Y"),
        wxBitmap(zoom_in),
        wxT("Enlarge y-scale (\"+\")"),
        wxITEM_NORMAL
        );
    scaleToolBar->AddTool(
        wxID_TOOL_YSHRINK,
        wxT("Shrink Y"),
        wxBitmap(zoom_out),
        wxT("Shrink y-scale (\"-\")"),
        wxITEM_NORMAL
        );
    scaleToolBar->AddSeparator();
    scaleToolBar->AddTool(
        wxID_TOOL_CH1,
        wxT("Ch 1"),
        wxBitmap(ch_),
        wxT("Scaling applies to active (black) channel (\"1\")"),
        wxITEM_CHECK
        );
    scaleToolBar->AddTool(
        wxID_TOOL_CH2,
        wxT("Ch 2"),
        wxBitmap(ch2_),
        wxT("Scaling applies to inactive (red) channel (\"2\")"),
        wxITEM_CHECK
        );
    return scaleToolBar;
}

wxAuiToolBar* wxStfParentFrame::CreateEditTb() {
    wxAuiToolBar* tb4= new wxAuiToolBar(
        this,
        wxID_ANY,
        wxDefaultPosition,
        wxDefaultSize,
        wxAUI_TB_DEFAULT_STYLE
        );
    tb4->SetToolBitmapSize(wxSize(20,20));
    tb4->AddTool(
        wxID_AVERAGE,
        wxT("Mean"),
        wxBitmap(sum_new),
        wxT("Average of selected traces"),
        wxITEM_NORMAL
        );
    tb4->AddTool(
        wxID_ALIGNEDAVERAGE,
        wxT("Aligned"),
        wxBitmap(sum_new_aligned),
        wxT("Aligned average of selected traces"),
        wxITEM_NORMAL
        );
    tb4->AddTool(
        wxID_FIT,
        wxT("Fit"),
        wxBitmap(fit),//chart_line),
        wxT("Fit function to data"),
        wxITEM_NORMAL
        );
    tb4->AddTool(
        wxID_VIEWTABLE,
        wxT("Table"),
        wxBitmap(table),
        wxT("View current trace as a table"),
        wxITEM_NORMAL
        );
    return tb4;
}

wxAuiToolBar* wxStfParentFrame::CreateCursorTb() {
    wxAuiToolBar* cursorToolBar = new wxAuiToolBar(
        this,
        wxID_ANY,
        wxDefaultPosition,
        wxDefaultSize,
        wxAUI_TB_DEFAULT_STYLE
        );
    cursorToolBar->SetToolBitmapSize(wxSize(20,20));
    cursorToolBar->AddTool(
        wxID_TOOL_SELECT,
        wxT("Select"),
        wxBitmap( acceptbmp ),
        wxT("Select this trace (\"S\")"),
        wxITEM_NORMAL
        );
    cursorToolBar->AddTool(
        wxID_TOOL_REMOVE,
        wxT("Unselect"),
        wxBitmap( bin ),
        wxT("Unselect this trace (\"R\")"),
        wxITEM_NORMAL
        );
    cursorToolBar->AddTool(
        wxID_TOOL_SNAPSHOT,
        wxT("Snapshot"),
        wxBitmap(camera),
        wxT("Copy bitmap image to clipboard"),
        wxITEM_NORMAL
        );
#ifdef _WINDOWS
    cursorToolBar->AddTool(
        wxID_TOOL_SNAPSHOT_WMF,
        wxT("WMF Snapshot"),
        wxBitmap(camera_ps),
        wxT("Copy vectorized image to clipboard"),
        wxITEM_NORMAL
        );
#endif
    cursorToolBar->AddSeparator();
    cursorToolBar->AddTool(
        wxID_TOOL_MEASURE,
        _T("Measure"),
        wxBitmap(cursor),
        wxT("Mouse selects measurement (crosshair) cursor (\"M\")"),
        wxITEM_CHECK
        );
    cursorToolBar->AddTool(
        wxID_TOOL_PEAK,
        _T("Peak"),
        wxBitmap(___em_open),
        wxT("Mouse selects peak cursors (\"P\")"),
        wxITEM_CHECK
        );
    cursorToolBar->AddTool(
        wxID_TOOL_BASE,
        _T("Base"),
        wxBitmap(___em_down),
        wxT("Mouse selects base cursors (\"B\")"),
        wxITEM_CHECK
        );
    cursorToolBar->AddTool(
        wxID_TOOL_DECAY,
        _T("Fit"),
        wxBitmap(fit_lim),//chart_curve),
        wxT("Mouse selects fit cursors (\"D\")"),
        wxITEM_CHECK
        );
    cursorToolBar->AddTool(
        wxID_TOOL_LATENCY,
        _T("Latency"),
        wxBitmap(latency_lim),//chart_curve),
        wxT("Mouse selects latency cursors (\"L\")"),
        wxITEM_CHECK
        );
    cursorToolBar->AddTool(
        wxID_TOOL_ZOOM,
        _T("Zoom"),
        wxBitmap(zoom),
        wxT("Draw a zoom window with left mouse button (\"Z\")"),
        wxITEM_CHECK
        );
    cursorToolBar->AddTool(
        wxID_TOOL_EVENT,
        _T("Events"),
        wxBitmap(event),
        wxT( "Add, erase or extract events manually with right mouse button (\"E\")" ),
        wxITEM_CHECK
        );
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
	info.SetCopyright(wxT("(C) 2001-2008 Christoph Schmidt-Hieber <christsc@gmx.de>\n\
Christoph Schmidt-Hieber, Physiology Department, University of Freiburg\n\
Published under the GNU general public license (http://www.gnu.org/licenses/gpl.html)"));

	wxAboutBox(info);
}

void wxStfParentFrame::OnHelp(wxCommandEvent& WXUNUSED(event) )
{
    wxLaunchDefaultBrowser( wxT("file:///home/cs/stimfit/doc/sphinx/.build/html/index.html") );
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
#ifdef __LINUX__
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
    int width=800, height=600;
    parent->GetClientSize(&width, &height);

    // Non-retained graph
    wxStfGraph *graph = new wxStfGraph(
        view,
        parent,
#ifndef __WXMAC__
        wxPoint(0, 0),
#else
        wxDefaultPosition,
#endif
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

void wxStfParentFrame::OnToolPrevious(wxCommandEvent& event) {
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
    if (!m_scaleToolBar->GetToolToggled(wxID_TOOL_CH1) &&
        !m_scaleToolBar->GetToolToggled(wxID_TOOL_CH2)) {
            m_scaleToolBar->ToggleTool(wxID_TOOL_CH1,true);
    }
}

void wxStfParentFrame::OnToolCh2(wxCommandEvent& WXUNUSED(event)) {
    // activate channel 1 if no channel is active:
    if (!m_scaleToolBar->GetToolToggled(wxID_TOOL_CH1) &&
        !m_scaleToolBar->GetToolToggled(wxID_TOOL_CH2)) {
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
#ifndef __WXMAC__
            wxGetApp().set_isHires(true);
#else
            wxGetApp().set_isHires(false);
#endif
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
#if 0
    wxStfDoc* pDoc=wxGetApp().GetActiveDoc();
    if (pDoc!=NULL) {
        pDoc->Userdef(event.GetId()-wxID_USERDEF1);
    }
#endif
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
    // Save the current visibility state:
    bool old_state = m_mgr.GetPane(wxT("pythonShell")).IsShown();
    // Toggle python shell visibility:
    m_mgr.GetPane(wxT("pythonShell")).Show( !old_state );
    wxGetApp().wxWriteProfileInt( wxT("Settings"),wxT("ViewShell"), int(!old_state) );
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
        */
        wxGetApp().wxWriteProfileInt( wxT("Settings"), wxT("LatencyEndMode"), pDoc->GetLatencyEndMode() );
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
        */
        wxGetApp().wxWriteProfileInt( wxT("Settings"), wxT("LatencyEndMode"), pDoc->GetLatencyEndMode() );
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
        */
        wxGetApp().wxWriteProfileInt( wxT("Settings"), wxT("LatencyEndMode"), pDoc->GetLatencyEndMode() );
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
    if (m_cursorToolBar->GetToolToggled(wxID_TOOL_MEASURE))
        return stf::measure_cursor;
    if (m_cursorToolBar->GetToolToggled(wxID_TOOL_PEAK))
        return stf::peak_cursor;
    if (m_cursorToolBar->GetToolToggled(wxID_TOOL_BASE))
        return stf::base_cursor;
    if (m_cursorToolBar->GetToolToggled(wxID_TOOL_DECAY))
        return stf::decay_cursor;
    if (m_cursorToolBar->GetToolToggled(wxID_TOOL_LATENCY))
        return stf::latency_cursor;
    if (m_cursorToolBar->GetToolToggled(wxID_TOOL_ZOOM))
        return stf::zoom_cursor;
    if (m_cursorToolBar->GetToolToggled(wxID_TOOL_EVENT))
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

    m_cursorToolBar->Refresh();
}

stf::zoom_channels wxStfParentFrame::GetZoomQual() const {
    if (m_scaleToolBar->GetToolToggled(wxID_TOOL_CH1)) {
        if (m_scaleToolBar->GetToolToggled(wxID_TOOL_CH2)) {
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
    m_scaleToolBar->Refresh();
}
