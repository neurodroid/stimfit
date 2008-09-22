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

/*! \file frame.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares wxStfParentFrame and wxStfChildFrame.
 */

#ifndef _FRAME_H
#define _FRAME_H

/*! \addtogroup wxstf
 *  @{
 */

#include "wx/aui/aui.h"
#include "wx/grid.h"
#include "wx/dnd.h"

#include "./../core/core.h"

// Define a new frame
class wxStfGraph;
class wxStfTable;
class wxStfGrid;

//! Default perspective string.
/*! Can be loaded to restore the default AUI perspective. */
const wxString defaultPersp =
wxT("layout2| \
name=Results;caption=Results;state=2044;dir=1;layer=0;row=0;pos=1;prop=167270; \
bestw=200;besth=184;minw=-1;minh=-1;maxw=-1;maxh=-1;floatx=-1;floaty=-1;floatw=-1;floath=-1| \
name=Selection;caption=Trace selection;state=2044;dir=1;layer=0;row=0;pos=0;prop=32730; \
bestw=128;besth=64;minw=-1;minh=-1;maxw=-1;maxh=-1;floatx=-1;floaty=-1;floatw=-1;floath=-1| \
name=Traces;caption=Traces;state=18428;dir=5;layer=0;row=0;pos=0;prop=100000; \
bestw=20;besth=20;minw=-1;minh=-1;maxw=-1;maxh=-1;floatx=-1;floaty=-1;floatw=-1;floath=-1|");

//! Provides the child frame for displaying documents on separate windows.
/*! This class can only be used for MDI child frames. It is part of the document/view 
 *  framework supported by wxWidgets.
 */
class StfDll wxStfChildFrame : public wxDocMDIChildFrame
{
    DECLARE_CLASS( wxStfChildFrame )
public:
    //! Constructor
    /*! \param doc Pointer to the attached document.
     *  \param view Pointer to the attached view.
     *  \param parent Pointer to the parent frame.
     *  \param id Window id.
     *  \param title Window title string.
     *  \param pos Initial window position.
     *  \param size Initial window size.
     *  \param style Window style.
     *  \param name Name of this frame.
     */
    wxStfChildFrame(
            wxDocument* doc, 
            wxView* view, 
            wxDocMDIParentFrame* parent, 
            wxWindowID id, 
            const wxString& title, 
            const wxPoint& pos = wxDefaultPosition, 
            const wxSize& size = wxDefaultSize, 
            long style = wxDEFAULT_FRAME_STYLE, 
            const wxString& name = wxT("frame")
    );
    //! Destructor
    ~wxStfChildFrame();

    //! Adds a table to the results notebook
    /*! \param table The table to be added.
     *  \param caption The title of the new table in the notebook.
     */
    void ShowTable(const stf::Table& table,const wxString& caption);
    
    //! Retrieves the current trace from the trace selection combo box.
    /*! \return The 0-based index of the currently selected trace.
     */
    std::size_t GetCurTrace() const;

    //! Sets the current trace from the trace selection combo box.
    /*! \return The 0-based index of the trace to be selected.
     */
    void SetCurTrace(std::size_t);
    
    //! Creates the trace selection combo box.
    /*! \param value The number of traces in the combo box drop-down list.
     */
    void CreateComboTraces(std::size_t value);
    
    
    //! Creates the channel selection combo boxes.
    /*! \param value The channel names for the combo box drop-down list.
     */
    void CreateComboChannels( const wxArrayString& channelNames );

    //! Refreshes the trace selection string.
    /*! \param value The number of selected traces.
     */
    void SetSelected(std::size_t value);

    //! Sets the channels in the combo boxes. Checks and corrects equal channels in both boxes.
    /*! \param act Index of the active channel.
     *  \param inact Index of the inactive channel.
     */
    void SetChannels( std::size_t act, std::size_t inact );
    
    //! Updates the channels according to the current combo boy selection.
    void UpdateChannels( );

    //! Updates the results table.
    /*! Called from wxStfApp::OnPeakcalcexecMsg() to update the results table.
     *  Don't call this directly; use wxStfApp::OnPeakcalcexecMsg() instead.
     */
    void UpdateResults();
    
    //! Retrieve the wxAuiManager.
    /*! \return A pointer to the wxAuiManager.
     */
    wxAuiManager* GetMgr() {return &m_mgr;}

    //! Retrieve the wxStfGrid that contains the results table.
    /*! \return A pointer to the grid.
     */
    wxStfGrid* GetCopyGrid() {return m_table;}
    
    //! Write the current AUI perspective to the configuration
    void Saveperspective();

    //! Load the saved AUI perspective from the configuration
    void Loadperspective();

    //! Restore the default AUI perspective.
    void Restoreperspective();
    
    //! Indicates whether all selected traces should be plotted. 
    /*! \return true if they should be plotted, false otherwise.
     */
    bool PlotSelected() const {return pPlotSelected->IsChecked();}

    void ActivateGraph();

private:
    wxAuiManager m_mgr;
    wxAuiNotebook* m_notebook;
    long m_notebook_style;
    wxPanel* m_traceCounter, *m_channelCounter;
    wxStaticText *pSelected, *pSize, *pTraceIndex;
    wxComboBox *pTraces, *pActChannel, *pInactChannel;
    wxStfGrid* m_table;
    wxCheckBox* pPlotSelected;
    wxFlexGridSizer *pTraceSizer, *pChannelSizer;
    wxFlexGridSizer *pTraceNumberSizer;
    bool firstResize;

    wxAuiNotebook* CreateNotebook();
    wxPanel* CreateTraceCounter();
    wxPanel* CreateChannelCounter();
    wxStfGrid* CreateTable();
    void OnComboTraces(wxCommandEvent& event);
    void OnComboActChannel(wxCommandEvent& event);
    void OnComboInactChannel(wxCommandEvent& event);
    void OnMenuHighlight(wxMenuEvent& event);
    void OnPlotselected(wxCommandEvent& event);
    DECLARE_EVENT_TABLE()
};

#if wxUSE_DRAG_AND_DROP
class wxStfFileDrop : public wxFileDropTarget {
protected:
    virtual bool OnDropFiles(wxCoord x, wxCoord y, const wxArrayString& filenames);
};
#endif

//! Provides the top-level frame.
/*! It is part of the of the document/view framework implemented in wxWidgets. 
 *  This class can only be used for MDI parent frames.
 */
class wxStfParentFrame : public wxDocMDIParentFrame {
    DECLARE_CLASS(wxStfParentFrame)
public:
    //! Constructor
    /*! \param manager Pointer to the document manager.
     *  \param frame Pointer to the parent frame (should be NULL, because this is 
     *         the top-level frame
     *  \param title Title of this frame.
     *  \param pos Initial position of this frame.
     *  \param size Initial size of this frame. 
     *  \param style Window style.
     */
    wxStfParentFrame(wxDocManager *manager, wxFrame *frame, const wxString& title, const wxPoint& pos, const wxSize& size,
            long style);
    
    //! Destructor
    ~wxStfParentFrame();

    //! Shows the "About" dialog.
    /*! \param event The menu event that made the call.
     */
    void OnAbout(wxCommandEvent& event);
    
    
    //! Creates a new graph.
    /*! Called from view.cpp when a new drawing view is created.
     *  \param view Pointer to the attached view.
     *  \param parent Pointer to the child frame that will serve as a parent for the new graph.
     *  \return A pointer to the newly created graph.
     */
    wxStfGraph *CreateGraph(wxView *view, wxStfChildFrame *parent);
    
    //! Retrieve the current mouse mode.
    /*! \return The current mouse cursor mode.
     */
    stf::cursor_type GetMouseQual() const;

    //! Sets the current mouse cursor mode.
    /*! \param value The new mouse cursor mode.
     */
    void SetMouseQual(stf::cursor_type value);

    //! Retrieve which channels will be affected by scaling operations
    /*! \return The channels affected by scaling operations.
     */
    stf::zoom_channels GetZoomQual() const;

    //! Set the channels that will be affected by scaling operations
    /*! \param value The channels affected by scaling operations.
     */
    void SetZoomQual(stf::zoom_channels value);

    //! Set the zoom buttons to single- or multi-channel mode. 
    /*! \param value Set to true for single- or false for multi-channel mode.
     */
    void SetSingleChannel(bool value);

    //! Retrieves the print data
    /*! \return Pointer to the stored print data.
     */
    wxPrintData* GetPrintData() { return m_printData.get(); }

    //! Retrieves the page setup
    /*! \return Pointer to the page setup data.
     */
    wxPageSetupDialogData* GetPageSetup() { return m_pageSetupData.get(); }

private:
    wxAuiManager m_mgr;
    wxToolBar *m_cursorToolBar, *m_scaleToolBar;
    wxStfFileDrop* m_drop;
    wxString python_code2;

    // print data, to remember settings during the session
    boost::shared_ptr<wxPrintData> m_printData;

    // page setup data
    boost::shared_ptr<wxPageSetupDialogData> m_pageSetupData;
    bool firstResize;

    wxToolBar* CreateStdTb();
    wxToolBar* CreateScaleTb();
    wxToolBar* CreateEditTb();
    wxToolBar* CreateCursorTb();

    void RedirectStdio();
    wxWindow* DoPythonStuff(wxWindow* parent);
    
    void OnToolFirst(wxCommandEvent& event);
    void OnToolNext(wxCommandEvent& event);
    void OnToolPrevious(wxCommandEvent& event);
    void OnToolLast(wxCommandEvent& event);
    void OnToolXenl(wxCommandEvent& event); 
    void OnToolXshrink(wxCommandEvent& event); 
    void OnToolYenl(wxCommandEvent& event); 
    void OnToolYshrink(wxCommandEvent& event); 
    void OnToolUp(wxCommandEvent& event); 
    void OnToolDown(wxCommandEvent& event); 
    void OnToolFit(wxCommandEvent& event); 
    void OnToolLeft(wxCommandEvent& event); 
    void OnToolRight(wxCommandEvent& event); 
    void OnToolCh1(wxCommandEvent& event); 
    void OnToolCh2(wxCommandEvent& event); 
    void OnToolSnapshot(wxCommandEvent& event);

#ifdef _WINDOWS
    void OnToolSnapshotwmf(wxCommandEvent& event);
#endif

//    void OnSwapChannels(wxCommandEvent& event); 
    void OnCh2base(wxCommandEvent& event); 
    void OnCh2pos(wxCommandEvent& event); 
    void OnCh2zoom(wxCommandEvent& event);
    void OnCh2basezoom(wxCommandEvent& event);
    void OnAverage(wxCommandEvent& event); 
    void OnAlignedAverage(wxCommandEvent& event); 
    void OnExportfile(wxCommandEvent& event);
    void OnExportatf(wxCommandEvent& event);
    void OnExportigor(wxCommandEvent& event);
    void OnExportimage(wxCommandEvent& event);
    void OnExportps(wxCommandEvent& event);
    void OnExportlatex(wxCommandEvent& event);
#if wxCHECK_VERSION(2, 9, 0)
    void OnExportsvg(wxCommandEvent& event);
#endif
    void OnConvert(wxCommandEvent& event);
    void OnUserdef(wxCommandEvent& event);
    void OnScale(wxCommandEvent& event);
    void OnHires(wxCommandEvent& event);
    void OnPrint(wxCommandEvent& event);
    void OnPrintPreview(wxCommandEvent& event);
    void OnPageSetup(wxCommandEvent& event);
    void OnViewResults(wxCommandEvent& event);
    void OnSaveperspective(wxCommandEvent& event); 
    void OnLoadperspective(wxCommandEvent& event); 
    void OnRestoreperspective(wxCommandEvent& event); 
    void OnViewshell(wxCommandEvent& event); 
    void OnLStartMaxslope(wxCommandEvent& event); 
    void OnLStartHalfrise(wxCommandEvent& event); 
    void OnLStartPeak(wxCommandEvent& event); 
    void OnLStartManual(wxCommandEvent& event); 
    void OnLEndFoot(wxCommandEvent& event); 
    void OnLEndMaxslope(wxCommandEvent& event); 
    void OnLEndHalfrise(wxCommandEvent& event); 
    void OnLEndPeak(wxCommandEvent& event); 
    void OnLEndManual(wxCommandEvent& event); 
    void OnLWindow(wxCommandEvent& event); 
    DECLARE_EVENT_TABLE()
};

//! App information Dialog
class wxStfAppAboutDialog : public wxDialog 
{
private:
    wxStdDialogButtonSizer* m_sdbSizer;
    wxTextCtrl* m_textCtrl1;
    wxStaticBitmap* m_bitmap1;

public:
    //! Constructor
    /*! \param parent Pointer to parent window.
     *  \param id Window id.
     *  \param title Dialog title.
     *  \param pos Initial position.
     *  \param size Initial size.
     *  \param style Window style.
     */
    wxStfAppAboutDialog(
            wxWindow* parent,
            int id = wxID_ANY,
            wxString title = wxT("About Stimfit"),
            wxPoint pos = wxDefaultPosition,
            wxSize size = wxDefaultSize,
            int style = wxCAPTION
    );

};

/*@}*/

#endif

