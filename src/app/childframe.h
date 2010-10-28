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

/*! \file childframe.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares wxStfChildFrame.
 */

#ifndef _CHILDFRAME_H
#define _CHILDFRAME_H

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

//! child frame type; depends on whether aui is used for the doc/view interface
#ifdef WITH_AUIDOCVIEW
typedef wxAuiDocMDIChildFrame wxStfChildType;
#else
typedef wxDocMDIChildFrame wxStfChildType;
#endif

//! parent frame type; depends on whether aui is used for the doc/view interface
#ifdef WITH_AUIDOCVIEW
typedef wxAuiDocMDIParentFrame wxStfParentType;
#else
typedef wxDocMDIParentFrame wxStfParentType;
#endif

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

#if wxUSE_DRAG_AND_DROP
class wxStfFileDrop : public wxFileDropTarget {
protected:
    virtual bool OnDropFiles(wxCoord x, wxCoord y, const wxArrayString& filenames);
};
#endif

//! Provides the child frame for displaying documents on separate windows.
/*! This class can only be used for MDI child frames. It is part of the document/view
 *  framework supported by wxWidgets.
 */
class StfDll wxStfChildFrame : public wxStfChildType
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
            wxStfParentType* parent,
            wxWindowID id,
            const wxString& title,
            const wxPoint& pos = wxPoint(48,48),
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
    /*! \param channelNames The channel names for the combo box drop-down list.
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

    //! Indicates whether the second channel should be plotted.
    /*! \return true if it should be plotted, false otherwise.
     */
    bool ShowSecond() const {return pShowSecond->IsChecked();}

    //! Activated the current graph
    void ActivateGraph();

private:
    wxAuiManager m_mgr;
    wxAuiNotebook* m_notebook;
    long m_notebook_style;
    wxPanel *m_traceCounter;
    wxPanel *m_channelCounter;
    wxStaticText *pSelected, *pSize, *pTraceIndex;
    //wxComboBox *pTraces, *pActChannel, *pInactChannel;
    wxComboBox *pActChannel, *pInactChannel;
    wxSpinCtrl *trace_spinctrl;
    wxStfGrid* m_table;
    wxCheckBox *pPlotSelected, *pShowSecond;
    wxFlexGridSizer *pTraceSizer, *pChannelSizer;
    wxFlexGridSizer *pTraceNumberSizer;
    bool firstResize;

    wxAuiNotebook* CreateNotebook();
    wxPanel* CreateTraceCounter();
    wxPanel* CreateChannelCounter();
    wxStfGrid* CreateTable();
    void OnMenuHighlight(wxMenuEvent& event);
    void OnPlotselected(wxCommandEvent& event);
    void OnShowsecond(wxCommandEvent& event);
//  void OnComboTraces(wxCommandEvent& event);
    void OnSpinCtrlTraces(wxSpinEvent& event);
//    void OnSpinCtrlTracesText(wxCommandEvent& event);
    void OnComboActChannel(wxCommandEvent& event);
    void OnComboInactChannel(wxCommandEvent& event);

    DECLARE_EVENT_TABLE()
};

/*@}*/

#endif

