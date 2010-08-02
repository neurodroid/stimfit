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

/*! \file parentframe.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares wxStfParentFrame.
 */

#ifndef _PARENTFRAME_H
#define _PARENTFRAME_H

/*! \addtogroup wxstf
 *  @{
 */

#include "wx/aui/aui.h"
#include "wx/grid.h"
#include "wx/dnd.h"

#include "./../core/core.h"

class wxStfGraph;
class wxStfTable;
class wxStfGrid;
class wxStfFileDrop;
class wxProgressDialog;
    
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


#ifdef _WINDOWS
typedef wxAuiToolBar wxStfToolBar;
#else
typedef wxAuiToolBar wxStfToolBar;
#endif

//! Provides the top-level frame.
/*! It is part of the of the document/view framework implemented in wxWidgets.
 *  This class can only be used for MDI parent frames.
 */
class wxStfParentFrame : public wxStfParentType {
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

    //! Sets status of the toolbar's selection button.
    /*! \param selected The desired toggle status of the selection button.
     */
    void SetSelectedButton(bool selected);

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

    //! Retrieve the wxAuiManager.
    /*! \return A reference to the wxAuiManager.
     */
    wxAuiManager& GetMgr() { return m_mgr; }
    
    //! Checks for updates.
    /*! \param progDlg An optional progress dialog
     */
    void CheckUpdate( wxProgressDialog* progDlg=NULL ) const;
    
private:
    wxAuiManager m_mgr;
    wxStfToolBar *m_cursorToolBar, *m_scaleToolBar;
    wxStfFileDrop* m_drop;
#ifdef WITH_PYTHON
    wxString python_code2; // python import code
    void RedirectStdio();
    wxWindow* DoPythonStuff(wxWindow* parent);
#endif
    // print data, to remember settings during the session
    boost::shared_ptr<wxPrintData> m_printData;

    // page setup data
    boost::shared_ptr<wxPageSetupDialogData> m_pageSetupData;
    bool firstResize;

    wxStfToolBar* CreateStdTb();
    wxStfToolBar* CreateScaleTb();
    wxStfToolBar* CreateEditTb();
    wxStfToolBar* CreateCursorTb();

    void OnHelp(wxCommandEvent& event);
    void OnCheckUpdate(wxCommandEvent& event);
    
    void OnToggleSelect(wxCommandEvent& event);
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
    void OnToolMeasure(wxCommandEvent& event);
    void OnToolPeak(wxCommandEvent& event);
    void OnToolBase(wxCommandEvent& event);
    void OnToolDecay(wxCommandEvent& event);
    void OnToolLatency(wxCommandEvent& event);
    void OnToolZoom(wxCommandEvent& event);
    void OnToolEvent(wxCommandEvent& event);

#ifdef WITH_PSLOPE
    void OnToolPSlope(wxCommandEvent& event);
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
    void OnExporthdf5(wxCommandEvent& event);
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
#ifdef WITH_PYTHON
    void OnViewshell(wxCommandEvent& event);
#endif
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

/*@}*/

#endif

