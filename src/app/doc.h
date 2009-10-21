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

/*! \file doc.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares wxStfDoc.
 */

#ifndef _DOC_H
#define _DOC_H

/*! \addtogroup wxstf
 *  @{
 */

#include "wx/app.h"
#include "wx/cmdproc.h"

#include "./../core/core.h"
#include "./../core/recording.h"

//! The document class, derived from both wxDocument and Recording.
/*! The document class can be used to model an application’s file-based data.
 *  It is part of the document/view framework supported by wxWidgets.
 */
class StfDll wxStfDoc: public wxDocument, public Recording
{
#ifndef FROM_PYTHON
    DECLARE_DYNAMIC_CLASS(wxStfDoc)
#endif
private:
    bool peakAtEnd, initialized, progress;
    Recording Average;
    int InitCursors();
    void PostInit();
    bool ChannelSelDlg();
    void WriteToReg();
    bool outOfRange(std::size_t check) {
        return (check>=cur().size() || check<0);
    }
    void Focus();
    void OnNewfromselectedThisMenu( wxCommandEvent& event ) { OnNewfromselectedThis( ); }
    void Selectsome(wxCommandEvent& event);
    void Unselectsome(wxCommandEvent& event);
    void Concatenate(wxCommandEvent& event);
    void OnAnalysisBatch( wxCommandEvent& event );
    void OnAnalysisIntegrate( wxCommandEvent& event );
    void OnAnalysisDifferentiate( wxCommandEvent& event );
    void OnSwapChannels( wxCommandEvent& event );
    void Multiply(wxCommandEvent& event);
    void SubtractBaseMenu( wxCommandEvent& event ) { SubtractBase( ); }
    void LFit(wxCommandEvent& event);
    void LnTransform(wxCommandEvent& event);
    void Filter(wxCommandEvent& event);
    void Spectrum(wxCommandEvent& event);
    void P_over_N(wxCommandEvent& event);
    void Plotcriterion(wxCommandEvent& event);
    void Plotcorrelation(wxCommandEvent& event);
    void MarkEvents(wxCommandEvent& event);
    void Threshold(wxCommandEvent& event);
    void Viewtable(wxCommandEvent& event);
    void Fileinfo(wxCommandEvent& event);
    Recording ReorderChannels();

    wxMenu* doc_file_menu;

protected:

public:

    //! Constructor.
    /*! Does nothing but initialising the member list.
     */
    wxStfDoc();
    //! Destructor.
    ~wxStfDoc();

    //! Override default file opening.
    /*! Attempts to identify the file type from the filter extension (such as "*.dat")
     *  \param filename Full path of the file.
     *  \return true if successfully opened, false otherwise.
     */
    virtual bool OnOpenDocument(const wxString& filename);

    //! Open document without progress dialog.
    /*! Attempts to identify the file type from the filter extension (such as "*.dat")
     *  \param filename Full path of the file.
     *  \return true if successfully opened, false otherwise.
     */
    virtual bool OnOpenPyDocument(const wxString& filename);

    //! Override default file saving.
    /*! \return true if successfully saved, false otherwise.
     */
    virtual bool SaveAs();

    //! Override default file saving.
    /*! \param filename Full path of the file.
     *  \return true if successfully saved, false otherwise.
     */
    virtual bool DoSaveDocument(const wxString& filename);

    //! Override default file closing.
    /*! Writes settings to the config file or registry before closing.
     *  \return true if successfully closed, false otherwise.
     */
    virtual bool OnCloseDocument();

    //! Override default file creation.
    /*! \return true if successfully closed, false otherwise.
     */
    virtual bool OnNewDocument();

    //! Sets the content of a newly created file.
    /*! \param c_Data The data that is used for the new file.
     *  \param Sender Pointer to the document that generated this file.
     *  \param title Title of the new document.
     */
    void SetData( const Recording& c_Data, const wxStfDoc* Sender, const wxString& title );

    //! Indicates whether an average has been created.
    /*! \return true if an average has been created, false otherwise.
     */
    bool GetIsAverage() const { return !Average.get().empty(); }

    //! Indicates whether the right peak cursor should always be at the end of a trace.
    /*! \return true if the right peak cursor should be at the end, false otherwise.
     */
    bool GetPeakAtEnd() const { return peakAtEnd; }

    //! Indicates whether the the document is fully initialised.
    /*! The document has to be fully initialized before other parts of the
     *  program start accessing it; for example, the graph might start reading out values
     *  before they exist.
     *  \return true if the document is fully initialised, false otherwise.
     */
    bool IsInitialized() const { return initialized; }

    //! Sets the right peak cursor to the end of a trace.
    /*! \param value determines whether the peak cursor should be at the end of a trace.
     */
    void SetPeakAtEnd(bool value) { peakAtEnd=value; }

    //! Retrieves the average trace(s).
    /*! \return The average trace as a Recording object.
     */
    const Recording& GetAverage() const { return Average; }

    //! Checks whether any cursor is reversed or out of range and corrects it if required.
    void CheckBoundaries();

    //! Updates the check marks in the latency mode menu
    void UpdateMenuCheckmarks();

    //! Sets the current section to the specified value
    /*! Checks for out-of-range errors
     *  \param section The 0-based index of the new section
     */
    bool SetSection(std::size_t section);

    //! Creates a new window containing the selected sections of this file.
    /*! \return true upon success, false otherwise.
     */
    bool OnNewfromselectedThis( );

    //! Selects all sections
    /*! \param event The menu event that made the call.
     */
    void Selectall(wxCommandEvent& event);

    //! Unselects all sections
    /*! \param event The menu event that made the call.
     */
    void Deleteselected(wxCommandEvent& event);

    //! Updates the status of the selection button
    void UpdateSelectedButton();

    //! Creates an average trace from the selected sections
    /*! \param calcSD Set to true if the standard deviation should be calculated as well, false otherwise
     *  \param align Set to true if traces should be aligned to the point of steepest rise of the inactive channel,
     *         false otherwise.
     */
    void CreateAverage( bool calcSD, bool align );

    //! Applies a user-defined function to the current data set
    /*! \param id The id of the user-defined function
     */
    void Userdef(std::size_t id);

    //! Toggles the selection status of the current section
    void ToggleSelect( );

    //! Selects the current section if previously unselected
    void Select();

    //! Unselects the current section if previously selected
    void Remove();

    //! Creates a new document from the checked events
    /*! \param event The menu event that made the call.
     */
    void Extract(wxCommandEvent& event);

    //! Erases all events, independent of whether they are checked or not
    /*! \param event The menu event that made the call.
     */
    void EraseEvents(wxCommandEvent& event);

    //! Adds an event at the current eventPos
    /*! \param event The menu event that made the call.
     */
    void AddEvent( wxCommandEvent& event );

    //! Subtracts the baseline of all selected traces.
    /*! \return true upon success, false otherwise.
     */
    bool SubtractBase( );

    //! Fit a function to the data.
    /*! \param event The menu event that made the call.
     */
    void FitDecay(wxCommandEvent& event);

    //! Sets a pointer to the file menu attached to this document.
    /*! \param menu The menu to be attached.
     */
    void SetFileMenu( wxMenu* menu ) { doc_file_menu = menu; }
    DECLARE_EVENT_TABLE()
};

/*@}*/

#endif

