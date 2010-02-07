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

/*! \file app.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares wxStfApp.
 */

#ifndef _APP_H
#define _APP_H

/*! \defgroup wxstf Stimfit classes and functions derived from wxWidgets
 *  @{
 */

//! Event ids
enum {
    wxID_TOOL_FIRST,
    wxID_TOOL_NEXT,
    wxID_TOOL_PREVIOUS,
    wxID_TOOL_LAST,
    wxID_TOOL_XENL,
    wxID_TOOL_XSHRINK,
    wxID_TOOL_YENL,
    wxID_TOOL_YSHRINK,
    wxID_TOOL_UP,
    wxID_TOOL_DOWN,
    wxID_TOOL_FIT,
    wxID_TOOL_LEFT,
    wxID_TOOL_RIGHT,
    wxID_TOOL_SELECT,
    wxID_TOOL_REMOVE,
    wxID_TOOL_MEASURE,
    wxID_TOOL_PEAK,
    wxID_TOOL_BASE,
    wxID_TOOL_DECAY,
    wxID_TOOL_LATENCY,
    wxID_TOOL_ZOOM,
    wxID_TOOL_EVENT,
    wxID_TOOL_CH1,
    wxID_TOOL_CH2,
    wxID_TOOL_SNAPSHOT,
#ifdef _WINDOWS
    wxID_TOOL_SNAPSHOT_WMF,
#endif
#ifdef WITH_PYTHON
    wxID_IMPORTPYTHON,
#endif
    wxID_VIEW_RESULTS,
    wxID_VIEW_MEASURE,
    wxID_VIEW_BASELINE,
    wxID_VIEW_BASESD,
    wxID_VIEW_THRESHOLD,
    wxID_VIEW_PEAKZERO,
    wxID_VIEW_PEAKBASE,
    wxID_VIEW_PEAKTHRESHOLD,
    wxID_VIEW_RT2080,
    wxID_VIEW_T50,
    wxID_VIEW_RD,
    wxID_VIEW_SLOPERISE,
    wxID_VIEW_SLOPEDECAY,
    wxID_VIEW_LATENCY,
    wxID_VIEW_CURSORS,
    wxID_VIEW_SHELL,
    wxID_FILEINFO,
    wxID_EXPORTIMAGE,
    wxID_EXPORTPS,
    wxID_EXPORTLATEX,
    wxID_EXPORTSVG,
    wxID_TRACES,
    wxID_PLOTSELECTED,
    wxID_SHOWSECOND,
    wxID_CURSORS,
    wxID_AVERAGE,
    wxID_ALIGNEDAVERAGE,
    wxID_FIT,
    wxID_LFIT,
    wxID_LOG,
    wxID_VIEWTABLE,
    wxID_BATCH,
    wxID_INTEGRATE,
    wxID_DIFFERENTIATE,
    wxID_CH2BASE,
    wxID_CH2POS,
    wxID_CH2ZOOM,
    wxID_CH2BASEZOOM,
    wxID_SWAPCHANNELS,
    wxID_SCALE,
    wxID_HIRES,
    wxID_ZOOMHV,
    wxID_ZOOMH,
    wxID_ZOOMV,
    wxID_EVENTADD,
    wxID_EVENTEXTRACT,
    wxID_APPLYTOALL,
    wxID_UPDATE,
    ID_CONVERT,
    wxID_LATENCYSTART_MAXSLOPE,
    wxID_LATENCYSTART_HALFRISE,
    wxID_LATENCYSTART_PEAK,
    wxID_LATENCYSTART_MANUAL,
    wxID_LATENCYEND_FOOT,
    wxID_LATENCYEND_MAXSLOPE,
    wxID_LATENCYEND_HALFRISE,
    wxID_LATENCYEND_PEAK,
    wxID_LATENCYEND_MANUAL,
    wxID_LATENCYWINDOW,
    WXPRINT_PRINT,
    WXPRINT_PAGE_SETUP,
    WXPRINT_PREVIEW,
    wxID_COPYINTABLE,
    wxID_MULTIPLY,
    wxID_SELECTSOME,
    wxID_UNSELECTSOME,
    wxID_MYSELECTALL,
    wxID_UNSELECTALL,
    wxID_NEWFROMSELECTED,
    wxID_NEWFROMSELECTEDTHIS,
    wxID_NEWFROMALL,
    wxID_CONCATENATE,
    wxID_SUBTRACTBASE,
    wxID_FILTER,
    wxID_SPECTRUM,
    wxID_POVERN,
    wxID_PLOTCRITERION,
    wxID_PLOTCORRELATION,
    wxID_EXTRACT,
    wxID_THRESHOLD,
    wxID_LOADPERSPECTIVE,
    wxID_SAVEPERSPECTIVE,
    wxID_RESTOREPERSPECTIVE,
    wxID_STFCHECKBOX,
    wxID_EVENT_ADDEVENT,
    wxID_EVENT_EXTRACT,
    wxID_EVENT_ERASE,
    wxCOMBOTRACES,
    wxCOMBOACTCHANNEL,
    wxCOMBOINACTCHANNEL
};

#include "wx/mdi.h"
#include "wx/docview.h"
#include "wx/docmdi.h"
#include "wx/config.h"
#include "wx/settings.h"

#ifdef __WXMAC__
#undef wxFontDialog
#include "wx/osx/fontdlg.h"
#endif

#include "./../core/stimdefs.h"
#ifdef WITH_PYTHON
    #include <Python.h>
    #include <wx/wxPython/wxPython.h>
#endif
#include <boost/shared_ptr.hpp>

class wxDocManager;
class wxStfDoc;
class wxStfView;
class wxStfCursorsDlg;
class wxStfParentFrame;
class wxStfChildFrame;
class Section;

//! The application, derived from wxApp
/*! This class is used to set and get application-wide properties,
 *  implement the windowing system message or event loop,
 *  initiate application processing via OnInit, and
 *  allow default processing of events not handled by other objects in the application.
 */
class StfDll wxStfApp: public wxApp
{
public:
    //! Constructor
    wxStfApp();

    //! Initialise the application
    /*! Initialises the document manager and the file-independent menu items,
     *  loads the user-defined plugin library and the least-squares function library,
     *  parses the command line and attempts to open a file if one was given
     *  at program startup by either double-clicking it or as a command-line
     *  argument.
     *  \return true upon successful initialisation, false otherwise.
     */
    virtual bool OnInit();

    //! Exit the application
    /*! Does nothing but calling the base class's wxApp::OnExit().
     *  \return The return value of wxApp::OnExit().
     */
    virtual int OnExit();

    //! Creates a new child frame
    /*! This is called from view.cpp whenever a child frame is created. If you
     *  want to pop up a new frame showing a new document, use NewChild() instead; this
     *  function will then be called by the newly created view.
     *  \param doc A pointer to the document that the new child frame should contain.
     *  \param view A pointer to the view corresponding to the document.
     *  \return A pointer to the newly created child frame.
     */
    wxStfChildFrame *CreateChildFrame(wxDocument *doc, wxView *view);

    //! Retrieves the currently active document.
    /*! \return A pointer to the currently active document.
     */
    wxStfDoc* GetActiveDoc() const;

    //! Retrieves the currently active view.
    /*! \return A pointer to the currently active view.
     */
    wxStfView* GetActiveView() const;

    //! Displays a message box when an error has occured.
    /*! You can use this function from almost anywhere using
     *  wxGetApp().ErrorMsg( wxT( "Error abc: xyz" ) );
     *  \param msg The message string to be shown.
     */
    void ErrorMsg(const wxString& msg) const {
        wxMessageBox(msg,wxT("An error has occured"),wxOK | wxICON_EXCLAMATION,NULL);
    }

    //! Displays a message box when an exception has occured.
    /*! You can use this function from almost anywhere using
     *  wxGetApp().ExceptMsg( wxT( "Exception description xyz" ) );
     *  \param msg The message string to be shown.
     */
    void ExceptMsg(const wxString& msg) const {
        wxMessageBox(msg,wxT("An exception was caught"),wxOK | wxICON_HAND,NULL);
    }

    //! Displays a message box with information.
    /*! You can use this function from almost anywhere using
     *  wxGetApp().InfoMsg( wxT( "Info xyz" ) );
     *  \param msg The message string to be shown.
     */
    void InfoMsg(const wxString& msg) const {
        wxMessageBox(msg,wxT("Information"), wxOK | wxICON_INFORMATION, NULL);
    }

    //! Indicates whether text files should be imported directly without showing an import settings dialog.
    /*! \return true if text files should be imported directly, false otherwise.
     */
    bool get_directTxtImport() const { return directTxtImport; }

    //! Determines whether text files should be imported directly without showing an import filter settings dialog.
    /*! \param directTxtImport_ Set to true if text files should be imported directly, false otherwise.
     */
    void set_directTxtImport(bool directTxtImport_) {
        directTxtImport=directTxtImport_;
    }

    //! Retrieves the text import filter settings.
    /*! \return A struct with the current text import filter settings.
     */
    const stf::txtImportSettings& GetTxtImport() const {
        return txtImport;
    }

    //! Sets the text import filter settings.
    /*! \param txtImport_ A struct with the new text import filter settings.
     */
    void set_txtImportSettings(const stf::txtImportSettings& txtImport_) {
        txtImport=txtImport_;
    }

    //! Retrieves the functions that are available for least-squares minimisation.
    /*! \return A vector containing the available functions.
     */
    const std::vector<stf::storedFunc>& GetFuncLib() const { return funcLib; }


    //! Retrieves a pointer to a function for least-squares minimisation.
    /*! \return A vector containing the available functions.
     */
    stf::storedFunc* GetFuncLibPtr(std::size_t at) { return &funcLib.at(at); }


    //! Retrieves a pointer to a function for least-squares minimisation.
    /*! \return A vector containing the available functions.
     */
    stf::storedFunc* GetLinFuncPtr( ) { return &storedLinFunc; }

	//! Retrieves the user-defined plugin functions.
    /*! \return A vector containing the user-defined functions.
     */
    const std::vector< stf::Plugin >& GetPluginLib() const { return pluginLib; }

    //! Retrieves the cursor settings dialog.
    /*! \return A pointer to the cursor settings dialog.
     */
    wxStfCursorsDlg* GetCursorsDialog() const { return CursorsDialog; }

    //! Retrieves all sections with fits
    /*! \return A vector containing pointers to all sections in which fits have been performed
     */
    std::vector<Section*> GetSectionsWithFits() const;

    //! Writes an integer value to the configuration.
    /*! \param main The main path within the configuration.
     *  \param sub The sub-path within the configuration.
     *  \param value The integer to write to the configuration.
     */
    void wxWriteProfileInt(const wxString& main,const wxString& sub, int value) const;

    //! Retrieves an integer value from the configuration.
    /*! \param main The main path within the configuration.
     *  \param sub The sub-path within the configuration.
     *  \param default_ The default integer to return if the configuration entry can't be read.
     *  \return The integer that is stored in /main/sub, or default_ if the entry couldn't
     *  be read.
     */
    int wxGetProfileInt(const wxString& main,const wxString& sub, int default_) const;

    //! Writes a string to the configuration.
    /*! \param main The main path within the configuration.
     *  \param sub The sub-path within the configuration.
     *  \param value The string to write to the configuration.
     */
    void wxWriteProfileString(
            const wxString& main, const wxString& sub, const wxString& value ) const;

    //! Retrieves a string from the configuration.
    /*! \param main The main path within the configuration.
     *  \param sub The sub-path within the configuration.
     *  \param default_ The default string to return if the configuration entry can't be read.
     *  \return The string that is stored in /main/sub, or default_ if the entry couldn't
     *  be read.
     */
    wxString wxGetProfileString(
            const wxString& main, const wxString& sub, const wxString& default_ ) const;

    //! Creates a new child window showing a new document.
    /*! \param NewData The new data to be shown in the new window.
     *  \param Sender The document that was at the origin of this new window.
     *  \param title A title for the new document.
     *  \return A pointer to the newly created document.
     */
    wxStfDoc* NewChild(
            const Recording& NewData,
            const wxStfDoc* Sender,
            const wxString& title = wxT("\0")
    );

    //! Execute all pending calculations.
    /*! Whenever settings that have an effect on measurements, such as
     *  cursor positions or trace selections, are modified, this function
     *  needs to be called to update the results table.
     */
    void OnPeakcalcexecMsg(wxStfDoc* actDoc = 0);

    //! Destroys the last cursor settings dialog when the last document is closed
    /*! Do not use this function directly. It only needs to be called from wxStfDoc::OnCloseDocument().
     */
    void OnCloseDocument();

    //! Closes all documents
    bool CloseAll() { return GetDocManager()->CloseDocuments(); }

    //! Opens a series of files. Optionally, files can be put into a single window.
    /*! \param fNameArray An array of file names to be opened.
     *  \return true upon successful opening of all files, false otherwise.
     */
    bool OpenFileSeries(const wxArrayString& fNameArray);

    //! Returns the number of currently opened documents.
    /*! \return The number of currently opened documents.
     */
    int GetDocCount() { return (int)GetDocManager()->GetDocuments().GetCount(); }

    //! Determine whether scale bars or coordinates should be shown.
    /*! \param value Set to true for scale bars, false for coordinates.
     */
    void set_isBars(bool value) { isBars=value; }

    //! Indicates whether scale bars or coordinates are shown.
    /*! \return true for scale bars, false for coordinates.
     */
    bool get_isBars() const { return isBars; }

    //! Determine whether a high or a low resolution should be used for drawing traces.
    /*! Will attempt to draw at most 50,000 points per trace.
     *  \param value Set to true for high resolution, false for low resolution.
     */
    void set_isHires(bool value) { isHires=value; }

    //! Indicates whether a high or a low resolution is used for drawing traces.
    /*! \return true for high resolution, false for low resolution.
     */
    bool get_isHires() const { return isHires; }

    //! Get a formatted version string.
    /*! \return A version string (stimfit x.y.z, release/debug build, date).
     */
    wxString GetVersionString() const;

    //! Open a new window showing all selected traces from all open files
    /*! \param event The associated menu event
     */
    void OnNewfromselected( wxCommandEvent& event );

    //! Access the document manager
    /*! \return A pointer to the document manager.
     */
    wxDocManager* GetDocManager() const { return wxDocManager::GetDocumentManager(); }
    
    virtual void OnInitCmdLine(wxCmdLineParser& parser);
    virtual bool OnCmdLineParsed(wxCmdLineParser& parser);

#ifdef WITH_PYTHON
    //! Opens a file in a new window, to be called from Python.
    /*! \param fNameArray An array of file names to be opened.
     *  \return true upon successful opening, false otherwise.
     */
    bool OpenFilePy(const wxString& fNameArray);
    
    //! Opens a dialog to import a Python module
    /*! \param event The associated menu event
     */
    void OnPythonImport( wxCommandEvent& event);
#endif
    
protected:

private:
    void OnCursorSettings( wxCommandEvent& event );
    void OnNewfromall( wxCommandEvent& event );
    void OnApplytoall( wxCommandEvent& event );
    void OnProcessCustom( wxCommandEvent& event );

#ifdef WITH_PYTHON
    void ImportPython(const wxString& modulelocation);
    bool Init_wxPython();
    bool Exit_wxPython();
#endif // WITH_PYTHON

#ifdef _WINDOWS
#pragma optimize( "", off )
#endif

#ifdef _WINDOWS
#pragma optimize( "", on )
#endif

    bool directTxtImport,isBars,isHires;
    stf::txtImportSettings txtImport;
    // Registry:
    boost::shared_ptr<wxConfig> config;
    std::vector<stf::storedFunc> funcLib;
    std::vector< stf::Plugin > pluginLib;
    // Pointer to the peak calculation dialog box
    wxStfCursorsDlg* CursorsDialog;
    wxDocTemplate* m_cfsTemplate, *m_hdf5Template, *m_txtTemplate,*m_abfTemplate,*m_atfTemplate,*m_axgTemplate,*m_sonTemplate;
    stf::storedFunc storedLinFunc;
    wxMenu* m_file_menu;
    wxString m_fileToLoad;
    
#ifdef WITH_PYTHON
    PyThreadState* m_mainTState;
#endif

    DECLARE_EVENT_TABLE()
};

#ifdef _WINDOWS
//! Returns a reference to the application.
extern StfDll wxStfApp& wxGetApp();
#else
DECLARE_APP(wxStfApp)
#endif

//! Retrieve the application's top-level frame
/*! \return A pointer to the top-level frame. */
extern wxStfParentFrame *GetMainFrame();

//! true if in single-window mode
extern bool singleWindowMode;

/*@}*/

#endif

