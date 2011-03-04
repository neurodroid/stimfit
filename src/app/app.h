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
    ID_TOOL_FIRST, // = wxID_HIGHEST+1, resulted in wrong events being fired
    ID_TOOL_NEXT,
    ID_TOOL_PREVIOUS,
    ID_TOOL_LAST,
    ID_TOOL_XENL,
    ID_TOOL_XSHRINK,
    ID_TOOL_YENL,
    ID_TOOL_YSHRINK,
    ID_TOOL_UP,
    ID_TOOL_DOWN,
    ID_TOOL_FIT,
    ID_TOOL_LEFT,
    ID_TOOL_RIGHT,
    ID_TOOL_SELECT,
    ID_TOOL_REMOVE,
    ID_TOOL_MEASURE,
    ID_TOOL_PEAK,
    ID_TOOL_BASE,
    ID_TOOL_DECAY,
    ID_TOOL_LATENCY,
#ifdef WITH_PSLOPE
    ID_TOOL_PSLOPE,
#endif
    ID_TOOL_ZOOM,
    ID_TOOL_EVENT,
    ID_TOOL_CH1,
    ID_TOOL_CH2,
    ID_TOOL_SNAPSHOT,
#ifdef _WINDOWS
    ID_TOOL_SNAPSHOT_WMF,
#endif
#ifdef WITH_PYTHON
    ID_IMPORTPYTHON,
#endif
    ID_VIEW_RESULTS,
    ID_VIEW_MEASURE,
    ID_VIEW_BASELINE,
    ID_VIEW_BASESD,
    ID_VIEW_THRESHOLD,
    ID_VIEW_PEAKZERO,
    ID_VIEW_PEAKBASE,
    ID_VIEW_PEAKTHRESHOLD,
    ID_VIEW_RT2080,
    ID_VIEW_T50,
    ID_VIEW_RD,
    ID_VIEW_SLOPERISE,
    ID_VIEW_SLOPEDECAY,
    ID_VIEW_LATENCY,
#ifdef WITH_PSLOPE
    ID_VIEW_PSLOPE,
#endif
    ID_VIEW_CURSORS,
    ID_VIEW_SHELL,
    ID_FILEINFO,
    ID_EXPORTIMAGE,
    ID_EXPORTPS,
    ID_EXPORTLATEX,
    ID_EXPORTSVG,
    ID_TRACES,
    ID_PLOTSELECTED,
    ID_SHOWSECOND,
    ID_CURSORS,
    ID_AVERAGE,
    ID_ALIGNEDAVERAGE,
    ID_FIT,
    ID_LFIT,
    ID_LOG,
    ID_VIEWTABLE,
    ID_BATCH,
    ID_INTEGRATE,
    ID_DIFFERENTIATE,
    ID_CH2BASE,
    ID_CH2POS,
    ID_CH2ZOOM,
    ID_CH2BASEZOOM,
    ID_SWAPCHANNELS,
    ID_SCALE,
    ID_HIRES,
    ID_ZOOMHV,
    ID_ZOOMH,
    ID_ZOOMV,
    ID_EVENTADD,
    ID_EVENTEXTRACT,
    ID_APPLYTOALL,
    ID_UPDATE,
    ID_CONVERT,
#if 0
    ID_LATENCYSTART_MAXSLOPE,
    ID_LATENCYSTART_HALFRISE,
    ID_LATENCYSTART_PEAK,
    ID_LATENCYSTART_MANUAL,
    ID_LATENCYEND_FOOT,
    ID_LATENCYEND_MAXSLOPE,
    ID_LATENCYEND_HALFRISE,
    ID_LATENCYEND_PEAK,
    ID_LATENCYEND_MANUAL,
#endif
    ID_LATENCYWINDOW,
    ID_PRINT_PRINT,
    ID_MPL,
    ID_PRINT_PAGE_SETUP,
    ID_PRINT_PREVIEW,
    ID_COPYINTABLE,
    ID_MULTIPLY,
    ID_SELECTSOME,
    ID_UNSELECTSOME,
    ID_MYSELECTALL,
    ID_UNSELECTALL,
    ID_NEWFROMSELECTED,
    ID_NEWFROMSELECTEDTHIS,
    ID_NEWFROMALL,
    ID_CONCATENATE,
    ID_SUBTRACTBASE,
    ID_FILTER,
    ID_SPECTRUM,
    ID_POVERN,
    ID_PLOTCRITERION,
    ID_PLOTCORRELATION,
    ID_EXTRACT,
    ID_THRESHOLD,
    ID_LOADPERSPECTIVE,
    ID_SAVEPERSPECTIVE,
    ID_RESTOREPERSPECTIVE,
    ID_STFCHECKBOX,
    ID_EVENT_ADDEVENT,
    ID_EVENT_EXTRACT,
    ID_EVENT_ERASE,
    ID_COMBOTRACES,
    ID_SPINCTRLTRACES,
    ID_ZERO_INDEX,
    ID_COMBOACTCHANNEL,
    ID_COMBOINACTCHANNEL,
    ID_USERDEF
};

#include <list>

#include "wx/mdi.h"
#include "wx/docview.h"
#include "wx/docmdi.h"
#include "wx/fileconf.h"
#include "wx/settings.h"

#ifdef __WXMAC__
#undef wxFontDialog
#include "wx/osx/fontdlg.h"
#endif

#include "./../core/stimdefs.h"

#ifdef WITH_PYTHON

#ifdef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE_WAS_DEF
#undef _POSIX_C_SOURCE
#endif
#ifdef _XOPEN_SOURCE
#define _XOPEN_SOURCE_WAS_DEF
#undef _XOPEN_SOURCE
#endif
#include <Python.h>
#ifdef _POSIX_C_SOURCE_WAS_DEF
  #ifndef _POSIX_C_SOURCE
    #define _POSIX_C_SOURCE
  #endif
#endif
#ifdef _XOPEN_SOURCE_WAS_DEF
  #ifndef _XOPEN_SOURCE
    #define _XOPEN_SOURCE
  #endif
#endif

#if defined(__WXMAC__) || defined(__WXGTK__)
  #pragma GCC diagnostic ignored "-Wwrite-strings"
#endif
#include <wx/wxPython/wxPython.h>
// revert to previous behaviour
#if defined(__WXMAC__) || defined(__WXGTK__)
  #pragma GCC diagnostic warning "-Wwrite-strings"
#endif

#endif // WITH_PYTHON

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
     *  loads the user-defined extension library and the least-squares function library,
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

    //! Sets the currently active document.
    /*! \param pDoc A pointer to the currently active document.
     */
    /*void SetActiveDoc(wxStfDoc* pDoc);*/

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

    //! Retrieves the user-defined extension functions.
    /*! \return A vector containing the user-defined functions.
     */
    const std::vector< stf::Extension >& GetExtensionLib() const { return extensionLib; }

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
     *  \param pDoc Pointer to the document that is being closed.
     */
    void CleanupDocument(wxStfDoc* pDoc);

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
    void OnPythonImport( wxCommandEvent& event );
#endif
    
protected:

private:
    void OnCursorSettings( wxCommandEvent& event );
    void OnNewfromall( wxCommandEvent& event );
    void OnApplytoall( wxCommandEvent& event );
    void OnProcessCustom( wxCommandEvent& event );
    void OnUserdef(wxCommandEvent& event);
    void OnKeyDown( wxKeyEvent& event );
    
#ifdef WITH_PYTHON
    void ImportPython(const wxString& modulelocation);
    bool Init_wxPython();
    bool Exit_wxPython();
    std::vector<stf::Extension> LoadExtensions();
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
    boost::shared_ptr<wxFileConfig> config;
    std::vector<stf::storedFunc> funcLib;
    std::vector< stf::Extension > extensionLib;
    // Pointer to the cursors settings dialog box
    wxStfCursorsDlg* CursorsDialog;
    wxDocTemplate* m_cfsTemplate, *m_hdf5Template, *m_txtTemplate,*m_abfTemplate,
      *m_atfTemplate,*m_axgTemplate,*m_sonTemplate, *m_hekaTemplate, *m_biosigTemplate;
    stf::storedFunc storedLinFunc;
    // wxMenu* m_file_menu;
    wxString m_fileToLoad;
    /*std::list<wxStfDoc *> activeDoc;*/
    
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

