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

/*! \file stimdefs.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Common definitions and classes.
 * 
 * 
 *  Header file for common definitions and classes. 
 */

#ifndef _STF_H_
#define _STF_H_

#include <boost/function.hpp>
#include <vector>
#include <map>
#include <string>

#include "./gui/zoom.h"

#ifdef _MSC_VER
#pragma warning( disable : 4251 )  // Disable warning messages
#pragma warning( disable : 4996 )  // Disable warning messages
#endif

//! Defines dll export or import functions for Windows
#if defined(_WINDOWS) && !defined(__MINGW32__)
    #ifdef STFDLL
        #define StfDll __declspec( dllexport )
    #else
        #define StfDll __declspec( dllimport )
    #endif
#else
    #define StfDll
#endif

#ifndef MODULE_ONLY
    #include <wx/wxprec.h>
    #ifdef __BORLANDC__
        #pragma hdrstop
    #endif

    #ifndef WX_PRECOMP
        #include <wx/wx.h>
    #endif

    #include <wx/wfstream.h>
    #include <wx/progdlg.h>
#else
    typedef std::string wxString;
    typedef int wxWindow;
    #define wxT(x)  x
    #define wxCHECK_VERSION(major,minor,release) 0
#endif

#include "../libstfio/stfio.h"

//! The stimfit namespace.
/*! All essential core functions and classes are in this namespace. 
 *  Its purpose is to reduce name mangling problems.
 */
namespace stf {

/*! \addtogroup stfgen
 *  @{
 */

//! Progress Info interface adapter; maps to wxProgressDialog
class wxProgressInfo : public stfio::ProgressInfo {
public:
    wxProgressInfo(const std::string& title, const std::string& message, int maximum, bool verbose=true);
    bool Update(int value, const std::string& newmsg="", bool* skip=NULL);
private:
    wxProgressDialog pd;
};

std::string wx2std(const wxString& wxs);
wxString std2wx(const std::string& sst);

//! Get a Recording, do something with it, return the new Recording.
typedef boost::function<Recording(const Recording&,const Vector_double&,std::map<std::string, double>&)> PluginFunc;

//! A table used for printing information.
/*! Members will throw std::out_of_range if out of range.
 */
class StfDll Table {
public:
    //! Constructor
    /*! \param nRows Initial number of rows.
     *  \param nCols Initial number of columns.
     */
    Table(std::size_t nRows,std::size_t nCols);

    //! Constructor
    /*! \param map A map used to initialise the table.
     */
    Table(const std::map< std::string, double >& map);

    //! Range-checked access. Returns a copy. Throws std::out_of_range if out of range.
    /*! \param row 0-based row index.
     *  \param col 0-based column index.
     *  \return A copy of the double at row, col.
     */
    double at(std::size_t row,std::size_t col) const;

    //! Range-checked access. Returns a reference. Throws std::out_of_range if out of range.
    /*! \param row 0-based row index.
     *  \param col 0-based column index.
     *  \return A reference to the double at row, col.
     */
    double& at(std::size_t row,std::size_t col);
    
    //! Check whether a cell is empty.
    /*! \param row 0-based row index.
     *  \param col 0-based column index.
     *  \return true if empty, false otherwise.
     */
    bool IsEmpty(std::size_t row,std::size_t col) const;

    //! Empties or un-empties a cell.
    /*! \param row 0-based row index.
     *  \param col 0-based column index.
     *  \param value true if the cell should be empty, false otherwise.
     */
    void SetEmpty(std::size_t row,std::size_t col,bool value=true);

    //! Sets the label of a row.
    /*! \param row 0-based row index.
     *  \param label Row label string.
     */
    void SetRowLabel(std::size_t row,const std::string& label);

    //! Sets the label of a column.
    /*! \param col 0-based column index.
     *  \param label Column label string.
     */
    void SetColLabel(std::size_t col,const std::string& label);

    //! Retrieves the label of a row.
    /*! \param row 0-based row index.
     *  \return Row label string.
     */
    const std::string& GetRowLabel(std::size_t row) const;

    //! Retrieves the label of a column.
    /*! \param col 0-based column index.
     *  \return Column label string.
     */
    const std::string& GetColLabel(std::size_t col) const;

    //! Retrieves the number of rows.
    /*! \return The number of rows.
     */
    std::size_t nRows() const { return rowLabels.size(); }

    //! Retrieves the number of columns.
    /*! \return The number of columns.
     */
    std::size_t nCols() const { return colLabels.size(); }
    
    //! Appends rows to the table.
    /*! \param nRows The number of rows to be appended.
     */
    void AppendRows(std::size_t nRows);

private:
    // row major order:
    std::vector< std::vector<double> > values;
    std::vector< std::deque< bool > > empty;
    std::vector< std::string > rowLabels;
    std::vector< std::string > colLabels;
};
 
//! Represents user input from dialogs that can be used in plugins.
struct UserInput {
    std::vector<std::string> labels; /*!< Dialog entry labels. */
    Vector_double defaults; /*!< Default dialog entries. */
    std::string title;               /*!< Dialog title. */

    //! Constructor.
    /*! \param labels_ A vector of dialog entry label strings.
     *  \param defaults_ A vector of default dialog entries.
     *  \param title_ Dialog title.
     */
    UserInput(
            const std::vector<std::string>& labels_=std::vector<std::string>(0),
            const Vector_double& defaults_=Vector_double(0),
            std::string title_="\0"
    ) : labels(labels_),defaults(defaults_),title(title_)
    {
                if (defaults.size()!=labels.size()) {
                    defaults.resize(labels.size());
                    std::fill(defaults.begin(), defaults.end(), 0.0);
                }
    }
};

//! User-defined plugin
/*! Class used for extending Stimfit's functionality: 
 *  The client supplies a new menu entry and an ExtFunc 
 *  that will be called upon selection of that entry.
 */
struct Plugin {
    //! Constructor
    /*! \param menuEntry_ Menu entry string for this plugin.
     *  \param pluginFunc_ Function to be executed by this plugin.
     *  \param input_ Dialog entries required by this plugin.
     */
    Plugin(
            const wxString& menuEntry_,
            const PluginFunc& pluginFunc_,
            const UserInput& input_=UserInput()
    ) : menuEntry(menuEntry_),pluginFunc(pluginFunc_),input(input_)
    {
        id = n_plugins;
        n_plugins++;
    }
    
    //! Destructor
    ~Plugin() { }

    int id;                /*!< The plugin id; set automatically upon construction, so don't touch. */
    static int n_plugins;  /*!< Static plugin counter. Initialised in plugins/plugins.cpp. */
    wxString menuEntry;    /*!< Menu entry string for this plugin. */
    PluginFunc pluginFunc; /*!< The function to be executed by this plugin. */
    UserInput input;       /*!< Dialog entries */
};

//! User-defined Python extension
/*! Class used for extending Stimfit's functionality: 
 *  The client supplies a new menu entry and a Python function 
 *  that will be called upon selection of that entry.
 */
struct Extension {
    //! Constructor
    /*! \param menuEntry_ Menu entry string for this extension.
     *  \param pyFunc_ Python function to be called.
     *  \param description_  Description for this function.
     *  \param requiresFile_ Whether a file needs to be open for this function to work
     */
    Extension(const std::string& menuEntry_, void* pyFunc_,
              const std::string& description_, bool requiresFile_) :
        menuEntry(menuEntry_), pyFunc(pyFunc_),
        description(description_), requiresFile(requiresFile_)
    {
        id = n_extensions;
        n_extensions++;
    }
    
    //! Destructor
    ~Extension() { }

    int id;                /*!< The extension id; set automatically upon construction, so don't touch. */
    static int n_extensions;  /*!< Static extension counter. Initialised in extensions/extensions.cpp. */
    std::string menuEntry;    /*!< Menu entry string for this extension. */
    void* pyFunc;     /*!< Python function to be called. */
    std::string description;  /*!< Description for this function. */
    bool requiresFile;     /*!< Whether a file needs to be open for this function to work */
};

//! Resource manager for ifstream objects.
struct ifstreamMan {
    
    //! Constructor
    /*! See fstream documentation for details */
    ifstreamMan( const wxString& filename )
    : myStream( filename, wxT("r") ) 
    {}
    
    //! Destructor
    ~ifstreamMan() { myStream.Close(); }
    
    //! The managed stream.
    wxFFile myStream;
};

//! Resource manager for ofstream objects.
struct ofstreamMan {

    //! Constructor
    /*! See fstream documentation for details */
    ofstreamMan( const wxString& filename )
    : myStream( filename, wxT("w") ) 
    {}
    
    //! Destructor
    ~ofstreamMan() { myStream.Close(); }

    //! The managed stream.
    wxFFile myStream;
};
 
//! Describes the attributes of an event.
class Event {
public:
    //! Constructor
    explicit Event(std::size_t start, std::size_t peak, std::size_t size, bool discard_ = false) : 
        eventStartIndex(start), eventPeakIndex(peak), eventSize(size), discard(discard_) { }
    
    //! Destructor
    ~Event() {}

    //! Retrieves the start index of an event.
    /*! \return The start index of an event within a section. */
    std::size_t GetEventStartIndex() const { return eventStartIndex; }

    //! Retrieves the index of an event's peak.
    /*! \return The index of an event's peak within a section. */
    std::size_t GetEventPeakIndex() const { return eventPeakIndex; }

    //! Retrieves the size of an event.
    /*! \return The size of an event in units of data points. */
    std::size_t GetEventSize() const { return eventSize; }

    //! Indicates whether an event should be discarded.
    /*! \return true if it should be discarded, false otherwise. */
    bool GetDiscard() const { return discard; }

    //! Sets the start index of an event.
    /*! \param value The start index of an event within a section. */
    void SetEventStartIndex( std::size_t value ) { eventStartIndex = value; }

    //! Sets the index of an event's peak.
    /*! \param value The index of an event's peak within a section. */
    void SetEventPeakIndex( std::size_t value ) { eventPeakIndex = value; }

    //! Sets the size of an event.
    /*! \param value The size of an event in units of data points. */
    void SetEventSize( std::size_t value ) { eventSize = value; }

    //! Determines whether an event should be discarded.
    /*! \return true if it should be discarded, false otherwise. */
    void SetDiscard( bool value ) { discard = value; }

    //! Sets discard to true if it was false and vice versa.
    void ToggleStatus() { discard = !discard; }

private:
    std::size_t eventStartIndex;
    std::size_t eventPeakIndex;
    std::size_t eventSize;
    bool discard;
};

//! A marker that can be set from Python
/*! A pair of x,y coordinates
 */
struct PyMarker {
    //! Constructor
    /*! \param xv x-coordinate.
     *  \param yv y-coordinate.
     */
    PyMarker( double xv, double yv ) : x(xv), y(yv) {} 
    double x; /*!< x-coordinate in units of sampling points */
    double y; /*!< y-coordinate in trace units (e.g. mV) */
};

struct storedFunc;
 
struct StfDll SectionAttributes {
    SectionAttributes();
    std::vector<stf::Event> eventList;
    std::vector<stf::PyMarker> pyMarkers;
    bool isFitted,isIntegrated;
    stf::storedFunc *fitFunc;
    Vector_double bestFitP;
    Vector_double quad_p;
    std::size_t storeFitBeg;
    std::size_t storeFitEnd;
    std::size_t storeIntBeg;
    std::size_t storeIntEnd;
    stf::Table bestFit;
};

struct SectionPointer {
    SectionPointer(Section* pSec=NULL, const SectionAttributes& sa=SectionAttributes());
    Section* pSection;
    SectionAttributes sec_attr;
};

//! Add decimals if you are not satisfied.
const double PI=3.14159265358979323846;

//! Does what it says.
/*! \param toRound The double to be rounded.
 *  \return The rounded integer.
 */
int round(double toRound);

//! Mouse cursor types
enum cursor_type {
    measure_cursor,  /*!< Measurement cursor (crosshair). */
    peak_cursor,     /*!< Peak calculation limits cursor. */
    base_cursor,     /*!< Baseline calculation limits cursor. */
    decay_cursor,    /*!< Fit limits cursor. */
    latency_cursor,  /*!< Latency cursor. */
    zoom_cursor,     /*!< Zoom rectangle cursor. */
    event_cursor,    /*!< Event mode cursor. */
#ifdef WITH_PSLOPE
    pslope_cursor,   /*!< PSlope mode cursor. */
#endif
    undefined_cursor /*!< Undefined cursor. */
};

 
 
//! The direction of peak calculations
enum direction {
    up,                 /*!< Find positive-going peaks. */
    down,               /*!< Find negative-going peaks. */
    both,               /*!< Find negative- or positive-going peaks, whichever is larger. */
    undefined_direction /*!< Undefined direction. */
};
 
//! Determines which channels to scale
enum zoom_channels {
    zoomch1, /*!< Scaling applies to channel 1 only. */
    zoomch2, /*!< Scaling applies to channel 2 only. */
    zoomboth /*!< Scaling applies to both channels. */
};

//! Latency cursor settings
enum latency_mode {
    manualMode = 0, /*!< Set the corresponding latency cursor manually (by clicking on the graph). */ 
    peakMode = 1,   /*!< Set the corresponding latency cursor to the peak. */ 
    riseMode = 2,   /*!< Set the corresponding latency cursor to the maximal slope of rise. */ 
    halfMode = 3,   /*!< Set the corresponding latency cursor to the half-maximal amplitude. */ 
    footMode = 4,    /*!< Set the corresponding latency cursor to the beginning of an event. */ 
    undefinedMode   /*!< undefined mode. */
};

//! Latency window settings
enum latency_window_mode {
    defaultMode = 0,  /*!< Use the current peak cursor window for the active channel. */ 
    windowMode = 1  /*!< Use a window of 100 sampling points around the peak. */ 
};

#ifdef WITH_PSLOPE
//! PSlope start cursor settings
enum pslope_mode_beg {
    psBeg_manualMode, /*< Set the start Slope cursor manually. */
    psBeg_footMode,   /*< Set the start Slope cursor to the beginning of an event. */
    psBeg_thrMode,    /*< Set the start Slope cursor to a threshold. */
    psBeg_t50Mode,    /*< Set the start Slope cursor to the half-width of an event*/
    psBeg_undefined
};

//! PSlope end cursor settings
enum pslope_mode_end {
    psEnd_manualMode, /*< Set the end Slope cursor manually. */
    psEnd_t50Mode,   /*< Set the Slope cursor to the half-width of an event. */
    psEnd_DeltaTMode,  /*< Set the Slope cursor to a given distance from the first cursor. */
    psEnd_peakMode,    /*< Set the Slope cursor to the peak. */
    psEnd_undefined
};

#endif // WITH_PSLOPE

//! Deconvolution
enum extraction_mode {
    criterion,                 /*!< Clements & Bekkers criterion. */
    correlation,               /*!< Jonas et al. correlation coefficient. */
    deconvolution               /*!< Pernia-Andrade et al. deconvolution. */
};

/*@}*/

} // end of namespace

inline int stf::round(double toRound) {
    return toRound <= 0.0 ? int(toRound-0.5) : int(toRound+0.5);
}

typedef std::vector< wxString >::iterator       wxs_it;      /*!< std::string iterator */
typedef std::vector< wxString >::const_iterator c_wxs_it;    /*!< constant std::string iterator */
typedef std::vector< stf::Event      >::iterator       event_it;    /*!< stf::Event iterator */
typedef std::vector< stf::Event      >::const_iterator c_event_it;  /*!< constant stf::Event iterator */
typedef std::vector< stf::PyMarker   >::iterator       marker_it;   /*!< stf::PyMarker iterator */
typedef std::vector< stf::PyMarker   >::const_iterator c_marker_it; /*!< constant stf::PyMarker iterator */

// Doxygen-links to documentation of frequently used wxWidgets-classes

/*! \defgroup wxwidgets wxWidgets classes
 *  @{
 */

/*! \class wxApp
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxapp.html (wxWidgets documentation)
 */

/*! \class wxCheckBox
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxcheckbox.html (wxWidgets documentation)
 */

/*! \class wxCommandEvent
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxcommandevent.html (wxWidgets documentation)
 */

/*! \class wxDialog
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxdialog.html (wxWidgets documentation)
 */

/*! \class wxDocMDIChildFrame
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxdocmdichildframe.html (wxWidgets documentation)
 */

/*! \class wxDocMDIParentFrame
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxdocmdiparentframe.html (wxWidgets documentation)
 */

/*! \class wxDocument
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxdocument.html (wxWidgets documentation)
 */

/*! \class wxGrid
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxgrid.html (wxWidgets documentation)
 */

/*! \class wxGridCellCoordsArray
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxgridcellcoordsarray.html (wxWidgets documentation)
 */

/*! \class wxGridTableBase
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxgridtablebase.html (wxWidgets documentation)
 */

/*! \class wxPoint
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxpoint.html (wxWidgets documentation)
 */

/*! \class wxPrintout
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxprintout.html (wxWidgets documentation)
 */

/*! \class wxSize
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxsize.html (wxWidgets documentation)
 */

/*! \class wxString
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxstring.html (wxWidgets documentation)
 */

/*! \class wxThread
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxthread.html (wxWidgets documentation)
 */

/*! \class wxView
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxview.html (wxWidgets documentation)
 */

/*! \class wxWindow
 *  \brief See http://www.wxwidgets.org/manuals/stable/wx_wxwindow.html (wxWidgets documentation)
 */

/*@}*/


/*! \defgroup stdcpp C++ standard library classes
 *  @{
 */

/*! \namespace std
 *  \brief The namespace of the C++ standard library (libstdc++).
 */

/*! \class std::map
 *  \brief See http://www.sgi.com/tech/stl/Map.html (SGI's STL documentation)
 */

/*! \class std::vector
 *  \brief See http://gcc.gnu.org/onlinedocs/libstdc++/latest-doxygen/classstd_1_1valarray.html (gcc's libstdc++ documentation)
 */

/*! \class std::vector
 *  \brief See http://www.sgi.com/tech/stl/Vector.html (SGI's STL documentation)
 */

/*@}*/

#endif

