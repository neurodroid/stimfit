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

#ifndef _STIMDEFS_H_
#define _STIMDEFS_H_

#include <boost/function.hpp>
#include <vector>
#include <deque>
#include <map>
#include <string>

#ifdef _MSC_VER
#pragma warning( disable : 4251 )  // Disable warning messages
#pragma warning( disable : 4996 )  // Disable warning messages
#endif

//! Defines dll export or import functions for Windows
#ifdef _WINDOWS
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
#else
    typedef std::string wxString;
    typedef int wxWindow;
    #define wxT(x)  x
#endif


typedef std::vector<double > Vector_double;
typedef std::vector<float > Vector_float;

class Recording;
class Channel;
class Section;

//! The stimfit namespace.
/*! All essential core functions and classes are in this namespace. 
 *  Its purpose is to reduce name mangling problems.
 */
namespace stf {

/*! \addtogroup stfgen
 *  @{
 */
    Vector_double vec_scal_plus(const Vector_double& vec, double scalar);

    Vector_double vec_scal_minus(const Vector_double& vec, double scalar);

    Vector_double vec_scal_mul(const Vector_double& vec, double scalar);

    Vector_double vec_scal_div(const Vector_double& vec, double scalar);

    Vector_double vec_vec_plus(const Vector_double& vec1, const Vector_double& vec2);

    Vector_double vec_vec_minus(const Vector_double& vec1, const Vector_double& vec2);

    Vector_double vec_vec_mul(const Vector_double& vec1, const Vector_double& vec2);

    Vector_double vec_vec_div(const Vector_double& vec1, const Vector_double& vec2);


#ifndef MODULE_ONLY
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
    Table(const std::map< wxString, double >& map);

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
    void SetRowLabel(std::size_t row,const wxString& label);

    //! Sets the label of a column.
    /*! \param col 0-based column index.
     *  \param label Column label string.
     */
    void SetColLabel(std::size_t col,const wxString& label);

    //! Retrieves the label of a row.
    /*! \param row 0-based row index.
     *  \return Row label string.
     */
    const wxString& GetRowLabel(std::size_t row) const;

    //! Retrieves the label of a column.
    /*! \param col 0-based column index.
     *  \return Column label string.
     */
    const wxString& GetColLabel(std::size_t col) const;

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
    std::vector< wxString > rowLabels;
    std::vector< wxString > colLabels;
};

//! Information about parameters used in storedFunc
/*! Contains information about a function's parameters used 
 *  in storedFunc (see below). The client supplies a description 
 *  (desc) and determines whether the parameter is to be 
 *  fitted (toFit==true) or to be kept constant (toFit==false).
 */
struct parInfo {
    //! Default constructor
    parInfo()
    : desc(wxT("")),toFit(true), constrained(false), constr_lb(0), constr_ub(0) {}

    //! Constructor
    /*! \param desc_ Parameter description string
     *  \param toFit_ true if this parameter should be fitted, false if
     *         it should be kept fixed. 
     *  \param constrained_ true if this is a constrained fit
     *  \param constr_lb_ lower bound for constrained fit
     *  \param constr_ub_ upper bound for constrained fit
     */
    parInfo( const wxString& desc_, bool toFit_, bool constrained_ = false, 
            double constr_lb_ = 0, double constr_ub_ = 0)
    : desc(desc_),toFit(toFit_), constrained(false), constr_lb(constr_lb_), constr_ub(constr_ub_) {}

    wxString desc; /*!< Parameter description string */
    bool toFit;    /*!< true if this parameter should be fitted, false if it should be kept fixed. */
    bool constrained; /*!< true if this parameter should be fitted, false if it should be kept fixed. */
    double constr_lb; /*!< Lower boundary for box-constrained fits */
    double constr_ub; /*!< Upper boundary for box-constrained fits */
};

//! A function taking a double and a vector and returning a double.
/*! Type definition for a function (or, to be precise, any 'callable entity') 
 *  that takes a double (the x-value) and a vector of parameters and returns 
 *  the function's result (the y-value).
 */
typedef boost::function<double(double, const Vector_double&)> Func;

//! The jacobian of a stf::Func.
typedef boost::function<Vector_double(double, const Vector_double&)> Jac;

//! Dummy function, serves as a placeholder to initialize functions without a Jacobian.
Vector_double nojac( double x, const Vector_double& p);

//! Initialising function for the parameters in stf::Func to start a fit.
typedef boost::function<void(const Vector_double&,double,double,double,Vector_double&)> Init;

//! Print the output of a fit into a stf::Table.
typedef boost::function<Table(const Vector_double&,const std::vector<stf::parInfo>,double)> Output;

//! Get a Recording, do something with it, return the new Recording.
typedef boost::function<Recording(const Recording&,const Vector_double&,std::map<wxString, double>&)> PluginFunc;

//! Default fit output function, constructing a stf::Table from the parameters, their description and chisqr.
Table defaultOutput(
        const Vector_double& pars, 
        const std::vector<parInfo>& parsInfo,
        double chisqr
);

//! Function used for least-squares fitting.
/*! Objects of this class are used for fitting functions 
 *  to data. The client supplies a function (func), its 
 *  jacobian (jac), information about the function's parameters 
 *  (pInfo) and a function to initialize the parameters (init).
 */
struct StfDll storedFunc {

    //! Constructor
    /*! \param name_ Plain function name.
     *  \param pInfo_ A vector containing information about the function parameters.
     *  \param func_ The function that will be fitted to the data.
     *  \param jac_ Jacobian of func_.
     *  \param hasJac_ true if a Jacobian is available.
     *  \param init_ A function for initialising the parameters.
     *  \param output_ Output of the fit.
     */
    storedFunc( const wxString& name_, const std::vector<parInfo>& pInfo_,
            const Func& func_, const Init& init_, const Jac& jac_, bool hasJac_ = true,
            const Output& output_ = defaultOutput /*,
            bool hasId_ = true*/
    ) : name(name_),pInfo(pInfo_),func(func_),init(init_),jac(jac_),hasJac(hasJac_),output(output_) /*, hasId(hasId_)*/
    {
/*        if (hasId) {
            id = NextId();
            wxString new_name;
            new_name << id << wxT(": ") << name;
            name = new_name;
        } else
            id = 0;
*/    }
     
    //! Destructor
    ~storedFunc() { }

//    static int n_funcs;          /*!< Static function counter */
//    int id;                      /*!< Function id; set automatically upon construction, so don't touch. */
    wxString name;            /*!< Function name. */
    std::vector<parInfo> pInfo;  /*!< A vector containing information about the function parameters. */
    Func func;                   /*!< The function that will be fitted to the data. */
    Init init;                   /*!< A function for initialising the parameters. */
    Jac jac;                     /*!< Jacobian of func. */
    bool hasJac;                 /*!< True if the function has an analytic Jacobian. */
    Output output;               /*!< Output of the fit. */
//    bool hasId;                  /*!< Determines whether a function should have an id. */

};

//! Represents user input from dialogs that can be used in plugins.
struct UserInput {
    std::vector<wxString> labels; /*!< Dialog entry labels. */
    Vector_double defaults; /*!< Default dialog entries. */
    wxString title;               /*!< Dialog title. */

    //! Constructor.
    /*! \param labels_ A vector of dialog entry label strings.
     *  \param defaults_ A vector of default dialog entries.
     *  \param title_ Dialog title.
     */
    UserInput(
            const std::vector<wxString>& labels_=std::vector<wxString>(0),
            const Vector_double& defaults_=Vector_double(0),
            wxString title_=wxT("\0")
    ) : labels(labels_),defaults(defaults_),title(title_)
    {
                if (defaults.size()!=labels.size()) {
                    defaults.resize(labels.size());
                    std::fill(defaults.begin(), defaults.end(), 0.0);
                }
    }
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

//! The direction of peak calculations
enum direction {
    up,                 /*!< Find positive-going peaks. */
    down,               /*!< Find negative-going peaks. */
    both,               /*!< Find negative- or positive-going peaks, whichever is larger. */
    undefined_direction /*!< Undefined direction. */
};

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

#else
#endif // Module only


//! Text file import filter settings
struct txtImportSettings {
  txtImportSettings() : hLines(1),toSection(true),firstIsTime(true),ncolumns(2),
        sr(20),yUnits(wxT("mV")),yUnitsCh2(wxT("pA")),xUnits(wxT("ms")) {}

    int hLines;            /*!< Number of header lines. */
    bool toSection;        /*!< Import columns into separate sections rather than separate channels. */
    bool firstIsTime;      /*!< First column contains time. */
    int ncolumns;          /*!< Number of columns. */
    double sr;             /*!< Sampling rate. */
    wxString yUnits;    /*!< y units string. */
    wxString yUnitsCh2; /*!< y units string of second channel. */
    wxString xUnits;    /*!< x units string. */
};

//! File types
enum filetype {
    atf,   /*!< Axon text file. */
    abf,   /*!< Axon binary file. */
    axg,   /*!< Axograph binary file. */
    ascii, /*!< Generic text file. */
    cfs,   /*!< CED filing system. */
    igor,  /*!< Igor binary wave. */
    son,   /*!< CED Son files. */
    hdf5,  /*!< hdf5 files. */
    heka,  /*!< heka files. */
    none   /*!< Undefined file type. */
};


//! Add decimals if you are not satisfied.
const double PI=3.14159265358979323846;

//! Does what it says.
/*! \param toRound The double to be rounded.
 *  \return The rounded integer.
 */
int round(double toRound);

/*@}*/

} // end of namespace

#ifndef MODULE_ONLY
typedef std::vector< stf::Event      >::iterator       event_it;    /*!< stf::Event iterator */
typedef std::vector< stf::Event      >::const_iterator c_event_it;  /*!< constant stf::Event iterator */
typedef std::vector< stf::PyMarker   >::iterator       marker_it;   /*!< stf::PyMarker iterator */
typedef std::vector< stf::PyMarker   >::const_iterator c_marker_it; /*!< constant stf::PyMarker iterator */
typedef std::vector< wxString        >::iterator       wxs_it;      /*!< wxString iterator */
typedef std::vector< wxString        >::const_iterator c_wxs_it;    /*!< constant wxString iterator */
typedef std::vector< stf::storedFunc >::const_iterator c_stfunc_it; /*!< constant stf::storedFunc iterator */
#endif

typedef std::vector< std::size_t     >::const_iterator c_st_it;     /*!< constant size_t iterator */
typedef std::vector< int             >::iterator       int_it;      /*!< int iterator */
typedef std::vector< int             >::const_iterator c_int_it;    /*!< constant int iterator */
typedef std::vector< Channel         >::iterator       ch_it;       /*!< Channel iterator */
typedef std::vector< Channel         >::const_iterator c_ch_it;     /*!< constant Channel iterator */
typedef std::vector< Section         >::iterator       sec_it;      /*!< Section iterator */
typedef std::vector< Section         >::const_iterator c_sec_it;    /*!< constant Section iterator */

inline int stf::round(double toRound) {
    return toRound <= 0.0 ? int(toRound-0.5) : int(toRound+0.5);
}

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

/*! \class std::deque
 *  \brief See http://www.sgi.com/tech/stl/Deque.html (SGI's STL documentation)
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

