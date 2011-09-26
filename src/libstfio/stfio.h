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

/*! \file stfio.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2011-09-25
 *  \brief header file for libstfio
 * 
 * 
 *  Header file for libstfio
 */

#ifndef _STFIO_H_
#define _STFIO_H_

#include <boost/function.hpp>
#include <vector>
#include <deque>
#include <map>
#include <string>
#include <cmath>
#include "./zoom.h"

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

typedef std::vector<double > Vector_double;
typedef std::vector<float > Vector_float;

class Recording;
class Channel;
class Section;

//! The stfio namespace.
/*! All essential core functions and classes are in this namespace. 
 *  Its purpose is to reduce name mangling problems.
 */
namespace stfio {

/*! \addtogroup stfio
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


 //! A function taking a double and a vector and returning a double.
/*! Type definition for a function (or, to be precise, any 'callable entity') 
 *  that takes a double (the x-value) and a vector of parameters and returns 
 *  the function's result (the y-value).
 */
typedef boost::function<double(double, const Vector_double&)> Func;

//! The jacobian of a stf::Func.
typedef boost::function<Vector_double(double, const Vector_double&)> Jac;

//! Scaling function for fit parameters
typedef boost::function<double(double, double, double, double, double)> Scale;

//! Dummy function, serves as a placeholder to initialize functions without a Jacobian.
Vector_double nojac( double x, const Vector_double& p);

//! Dummy function, serves as a placeholder to initialize parameters without a scaling function.
double noscale(double param, double xscale, double xoff, double yscale, double yoff);
    
//! Information about parameters used in storedFunc
/*! Contains information about a function's parameters used 
 *  in storedFunc (see below). The client supplies a description 
 *  (desc) and determines whether the parameter is to be 
 *  fitted (toFit==true) or to be kept constant (toFit==false).
 */
struct parInfo {
    //! Default constructor
    parInfo()
    : desc(""),toFit(true), constrained(false), constr_lb(0), constr_ub(0), scale(noscale), unscale(noscale) {}

    //! Constructor
    /*! \param desc_ Parameter description string
     *  \param toFit_ true if this parameter should be fitted, false if
     *         it should be kept fixed. 
     *  \param constrained_ true if this is a constrained fit
     *  \param constr_lb_ lower bound for constrained fit
     *  \param constr_ub_ upper bound for constrained fit
     *  \param scale_ scaling function
     *  \param unscale_ unscaling function
     */
  parInfo( const std::string& desc_, bool toFit_, bool constrained_ = false, 
             double constr_lb_ = 0, double constr_ub_ = 0, Scale scale_ = noscale, Scale unscale_ = noscale)
    : desc(desc_),toFit(toFit_),
        constrained(false), constr_lb(constr_lb_), constr_ub(constr_ub_),
        scale(scale_), unscale(unscale_)
    {}

    std::string desc; /*!< Parameter description string */
    bool toFit;    /*!< true if this parameter should be fitted, false if it should be kept fixed. */
    bool constrained; /*!< true if this parameter should be constrained */
    double constr_lb; /*!< Lower boundary for box-constrained fits */
    double constr_ub; /*!< Upper boundary for box-constrained fits */
    Scale scale; /*!< Scaling function for this parameter */
    Scale unscale; /*!< Unscaling function for this parameter */
};

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

//! Initialising function for the parameters in stf::Func to start a fit.
typedef boost::function<void(const Vector_double&,double,double,double,Vector_double&)> Init;

//! Print the output of a fit into a stf::Table.
typedef boost::function<Table(const Vector_double&,const std::vector<stfio::parInfo>,double)> Output;

 
//! Solves a linear equation system using LAPACK.
/*! Uses column-major order for matrices. For an example, see
 *  Section::SetIsIntegrated()
 *  \param m Number of rows of the matrix \e A.
 *  \param n Number of columns of the matrix \e A.
 *  \param nrhs Number of columns of the matrix \e B.
 *  \param A On entry, the left-hand-side matrix. On exit, 
 *         the factors L and U from the factorization
 *         A = P*L*U; the unit diagonal elements of L are not stored. 
 *  \param B On entry, the right-hand-side matrix. On exit, the
 *           solution to the linear equation system.
 *  \return At present, always returns 0.
 */
int
linsolv(
        int m,
        int n,
        int nrhs,
        Vector_double& A,
        Vector_double& B
);

//! Default fit output function, constructing a stf::Table from the parameters, their description and chisqr.
Table defaultOutput(const Vector_double& pars, 
                    const std::vector<parInfo>& parsInfo,
                    double chisqr);

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
    storedFunc( const std::string& name_, const std::vector<parInfo>& pInfo_,
            const Func& func_, const Init& init_, const Jac& jac_, bool hasJac_ = true,
            const Output& output_ = defaultOutput /*,
            bool hasId_ = true*/
    ) : name(name_),pInfo(pInfo_),func(func_),init(init_),jac(jac_),hasJac(hasJac_),output(output_) /*, hasId(hasId_)*/
    {
/*        if (hasId) {
            id = NextId();
            std::string new_name;
            new_name << id << ": " << name;
            name = new_name;
        } else
            id = 0;
*/    }
     
    //! Destructor
    ~storedFunc() { }

//    static int n_funcs;          /*!< Static function counter */
//    int id;                      /*!< Function id; set automatically upon construction, so don't touch. */
    std::string name;            /*!< Function name. */
    std::vector<parInfo> pInfo;  /*!< A vector containing information about the function parameters. */
    Func func;                   /*!< The function that will be fitted to the data. */
    Init init;                   /*!< A function for initialising the parameters. */
    Jac jac;                     /*!< Jacobian of func. */
    bool hasJac;                 /*!< True if the function has an analytic Jacobian. */
    Output output;               /*!< Output of the fit. */
//    bool hasId;                  /*!< Determines whether a function should have an id. */

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

//! ProgressInfo class
/*! Abstract class to be used as an interface for the file io read/write functions
 *  Can be a GUI Dialog or stdout messages
 */
 class ProgressInfo {
 public:
     //! Constructor
     /*! \param title Dialog title
      *  \param message Message displayed
      *  \param maximum Maximum value for the progress meter
      *  \param verbose Whether or not to emit a lot of noise
      */
     ProgressInfo(const std::string& title, const std::string& message, int maximum, bool verbose) {};

     //! Updates the progress info
     /*! \param value New value of the progress meter
      *  \param newmsg New message for the info text
      *  \param skip This is set to true if the user has chosen to skip the operation
      *  \return True unless the operation was cancelled.
      */
     virtual bool Update(int value, const std::string& newmsg="", bool* skip=NULL) = 0;
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

 
//! Text file import filter settings
struct txtImportSettings {
  txtImportSettings() : hLines(1),toSection(true),firstIsTime(true),ncolumns(2),
        sr(20),yUnits("mV"),yUnitsCh2("pA"),xUnits("ms") {}

    int hLines;            /*!< Number of header lines. */
    bool toSection;        /*!< Import columns into separate sections rather than separate channels. */
    bool firstIsTime;      /*!< First column contains time. */
    int ncolumns;          /*!< Number of columns. */
    double sr;             /*!< Sampling rate. */
    std::string yUnits;    /*!< y units string. */
    std::string yUnitsCh2; /*!< y units string of second channel. */
    std::string xUnits;    /*!< x units string. */
};

//! File types
enum filetype {
    atf,    /*!< Axon text file. */
    abf,    /*!< Axon binary file. */
    axg,    /*!< Axograph binary file. */
    ascii,  /*!< Generic text file. */
    cfs,    /*!< CED filing system. */
    igor,   /*!< Igor binary wave. */
    son,    /*!< CED Son files. */
    hdf5,   /*!< hdf5 files. */
    heka,   /*!< heka files. */
#ifdef WITH_BIOSIG
    biosig, /*!< biosig files. */
#endif
    none    /*!< Undefined file type. */
};

  
//! Attempts to determine the filetype from the filter extension.
/*! \param ext The filter extension to be tested (in the form wxT("*.ext")).
 *  \return The corresponding file type.
 */
stfio::filetype
findType(const std::string& ext);

//! Generic file import.
/*! \param fName The full path name of the file. 
 *  \param type The file type. 
 *  \param ReturnData Will contain the file data on return.
 *  \param txtImport The text import filter settings.
 *  \param progress Set to true if a progress dialog should be shown.
 *  \return true if the file has successfully been read, false otherwise.
 */
bool
importFile(
        const std::string& fName,
        stfio::filetype type,
        Recording& ReturnData,
        const stfio::txtImportSettings& txtImport,
        stfio::ProgressInfo& progDlg
);

//! Generic file export.
/*! \param fName The full path name of the file. 
 *  \param type The file type. 
 *  \param Data Data to be written
 *  \param progress Set to true if a progress dialog should be shown.
 *  \return true if the file has successfully been written, false otherwise.
 */
bool
exportFile(const std::string& fName, stfio::filetype type, const Recording& Data,
           ProgressInfo& progDlg);

/*@}*/

} // end of namespace

typedef std::vector< stfio::Event      >::iterator       event_it;    /*!< stfio::Event iterator */
typedef std::vector< stfio::Event      >::const_iterator c_event_it;  /*!< constant stfio::Event iterator */
typedef std::vector< stfio::PyMarker   >::iterator       marker_it;   /*!< stfio::PyMarker iterator */
typedef std::vector< stfio::PyMarker   >::const_iterator c_marker_it; /*!< constant stfio::PyMarker iterator */
typedef std::vector< std::string        >::iterator       sst_it;      /*!< std::string iterator */
typedef std::vector< std::string        >::const_iterator c_sst_it;    /*!< constant std::string iterator */
typedef std::vector< stfio::storedFunc >::const_iterator c_stfunc_it; /*!< constant stfio::storedFunc iterator */
typedef std::vector< std::size_t     >::const_iterator c_st_it;     /*!< constant size_t iterator */
typedef std::vector< int             >::iterator       int_it;      /*!< int iterator */
typedef std::vector< int             >::const_iterator c_int_it;    /*!< constant int iterator */
typedef std::vector< Channel         >::iterator       ch_it;       /*!< Channel iterator */
typedef std::vector< Channel         >::const_iterator c_ch_it;     /*!< constant Channel iterator */
typedef std::vector< Section         >::iterator       sec_it;      /*!< Section iterator */
typedef std::vector< Section         >::const_iterator c_sec_it;    /*!< constant Section iterator */

#endif
