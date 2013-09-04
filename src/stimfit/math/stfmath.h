// Header file for the stimfit namespace
// General-purpose routines
// last revision: 08-08-2006
// C. Schmidt-Hieber, christsc@gmx.de

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

/*! \file stfmath.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Math functions.
 */

#ifndef _CORE_H
#define _CORE_H

#ifdef _WINDOWS
#pragma warning( disable : 4251 )  // Disable warning messages
#endif

#include <vector>
#include <complex>
#include <deque>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#define INFINITY (DBL_MAX+DBL_MAX)
#define NAN (INFINITY-INFINITY)
#endif

#include "./../stf.h"

// header for the fourier transform:
#include "fftw3.h"
#include "./spline.h"

namespace stf {

/*! \addtogroup stfgen
 *  @{
 */

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

//! Print the output of a fit into a stf::Table.
typedef boost::function<Table(const Vector_double&,const std::vector<stf::parInfo>,double)> Output;
 
//! Default fit output function, constructing a stf::Table from the parameters, their description and chisqr.
Table defaultOutput(const Vector_double& pars, 
                    const std::vector<parInfo>& parsInfo,
                    double chisqr);

//! Initialising function for the parameters in stf::Func to start a fit.
typedef boost::function<void(const Vector_double&, double, double, double, double, double, Vector_double&)> Init;

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

//! Calculates the square of a number.
/*! \param a Argument of the function.
 *  \return \e a ^2
 */
template <typename T>
T SQR (T a);

//! Convolves a data set with a filter function.
/*! \param toFilter The valarray to be filtered.
 *  \param filter_start The index from which to start filtering.
 *  \param filter_end The index at which to stop filtering.
 *  \param a A valarray of parameters for the filter function.
 *  \param SR The sampling rate.
 *  \param func The filter function in the frequency domain.
 *  \param inverse true if (1- \e func) should be used as the filter function, false otherwise
 *  \return The convolved data set.
 */
Vector_double
filter(
        const Vector_double& toFilter,
        std::size_t filter_start,
        std::size_t filter_end,  
        const Vector_double &a,
        int SR,
        stf::Func func,
        bool inverse = false
);

//! Computes a histogram
/*! \param data The signal
 *  \param nbins Number of bins in the histogram.
 *  \return A map with lower bin limits as keys, number of observations as values.
 */
std::map<double, int>
histogram(const Vector_double& data, int nbins=-1);

//! Deconvolves a template from a signal
/*! \param data The input signal
 *  \param templ The template
 *  \param SR The sampling rate in kHz.
 *  \param hipass Highpass filter cutoff frequency in kHz
 *  \param lopass Lowpass filter cutoff frequency in kHz
 *  \return The result of the deconvolution
 */
StfDll Vector_double
deconvolve(const Vector_double& data, const Vector_double& templ,
           int SR, double hipass, double lopass, stfio::ProgressInfo& progDlg);

//! Interpolates a dataset using cubic splines.
/*! \param y The valarray to be interpolated.
 *  \param oldF The original sampling frequency.
 *  \param newF The new frequency of the interpolated array.
 *  \return The interpolated data set.
 */
template <class T>
std::vector<T>
cubicSpline(
        const std::vector<T>& y,
        T oldF,
        T newF
);

//! Converts a Section to a wxString.
/*! \param section The Section to be written to a string.
 *  \return A string containing the x- and y-values of the section in two columns.
 */
wxString sectionToString(const Section& section);

 //! Differentiate data.
/* \param input The valarray to be differentiated.
 * \param x_scale The sampling interval.
 * \return The result of the differentiation.
 */
template <class T>
std::vector<T> diff(const std::vector<T>& input, T x_scale);

//! Integration using Simpson's rule.
/*! \param input The valarray to be integrated.
 *  \param a Start of the integration interval.
 *  \param b End of the integration interval.
 *  \param x_scale Sampling interval.
 *  \return The integral of \e input between \e a and \e b.
*/
double integrate_simpson(
        const Vector_double& input,
        std::size_t a,
        std::size_t b,
        double x_scale
);

//! Integration using the trapezium rule.
/*! \param input The valarray to be integrated.
 *  \param a Start of the integration interval.
 *  \param b End of the integration interval.
 *  \param x_scale Sampling interval.
 *  \return The integral of \e input between \e a and \e b.
*/
double integrate_trapezium(
        const Vector_double& input,
        std::size_t a,
        std::size_t b,
        double x_scale
);

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

//! Solve quadratic equations for 3 adjacent sampling points
/*! \param data The data vector
 *  \param begin Start of interval to be used
 *  \param end End of interval to be used
 *  \return Parameters of quadratic equation
 */
Vector_double
quad(const Vector_double& data, std::size_t begin, std::size_t end);
 

//! Computes the event detection criterion according to Clements & Bekkers (1997).
/*! \param data The valarray from which to extract events.
 *  \param templ A template waveform that is used for event detection.
 *  \return The detection criterion for every value of \e data.
 */
StfDll Vector_double
detectionCriterion(
        const Vector_double& data,
        const Vector_double& templ,
        stfio::ProgressInfo& progDlg
);

// TODO: Add negative-going peaks.
//! Searches for positive-going peaks.
/*! \param data The valarray to be searched for peaks.
 *  \param threshold Minimal amplitude of a peak.
 *  \param minDistance Minimal distance between subsequent peaks.
 *  \return A vector of indices where peaks have occurred in \e data.
 */
StfDll std::vector<int> peakIndices(const Vector_double& data, double threshold, int minDistance);

//! Computes the linear correlation between two arrays.
/*! \param va1 First array.
 *  \param va2 Second array.
 *  \return The linear correlation between the two arrays for each data point of \e va1.
 */
StfDll Vector_double linCorr(const Vector_double& va1, const Vector_double& va2, stfio::ProgressInfo& progDlg); 

//! Computes a Gaussian that can be used as a filter kernel.
/*! \f[
 *      f(x) = \mathrm{e}^{-0.3466 \left( \frac{x}{p_{0}} \right) ^2}   
 *  \f]
 *  \param x Argument of the function.
 *  \param p Function parameters, where \n
 *         \e p[0] is the corner frequency (-3 dB according to Colquhoun)
 *  \return The evaluated function.
 */
double fgaussColqu(double x, const Vector_double& p);

//! Computes a Boltzmann function.
/*! \f[f(x)=\frac{1}{1+\mathrm{e}^{\frac{p_0-x}{p_1}}}\f] 
 *  \param x Argument of the function.
 *  \param p Function parameters, where \n
 *         \e p[0] is the midpoint and \n
 *         \e p[1] is the slope of the function. \n
 *  \return The evaluated function.
 */
double fboltz(double x, const Vector_double& p);

//! Computes a Bessel polynomial.
/*! \f[
 *     f(x, n) = \sum_{k=0}^n \frac{ \left( 2n - k \right) ! }{ \left( n - k \right) ! k! } \frac{x^k}{ 2^{n-k} }
 *  \f] 
 *  \param x Argument of the function.
 *  \param n Order of the polynomial. \n
 *  \return The evaluated function.
 */
double fbessel(double x, int n);

//! Computes a 4th-order Bessel polynomial that can be used as a filter kernel.
/*! \f[
 *     f(x) = \frac{b(0,4)}{b(\frac{0.355589x}{p_0},4)}
 *  \f] 
 *  where \f$ b(a,n) \f$ is the bessel polynomial stf::fbessel().
 *  \param x Argument of the function.
 *  \param p Function parameters, where \n
 *         \e p[0] is the corner frequency (-3 dB attenuation)
 *  \return The evaluated function.
 */
double fbessel4(double x, const Vector_double& p);

//! Creates a preview of a text file.
/*! \param fName Full path name of the file.
 *  \return A string showing at most the initial 100 lines of the text file.
 */
wxString CreatePreview(const wxString& fName);

//! Computes the faculty of an integer.
/*! \param arg Argument of the function.
 *  \return The faculty of \e arg.
 */
int fac(int arg);

//! Computes \f$ 2^{arg} \f$. Uses the bitwise-shift operator (<<).
/*! \param arg Argument of the function.
 *  \return \f$ 2^{arg} \f$.
 */
int pow2(int arg);

//! Strips the directory off a full path name, returns only the filename.
/*! \param fName The full path of a file.
 *  \return The file name without the directory.
 */
wxString noPath(const wxString& fName);

/*@}*/

}

typedef std::vector< stf::storedFunc >::const_iterator c_stfunc_it; /*!< constant stf::storedFunc iterator */

inline int stf::pow2(int arg) {return 1<<arg;}

//! Swaps \e s1 and \e s2.
/*! \param s1 will be swapped with 
 *  \param s2
 */
template <typename T>
void SWAP(T s1, T s2) {
    T aux=s1;
    s1=s2;
    s2=aux;
}

template <class T>
std::vector<T>
stf::cubicSpline(const std::vector<T>& y,
        T oldF,
        T newF)
{
    double factor_i=newF/oldF;
    int size=(int)y.size();
    // size of interpolated data:
    int size_i=(int)(size*factor_i);
    Vector_double x(size);
    Vector_double y_d(size);
    for (int n_p=0; n_p < size; ++n_p) {
        x[n_p]=n_p;
        y_d[n_p]=y[n_p];
    }
    Vector_double y_i(stf::spline_cubic_set(x,y_d,0,0,0,0));

    std::vector<T> y_if(size_i);
    Vector_double x_i(size_i);

    //Cubic spline interpolation:
    for (int n_i=0; n_i < size_i; ++n_i) {
        x_i[n_i]=(double)n_i * (double)size/(double)size_i;
        double yp, ypp;
        y_if[n_i]=(T)stf::spline_cubic_val(x,x_i[n_i],y_d,y_i,yp,ypp);
    }
    return y_if;
}

template <class T>
std::vector<T> stf::diff(const std::vector<T>& input, T x_scale) {
    std::vector<T> diffVA(input.size()-1);
    for (unsigned n=0;n<diffVA.size();++n) {
        diffVA[n]=(input[n+1]-input[n])/x_scale;
    }
    return diffVA;
}

template <typename T>
inline T stf::SQR(T a) {return a*a;}

#endif
