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

/*! \file core.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Core components.
 */

#ifndef _CORE_H
#define _CORE_H

#ifdef _WINDOWS
#pragma warning( disable : 4251 )  // Disable warning messages
#endif

#include <vector>
#include <valarray>
#include <complex>
#include <deque>
#ifdef _OPENMP
#include <omp.h>
#endif
// header for the fourier transform:
#ifndef TEST_MINIMAL
#include "fftw3.h"
#endif

#include "./stimdefs.h"
#include "./spline.h"

#include "./section.h"
#include "./recording.h"

namespace stf {

/*! \addtogroup stfgen
 *  @{
 */

//! Computes a spectral estimate using Welch's method. 
/*! \param data An input valarray of complex numbers.
 *  \param K \e data will be split into \e K windows.
 *  \param f_n On return, this contains the frequency step between adjacent
 *         indices in the spectrum, in units of 1/index_data.
 *  \return A valarray containing the spectrum.
 */
std::valarray<double>
spectrum(const std::valarray<std::complex<double> >& data,int K,double& f_n);

//! Window function for psd estimation
/*! \param n Argument of the window function.
 *  \param N Width of the window.
 *  \return Result of the window function.
 */
double window(double n, double N);

//! Calculates the square of a number.
/*! \param a Argument of the function.
 *  \return \e a ^2
 */
template <typename T>
T SQR (T a);

//! Convolutes a data set with a filter function.
/*! \param toFilter The valarray to be filtered.
 *  \param filter_start The index from which to start filtering.
 *  \param filter_end The index at which to stop filtering.
 *  \param a A valarray of parameters for the filter function.
 *  \param SR The sampling rate.
 *  \param func The filter function in the frequency domain.
 *  \param inverse true if (1- \e func) should be used as the filter function, false otherwise
 *  \return The convoluted data set.
 */
std::valarray<double>
filter(
        const std::valarray<double>& toFilter,
        std::size_t filter_start,
        std::size_t filter_end,  
        const std::valarray<double> &a,
        int SR,
        Func func,
        bool inverse = false
);

//! Interpolates a dataset using cubic splines.
/*! \param y The valarray to be interpolated.
 *  \param oldF The original sampling frequency.
 *  \param newF The new frequency of the interpolated array.
 *  \return The interpolated data set.
 */
template <class T>
std::valarray<T>
cubicSpline(
        const std::valarray<T>& y,
        T oldF,
        T newF
);

//! Converts a Section to a wxString.
/*! \param section The Section to be written to a string.
 *  \return A string containing the x- and y-values of the section in two columns.
 */
wxString sectionToString(const Section& section);

//! Strips the directory off a full path name, returns only the filename.
/*! \param fName The full path of a file.
 *  \return The file name without the directory.
 */
wxString noPath(const wxString& fName);

//! Attempts to determine the filetype from the filter extension.
/*! \param ext The filter extension to be tested (in the form wxT("*.ext")).
 *  \return The corresponding file type.
 */
stf::filetype
findType(const wxString& ext);

//! Generic file import.
/*! \param fName The full path name of the file. 
 *  \param type The file type. 
 *  \param ReturnData Will contain the file data on return.
 *  \param txtImport The text import filter settings.
 *  \return true if the file has successfully been read, false otherwise.
 */
bool
importFile(
        const wxString& fName,
        stf::filetype type,
        Recording& ReturnData,
        const stf::txtImportSettings& txtImport
);

//! Differentiate data.
/* \param input The valarray to be differentiated.
 * \param x_scale The sampling interval.
 * \return The result of the differentiation.
 */
template <class T>
std::valarray<T> diff(const std::valarray<T>& input, T x_scale);

//! Integration using Simpson's rule.
/*! \param input The valarray to be integrated.
 *  \param a Start of the integration interval.
 *  \param b End of the integration interval.
 *  \param x_scale Sampling interval.
 *  \return The integral of \e input between \e a and \e b.
*/
double integrate_simpson(
        const std::valarray<double>& input,
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
        const std::valarray<double>& input,
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
        std::valarray<double>& A,
        std::valarray<double>& B
);

//! Computes the event detection criterion according to Clements & Bekkers (1997).
/*! \param data The valarray from which to extract events.
 *  \param templ A template waveform that is used for event detection.
 *  \return The detection criterion for every value of \e data.
 */
std::valarray<double>
detectionCriterion(
        const std::valarray<double>& data,
        const std::valarray<double>& templ
);

// TODO: Add negative-going peaks.
//! Searches for positive-going peaks.
/*! \param data The valarray to be searched for peaks.
 *  \param threshold Minimal amplitude of a peak.
 *  \param minDistance Minimal distance between subsequent peaks.
 *  \return A vector of indices where peaks have occurred in \e data.
 */
std::vector<int> peakIndices(const std::valarray<double>& data, double threshold, int minDistance);

//! Computes the linear correlation between two arrays.
/*! \param va1 First array.
 *  \param va2 Second array.
 *  \return The linear correlation between the two arrays for each data point of \e va1.
 */
std::valarray<double> linCorr(const std::valarray<double>& va1, const std::valarray<double>& va2); 

//! Computes the sum of an arbitrary number of Gaussians.
/*! \f[
 *      f(x) = \sum_{i=0}^{n-1}p_{3i}\mathrm{e}^{- \left( \frac{x-p_{3i+1}}{p_{3i+2}} \right) ^2}
 *  \f] 
 *  \param x Argument of the function.
 *  \param p A valarray of function parameters of size 3\e n, where \n
 *         \e p[3<em>i</em>] is the amplitude of the Gaussian \n
 *         \e p[3<em>i</em>+1] is the position of the center of the peak, \n
 *         \e p[3<em>i</em>+2] is the width of the Gaussian, \n
 *         \e n is the number of Gaussian functions and \n
 *         \e i is the 0-based index of the i-th Gaussian.
 *  \return The evaluated function.
 */
double fgauss(double x, const std::valarray<double>& p);

//! Computes a Gaussian that can be used as a filter kernel.
/*! \f[
 *      f(x) = \mathrm{e}^{-0.3466 \left( \frac{x}{p_{0}} \right) ^2}   
 *  \f]
 *  \param x Argument of the function.
 *  \param p Function parameters, where \n
 *         \e p[0] is the corner frequency (-3 dB according to Colquhoun)
 *  \return The evaluated function.
 */
double fgaussColqu(double x, const std::valarray<double>& p);

//! Computes a Boltzmann function.
/*! \f[f(x)=\frac{1}{1+\mathrm{e}^{\frac{p_0-x}{p_1}}}\f] 
 *  \param x Argument of the function.
 *  \param p Function parameters, where \n
 *         \e p[0] is the midpoint and \n
 *         \e p[1] is the slope of the function. \n
 *  \return The evaluated function.
 */
double fboltz(double x, const std::valarray<double>& p);

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
double fbessel4(double x, const std::valarray<double>& p);

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

/*@}*/

}

inline double stf::window(double n, double N) {
    return 1.0-(pow((2.0*n-N)/N,2.0));
}

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
std::valarray<T>
stf::cubicSpline(const std::valarray<T>& y,
        T oldF,
        T newF)
{
    double factor_i=newF/oldF;
    int size=(int)y.size();
    // size of interpolated data:
    int size_i=(int)(size*factor_i);
    std::valarray<double> x(size);
    std::valarray<double> y_d(size);
    for (int n_p=0; n_p < size; ++n_p) {
        x[n_p]=n_p;
        y_d[n_p]=y[n_p];
    }
    std::valarray<double> y_i(stf::spline_cubic_set(x,y_d,0,0,0,0));

    std::valarray<T> y_if(size_i);
    std::valarray<double> x_i(size_i);

    //Cubic spline interpolation:
    for (int n_i=0; n_i < size_i; ++n_i) {
        x_i[n_i]=(double)n_i * (double)size/(double)size_i;
        double yp, ypp;
        y_if[n_i]=(T)stf::spline_cubic_val(x,x_i[n_i],y_d,y_i,yp,ypp);
    }
    return y_if;
}

template <class T>
std::valarray<T> stf::diff(const std::valarray<T>& input, T x_scale) {
    std::valarray<T> diffVA(input.size()-1);
    for (unsigned n=0;n<diffVA.size();++n) {
        diffVA[n]=(input[n+1]-input[n])/x_scale;
    }
    return diffVA;
}

template <typename T>
inline T stf::SQR(T a) {return a*a;}

#endif
