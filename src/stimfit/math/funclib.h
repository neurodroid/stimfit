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

/*! \file funclib.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-20
 *  \brief User-defined functions for the Levenberg-Marquardt non-linear regression.
 */

#ifndef _FUNCLIB_H
#define _FUNCLIB_H

// common definitions needed here:
#include "./../stf.h"

namespace stf {

/*! \addtogroup stfgen
 *  @{
 */

    //! Sum of \e n exponential functions.
    /*! \f[f(x)=p_{2n} + \sum_{i=0}^{2 n - 1}p_{2i}\mathrm{e}^{\frac{x}{p_{2i + 1}}}\f] 
     *  \param x Function argument.
     *  \param p A valarray of parameters of size 2<em>n</em>+1, where \n
     *         \e n is the number of exponential terms, \n
     *         \e p[2<em>i</em>] is the amplitude term, \n
     *         \e p[2<em>i</em>+1] is the time constant, \n
     *         \e p[2<em>n</em>], the last element, contains the offset and \n
     *         \e i denotes the <em>i</em> -th exponential term (running from 0 to <em>n</em>-1).
     *  \return The evaluated function.
     */
    double fexp(double x, const Vector_double& p);
    
    //! Computes the Jacobian of stf::fexp().
    /*! \f{eqnarray*}
     *   j_{2i}(x) &=& \frac{df(x)}{dp_{2i}} = \mathrm{e}^{\frac{-x}{p_{2i+1}}} \\
     *   j_{2i+1}(x) &=& \frac{df(x)}{dp_{2i+1}} = \frac{p_{2i}}{p_{2i+1}^2} x \mathrm{e}^{\frac{-x}{p_{2i+1}}} \\
     *   j_n(x) &=& \frac{df(x)}{dp_{n}} = 1
     *  \f} 
     *  \param x Function argument.
     *  \param p A valarray of parameters of size 2<em>n</em>+1, where \n
     *         \e n is the number of exponential terms, \n
     *         \e p[2<em>i</em>] is the amplitude term, \n
     *         \e p[2<em>i</em>+1] is the time constant, \n
     *         \e p[2<em>n</em>], the last element, contains the offset and \n
     *         \e i denotes the <em>i</em> -th exponential term (running from 0 to <em>n</em>-1).
     *  \return A valarray \e j of size 2<em>n</em>+1 with the evaluated Jacobian, where \n
     *          \e j[2<em>i</em>] contains the derivative with respect to \e p[2<em>i</em>], \n
     *          \e j[2<em>i</em>+1] contains the derivative with respect to \e p[2<em>i</em>+1] and \n
     *          \e j[2<em>n</em>], the last element, contains the derivative with respect to \e p[2<em>n</em>].
     */
    Vector_double fexp_jac(double x, const Vector_double& p);

    //! Initialises parameters for fitting stf::fexp() to \e data.
    /*! This needs to be made more robust.
     *  \param data The waveform of the data for the fit.
     *  \param base Baseline of \e data.
     *  \param peak Peak value of \e data.
     *  \param dt The sampling interval.
     *  \param pInit On entry, pass a valarray of size 2<em>n</em>+1, where \e n is the
     *         number of exponential functions. On exit, will contain initial parameter
     *         estimates.
     */
    void fexp_init(const Vector_double& data, double base, double peak, double RTLoHi, double HalfWidth, double dt, Vector_double& pInit );

    //! Yet another initialiser for fitting stf::fexp() to \e data.
    /*! In this case, one of the amplitude terms will have another sign than the others, making
     *  it more suitable for fitting PSCs or PSPs. However, this often fails to work in practice.
     *  \param data The waveform of the data for the fit.
     *  \param base Baseline of \e data.
     *  \param peak Peak value of \e data.
     *  \param dt The sampling interval.
     *  \param pInit On entry, pass a valarray of size 2<em>n</em>+1, where \e n is the
     *         number of exponential functions. On exit, will contain initial parameter
     *         estimates.
     */
    void fexp_init2(const Vector_double& data, double base, double peak, double RTLoHi, double HalfWidth, double dt, Vector_double& pInit );
    
    //! Monoexponential function with delay. 
    /*! \f{eqnarray*}
     *      f(x)=
     *      \begin{cases}
     *          p_0, & \mbox{if }x < p_3 \\ 
     *          \left( p_0 - p_2 \right) \mathrm{e}^{\frac{p_3 - x}{p_1}} + p_2, & \mbox{if }x \geq p_3
     *      \end{cases}
     *  \f} 
     *  \param x Function argument.
     *  \param p A valarray of parameters, where \n
     *         \e p[0] is the baseline, \n
     *         \e p[1] is the time constant, \n
     *         \e p[2] is the amplitude and \n
     *         \e p[3] is the delay.
     *  \return The evaluated function.
     */
    double fexpde(double x, const Vector_double& p);

#if 0
    //! Computes the Jacobian of stf::fexpde().
    /*! \f{eqnarray*}
     *      j_0(x)&=& \frac{df(x)}{dp_0} = 
     *      \begin{cases}
     *          1, & \mbox{if }x < p_3 \\ 
     *          \mathrm{e}^{\frac{p_3 - x}{p_1}}, & \mbox{if }x \geq p_3
     *      \end{cases} \\
     *      j_1(x)&=& \frac{df(x)}{dp_1} = 
     *      \begin{cases}
     *          0, & \mbox{if }x < p_3 \\ 
     *          \left( p_0-p_2 \right) \left( p_3-x \right) \frac{-1}{p_1^2} \mathrm{e}^{\frac{p_3 - x}{p_1}}, & \mbox{if }x \geq p_3
     *      \end{cases} \\
     *      j_2(x)&=& \frac{df(x)}{dp_2} = 
     *      \begin{cases}
     *          0, & \mbox{if }x < p_3 \\ 
     *          1 - \mathrm{e}^{\frac{p_3 - x}{p_1}}, & \mbox{if }x \geq p_3
     *      \end{cases} \\
     *      j_3(x)&=& \frac{df(x)}{dp_3} = 
     *      \begin{cases}
     *          0, & \mbox{if }x < p_3 \\ 
     *          \left( p_0-p_2 \right) \frac{1}{p_1} \mathrm{e}^{\frac{p_3 - x}{p_1}}, & \mbox{if }x \geq p_3
     *      \end{cases}
     *  \f} 
     *  \param x Function argument.
     *  \param p A valarray of parameters, where \n
     *         \e p[0] is the baseline, \n
     *         \e p[1] is the time constant, \n
     *         \e p[2] is the amplitude and \n
     *         \e p[3] is the delay.
     *  \return A valarray \e j with the evaluated Jacobian, where \n
     *          \e j[0] contains the derivative with respect to \e p[0], \n
     *          \e j[1] contains the derivative with respect to \e p[1], \n
     *          \e j[2] contains the derivative with respect to \e p[2] and \n
     *          \e j[3] contains the derivative with respect to \e p[3].
     */
    Vector_double fexpde_jac(double x, const Vector_double& p);
#endif
    
    //! Initialises parameters for fitting stf::fexpde() to \e data.
    /*! \param data The waveform of the data for the fit.
     *  \param base Baseline of \e data.
     *  \param peak Peak value of \e data.
     *  \param dt The sampling interval.
     *  \param pInit On entry, pass a valarray of size 4.
     *         On exit, will contain initial parameter estimates.
     */
    void fexpde_init(const Vector_double& data, double base, double peak, double RTLoHi, double HalfWidth, double dt,  Vector_double& pInit );
    
    
    //! Biexponential function with delay. 
    /*! \f{eqnarray*}
     *      f(x)=
     *      \begin{cases}
     *          p_0, & \mbox{if }x < p_1 \\ 
     *          n p_2 \left( \mathrm{e}^{\frac{p_1 - x}{p_2}} - \mathrm{e}^{\frac{p_1 - x}{p_4}} \right) + p_0, & \mbox{if }x \geq p_1
     *      \end{cases}
     *  \f}
     *  \e n is a normalization factor, \n 
     *  \param x Function argument.
     *  \param p A valarray of parameters, where \n
     *         \e p[0] is the baseline, \n
     *         \e p[1] is the delay, \n
     *         \e p[2] is the later (slower) time constant, \n
     *         \e p[3] is the amplitude and \n
     *         \e p[4] is the earlier (faster) time constant, \n
     *  \return The evaluated function.
     */
    double fexpbde(double x, const Vector_double& p);

#if 0
    //! Computes the Jacobian of stf::fexpde().
    /*! \f{eqnarray*}
     *      j_0(x)&=& \frac{df(x)}{dp_0} = 
     *      \begin{cases}
     *          1, & \mbox{if }x < p_3 \\ 
     *          \mathrm{e}^{\frac{p_3 - x}{p_1}}, & \mbox{if }x \geq p_3
     *      \end{cases} \\
     *      j_1(x)&=& \frac{df(x)}{dp_1} = 
     *      \begin{cases}
     *          0, & \mbox{if }x < p_3 \\ 
     *          \left( p_0-p_2 \right) \left( p_3-x \right) \frac{-1}{p_1^2} \mathrm{e}^{\frac{p_3 - x}{p_1}}, & \mbox{if }x \geq p_3
     *      \end{cases} \\
     *      j_2(x)&=& \frac{df(x)}{dp_2} = 
     *      \begin{cases}
     *          0, & \mbox{if }x < p_3 \\ 
     *          1 - \mathrm{e}^{\frac{p_3 - x}{p_1}}, & \mbox{if }x \geq p_3
     *      \end{cases} \\
     *      j_3(x)&=& \frac{df(x)}{dp_3} = 
     *      \begin{cases}
     *          0, & \mbox{if }x < p_3 \\ 
     *          \left( p_0-p_2 \right) \frac{1}{p_1} \mathrm{e}^{\frac{p_3 - x}{p_1}}, & \mbox{if }x \geq p_3
     *      \end{cases}
     *  \f} 
     *  \param x Function argument.
     *  \param p A valarray of parameters, where \n
     *         \e p[0] is the baseline, \n
     *         \e p[1] is the time constant, \n
     *         \e p[2] is the amplitude and \n
     *         \e p[3] is the delay.
     *  \return A valarray \e j with the evaluated Jacobian, where \n
     *          \e j[0] contains the derivative with respect to \e p[0], \n
     *          \e j[1] contains the derivative with respect to \e p[1], \n
     *          \e j[2] contains the derivative with respect to \e p[2] and \n
     *          \e j[3] contains the derivative with respect to \e p[3].
     */
    Vector_double fexpbde_jac(double x, const Vector_double& p);
#endif
    
    //! Initialises parameters for fitting stf::fexpde() to \e data.
    /*! \param data The waveform of the data for the fit.
     *  \param base Baseline of \e data.
     *  \param peak Peak value of \e data.
     *  \param dt The sampling interval.
     *  \param pInit On entry, pass a valarray of size 4.
     *         On exit, will contain initial parameter estimates.
     */
    void fexpbde_init(const Vector_double& data, double base, double peak, double RTLoHi, double HalfWidth, double dt,  Vector_double& pInit );

    //! Alpha function.
    /*! \f[f(x)=p_0 p_1^2 x \mathrm{e}^{-p_1 x} + p_2\f]
     *  \param x Function argument.
     *  \param p A valarray of parameters, where \n
     *         \e p[0] is the amplitude, \n
     *         \e p[1] is the rate and \n
     *         \e p[2] is the offset.
     *  \return The evaluated function.
     */
    double falpha(double x, const Vector_double& p);
    
    //! Computes the Jacobian of stf::falpha().
    /*! \f{eqnarray*}
     *   j_0(x) &=& \frac{df(x)}{dp_0} = p_1^2 x \mathrm{e}^{-p_1 x} \\
     *   j_1(x) &=& \frac{df(x)}{dp_1} = p_0 p_1 x \left( 2 \mathrm{e}^{-p_1 x} - p_1 x \mathrm{e}^{-p_1 x} \right) \\
     *   j_2(x) &=& \frac{df(x)}{dp_2} = 1.0
     *  \f} 
     *  \param x Function argument.
     *  \param p A valarray of parameters, where \n
     *         \e p[0] is the amplitude, \n
     *         \e p[1] is the rate and \n
     *         \e p[2] is the offset.
     *  \return A valarray \e j with the evaluated Jacobian, where \n
     *          \e j[0] contains the derivative with respect to \e p[0], \n
     *          \e j[1] contains the derivative with respect to \e p[1] and \n
     *          \e j[2] contains the derivative with respect to \e p[2].
     */
    Vector_double falpha_jac(double x, const Vector_double& p);
    
    //! Hodgkin-Huxley sodium conductance function.
    /*! \f[f(x)=p_0\left(1-\mathrm{e}^{\frac{-x}{p_1}}\right)^3\mathrm{e}^{\frac{-x}{p_2}} + p_3\f]
     *  \param x Function argument.
     *  \param p A valarray of parameters, where \n
     *         \e p[0] is the amplitude \f$g'_{Na}\f$, \n
     *         \e p[1] is \f$\tau_m\f$, \n
     *         \e p[2] is \f$\tau_h\f$ and \n
     *         \e p[3] is the offset. \n
     *  \return The evaluated function.
     */
    double fHH(double x, const Vector_double& p);

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
    double fgauss(double x, const Vector_double& p);
    
    //! power of 1 sodium conductance function.
    /*! \f[f(x)=p_0\left(1-\mathrm{e}^{\frac{-x}{p_1}}\right)\mathrm{e}^{\frac{-x}{p_2}} + p_3\f]
     *  \param x Function argument.
     *  \param p A valarray of parameters, where \n
     *         \e p[0] is the amplitude \f$g'_{Na}\f$, \n
     *         \e p[1] is \f$\tau_m\f$, \n
     *         \e p[2] is \f$\tau_h\f$ and \n
     *         \e p[3] is the offset. \n
     *  \return The evaluated function.
     */
    double fgnabiexp(double x, const Vector_double& p);

    //! Initialises parameters for fitting stf::falpha() to \e data.
    /*! \param data The waveform of the data for the fit.
     *  \param base Baseline of \e data.
     *  \param peak Peak value of \e data.
     *  \param dt The sampling interval.
     *  \param pInit On entry, pass a valarray of size 3. On exit, will contain initial parameter
     *         estimates.
     */
    void falpha_init(const Vector_double& data, double base, double peak, double RTLoHI, double HalfWidth, double dt, Vector_double& pInit );

    //! Initialises parameters for fitting stf::fgauss() to \e data.
    /*! \param data The waveform of the data for the fit.
     *  \param base Baseline of \e data.
     *  \param peak Peak value of \e data.
     *  \param dt The sampling interval.
     *  \param pInit On entry, pass a valarray of size 3. On exit, will contain initial parameter
     *         estimates.
     */
    void fgauss_init(const Vector_double& data, double base, double peak, double RTLoHI, double HalfWidth, double dt, Vector_double& pInit );

    //! Initialises parameters for fitting stf::falpha() to \e data.
    /*! \param data The waveform of the data for the fit.
     *  \param base Baseline of \e data.
     *  \param peak Peak value of \e data.
     *  \param dt The sampling interval.
     *  \param pInit On entry, pass a valarray of size 4. On exit, will contain initial parameter
     *         estimates.
     */
    void fHH_init(const Vector_double& data, double base, double peak, double RTLoHi, double HalfWidth, double dt, Vector_double& pInit );

    //! Initialises parameters for fitting stf::falpha() to \e data.
    /*! \param data The waveform of the data for the fit.
     *  \param base Baseline of \e data.
     *  \param peak Peak value of \e data.
     *  \param dt The sampling interval.
     *  \param pInit On entry, pass a valarray of size 4. On exit, will contain initial parameter
     *         estimates.
     */
    void fgnabiexp_init(const Vector_double& data, double base, double peak, double RTLoHi, double HalfWidth, double dt, Vector_double& pInit );

    //! Scales a parameter that linearly depends on x
    /*! \param The parameter to scale
     *  \param xscale x scaling factor
     *  \param xoff x offset
     *  \param yscale y scaling factor
     *  \param yoff y offset
     *  \return Scaled parameter
     */
    double xscale(double param, double xscale, double xoff, double yscale, double yoff);

    //! Unscales a parameter that linearly depends on x
    /*! \param The parameter to scale
     *  \param xscale x scaling factor
     *  \param xoff x offset
     *  \param yscale y scaling factor
     *  \param yoff y offset
     *  \return Unscaled parameter
     */
    double xunscale(double param, double xscale, double xoff, double yscale, double yoff);

    //! Scales a parameter that linearly depends on y
    /*! \param The parameter to scale
     *  \param xscale x scaling factor
     *  \param xoff x offset
     *  \param yscale y scaling factor
     *  \param yoff y offset
     */
    double yscale(double param, double xscale, double xoff, double yscale, double yoff);

    //! Scales a parameter that linearly depends on y and adds an offset
    /*! \param The parameter to scale
     *  \param xscale x scaling factor
     *  \param xoff x offset
     *  \param yscale y scaling factor
     *  \param yoff y offset
     */
    double yscaleoffset(double param, double xscale, double xoff, double yscale, double yoff);

    //! Unscales a parameter that linearly depends on y
    /*! \param The parameter to scale
     *  \param xscale x scaling factor
     *  \param xoff x offset
     *  \param yscale y scaling factor
     *  \param yoff y offset
     *  \return Unscaled parameter
     */
    double yunscale(double param, double xscale, double xoff, double yscale, double yoff);

    //! Unscales a parameter that linearly depends on y and removes the offset
    /*! \param The parameter to scale
     *  \param xscale x scaling factor
     *  \param xoff x offset
     *  \param yscale y scaling factor
     *  \param yoff y offset
     *  \return Unscaled parameter
     */
    double yunscaleoffset(double param, double xscale, double xoff, double yscale, double yoff);

    //! Creates stf::parInfo structs for n-exponential functions.
    /*! \param n_exp Number of exponential terms.
     *  \return A vector of parameter information structs.
     */
    std::vector<stf::parInfo> getParInfoExp(int n_exp);
    
    //! Calculates a weighted time constant.
    /*! \param p Parameters of an exponential function (see stf::fexp()).
     *  \param parsInfo Information about the parameters \e p.
     *  \param chisqr The sum of squared errors, as returned from a least-squares fit.
     *  \return A formatted table of results.
     */
    stf::Table outputWTau(const Vector_double& p, const std::vector<stf::parInfo>& parsInfo, double chisqr);
    
    //! Finds the index of \e data where \e value is encountered for the first time.
    /*! \param data The waveform to be searched.
     *  \param value The value to be found.
     *  \return The index of \e data right after \e value has been crossed.
     */
    std::size_t whereis(const Vector_double& data, double value);

    //! Returns the library of functions for non-linear regression.
    /*! \return A vector of non-linear regression functions.
     */
    std::vector<stf::storedFunc> GetFuncLib();

    /*@}*/

}

#endif
