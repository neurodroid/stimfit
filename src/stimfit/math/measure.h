// Header file for the stimfit namespace
// Routines for measuring basic event properties
// last revision: 24-Jan-2011
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

/*! \file measure.h
 *  \author Christoph Schmidt-Hieber, Peter Jonas
 *  \date 2011-01-24
 *  \brief Functions for measuring kinetics of events within waveforms.
 * 
 * 
 *  For an example how to use these functions, see Recording::Measure().
 */

#ifndef _MEASLIB_H
#define _MEASLIB_H

#include <vector>

namespace stf {

/*! \addtogroup stfgen
 *  @{
 */

//! Calculate the average of all sampling points between and including \e llb and \e ulb.
/*! \param method: 0: mean and s.d.; 1: median
 *  \param var Will contain the variance on exit (only when method=0).
 *  \param data The data waveform to be analysed.
 *  \param llb Averaging will be started at this index.
 *  \param ulb Index of the last data point included in the average (legacy of the PASCAL version).
 *  \param llp Lower limit of the peak window (see stf::peak()).
 *  \param ulp Upper limit of the peak window (see stf::peak()). 
 *  \return The baseline value - either the mean or the median depending on method.
 */
double base(enum stf::baseline_method method, double& var, const std::vector<double>& data, std::size_t llb, std::size_t ulb);


//! Find the peak value of \e data between \e llp and \e ulp.
/*! Note that peaks will be detected by measuring from \e base, but the return value
 *  is given from 0. Data points at both \e llp and \e ulp will be included in the search 
 *  (legacy of Stimfit for PASCAL).
 *  \param data The data waveform to be analysed.
 *  \param base The baseline value.
 *  \param llp Lower limit of the peak window.
 *  \param ulp Upper limit of the peak window. 
 *  \param pM If \e pM > 1, a sliding (boxcar) average of width \e pM will be used
 *         to measure the peak.
 *  \param dir Can be \n
 *         stf::up for positive-going peaks, \n
 *         stf::down for negative-going peaks or \n
 *         stf::both for negative- or positive-going peaks, whichever is larger.
 *  \param maxT On exit, the index of the peak value. May be interpolated if \e pM > 1.
 *  \return The peak value, measured from 0.
 */
double peak( const std::vector<double>& data, double base, std::size_t llp, std::size_t ulp,
        int pM, stf::direction, double& maxT);
 
//! Find the value within \e data between \e llp and \e ulp at which \e slope is exceeded.
/*! \param data The data waveform to be analysed.
 *  \param llp Lower limit of the peak window.
 *  \param ulp Upper limit of the peak window. 
 *  \param thrT On exit, The interpolated time point of the threshold crossing
 *              in units of sampling points, or a negative value if the threshold
                wasn't found.
 *  \param windowLength is the distance (in number of samples) used to compute the difference,
                the default value is 1.
 *  \return The interpolated threshold value.
 */
 double threshold( const std::vector<double>& data, std::size_t llp, std::size_t ulp, double slope, double& thrT, std::size_t windowLength );

//! Find 20 to 80% rise time of an event in \e data.
/*! Although t80real is not explicitly returned, it can be calculated
 *  from t20Real+risetime.
 *  \param data The data waveform to be analysed.
 *  \param base The baseline value.
 *  \param ampl The amplitude of the event (typically, peak-base).

 *  \param left Delimits the search to the left.
 *  \param right Delimits the search to the right.
 *  \param t20Id On exit, the index wich is closest to the 20%-point.
 *  \param t80Id On exit, the index wich is closest to the 80%-point.
 *  \param t20Real the linearly interpolated 20%-timepoint in
 *         units of sampling points.

 *  \return The rise time.
 */
double risetime(const std::vector<double>& data, double base, double ampl,
                double left, double right, double frac, std::size_t& tLoId, std::size_t& tHiId,
                double& tLoReal);

//! Find 20 to 80% rise time of an event in \e data.
/*! Although t80real is not explicitly returned, it can be calculated
 *  from t20Real+risetime.
 *  \param data The data waveform to be analysed.
 *  \param base The baseline value.
 *  \param ampl The amplitude of the event (typically, peak-base).

 *  \param left Delimits the search to the left.
 *  \param right Delimits the search to the right.
 *  \param innerTLoReal interpolated starting point of the inner risetime
 *  \param innerTHiReal interpolated end point of the inner risetime
 *  \param outerTLoReal interpolated starting point of the outer risetime
 *  \param outerTHiReal interpolated end point of the outer risetime
    the inner rise time is (innerTHiReal-innerTLoReal),
    the outer rise time is (outerTHiReal-outerTLoReal),
    in case of noise free data, inner and outer rise time are the same.

 *  \return The inner rise time.
 */
double risetime2(const std::vector<double>& data, double base, double ampl,
                double left, double right, double frac,
                double& innerTLoReal, double& innerTHiReal, double& outerTLoReal, double& outerTHiReal );

//! Find the full width at half-maximal amplitude of an event within \e data.
/*! Although t50RightReal is not explicitly returned, it can be calculated
 *  from t50LeftReal+t_half.
 *  \param data The data waveform to be analysed.
 *  \param base The baseline value.
 *  \param ampl The amplitude of the event (typically, peak-base).
 *  \param left Delimits the search to the left.
 *  \param right Delimits the search to the right.
 *  \param center The estimated center of an event from which to start
 *         searching to the left and to the right (typically, the index
 *         of the peak).
 *  \param t50LeftId On exit, the index wich is closest to the left 50%-point.
 *  \param t50RightId On exit, the index wich is closest to the right 50%-point.
 *  \param t50LeftReal the linearly interpolated left 50%-timepoint in 
 *         units of sampling points.
 *  \return The full width at half-maximal amplitude.
 */
double t_half( const std::vector<double>& data, double base, double ampl, double left, double right,
               double center, std::size_t& t50LeftId, std::size_t& t50RightId, double& t50LeftReal );

//! Find the maximal slope during the rising phase of an event within \e data.
/*! \param data The data waveform to be analysed.
 *  \param left Delimits the search to the left.
 *  \param right Delimits the search to the right.
 *  \param maxRiseT The interpolated time point of the maximal slope of rise
 *         in units of sampling points.
 *  \param maxRiseY The interpolated value of \e data at \e maxRiseT.
 *  \param windowLength is the distance (in number of samples) used to compute
           the slope, the default value is 1.
 *  \return The maximal slope during the rising phase.
 */
double  maxRise( const std::vector<double>& data, double left, double right, double& maxRiseT,
                 double& maxRiseY, std::size_t windowLength);

//! Find the maximal slope during the decaying phase of an event within \e data.
/*! \param data The data waveform to be analysed.
 *  \param left Delimits the search to the left.
 *  \param right Delimits the search to the right.
 *  \param maxDecayT The interpolated time point of the maximal slope of decay
 *         in units of sampling points.
 *  \param maxDecayY The interpolated value of \e data at \e maxDecayT.
 *  \param windowLength is the distance (in number of samples) used to compute
           the slope, the default value is 1.
 *  \return The maximal slope during the decaying phase.
 */
double  maxDecay( const std::vector<double>& data, double left, double right, double& maxDecayT,
                  double& maxDecayY, std::size_t windowLength);

#ifdef WITH_PSLOPE
//! Find the slope an event within \e data.
/*! \param data The data waveform to be analysed.
 *  \param left delimits the search to the left.
 *  \param right delimits the search to the right.
 *  \return The slope during the limits defined in left and right.
 */
double pslope( const std::vector<double>& data, std::size_t left, std::size_t right);

#endif
/*@}*/

}

#endif
