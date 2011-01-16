// Header file for the stimfit namespace
// Routines for measuring basic event properties
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

/*! \file measlib.h
 *  \author Christoph Schmidt-Hieber, Peter Jonas
 *  \date 2008-01-16
 *  \brief Functions for measuring kinetics of events within waveforms.
 * 
 * 
 *  For an example how to use these functions, see Recording::Measure().
 */

#ifndef _MEASLIB_H
#define _MEASLIB_H

#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace stf {

/*! \addtogroup stfgen
 *  @{
 */

//! Calculate the average of all sampling points between and including \e llb and \e ulb.
/*! \param var Will contain the variance on exit.
 *  \param data The data waveform to be analysed.
 *  \param llb Averaging will be started at this index.
 *  \param ulb Index of the last data point included in the average (legacy of the PASCAL version).
 *  \param llp Lower limit of the peak window (see stf::peak()).
 *  \param ulp Upper limit of the peak window (see stf::peak()). 
 *  \return The baseline value.
 */
template <typename T>
T base( T& var, const std::vector<T>& data, std::size_t llb, std::size_t ulb,
        std::size_t llp=0, std::size_t ulp=0 );

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
template <typename T>
T peak( const std::vector<T>& data, T base, std::size_t llp, std::size_t ulp,
        int pM, stf::direction, T& maxT);
 
//! Find the value within \e data between \e llp and \e ulp at which \e slope is exceeded.
/*! \param data The data waveform to be analysed.
 *  \param llp Lower limit of the peak window.
 *  \param ulp Upper limit of the peak window. 
 *  \param thrT On exit, The interpolated time point of the threshold crossing
 *              in units of sampling points, or a negative value if the threshold
                wasn't found.
 *  \return The interpolated threshold value.
 */
template <typename T>
T threshold( const std::vector<T>& data, std::size_t llp, std::size_t ulp, T slope, T& thrT );

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
template <typename T>
T risetime( const std::vector<T>& data, T base, T ampl, T left, T right,
            std::size_t& t20Id, std::size_t& t80Id, T& t20Real );

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
template <typename T>
T t_half( const std::vector<T>& data, T base, T ampl, T left, T right,
          T center, std::size_t& t50LeftId, std::size_t& t50RightId,
          T& t50LeftReal );

//! Find the maximal slope during the rising phase of an event within \e data.
/*! \param data The data waveform to be analysed.
 *  \param left Delimits the search to the left.
 *  \param right Delimits the search to the right.
 *  \param maxRiseT The interpolated time point of the maximal slope of rise
 *         in units of sampling points.
 *  \param maxRiseY The interpolated value of \e data at \e maxRiseT.
 *  \return The maximal slope during the rising phase.
 */
template <typename T>
T  maxRise( const std::vector<T>& data, T left, T right, T& maxRiseT,
            T& maxRiseY);

//! Find the maximal slope during the decaying phase of an event within \e data.
/*! \param data The data waveform to be analysed.
 *  \param left Delimits the search to the left.
 *  \param right Delimits the search to the right.
 *  \param maxDecayT The interpolated time point of the maximal slope of decay
 *         in units of sampling points.
 *  \param maxDecayY The interpolated value of \e data at \e maxDecayT.
 *  \return The maximal slope during the decaying phase.
 */
template <typename T>
T  maxDecay( const std::vector<T>& data, T left, T right, T& maxDecayT,
             T& maxDecayY);

#ifdef WITH_PSLOPE
//! Find the slope an event within \e data.
/*! \param data The data waveform to be analysed.
 *  \param left delimits the search to the left.
 *  \param right delimits the search to the right.
 *  \return The slope during the limits defined in left and right.
 */
template <typename T>
T pslope( const std::vector<T>& data, std::size_t left, std::size_t right);

#endif
/*@}*/

}

template <typename T>
T stf::base( T& var, const std::vector<T>& data, std::size_t llb, std::size_t ulb,
             std::size_t llp, std::size_t ulp)
{
    if (data.size()==0) return 0;
    if (llb>ulb || ulb>=data.size()) {
        throw (std::out_of_range("Exception:\n Index out of range in stf::base()"));
    }
    T base=0.0;

    T sumY=0.0;
    //according to the pascal version, every value 
    //within the window shall be summed up:
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sumY)
#endif
    for (int i=(int)llb; i<=(int)ulb;++i) {
        sumY+=data[i];
    }
    int n=(int)(ulb-llb+1);
    base=sumY/n;
    // second pass to calculate the variance:
    T varS=0.0;
    T corr=0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:varS,corr)
#endif
    for (int i=(int)llb; i<=(int)ulb;++i) {
        T diff=data[i]-base;
        varS+=diff*diff;
        // correct for floating point inaccuracies:
        corr+=diff;
    }
    corr=(corr*corr)/n;
    var = (varS-corr)/(n-1);

    return base;
}

template <typename T>
T stf::peak(const std::vector<T>& data, T base, std::size_t llp, std::size_t ulp,
            int pM, stf::direction dir, T& maxT)
{
    if (llp>ulp || ulp>data.size()) {
        throw (std::out_of_range("Exception:\n Index out of range in stf::peak()"));
    }
    
    T max=data[llp];
    maxT=(double)llp;
    T peak=0.0;

    if (pM > 0) {
        for (std::size_t i=llp+1; i <=ulp; i++) {
            //Calculate peak as the average over pM points around the point i
            peak=0.0;
            div_t Div1=div((int)pM-1, 2);
            for (std::size_t j=i-Div1.quot; j <=i-Div1.quot+pM-1; j++) 
                peak+=data[j];
            peak /= pM;
            //Set peak for BOTH
            if (dir == stf::both && fabs(peak-base) > fabs (max-base))
            {
                max = peak;
                maxT = (double)i;
            }
            //Set peak for UP
            if (dir == stf::up && peak-base > max-base)
            {
                max = peak;
                maxT = (double)i;
            }
            //Set peak for DOWN
            if (dir == stf::down && peak-base < max-base)
            {
                max = peak;
                maxT = (double)i;
            }
        }	//End loop: data points
        peak = max;
        //End peak and base calculation
        //-------------------------------
    } else {
        if (pM==-1) { // calculate the average within the peak window
            T sumY=0; 
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sumY)
#endif
            for (int i=(int)llp; i<=(int)ulp;++i) {
                sumY+=data[i];
            }
            int n=(int)(ulp-llp+1);
            peak=sumY/n;
            maxT=(T)((llp+ulp)/2.0);
        } else {
            throw (std::out_of_range(
                    "mean peak points out of range in stf::peak()")
            );
        }
    }
    return peak;
}

template <typename T>
T stf::threshold( const std::vector<T>& data, std::size_t llp, std::size_t ulp, T slope, T& thrT )
{
    thrT = -1;
    
    if (data.size()==0) return 0;

    // ulb has to be < data.size()-1 (data[i+1] will be used)
    if (llp > ulp || ulp >= data.size()) {
        throw (std::out_of_range("Exception:\n Index out of range in stf::threshold()"));
    }
    
    T threshold = 0.0;

    // find Slope within peak window:
    for (std::size_t i=llp; i < ulp; ++i) {
        T diff=data[i+1]-data[i];
        if (diff>slope) {
            threshold=(data[i+1]+data[i])/(T)2.0;
            thrT=(T)(i+0.5);
            break;
        }
    }

    return threshold;
}

template <typename T>
T stf::risetime(const std::vector<T>& data,
        T base,
        T ampl,
        T left,
        T right,
        std::size_t& t20Id,
        std::size_t& t80Id,
        T& t20Real)
{
    //20%of peak
    if (right<0 || left<0 || right>=data.size()) {
        throw std::out_of_range("Index out of range in stf::risetime");
    }
    t20Id=(int)right>=1? (int)right:1;
    do {
        --t20Id;
    } 
    while (fabs(data[t20Id]-base)>fabs(0.2*ampl) && t20Id>left);

    //80%of peak
    t80Id=t20Id;
    do {
        ++t80Id;
    }
    while (fabs(data[t80Id]-base)<fabs(0.8*ampl) && t80Id<right);

    //Calculation of real values by linear interpolation: 
    //20%of peak
    //there was a bug in Stimfit for DOS before 2002 that I used
    //as a template
    //corrected 03/01/2006
    T yLong2=data[ t20Id+1];
    T yLong1=data[ t20Id];
    t20Real=0.0;
    T t80Real=0.0;
    if (yLong2-yLong1 !=0)
    {
        t20Real=(double)((double)t20Id+
                fabs((0.2*ampl+base-yLong1)/(yLong2-yLong1)));
    } 
    else t20Real=(double)t20Id;
    //80%of peak
    yLong2=data[ t80Id];
    yLong1=data[ t80Id-1];	
    if (yLong2-yLong1 !=0) 
    {
        t80Real=(double)((double)t80Id-
                fabs(((yLong2-base)-0.8*ampl)/(yLong2-yLong1)));
    } 
    else t80Real=(double)t80Id;

    T rt2080=(t80Real-t20Real);
    return rt2080;  
}

template <typename T>
T   stf::t_half(const std::vector<T>& data,
        T base,
        T ampl,
        T left,
        T right,
        T center,
        std::size_t& t50LeftId,
        std::size_t& t50RightId,
        T& t50LeftReal)
{
    if (center<0 || center>=data.size()) {
        throw std::out_of_range("Index out of range in stf::thalf()");
    }
    t50LeftId=(int)center>=1? (int)center:1;
    do {
        --t50LeftId;
    } while (fabs(data[t50LeftId]-base)>fabs(0.5 * ampl) &&
            t50LeftId > left);
    //Right side half duration
    t50RightId=(int)center<=(int)data.size()-2? (int)center:data.size()-2;
    if ((int)right>(int)data.size()-1) right=data.size()-1;  
    do {
        ++t50RightId;
    } while (fabs(data[t50RightId]-base)>fabs(0.5 * ampl) &&
            t50RightId < right);

    //calculation of real values by linear interpolation: 
    //Left side
    T yLong2=data[t50LeftId+1];
    T yLong1=data[t50LeftId];
    if (yLong2-yLong1 !=0) {
        t50LeftReal=(double)(t50LeftId+
                fabs((0.5*ampl-(yLong1-base))/(yLong2-yLong1)));
    } else {
        t50LeftReal=(double)t50LeftId;
    }
    //Right side
    yLong2=data[t50RightId];
    yLong1=data[t50RightId-1];
    T t50RightReal=0.0;
    if (yLong2-yLong1 !=0) {
        t50RightReal=(double)(t50RightId-
                fabs((0.5*ampl-(yLong2-base))/fabs(yLong2-yLong1)));
    } else {
        t50RightReal=(double)t50RightId;
    }
    return t50RightReal-t50LeftReal;
}

template <typename T>
T   stf::maxRise(const std::vector<T>& data,
        T left,
        T right,
        T& maxRiseT,
        T& maxRiseY)
{
    if (left<0 || right<0 || left>=data.size()-1 || right>=data.size()) {
        throw std::out_of_range("Index out of range in stf::maxRise");
    }
    if (right==0)
        right=1;
    //Maximal rise
    T maxRise=fabs(data[(int)right]-data[(int)right-1]);
    maxRiseT=right-(double)0.5;
    int i=(int)right-1;
    do {
        T diff=fabs(data[i]-data[i-1]);
        if (maxRise<diff) {
            maxRise=diff;
            maxRiseY=data[i]/(T)2.0+data[i-1]/(T)2.0;
            maxRiseT=(T)(i-0.5);
        }
        --i;
    } while (i>=left);
    return maxRise;
}

template <typename T>
T stf::maxDecay(const std::vector<T>& data,
        T left,
        T right,
        T& maxDecayT,
        T& maxDecayY)
{
    if (left<0 || right<0 || left>=data.size()-2 || right>=data.size()) {
        throw std::out_of_range("Index out of range in stf::maxDecay");
    }
    if (right==0)
        right=1;
            
    //Maximal decay
    T maxDecay=fabs(data[(int)left+1]-data[(int)left]);
    maxDecayT=left+(double)0.5;
    int i=(int)left+2;
    do {
        double diff=fabs(data[i]-data[i-1]);
        if (maxDecay<diff) {
            maxDecay=diff;
            // maxDecayY = ( data[i]+data[i-1] )/(T)2.0;
            maxDecayY=data[i]/(T)2.0+data[i-1]/(T)2.0;
            maxDecayT=(T)(i-0.5);
        }
        ++i;
    } while (i<right);
    return maxDecay;
}

#ifdef WITH_PSLOPE
template <typename T>
T stf::pslope(const std::vector<T>& data, std::size_t left, std::size_t right) {

    // data testing not zero 
    //if (!data.size()) return 0;
    if (data.size()==0) return 0;

    // cursor testing out of bounds
    if (left>right || right>data.size()) {
        throw (std::out_of_range("Exception:\n Index out of range in stf::pslope()"));
    }
    // use interpolated data
    T y2 = ( data[right]+data[right+1] )/(T)2.0;
    T y1 = ( data[left]+data[left+1] )/(T)2.0;
    T t2 = (T)(right-0.5);
    T t1 = (T)(left-0.5);

    T SlopeVal = (y2-y1)/(t2-t1);

    return SlopeVal;
}
#endif // WITH_PSLOPE
#endif

