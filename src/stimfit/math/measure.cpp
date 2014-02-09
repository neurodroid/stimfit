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

/*! \file measlib.cpp
 *  \author Christoph Schmidt-Hieber, Peter Jonas
 *  \date 2011-01-24
 *  \brief Functions for measuring kinetics of events within waveforms.
 * 
 * 
 *  For an example how to use these functions, see Recording::Measure().
 */

#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../stf.h"
#include "./measure.h"
#include "../../libstfio/stfio.h"

double stf::base( double& var, const std::vector<double>& data, std::size_t llb, std::size_t ulb)
{
    if (data.size()==0) return 0;
    if (llb>ulb || ulb>=data.size()) {
        return NAN;
    }
    double base=0.0;

    double sumY=0.0;
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
    double varS=0.0;
    double corr=0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:varS,corr)
#endif
    for (int i=(int)llb; i<=(int)ulb;++i) {
        double diff=data[i]-base;
        varS+=diff*diff;
        // correct for floating point inaccuracies:
        corr+=diff;
    }
    corr=(corr*corr)/n;
    var = (varS-corr)/(n-1);

    return base;
}

double stf::peak(const std::vector<double>& data, double base, std::size_t llp, std::size_t ulp,
            int pM, stf::direction dir, double& maxT)
{
    if (llp>ulp || ulp>=data.size()) {
        maxT = NAN;
        return NAN;
    }
    
    double max=data[llp];
    maxT=(double)llp;
    double peak=0.0;

    if (pM > 0) {
        for (std::size_t i=llp+1; i <=ulp; i++) {
            //Calculate peak as the average over pM points around the point i
            peak=0.0;
            div_t Div1=div((int)pM-1, 2);
            int counter = 0;
            int start = i-Div1.quot;
            if (start < 0)
                start = 0;
            for (counter=start; counter <= start+pM-1 && counter < (int)data.size(); counter++)
                peak+=data[counter];
            peak /= (counter-start);
            
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
            double sumY=0; 
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sumY)
#endif
            for (int i=(int)llp; i<=(int)ulp;++i) {
                sumY+=data[i];
            }
            int n=(int)(ulp-llp+1);
            peak=sumY/n;
            maxT=(double)((llp+ulp)/2.0);
        } else {
            maxT = NAN;
            peak = NAN;
        }
    }
    return peak;
}

double stf::threshold( const std::vector<double>& data, std::size_t llp, std::size_t ulp, double slope, double& thrT, std::size_t windowLength )
{
    thrT = -1;
    
    if (data.size()==0) return 0.0;

    // lower limit peak (ulb) has to be zero at least
    // upper limit peak (ulb) has to be < data.size()-windowLength (data[i+windowLength] will be used)
    if (llp > ulp || ulp >= data.size()) {
        thrT = NAN;
        return NAN;
    }
    if (ulp + windowLength > data.size()) {
        thrT = NAN;
        return NAN;
    }

    double threshold = 0.0;

    // find Slope within peak window:
    for (std::size_t i=llp; i < ulp; ++i) {
        double diff = data[i + windowLength] - data[i];
        if (diff > slope * windowLength) {
            threshold=(data[i+windowLength] + data[i]) / 2.0;
            thrT = i + windowLength/2.0;
            break;
        }
    }

    return threshold;
}

double stf::risetime(const std::vector<double>& data, double base, double ampl,
                     double left, double right, double frac, std::size_t& tLoId, std::size_t& tHiId,
                     double& tLoReal)
{
    if (frac <= 0 || frac >=0.5) {
        tLoReal = NAN;
        return NAN;
    }
    
    double lo = frac;
    double hi = 1.0-frac;
    
    //Lo%of peak
    if (right<0 || left<0 || right>=data.size()) {
        tLoReal = NAN;
        return NAN;
    }
    tLoId=(int)right>=1? (int)right:1;
    do {
        --tLoId;
    } 
    while (fabs(data[tLoId]-base)>fabs(lo*ampl) && tLoId>left);

    //Hi%of peak
    tHiId=tLoId;
    do {
        ++tHiId;
    }
    while (fabs(data[tHiId]-base)<fabs(hi*ampl) && tHiId<right);

    //Calculation of real values by linear interpolation: 
    //Lo%of peak
    //there was a bug in Stimfit for DOS before 2002 that I used
    //as a template
    //corrected 03/01/2006
    double yLong2=data[ tLoId+1];
    double yLong1=data[ tLoId];
    tLoReal=0.0;
    double tHiReal=0.0;
    if (yLong2-yLong1 !=0)
    {
        tLoReal=(double)((double)tLoId+
                fabs((lo*ampl+base-yLong1)/(yLong2-yLong1)));
    } 
    else tLoReal=(double)tLoId;
    //Hi%of peak
    yLong2=data[ tHiId];
    yLong1=data[ tHiId-1];	
    if (yLong2-yLong1 !=0) 
    {
        tHiReal=(double)((double)tHiId-
                fabs(((yLong2-base)-hi*ampl)/(yLong2-yLong1)));
    } 
    else tHiReal=(double)tHiId;

    double rtLoHi=(tHiReal-tLoReal);
    return rtLoHi;  
}

double stf::risetime2(const std::vector<double>& data, double base, double ampl,
                     double left, double right, double frac,
                     double& innerTLoReal, double& innerTHiReal, double& outerTLoReal, double& outerTHiReal )
{
    if (frac <= 0 || frac >=0.5) {
        innerTLoReal = NAN;
        innerTHiReal = NAN;
        outerTLoReal = NAN;
        outerTHiReal = NAN;
        return NAN;
    }

#define NDEBUG
#ifndef NDEBUG
	fprintf(stdout,"%s %i:RISETIME2\n",__FILE__,__LINE__);
#endif

    double lo = frac;
    double hi = 1.0-frac;

    /*
		outer_tLoId	first index from left which is above lo*ampl
		outer_tHiId	last  index from left which is below hi*ampl

		inner_tLoId	last  index from left which is below lo*ampl
		inner_tHiId	first index from left which is above hi*ampl

		Note: in noise free case (outer_tHiId==inner_tHiId-1) and
		(outer_tLoId==inner_tLoId+1) are true.
    */
    long outer_tLoId=-1, outer_tHiId=-1, inner_tLoId=-1, inner_tHiId=-1;
	long k;

    //Lo%of peak
    if (right<0 || left<0 || right>=data.size()) {
        innerTLoReal = NAN;
        innerTHiReal = NAN;
        outerTLoReal = NAN;
        outerTHiReal = NAN;
        return NAN;
    }

#ifndef NDEBUG
	fprintf(stdout,"%s %i:RISETIME2\n",__FILE__,__LINE__);
#endif

    for (k=(long)left; k<=(long)right; k++) {
		double v = fabs(data[k]-base);
		if (v < fabs(lo*ampl)) inner_tLoId = k;
		if (v < fabs(hi*ampl)) outer_tHiId = k;
    }
#ifndef NDEBUG
	fprintf(stdout,"%s %i:RISETIME2 r:%f l:%f \n",__FILE__,__LINE__,right,left);
#endif

    for (k=(long)right; k>=(long)left; k--) {
		double v = fabs(data[k]-base);
		if (v > fabs(lo*ampl)) outer_tLoId = k;
		if (v > fabs(hi*ampl)) inner_tHiId = k;
    }

#ifndef NDEBUG
	fprintf(stdout,"%s %i:RISETIME2: %i %i %i %i\n",__FILE__,__LINE__,(int)outer_tLoId,(int)inner_tLoId,(int)inner_tHiId,(int)outer_tHiId);
#endif

    //*** Inner Risetime ***/
    if (inner_tLoId < 0)
        innerTLoReal = NAN;
    else {
        double yLong2 = data[inner_tLoId+1];
        double yLong1 = data[inner_tLoId];
        if (yLong2-yLong1 != 0)
            innerTLoReal = inner_tLoId + fabs((lo*ampl+base-yLong1)/(yLong2-yLong1));
        else
            innerTLoReal=(double)inner_tLoId;
    }

    //Hi%of peak
    if (inner_tHiId < 1)
        innerTHiReal = NAN;
    else {
        double yLong2 = data[inner_tHiId];
        double yLong1 = data[inner_tHiId-1];
        if (yLong2 - yLong1 != 0)
            innerTHiReal = inner_tHiId - fabs(((yLong2-base)-hi*ampl)/(yLong2-yLong1));
        else
            innerTHiReal=(double)inner_tHiId;
    }

#ifndef NDEBUG
	fprintf(stdout,"%s %i:RISETIME2 %f %s\n",__FILE__,__LINE__,innerTHiReal-innerTLoReal,innerTHiReal<innerTLoReal ?"!!! inner Rise time is invalid !!!":"");
#endif

    //*** Outer Risetime ***/
    if (outer_tLoId < 1)
        outerTLoReal = NAN;
    else {
        double yLong2 = data[outer_tLoId];
        double yLong1 = data[outer_tLoId-1];
        if (yLong2 - yLong1 != 0)
            outerTLoReal = outer_tLoId - fabs(((yLong2-base)-lo*ampl)/(yLong2-yLong1));
        else
            outerTLoReal=(double)outer_tLoId;
    }

    if (outer_tHiId < 0)
        outerTHiReal = NAN;
    else {
        //Hi%of peak
        double yLong2 = data[outer_tHiId+1];
        double yLong1 = data[outer_tHiId];
        if (yLong2-yLong1 != 0 )
            outerTHiReal = outer_tHiId + fabs((hi*ampl+base-yLong1) / (yLong2-yLong1));
        else
            outerTHiReal = (double)outer_tHiId;
    }

#ifndef NDEBUG
	fprintf(stdout,"%s %i:RISETIME2 %f %f %f %f\n",__FILE__,__LINE__,outerTLoReal,innerTLoReal,innerTHiReal,outerTHiReal);
	fprintf(stdout,"%s %i:RISETIME2 %f %s\n",__FILE__,__LINE__,outerTHiReal-outerTLoReal,outerTHiReal<outerTLoReal ?"!!! outer Rise time is invalid !!!":"");
#endif

    return (innerTHiReal-innerTLoReal);
}

double   stf::t_half(const std::vector<double>& data,
        double base,
        double ampl,
        double left,
        double right,
        double center,
        std::size_t& t50LeftId,
        std::size_t& t50RightId,
        double& t50LeftReal)
{
    if (center<0 || center>=data.size() || data.size()<=2 || left<-1) {
        t50LeftReal = NAN;
        return NAN;
    }
    t50LeftId=(int)center>=1? (int)center:1;
    if (t50LeftId-1 >= data.size()) {
        
#ifndef NDEBUG
        std::cout << "t50LeftId-1 >= data.size()" << t50LeftId-1 << " "
                  << data.size() << std::endl;
#endif
        return NAN;
    }
    do {
        --t50LeftId;
    } while (fabs(data[t50LeftId]-base)>fabs(0.5 * ampl) &&
            t50LeftId > left);
    //Right side half duration
    if ((std::size_t)center <= data.size()-2) {
        t50RightId = center;
    } else {
        t50RightId = data.size() >= 2? data.size()-2 : 0;
    }
    if (right >= data.size()-1 || t50RightId+1 >= data.size()) {
#ifndef NDEBUG
        std::cout << "right, data.size(), t50RightId+1 " << right << " "
                  << data.size() << " " << t50RightId+1 << std::endl;
#endif
        return NAN;
    }
    do {
        ++t50RightId;
    } while (fabs(data[t50RightId]-base)>fabs(0.5 * ampl) &&
            t50RightId < right);

    //calculation of real values by linear interpolation: 
    //Left side
    double yLong2=data[t50LeftId+1];
    double yLong1=data[t50LeftId];
    if (yLong2-yLong1 !=0) {
        t50LeftReal=(double)(t50LeftId+
                fabs((0.5*ampl-(yLong1-base))/(yLong2-yLong1)));
    } else {
        t50LeftReal=(double)t50LeftId;
    }
    //Right side
    yLong2=data[t50RightId];
    yLong1=data[t50RightId-1];
    double t50RightReal=0.0;
    if (yLong2-yLong1 !=0) {
        t50RightReal=(double)(t50RightId-
                fabs((0.5*ampl-(yLong2-base))/fabs(yLong2-yLong1)));
    } else {
        t50RightReal=(double)t50RightId;
    }
    return t50RightReal-t50LeftReal;
}

double   stf::maxRise(const std::vector<double>& data,
        double left,
        double right,
        double& maxRiseT,
        double& maxRiseY,
        std::size_t    windowLength)
{

    size_t rightc = lround(right);
    size_t leftc  = lround(left);
    if (leftc >= data.size()-windowLength) {
        leftc = data.size()-windowLength-1;
    }
    if (rightc >= data.size() || data.size() < windowLength) {
        maxRiseY = NAN;
        maxRiseT = NAN;
        return NAN;
    }
    double maxRise = -INFINITY;  // -Infinity
    maxRiseT = NAN;		// non-a-number
    size_t i,j;
    for (i = leftc, j = leftc + windowLength; j <= rightc; i++, j++) {
        double diff = fabs( data[i] - data[j] );
        if (maxRise<diff) {
            maxRise=diff;
            maxRiseY=(data[i]+data[j])/2.0;
            maxRiseT=(i+windowLength/2.0);
        }
    }
    return maxRise/windowLength;
}

double stf::maxDecay(const std::vector<double>& data,
        double left,
        double right,
        double& maxDecayT,
        double& maxDecayY,
        std::size_t    windowLength)
{
    size_t rightc = lround(right);
    size_t leftc  = lround(left);
    if (leftc >= data.size()-windowLength) {
        leftc = data.size()-windowLength-1;
    }
    if (rightc >= data.size() || data.size() < windowLength) {
        maxDecayT = NAN;
        maxDecayY = NAN;
        return NAN;
    }
    double maxDecay = -INFINITY;  // -Infinity
    maxDecayT = NAN;		// non-a-number
    size_t i,j;
    for (j = leftc, i = leftc + windowLength; i < rightc; i++, j++) {
        double diff = fabs( data[i] - data[j] );
        if (maxDecay<diff) {
            maxDecay=diff;
            maxDecayY=(data[i]+data[j])/2.0;
            maxDecayT=(j+windowLength/2.0);
        }
    }
    return maxDecay/windowLength;
}

#ifdef WITH_PSLOPE
double stf::pslope(const std::vector<double>& data, std::size_t left, std::size_t right) {

    // data testing not zero 
    //if (!data.size()) return 0;
    if (data.size()==0) return 0;

    // cursor testing out of bounds
    if (left>right || right>data.size()) {
        return NAN;
    }
    // use interpolated data
    double y2 = ( data[right]+data[right+1] )/(double)2.0;
    double y1 = ( data[left]+data[left+1] )/(double)2.0;
    double t2 = (double)(right-0.5);
    double t1 = (double)(left-0.5);

    double SlopeVal = (y2-y1)/(t2-t1);

    return SlopeVal;
}
#endif // WITH_PSLOPE


