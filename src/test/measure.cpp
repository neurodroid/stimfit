#include "../stimfit/stf.h"
#include "../stimfit/math/measure.h"
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>

#define PI  3.14159265f

const double tol = 0.1; /* tolerance value */
const static float dt = 1/500.0; /* sampling interval */


void save_txt(const char *fname, Vector_double &mydata){

    std::ofstream output_file;
    output_file.open(fname);

    Vector_double::iterator it;

    for (it = mydata.begin(); it != mydata.end(); ++it){
        output_file << *it << std::endl;
    }

    output_file.close();
}
//=========================================================================
// evaluates if the measurement is within the expected value for a given
// tolerance level that corresponds to 
//=========================================================================
void pass_test(double measurement, double expected, double tolerance){
    EXPECT_NEAR( measurement, expected, fabs(expected*tolerance) );
}

//=========================================================================
// a sine wave to test basic Stimfit measurements
// the sine function has well defined maxima and minima that we use 
// to test the peak algorithm in both directions.
// In addition, because derivative of the sine is known (cosine)
// we can test easily the max slope of rise and decay.
// The maximal slope of rise correspond to the point where cosine is one 
// (at 0, 2*PI) and the max slope of decay where the cosine is minus one
// (at PI, 3*PI). Finally, the 20-80% rise-time should be 0.7, which
// is the result of calculating arcsin(0.8)-arcsin(0.2)
//
// <length> length of the wave in sampling points (e.g long(2*PI))
//=========================================================================
std::vector<double> sinwave(long length){
    std::vector<double> mydata(length);

    for(std::vector<int>::size_type x=0; x != mydata.size() ; ++x){
        mydata[x] = sin( x*dt ); /* see sampling interval */
    }

    return mydata;
}
//=========================================================================
// Exponential function to test the Stimfit threshold measurement
// we can easily evalute the value and position of the threshold
// because the slope we fix for the threshold is simply the exponential 
// for example, for a slope of 1.5, we will find that
// the obtained threshold obtained is 1.5. The time of the
// threshold (thrT) can be evaluated at exp(thrT) and give the slope value
// the function starts at x=0 (y=1), therefore, the slope values
// to be tested should be greater or equal to ONE!!!.
//=========================================================================
std::vector<double> expwave(long length){
    std::vector<double> mydata(length);

    for(std::vector<int>::size_type x=0; x != mydata.size() ; ++x){
        mydata[x] = exp( x*dt ); /* see sampling interval */
    }

    return mydata;
}


//=========================================================================
// test baseline (base)
//=========================================================================
TEST(measlib_test, baseline_basic) {

    std::vector<double> data(32768);
    double var = 0;

    EXPECT_EQ(stf::base(var, data, 0, data.size()-1), 0);
    EXPECT_EQ(var, 0);

}

//=========================================================================
// test baseline out of range exceptions 
//=========================================================================
TEST(measlib_test, baseline_out_of_range_exceptions) {

    std::vector<double> data(32768);
    double var;

    /* Out of range: after last point */
    EXPECT_THROW(stf::base(var, data, 0, data.size()),\
         std::out_of_range);

    /* Out of range: before first point */
    EXPECT_THROW(stf::base(var, data, -1, data.size()-1),\
         std::out_of_range);

}

//=========================================================================
// test peak 
//=========================================================================
TEST(measlib_test, peak_basic) {

    /* 1.- Test with a basic example */
    std::vector<double> data(32768);
    data[16385] = 1.0;
    double maxT;
    
    /* Find positive going peaks */
    double peak_up = stf::peak(data, 0.0, 0, data.size()-1, \
        1, stf::up, maxT);
    EXPECT_EQ(peak_up, 1.0);

    /* Find negative going peaks */
    double peak_down = stf::peak(data, 0.0, 0, data.size()-1, \
        1, stf::down, maxT);
    EXPECT_EQ(peak_down, 0.0);

    /* Find either positive or negative going peaks */
    double peak_both = stf::peak(data, 0.0, 0, data.size()-1,\
         1, stf::both, maxT);
    EXPECT_EQ(peak_both, 1.0); /* take larger value */
    EXPECT_EQ(maxT, 16385);
}

//=========================================================================
// test peak out of range exceptions
//=========================================================================
TEST(measlib_test, peak_out_of_range_exceptions) {
    std::vector<double> data(32768);
    double maxT;

    /* Out of range: before first point */
    EXPECT_THROW(stf::peak(data, 0.0, 0, data.size(), 
        1, stf::both, maxT), std::out_of_range);

    /* Out of range: before first point */
    EXPECT_THROW(stf::peak(data, 0.0, -1, data.size()-1, 
        1, stf::both, maxT), std::out_of_range);
}

//=========================================================================
// test peak direction
//=========================================================================
TEST(measlib_test, peak_direction) {

    /* Sin wave between 0 and 2PI */
    std::vector<double> mywave = sinwave( long(2*PI/dt) );

    /* positive peak is at one, located at PI/2 */
    double maxT;
    double peak = stf::peak(mywave, 0.0, 0, long(2*PI/dt)-1, \
        1, stf::up, maxT);

    double peak_xpted = 1.0;               /* peak is at 1.0      */
    double maxT_xpted = (PI/2.0)/dt;      /* maxT located at PI/2 */
    EXPECT_NEAR(peak, peak_xpted,  0.1);      
    EXPECT_NEAR(maxT, maxT_xpted, fabs(maxT_xpted*tol)); 

    /* look for negative peak between zero and 2*PI */
    double drop = stf::peak(mywave, 0.0, 0, long(2*PI/dt)-1, \
        1, stf::down, maxT);

    peak_xpted = -1.0;               /* drop is at -1.0    */
    maxT_xpted = (3*PI/2)/dt;        /* maxT located at 3*PI/2 */
    EXPECT_NEAR(drop, peak_xpted,  0.1);      
    EXPECT_NEAR(maxT, maxT_xpted, fabs(maxT_xpted*tol)); 

    /* Cursors between 0 and PI give only possitive peak values*/
    double p1 = stf::peak(mywave, 0.0, 0, long(PI/dt)-1, \
        1, stf::down, maxT);
    EXPECT_TRUE(p1 >= 0);

    /* Cursors between PI and 2*PI give only negative peak values*/
    double p2 = stf::peak(mywave, 0.0, long(PI/dt), long(2*PI/dt)-1, \
        1, stf::down, maxT);
    EXPECT_TRUE(p2 <= 0);

}

//=========================================================================
// test threshold 
//=========================================================================
TEST(measlib_test, threshold){
    
    /* exp wave between 0 and 1 */
    std::vector<double> mywave = expwave(1/dt);

    double thrT;
    double slope = 1.2; /* y-units/sample, choose always >= 1 */
    int windowLength = 1;
    
    /* check threshold value at the given slope */
    double threshold = stf::threshold(mywave, 1, 
        long(1/dt)-1, slope*dt, thrT, windowLength);

    /* the threshold should be exactly the slope value */
    EXPECT_NEAR(threshold, slope, fabs(slope*tol)); 

    /* exp(t) should give the slope value */
    EXPECT_NEAR(std::exp(thrT*dt), slope, fabs(slope*tol)); 
    
}
//=========================================================================
// test threshold windowLength exceptions
//=========================================================================
TEST(measlib_test, threshold_windowLength_exceptions){
    
    std::vector<double> data = expwave(1/dt);

    double thrT;
    double slope = 1.2; /* y-units/sample, choose always >= 1 */
    int mywindowLength = 10;
    
    /* Right peak cursor must be larger than windowLength */
    long myRightPeakCursor = mywindowLength-1;
    EXPECT_THROW(stf::threshold(data, 0, myRightPeakCursor, 
        slope*dt, thrT, mywindowLength), 
        std::out_of_range);

    /* Left peak cursor must be smaller than data.size()-windowLength */
    long myLeftPeakCursor = data.size()-mywindowLength; 
    EXPECT_THROW(stf::threshold(data, myLeftPeakCursor, data.size()-1, 
        slope*dt, thrT, mywindowLength), 
        std::out_of_range);

    /* Data size itself must be smaller than windowLength */
    mywindowLength = data.size()+1;
    EXPECT_THROW(stf::threshold(data, 0, data.size()-1, \
        slope*dt, thrT, mywindowLength), 
        std::out_of_range);

}
//=========================================================================
// test threshold out of range exceptions
//=========================================================================
TEST(measlib_test, threshold_out_of_range){
   
    std::vector<double> mywave = expwave(1/dt); 

    double thrT;
    double slope = 0.2; /* y-units/sample, choose always >= 1 */
    int windowLength = 1;
    
    /* Out of range: after last point*/
    EXPECT_THROW(stf::threshold(mywave, 1, mywave.size(), 
        slope*dt, thrT, windowLength),
        std::out_of_range);
    EXPECT_EQ(thrT,-1); 

    /* Out of range: before first point*/
    EXPECT_THROW(stf::threshold(mywave, -1, mywave.size()-1, 
        slope*dt, thrT, windowLength),
        std::out_of_range);
    EXPECT_EQ(thrT,-1); 

    
}

//=========================================================================
// test risetime values 
//=========================================================================
TEST(measlib_test, risetime_values){
    
    /* a sine wave between 0 and PI */
    std::vector<double> mywave = sinwave( long(PI/dt) );

    std::size_t t20, t80;
    double t20Real;
     
    /* check rise time between 0 and PI/2 */
    double risetime = stf::risetime(mywave, 0.0, 1.0, 1, 
        long((PI/2)/dt)-1, 0.2, t20, t80, t20Real);

    /* t20 and t80 correspond to 0.2 and 0.8 respectively */
    EXPECT_NEAR( std::sin(t20*dt), 0.2, 0.02 ); /* sin(t20) = 0.2 */
    EXPECT_NEAR( std::sin(t80*dt), 0.8, 0.08 ); /* sin(t80) = 0.8 */

    /* the risetime is the arcsin(t80)-arcsin(t20) */
    double risetime_xpted = std::asin(0.8) - std::asin(0.2); 
    EXPECT_NEAR(risetime*dt, risetime_xpted, \
        fabs(risetime_xpted*tol) ); 
}

//=========================================================================
// test half_duration 
//=========================================================================
TEST(measlib_test, half_duration){

    /* a sine wave between 0 and PI */
    std::vector <double> mywave = sinwave( long(PI/dt) );
    
    std::size_t t50LeftId, t50RigthId;
    double t50Real;
    
    /* check half duration between 0 and PI */
    double half_dur = stf::t_half(mywave, 0.0, 1.0, 1,
        long(PI/dt)-1, long((PI/2)/dt),t50LeftId, t50RigthId, t50Real);

    /* t50Left and t50Rigth correspond to 0.5 */
    EXPECT_NEAR( std::sin(t50LeftId*dt),  0.5, 0.05); /* sin(t50) = 0.5 */
    EXPECT_NEAR( std::sin(t50RigthId*dt), 0.5, 0.05);

    /* half-duration is arcsin(0.5)+ arcsin(1) */
    double half_dur_xpted = std::asin(0.5)+std::asin(1.0);
    EXPECT_NEAR(half_dur*dt, half_dur_xpted, 
        fabs(half_dur_xpted*tol) );

}

//=========================================================================
// test half_duration exceptions
//=========================================================================
TEST(measlib_test, half_duration_out_of_range_exceptions){

    /* a sine wave between 0 and PI */
    std::vector <double> mywave = sinwave( long(PI/dt) );
    
    std::size_t t50LeftId, t50RigthId;
    double t50Real;
    double center = -1.0; /* index of the peak */ 
    
    /* Out of range: if center <0 */
    EXPECT_THROW( stf::t_half(mywave, 0.0, 1.0, 1,
        long(PI/dt)-1, center, t50LeftId, t50RigthId, t50Real),
        std::out_of_range);

    /* Out of range: if center > recording length */
    center = mywave.size();
    EXPECT_THROW( stf::t_half(mywave, 0.0, 1.0, 1,
        long(PI/dt)-1, center, t50LeftId, t50RigthId, t50Real),
        std::out_of_range);

}

//=========================================================================
// test maximal slope of rise
//=========================================================================
TEST(measlib_test, maxrise_basic) {

    std::vector<double> data(32768);
    data[16385] = 1.0;

    double maxRiseT, maxRiseY;
    double maxrise = stf::maxRise(data, 1, data.size()-1, \
        maxRiseT, maxRiseY, 1);
    EXPECT_EQ(maxrise, 1.0);
    EXPECT_EQ(maxRiseT, 16385.5);
    EXPECT_EQ(maxRiseY, 0.5);

}

//=========================================================================
// test maximal slope of rise out of range exceptions
//=========================================================================
TEST(measlib_test, maxrise_out_of_range_exceptions) {

    std::vector<double> data(32768);
    double maxRiseT, maxRiseY;
    
    /* Out of range: peak cursor after last point */
    EXPECT_THROW(stf::maxRise(data, 0, data.size(), \
        maxRiseT, maxRiseY, 1), std::out_of_range);

    /* Out of range: peak cursor before first point */
    EXPECT_THROW(stf::maxRise(data, -1, data.size()-1, \
        maxRiseT, maxRiseY, 1), std::out_of_range);

}

//=========================================================================
// test maximal slope of rise windowLength exceptions 
//=========================================================================
TEST(measlib_test, maxrise_windowLength_exceptions){

    std::vector<double> data(32768);
    double maxRiseT, maxRiseY;
    long mywindowLength; /* fixed time interval (in sampling points) */

    /* Right peak cursor must be larger than windowLength */
    mywindowLength = 10;
    long myRightPeakCursor = mywindowLength-1;
    EXPECT_THROW(stf::maxRise(data, 0, myRightPeakCursor, \
        maxRiseT, maxRiseY, mywindowLength), std::out_of_range);

    /* Left peak cursor must be smaller than data.size()-windowLength */
    long myLeftPeakCursor = data.size()-mywindowLength; 
    EXPECT_THROW(stf::maxRise(data, myLeftPeakCursor, data.size()-1 , \
        maxRiseT, maxRiseY, mywindowLength), std::out_of_range);

    /* Data size itself must be smaller than windowLength */
    mywindowLength = data.size()+1;
    EXPECT_THROW(stf::maxRise(data, 0, data.size()-1, \
        maxRiseT, maxRiseY, mywindowLength), std::out_of_range);

}

//=========================================================================
// test maximal slope of rise with sine wave
//=========================================================================
TEST(measlib_test, maxrise_values) {

    /* sine wave between 0 and 3*PI */
    std::vector<double> mywave = sinwave( long(3*PI/dt) );
    double maxRiseT, maxRiseY;
    
    /* check max rise from peak to peak */
    int windowLength = 1;
    double maxrise = stf::maxRise(mywave, long((PI/2)/dt), 
        long((5*PI/2)/dt)-1, maxRiseT, maxRiseY, windowLength);

    /* Max slope of rise should be in 2*PI and give value 0 */
    double maxRiseT_xpkted = 2*PI/dt;
    EXPECT_NEAR(maxRiseY, 0 , 0.1);
    EXPECT_NEAR( maxRiseT, maxRiseT_xpkted, fabs(maxRiseT_xpkted*tol) );
}

//=========================================================================
// test maximal slope of decay
//=========================================================================
TEST(measlib_test, maxdecay_basic) {

    std::vector<double> data(32768);
    data[16385] = 1.0;

    double maxDecayT, maxDecayY;
    double maxdecay = stf::maxDecay(data, 0, data.size()-1, \
        maxDecayT, maxDecayY, 1);
    EXPECT_EQ(maxdecay, 1.0);
    EXPECT_EQ(maxDecayT, 16385.5);
    EXPECT_EQ(maxDecayY, 0.5);

}

//=========================================================================
// test maximal slope of decay out of range exceptions
//=========================================================================
TEST(measlib_test, maxdecay_out_of_range_exceptions) {

    std::vector<double> data(32768);
    double maxDecayT, maxDecayY;

    /* Out of range: peak cursor after last point */
    EXPECT_THROW(stf::maxRise(data, 0, data.size(), \
        maxDecayT, maxDecayY, 1), std::out_of_range);

    /* Out of range: peak cursor before first point */
    EXPECT_THROW(stf::maxRise(data, -1, data.size()-1, \
        maxDecayT, maxDecayY, 1), std::out_of_range);

}

//=========================================================================
// test maximal slope of decay windowLength exceptions
//=========================================================================
TEST(measlib_test, maxdecay_windowLength_exceptions) {
    
    std::vector<double> data(32768);
    double maxDecayT, maxDecayY;
    long mywindowLength; /* fixed time interval (in sampling points) */

    /* Right peak cursor must be larger than windowLength */
    mywindowLength = 10;
    long myRightPeakCursor = mywindowLength-1;
    EXPECT_THROW(stf::maxDecay(data, 0, myRightPeakCursor, \
        maxDecayT, maxDecayY, mywindowLength), std::out_of_range);

    /* Left peak cursor must be smaller than data.size()-windowLength */
    long myLeftPeakCursor = data.size()-mywindowLength; 
    EXPECT_THROW(stf::maxRise(data, myLeftPeakCursor, data.size()-1 , \
        maxDecayT, maxDecayY, mywindowLength), std::out_of_range);

    /* Data size itself must be smaller than windowLength */
    mywindowLength = data.size()+1;
    EXPECT_THROW(stf::maxRise(data, 0, data.size()-1, \
        maxDecayT, maxDecayY, mywindowLength), std::out_of_range);
}

//=========================================================================
// test maximal slope of decay with a sine wave 
//=========================================================================
TEST(measlib_test, maxdecay_values){

    /* a sine wave between 0 and 2*PI */
    std::vector<double> mywave = sinwave( long(2*PI/dt) );
    double maxDecayT, maxDecayY;
    
    int windowLength = 1; 
    /* compute max slope of decay between 0 and 3*PI/2 */
    long endCursor = (3*PI/2)/dt ;
    double maxdecay = stf::maxDecay(mywave, 1, endCursor, \
        maxDecayT, maxDecayY, windowLength);

    /* Max slope of decay should be in PI and give value 0 */
    EXPECT_NEAR(maxDecayY, 0 , 0.1);      
    double maxDecayT_xpkted = PI/dt;
    EXPECT_NEAR(maxDecayT, maxDecayT_xpkted, fabs(maxDecayT_xpkted*tol));

}

/*
TEST(measlib_test, checks) {
    std::vector<double> data(32768);
    double var = 0;
    EXPECT_EQ(stf::base(var, data, 0, data.size()-1), 0);
    EXPECT_EQ(var, 0);
    EXPECT_THROW(stf::base(var, data, 0, data.size()), std::out_of_range);

    data[16385] = 1.0;
    double maxT;
    double peak = stf::peak(data, 0.0, 0, data.size()-1, 1, stf::both, maxT);
    EXPECT_EQ(peak, 1.0);
    EXPECT_EQ(maxT, 16385);
    EXPECT_THROW(stf::peak(data, 0.0, 0, data.size(), 1, stf::both, maxT), std::out_of_range);

    double maxRiseT, maxRiseY;
    long windowLength = 1; // number of sampling points to calculate slopes
    double maxrise = stf::maxRise(data, 0, data.size()-1, maxRiseT, maxRiseY, windowLength);
    EXPECT_EQ(maxrise, 1.0);
    EXPECT_EQ(maxRiseT, 16385.5);
    EXPECT_EQ(maxRiseY, 0.5);
    EXPECT_THROW(stf::maxRise(data, 0, data.size(), maxRiseT, maxRiseY, windowLength),
                 std::out_of_range);
    EXPECT_THROW(stf::maxRise(data, data.size(), data.size()-1, maxRiseT, maxRiseY, windowLength),
                 std::out_of_range);

    double maxDecayT, maxDecayY;
    double maxdecay = stf::maxDecay(data, 0, data.size()-1, maxDecayT, maxDecayY, windowLength);
    EXPECT_EQ(maxdecay, 1.0);
    EXPECT_EQ(maxDecayT, 16384.5);
    EXPECT_EQ(maxDecayY, 0.5);
    EXPECT_THROW(stf::maxDecay(data, 0, data.size(), maxDecayT, maxDecayY, windowLength),
                 std::out_of_range);
    EXPECT_THROW(stf::maxDecay(data, data.size(), data.size()-1, maxDecayT, maxDecayY, windowLength),
                 std::out_of_range);

}
*/
