#include "../stimfit/stf.h"
#include "../stimfit/math/measure.h"
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#define PI  3.14159265f
#define N_MAX 10000

const double tol = 0.01; /* tolerance value */
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
// is the result of calculating arcsin(0.8)-arcsin(0.2) and the half-with
// is two times the value of arcsin(1)-arcsin(0.5) 
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
// sine wave as function of amplitude and lambda
//=========================================================================
std::vector<double> sinwave(double amp, double lambda, long length){
    std::vector<double> mydata(length);

    for(std::vector<int>::size_type x=0; x != mydata.size() ; ++x){
        mydata[x] = amp*sin( 2*PI*x*dt/lambda ); /* see sampling interval */
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

std::vector<double> expwave(double tau, long length){
    
    std::vector<double> mydata(length);

    for(std::vector<int>::size_type x=0; x != mydata.size() ; ++x){
        mydata[x] = exp( x*dt/tau ); /* see sampling interval */
    }

    return mydata;
}
//=========================================================================
// A vector with random numbers between 0 and 1
//=========================================================================
std::vector<double> rand(long size){
    /* seed the random number generator */
    int seed = time(NULL);
    srand(seed);

    std::vector<double> myrand(size);
    for (int i=0; i<size;i++){
        myrand[i] = (double) rand()/(double) RAND_MAX;
    }
    return myrand;
    
}

//=========================================================================
// A vector of size N_MAX with random numbers from a normal distribution
//=========================================================================
std::vector<double> norm(double mean, double stddev){


    boost::mt19937 rng; /* seed? */
    boost::normal_distribution<> norm(mean, stddev);
    boost::variate_generator<boost::mt19937&,
        boost::normal_distribution<> > rand_val(rng, norm);
    
    std::vector<double> myrand(N_MAX);

    for (int i=0; i<N_MAX; ++i){
        myrand[i] = rand_val();
    }

    return myrand;
    
}

//=========================================================================
// test baseline random 
//=========================================================================
TEST(measlib_test, baseline_random) {

    double var;

    std::vector<double> mybase(N_MAX);

    for (int i=0; i<N_MAX; i++){
        std::vector<double> myrand = rand(N_MAX); 
        mybase[i] = stf::base(var, myrand, 0, N_MAX-1);
        EXPECT_NEAR(mybase[i], 1/2., 0.05); /* expected mean = 1/2   */
        EXPECT_NEAR(var, 1/12., (1/12.)*0.1); /* expected var = 1/12 */
    }

    //save_txt("base.out", mybase);
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
// Peak random
//=========================================================================
TEST(measlib_test, peak_random) {
    double maxT;
    
    std::vector<double> mypeak(N_MAX);
    std::vector<double> myrand = rand(N_MAX);

    for (int i=0; i<10; i++){
        /* A*sin(2*PI*x/lambda) */
        std::vector<double> mywave = sinwave(myrand[i], long(2*PI),long(2*PI/dt) ); 
        mypeak[i] = stf::peak(mywave, 0.0, 0, long(2*PI/dt)-1,
            1, stf::up, maxT);
        EXPECT_NEAR(mypeak[i], myrand[i], fabs(myrand[i]*tol));
    }

    save_txt("peaks.out", mypeak);
    
    
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
    EXPECT_NEAR(threshold*1, slope, fabs(slope*tol)); 

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



//=========================================================================
// test baseline N_MAX random traces
//=========================================================================
TEST(measlib_validation, baseline) {
    double var;
    double tol = 0.1; /* for this case only, to account for the variance */

    /* measurement results for base */
    std::vector<double> mybase(N_MAX);
    /* random values from a normal dist. */
    std::vector<double> myrand = norm(0, 1); 

    /* we check the measurement N_MAX times */
    for (int i=0; i<N_MAX; i++){
        double mean = myrand[i];
        double stddev = fabs(myrand[i]*2.5);
        /* the dataset is a normal distribution */
        std::vector<double> mytrace = norm(mean, stddev);
        /* calculate base between start and end */
        mybase[i] = stf::base(var, mytrace, 0, mytrace.size()-1);
        EXPECT_NEAR(mybase[i], myrand[i], fabs(myrand[i]*tol));
        EXPECT_NEAR(std::sqrt(var),stddev, stddev*tol );
    }

    save_txt("base.val", mybase);
}

//=========================================================================
// test peak N_MAX random traces
//=========================================================================
TEST(measlib_validation, peak) {
    double maxT;

    /* measurement results for peak */
    std::vector<double> mypeak(N_MAX);
    /* random values from a normal dist. */
    std::vector<double> myrand = norm(10, 1); 


    /* we check the measurement N_MAX times */
    for (int i=0; i<N_MAX; i++){
        double peak = myrand[i];
        /* the dataset is a sine wave with random amplitude */
        std::vector<double> mytrace = sinwave(peak, 9.5,  long(9.5/dt));
        /* calculate peak between start and end */
        mypeak[i] = stf::peak(mytrace, 0.0, 0, mytrace.size()-1,
            1, stf::up, maxT);
        EXPECT_NEAR(mypeak[i], myrand[i], fabs(myrand[i]*tol));
    }

    save_txt("peak.val", mypeak);
}

//=========================================================================
// test risetime N_MAX random traces
//=========================================================================
TEST(measlib_validation, risetime) {
    std::size_t t20, t80;
    double t20Real;

    /* measurement results for risetime */
    std::vector<double> myrisetime(N_MAX);
    /* N_MAX random values from a normal dist. */
    std::vector<double> myrand = norm(20., 2.); 

    /* we check the measurement N_MAX times */
    for (int i=0; i<N_MAX; i++){
        double lambda = myrand[i];
        /* the dataset is a sine wave with random wavelength */
        std::vector<double> mytrace = sinwave(1.0, lambda, long(lambda/dt));
        /* calculate risetime between start and peak (lambda/4) */
        myrisetime[i] = stf::risetime(mytrace, 0.0, 1.0, 1, 
            long((lambda/4)/dt), 0.2, t20, t80, t20Real);
        double l = 2*PI/lambda;
        double risetime_xpted = (std::asin(.8)-std::asin(.2))/l;
        EXPECT_NEAR(myrisetime[i]*dt, risetime_xpted, 
            fabs(risetime_xpted*tol));
        myrisetime[i] *=dt; /* to save real values in a file */
    }

    save_txt("risetime.val", myrisetime);
}


//=========================================================================
// test half_t N_MAX random traces
//=========================================================================
TEST(measlib_validation, half_duration) {
    std::size_t t50LeftId, t50RigthId;
    double t50Real;

    /* measurement results for risetime */
    std::vector<double> myhalf_width(N_MAX);
    /* N_MAX random values from a normal dist. */
    std::vector<double> myrand = norm(20., 2.); 

    /* we check the measurement N_MAX times */
    for (int i=0; i<N_MAX; i++){
        double lambda = myrand[i];
        /* the dataset is a sine wave with random wavelength */
        std::vector<double> mytrace = sinwave(1.0, lambda, long(lambda/dt));
        /* calculate half width starting form start and entering peak (lambda/4) */
        myhalf_width[i] = stf::t_half(mytrace, 0.0, 1.0, 1, 
            long(lambda/dt), long((lambda/4)/dt), t50LeftId, t50RigthId, t50Real);
        double l = 2*PI/lambda;
        double half_width_xpted = 2*(std::asin(1.)-std::asin(.5))/l;
        EXPECT_NEAR(myhalf_width[i]*dt, half_width_xpted, 
            fabs(half_width_xpted*tol));
    }

    save_txt("half_width.val", myhalf_width);
}

//=========================================================================
// test slope_rise N_MAX random traces
//=========================================================================
TEST(measlib_validation, maxrise) {
    double maxRiseT, maxRiseY;

    /* measurement results for maxrise */
    std::vector<double> mymaxrise(N_MAX);
    /* N_MAX random values from a normal dist. */
    std::vector<double> myrand = norm(10., 2.); 

    /* we check measurements N_MAX times */
    for (int i=0; i<N_MAX; i++){
        double lambda = myrand[i];
        /* the dataset is a sine wave with random wavelength */
        std::vector<double> mytrace = sinwave(1.0, lambda, long(2*lambda/dt));
        /* calculate half width cursors .25 around lambda */
        mymaxrise[i] = stf::maxRise(mytrace, long(0.75*lambda/dt), 
            long(1.25*lambda/dt), maxRiseT, maxRiseY, 1);
        double maxRiseT_xpted = lambda;
        EXPECT_NEAR(maxRiseT*dt, maxRiseT_xpted, 
            fabs(maxRiseT_xpted*tol));
        mymaxrise[i] *=dt; /* to save reeal values in a file */
    }

    save_txt("max_rise.val", mymaxrise);
}

//=========================================================================
// test threshold N_MAX random traces
//=========================================================================
TEST(measlib_validation, threshold) {
    double thrT;
    
    /* measurements results for threshold */
    std::vector<double> mythreshold(N_MAX);

    /* N_MAX random values from a normal dist. */
    std::vector<double> myrand = norm(10., 2.); 

    /* fix a slope and to look for it in different traces */
    /* this could  be any value between 1 and e (2.718281...) */
    const double myslope = 2.0;
    double tau = 10.0;/*random */

    /* we check measurements N_MAX times */
    for (int i=0; i<N_MAX; i++){
        double tau = myrand[i];
        /* the dataset is an exponential with random tau */
        std::vector<double> mytrace = expwave(tau, 5*long(tau/dt));
        /* calculate thresholds */
        mythreshold[i] = stf::threshold(mytrace, 1, mytrace.size()-1,
            myslope*dt, thrT, 1);
        
        /* Threshold is the slope value times tau */
        double thr_xpted = myslope*tau; 
        EXPECT_NEAR(mythreshold[i], thr_xpted, fabs(thr_xpted*tol)); 
        /* sanity check */
        /* The differential of the exponential function is e^(x/tau)/tau
        at x=thrT should give us the slope that we used as threshold */
        double slope_xpted = std::exp(thrT*dt/tau)/tau;
    }

    save_txt("threshold.val", mythreshold);
    

}
