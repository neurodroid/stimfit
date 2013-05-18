#include "../stimfit/stf.h"
#include "../stimfit/math/measure.h"
#include <gtest/gtest.h>

//=========================================================================
// test baseline (base)
//=========================================================================
TEST(measlib_test, baseline) {

    std::vector<double> data(32768);
    double var = 0;

    EXPECT_EQ(stf::base(var, data, 0, data.size()-1), 0);
    EXPECT_EQ(var, 0);

    /* Check out exceptions */

    /* check out of range */
    EXPECT_THROW(stf::base(var, data, 0, data.size()),\
         std::out_of_range);

}

//=========================================================================
// test peak 
//=========================================================================
TEST(measlib_test, peak) {

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

    /* Check out exceptions */
    /* check out of range */
    EXPECT_THROW(stf::peak(data, 0.0, 0, data.size(), 
        1, stf::both, maxT), std::out_of_range);

}

//=========================================================================
// test threshold 
//=========================================================================
// TODO
//=========================================================================
// test rise time 
//=========================================================================
// TODO

//=========================================================================
// test half-width 
//=========================================================================
// TODO

//=========================================================================
// test maximal slope of rise
//=========================================================================
TEST(measlib_test, max_slope_rise) {

    std::vector<double> data(32768);
    data[16385] = 1.0;

    double maxRiseT, maxRiseY;
    double maxrise = stf::maxRise(data, 1, data.size()-1, \
        maxRiseT, maxRiseY, 1);
    EXPECT_EQ(maxrise, 1.0);
    EXPECT_EQ(maxRiseT, 16385.5);
    EXPECT_EQ(maxRiseY, 0.5);

    /* Check out exceptions */
    
    /* Out of range */
    EXPECT_THROW(stf::maxRise(data, 1, data.size(), \
        maxRiseT, maxRiseY, 1), std::out_of_range);

    EXPECT_THROW(stf::maxRise(data, data.size(), data.size()-1,\
        maxRiseT, maxRiseY, 1), std::out_of_range);

    /* Not possible to compute slope from the 1st sampling point */
    EXPECT_THROW(stf::maxRise(data, 0, data.size(), \
        maxRiseT, maxRiseY, 1), std::out_of_range);

}

//=========================================================================
// test maximal slope of decay
//=========================================================================
TEST(measlib_test, max_slope_decay) {

    std::vector<double> data(32768);
    data[16385] = 1.0;

    double maxDecayT, maxDecayY;
    double maxdecay = stf::maxDecay(data, 0, data.size()-1, \
        maxDecayT, maxDecayY, 1);
    EXPECT_EQ(maxdecay, 1.0);
    EXPECT_EQ(maxDecayT, 16385.5);
    EXPECT_EQ(maxDecayY, 0.5);

    /* Check out exceptions */
    /* check out of range */
    EXPECT_THROW(stf::maxDecay(data, 0, data.size(), \
        maxDecayT, maxDecayY, 1), std::out_of_range);
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
