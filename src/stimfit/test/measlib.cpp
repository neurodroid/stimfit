#include "../core/stimdefs.h"
#include "../core/measlib.h"
#include <gtest/gtest.h>

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
    double maxrise = stf::maxRise(data, 0, data.size()-1, maxRiseT, maxRiseY);
    EXPECT_EQ(maxrise, 1.0);
    EXPECT_EQ(maxRiseT, 16385.5);
    EXPECT_EQ(maxRiseY, 0.5);
    EXPECT_THROW(stf::maxRise(data, 0, data.size(), maxRiseT, maxRiseY),
                 std::out_of_range);
    EXPECT_THROW(stf::maxRise(data, data.size(), data.size()-1, maxRiseT, maxRiseY),
                 std::out_of_range);

    double maxDecayT, maxDecayY;
    double maxdecay = stf::maxDecay(data, 0, data.size()-1, maxDecayT, maxDecayY);
    EXPECT_EQ(maxdecay, 1.0);
    EXPECT_EQ(maxDecayT, 16384.5);
    EXPECT_EQ(maxDecayY, 0.5);
    EXPECT_THROW(stf::maxDecay(data, 0, data.size(), maxDecayT, maxDecayY),
                 std::out_of_range);
    EXPECT_THROW(stf::maxDecay(data, data.size(), data.size()-1, maxDecayT, maxDecayY),
                 std::out_of_range);

}
