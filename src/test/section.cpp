#include "../libstfio/stfio.h"
#include <gtest/gtest.h>

TEST(Section_test, constructors) {
    Section sec0;
    EXPECT_EQ( sec0.size(), 0 );

    Section sec1(Vector_double(32768, 0), "Test section");
    EXPECT_EQ( sec1.size(), 32768 );

    Section sec2(32768, "Test section");
    EXPECT_EQ( sec2.size(), 32768 );
}

TEST(Section_test, data_access) {

    Section sec1(Vector_double(32768, 0), "Test section");
    EXPECT_EQ( sec1[sec1.size()-1], 0 );
    EXPECT_THROW( sec1.at( sec1.size() ), std::out_of_range );

    Section sec2(32768, "Test section");
    EXPECT_EQ( sec2[sec2.size()-1], 0 );
    EXPECT_THROW( sec2.at( sec2.size() ), std::out_of_range );
}
