#include "../libstfio/stfio.h"
#include <gtest/gtest.h>

TEST(Channel_test, constructors)
{
    Channel ch0;
    EXPECT_EQ( ch0.size(), 0 );

    Section sec(32768);

    Channel ch1(sec);
    EXPECT_EQ( ch1.size(), 1 );
    EXPECT_EQ( ch1[0].size(), 32768 );

    std::vector<Section> sec_list(16, Section(32768));
    Channel ch2(sec_list);
    EXPECT_EQ( ch2.size(), 16 );
    EXPECT_EQ( ch2[ch2.size()-1].size(), 32768 );

    Channel ch3(16, 32768);
    EXPECT_EQ( ch3.size(), 16 );
    EXPECT_EQ( ch3[ch3.size()-1].size(), 32768 );
}

TEST(Channel_test, data_access)
{
    Channel ch1(Section(32768));
    EXPECT_EQ( ch1[0][ch1[0].size()-1], 0 );
    EXPECT_THROW( ch1.at( ch1.size() ), std::out_of_range );
    EXPECT_THROW( ch1[0].at(ch1[0].size()), std::out_of_range );

    std::vector<Section> sec_list(16, Section(32768));
    Channel ch2(sec_list);
    EXPECT_EQ( ch2[ch2.size()-1][ch2[ch2.size()-1].size()-1], 0 );
    EXPECT_THROW( ch2.at( ch2.size() ), std::out_of_range );
    EXPECT_THROW( ch2[ch2.size()-1].at(ch2[ch2.size()-1].size()), std::out_of_range );

    Channel ch3(16, 32768);
    EXPECT_EQ( ch3[ch3.size()-1][ch3[ch3.size()-1].size()-1], 0 );
    EXPECT_THROW( ch3.at( ch3.size() ), std::out_of_range );
    EXPECT_THROW( ch3[ch3.size()-1].at(ch3[ch3.size()-1].size()), std::out_of_range );
}
