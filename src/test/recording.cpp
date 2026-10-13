#include "../libstfio/recording.h"
#include <cstddef>
#include <gtest/gtest.h>

TEST(Recording_test, constructors)
{
    Recording rec0;
    EXPECT_EQ( rec0.size(), std::size_t{0} );

    std::deque<Section> sec_list(16, Section(32768));
    Channel ch(sec_list);

    Recording rec1(ch);
    EXPECT_EQ( rec1.size(), std::size_t{1} );
    EXPECT_EQ( rec1[0].size(), std::size_t{16} );
    EXPECT_EQ( rec1[0][0].size(), std::size_t{32768} );

    std::deque<Channel> ch_list(4, Channel(16, 32768));
    Recording rec2(ch_list);
    EXPECT_EQ( rec2.size(), std::size_t{4} );
    EXPECT_EQ( rec2[rec2.size()-1].size(), std::size_t{16} );
    EXPECT_EQ( rec2[rec2.size()-1][rec2[rec2.size()-1].size()-1].size(), std::size_t{32768} );

    Recording rec3(4, 16, 32768);
    EXPECT_EQ( rec3.size(), std::size_t{4} );
    EXPECT_EQ( rec3[rec3.size()-1].size(), std::size_t{16} );
    EXPECT_EQ( rec3[rec3.size()-1][rec3[rec3.size()-1].size()-1].size(), std::size_t{32768} );
}

TEST(Recording_test, data_access)
{
    std::deque<Section> sec_list(16, Section(32768));
    Channel ch(sec_list);

    Recording rec1(ch);
    std::size_t chsize = rec1[0].size();
    std::size_t secsize = rec1[0][rec1[0].size()-1].size();
    EXPECT_EQ( rec1[0][chsize-1][secsize-1], 0 );
    EXPECT_THROW( rec1.at(1), std::out_of_range );
    EXPECT_THROW( rec1[0].at(chsize), std::out_of_range );
    EXPECT_THROW( rec1[0][chsize-1].at(secsize), std::out_of_range );

    std::deque<Channel> ch_list(4, Channel(16, 32768));
    Recording rec2(ch_list);
    std::size_t recsize = rec2.size();
    chsize = rec2[recsize-1].size();
    secsize = rec2[recsize-1][rec2[recsize-1].size()-1].size();
    EXPECT_EQ( rec2[recsize-1][chsize-1][secsize-1], 0 );
    EXPECT_THROW( rec2.at(recsize), std::out_of_range );
    EXPECT_THROW( rec2[recsize-1].at(chsize), std::out_of_range );
    EXPECT_THROW( rec2[recsize-1][chsize-1].at(secsize), std::out_of_range );

    Recording rec3(4, 16, 32768);
    recsize = rec3.size();
    chsize = rec3[recsize-1].size();
    secsize = rec3[recsize-1][rec3[recsize-1].size()-1].size();
    EXPECT_EQ( rec3[recsize-1][chsize-1][secsize-1], 0 );
    EXPECT_THROW( rec3.at(recsize), std::out_of_range );
    EXPECT_THROW( rec3[recsize-1].at(chsize), std::out_of_range );
    EXPECT_THROW( rec3[recsize-1][chsize-1].at(secsize), std::out_of_range );
}
