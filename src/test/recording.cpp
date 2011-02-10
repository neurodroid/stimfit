#include "../core/recording.h"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE( recording_suite )

BOOST_AUTO_TEST_CASE( constructors_test )
{
    Recording rec0;
    BOOST_CHECK_EQUAL( rec0.size(), 0 );

    std::vector<Section> sec_list(16, Section(32768));
    Channel ch(sec_list);

    Recording rec1(ch);
    BOOST_CHECK_EQUAL( rec1.size(), 1 );
    BOOST_CHECK_EQUAL( rec1[0].size(), 16 );
    BOOST_CHECK_EQUAL( rec1[0][0].size(), 32768 );

    std::vector<Channel> ch_list(4, Channel(16, 32768));
    Recording rec2(ch_list);
    BOOST_CHECK_EQUAL( rec2.size(), 4 );
    BOOST_CHECK_EQUAL( rec2[rec2.size()-1].size(), 16 );
    BOOST_CHECK_EQUAL( rec2[rec2.size()-1][rec2[rec2.size()-1].size()-1].size(), 32768 );

    Recording rec3(4, 16, 32768);
    BOOST_CHECK_EQUAL( rec3.size(), 4 );
    BOOST_CHECK_EQUAL( rec3[rec3.size()-1].size(), 16 );
    BOOST_CHECK_EQUAL( rec3[rec3.size()-1][rec3[rec3.size()-1].size()-1].size(), 32768 );
}

BOOST_AUTO_TEST_CASE( data_access_test )
{
    std::vector<Section> sec_list(16, Section(32768));
    Channel ch(sec_list);

    Recording rec1(ch);
    int chsize = rec1[0].size();
    int secsize = rec1[0][rec1[0].size()-1].size();
    BOOST_CHECK_EQUAL( rec1[0][chsize-1][secsize-1], 0 );
    BOOST_CHECK_THROW( rec1.at(1), std::out_of_range );
    BOOST_CHECK_THROW( rec1[0].at(chsize), std::out_of_range );
    BOOST_CHECK_THROW( rec1[0][chsize-1].at(secsize), std::out_of_range );

    std::vector<Channel> ch_list(4, Channel(16, 32768));
    Recording rec2(ch_list);
    int recsize = rec2.size();
    chsize = rec2[recsize-1].size();
    secsize = rec2[recsize-1][rec2[recsize-1].size()-1].size();
    BOOST_CHECK_EQUAL( rec2[recsize-1][chsize-1][secsize-1], 0 );
    BOOST_CHECK_THROW( rec2.at(recsize), std::out_of_range );
    BOOST_CHECK_THROW( rec2[recsize-1].at(chsize), std::out_of_range );
    BOOST_CHECK_THROW( rec2[recsize-1][chsize-1].at(secsize), std::out_of_range );

    Recording rec3(4, 16, 32768);
    recsize = rec3.size();
    chsize = rec3[recsize-1].size();
    secsize = rec3[recsize-1][rec3[recsize-1].size()-1].size();
    BOOST_CHECK_EQUAL( rec3[recsize-1][chsize-1][secsize-1], 0 );
    BOOST_CHECK_THROW( rec3.at(recsize), std::out_of_range );
    BOOST_CHECK_THROW( rec3[recsize-1].at(chsize), std::out_of_range );
    BOOST_CHECK_THROW( rec3[recsize-1][chsize-1].at(secsize), std::out_of_range );
}

BOOST_AUTO_TEST_SUITE_END()
