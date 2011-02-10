#include "../core/channel.h"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE( channel_suite )

BOOST_AUTO_TEST_CASE( constructors_test )
{
    Channel ch0;
    BOOST_CHECK_EQUAL( ch0.size(), 0 );

    Section sec(32768);

    Channel ch1(sec);
    BOOST_CHECK_EQUAL( ch1.size(), 1 );
    BOOST_CHECK_EQUAL( ch1[0].size(), 32768 );

    std::vector<Section> sec_list(16, Section(32768));
    Channel ch2(sec_list);
    BOOST_CHECK_EQUAL( ch2.size(), 16 );
    BOOST_CHECK_EQUAL( ch2[ch2.size()-1].size(), 32768 );

    Channel ch3(16, 32768);
    BOOST_CHECK_EQUAL( ch3.size(), 16 );
    BOOST_CHECK_EQUAL( ch3[ch3.size()-1].size(), 32768 );
}

BOOST_AUTO_TEST_CASE( data_access_test )
{
    Channel ch1(Section(32768));
    BOOST_CHECK_EQUAL( ch1[0][ch1[0].size()-1], 0 );
    BOOST_CHECK_THROW( ch1.at( ch1.size() ), std::out_of_range );
    BOOST_CHECK_THROW( ch1[0].at(ch1[0].size()), std::out_of_range );

    std::vector<Section> sec_list(16, Section(32768));
    Channel ch2(sec_list);
    BOOST_CHECK_EQUAL( ch2[ch2.size()-1][ch2[ch2.size()-1].size()-1], 0 );
    BOOST_CHECK_THROW( ch2.at( ch2.size() ), std::out_of_range );
    BOOST_CHECK_THROW( ch2[ch2.size()-1].at(ch2[ch2.size()-1].size()), std::out_of_range );

    Channel ch3(16, 32768);
    BOOST_CHECK_EQUAL( ch3[ch3.size()-1][ch3[ch3.size()-1].size()-1], 0 );
    BOOST_CHECK_THROW( ch3.at( ch3.size() ), std::out_of_range );
    BOOST_CHECK_THROW( ch3[ch3.size()-1].at(ch3[ch3.size()-1].size()), std::out_of_range );
}

BOOST_AUTO_TEST_SUITE_END()
