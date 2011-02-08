#include "../core/section.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Section test
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE( constructors_test )
{
    Section sec0;
    BOOST_CHECK_EQUAL( sec0.size(), 0 );

    Section sec1(Vector_double(32768, 0), "Test section");
    BOOST_CHECK_EQUAL( sec1.size(), 32768 );
    BOOST_CHECK_EQUAL( sec1[32767], 0 );
    BOOST_CHECK_EQUAL( sec1.GetSectionDescription(), "Test section" );
    BOOST_CHECK_THROW( sec1.at( sec1.size() ), std::out_of_range );

    Section sec2(32768, "Test section");
    BOOST_CHECK_EQUAL( sec2.size(), 32768 );
    BOOST_CHECK_EQUAL( sec2[32767], 0 );
    BOOST_CHECK_EQUAL( sec2.GetSectionDescription(), "Test section" );
    BOOST_CHECK_THROW( sec2.at( sec2.size() ), std::out_of_range );
}
