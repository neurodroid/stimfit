#include "section.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Section test
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE( constructors_test )
{
    Section sec0;                                                 // 1 //
    BOOST_CHECK_EQUAL( sec0.size(), 0 );
}
