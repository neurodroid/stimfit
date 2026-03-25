// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

/*! \file core.h
 *  \author Stimfit contributors
 *  \brief Lightweight core definitions shared by libstfio model classes.
 */

#ifndef _STFIO_CORE_H_
#define _STFIO_CORE_H_

#include <cmath>
#include <cfloat>
#include <cstddef>
#include <deque>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#if (__cplusplus < 201103)
#  include <boost/function.hpp>
#else
#  include <algorithm>
#  include <functional>
#endif

#ifdef _MSC_VER
#pragma warning( disable : 4251 )
#pragma warning( disable : 4996 )
#endif

//! Defines dll export or import functions for Windows
#if defined(_WINDOWS) && !defined(__MINGW32__)
    #ifdef STFIODLL
        #define StfioDll __declspec( dllexport )
    #else
        #define StfioDll __declspec( dllimport )
    #endif
#else
    #define StfioDll
#endif

typedef std::vector<double > Vector_double;
typedef std::vector<float > Vector_float;

#ifdef _MSC_VER
    #ifndef NAN
        static const unsigned long __nan[2] = {0xffffffff, 0x7fffffff};
        #define NAN (*(const float *) __nan)
    #endif
    #ifndef INFINITY
        #define INFINITY (DBL_MAX+DBL_MAX)
    #endif
    #define snprintf _snprintf
#endif

#endif
