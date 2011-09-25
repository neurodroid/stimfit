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

/*! \file stfio.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2011-09-25
 *  \brief header file for libstfio
 * 
 * 
 *  Header file for libstfio
 */

#ifndef _STFIO_H_
#define _STFIO_H_

#include <string>

#ifdef _MSC_VER
#pragma warning( disable : 4251 )  // Disable warning messages
#pragma warning( disable : 4996 )  // Disable warning messages
#endif

//! Defines dll export or import functions for Windows
#ifdef _WINDOWS
    #ifdef STFDLL
        #define StfDll __declspec( dllexport )
    #else
        #define StfDll __declspec( dllimport )
    #endif
#else
    #define StfDll
#endif

//! The stimfit namespace.
/*! All essential core functions and classes are in this namespace. 
 *  Its purpose is to reduce name mangling problems.
 */
namespace stfio {

/*! \addtogroup stfio
 *  @{
 */

//! Abstract ProgressInfo class
/*! Abstract class to be used as an interface for the file io read/write functions
 *  Can be a GUI Dialog or stdout messages
 */
 class ProgressInfo {
     //! Constructor
     /*! \param title Dialog title
      *  \param message Message displayed
      *  \param maximum Maximum value for the progress meter
      */
     ProgressInfo(const std::string& title, const std::string& message, int maximum);

     //! Updates the progress info
     /*! \param value New value of the progress meter
      *  \param newmsg New message for the info text
      *  \param skip This is set to true if the user has chosen to skip the operation
      *  \return True unless the operation was cancelled.
      */
     virtual bool Update(int value, const std::string& newmsg="", bool* skip=NULL) = 0;
 };

/*@}*/

} // end of namespace

#endif

