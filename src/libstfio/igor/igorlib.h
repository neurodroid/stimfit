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


/*! \file igorlib.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-23
 *  \brief Export Igor binary waves.
 */

#ifndef _IGORLIB_H
#define _IGORLIB_H

#include "./../stfio.h"

class Recording;

namespace stfio {

//! Export a Recording to an Igor binary wave.
/*! \param fName Full path to the file to be written.
 *  \param WData The data to be exported.
 *  \return At present, always returns 0.
 */
bool
    exportIGORFile(const std::string& fName, const Recording& WData, ProgressInfo& progDlg);

}

#endif
