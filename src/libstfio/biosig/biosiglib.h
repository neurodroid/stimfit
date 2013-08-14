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

/*! \file biosiglib.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2011-02-18
 *  \brief Use biosig to read files
 */

#ifndef _BIOSIGLIB_H
#define _BIOSIGLIB_H

#include "../stfio.h"

namespace stfio {

//! Open an BIOSIG file and store its contents to a Recording object.
/*! \param fName The full path to the file to be opened.
 *  \param ReturnData On entry, an empty Recording object. On exit,
 *         the data stored in \e fName.
 *  \param progress True if the progress dialog should be updated.
 */
    void importBSFile(const std::string& fName, Recording& ReturnData, ProgressInfo& progDlg);

//! Export a Recording to a GDF file using biosig.
/*! \param fName Full path to the file to be written.
 *  \param WData The data to be exported.
 */
StfioDll bool exportBiosigFile(const std::string& fName, const Recording& WData, ProgressInfo& progDlg);


}

#endif
