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

/*! \file cfslib.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-23
 *  \brief Import from and export to the CED filing system.
 */

#ifndef _CFSLIB_H
#define _CFSLIB_H

#include "../core.h"

namespace stf {

//! Open a CFS file and store its contents to a Recording object.
/*! \param fName Full path to the file to be read.
 *  \param ReturnData On entry, an empty Recording object. On exit,
 *         the data stored in \e fName.
 *  \param progress Set tu true if a progress dialog should be shown.
 */
void importCFSFile(const wxString& fName, Recording& ReturnData, bool progress = true );

//! Export a Recording to a CFS file.
/*! \param fName Full path to the file to be written.
 *  \param WData The data to be exported.
 *  \return The CFS file handle.
 */
int exportCFSFile(const wxString& fName, const Recording& WData);

}

#endif
