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

/*! \file asciilib.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-23
 *  \brief Import and export plain text files.
 */

#ifndef _ASCIILIB_H
#define _ASCIILIB_H

#include "../stfio.h"
#include "../recording.h"

namespace stfio {

//! Open an ASCII file and store its contents to a Recording object.
/*! \param fName Full path to the file to be read.
 *  \param hLinesToSkip Header lines to skip.
 *  \param nColumns Number of columns.
 *  \param firstIsTime true if the first column contains time values, false otherwise.
 *  \param toSection true if the columns should be put into different sections,
 *         false if they should be put into different channels.
 *  \param ReturnRec On entry, an empty Recording object. On exit,
 *         the data stored in \e fName.
 *  \param progress True if the progress dialog should be updated.
 */
void importASCIIFile(const std::string& fName,
        int hLinesToSkip,
        int nColumns,
        bool firstIsTime,
        bool toSection,
        Recording& ReturnRec,
        ProgressInfo& progDlg);

//! Export a Section to a text file.
/*! \param fName Full path to the file to be written.
 *  \param Export The section to be exported.
 *  \return true upon success, false otherwise.
 */
bool exportASCIIFile(const std::string& fName, const Section& Export);

//! Export a Channel to a text file.
/*! \param fName Full path to the file to be written.
 *  \param Export The channel to be exported.
 *  \return true upon success, false otherwise.
 */
bool exportASCIIFile(const std::string& fName, const Channel& Export);
 
#if 0
std::string NextWord( std::string& str );
#endif

}

#endif
