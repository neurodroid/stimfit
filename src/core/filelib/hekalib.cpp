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

#include <string>
#include <cstdio>
#include <iomanip>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <wx/wx.h>
#include <wx/progdlg.h>

#include "./../core.h"
#include "./hekalib.h"

void stf::importHEKAFile(const wxString &fName, Recording &ReturnData, bool progress) {
    wxProgressDialog progDlg( wxT("HEKA binary file import"), wxT("Starting file import"),
                              100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );
    wxString errorMsg(wxT("Exception while calling importHEKAFile():\n"));
    wxString yunits;

    // Open file
    FILE* pgf_fh = fopen(fName.utf8_str(), "r");
    std::cout << pgf_fh << std::endl;
    
    // Close file
    fclose(pgf_fh);
}
