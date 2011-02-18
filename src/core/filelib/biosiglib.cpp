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
#include <iomanip>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#ifndef MODULE_ONLY
#include <wx/wx.h>
#include <wx/progdlg.h>
#else
#include <iostream>
#endif
#include <biosig.h>

#include "./../core.h"
#include "./biosiglib.h"

class BiosigHDR {
  public:
    BiosigHDR(unsigned int NS, unsigned int N_EVENT) {
        pHDR = constructHDR(NS, N_EVENT);
    }
    ~BiosigHDR() {
        destructHDR(pHDR);
    }

  private:
    HDRTYPE* pHDR;
};

void stf::importBSFile(const wxString &fName, Recording &ReturnData, bool progress, wxWindow* parent) {
#ifndef MODULE_ONLY
    wxProgressDialog progDlg( wxT("Biosig binary file import"), wxT("Starting file import"),
                              100, parent, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL | wxPD_CAN_SKIP );
#endif
    std::string errorMsg("Exception while calling std::importBSFile():\n");
    std::string yunits;
    // =====================================================================================================================
    //
    // Open an AxoGraph file and read in the data
    //
    // =====================================================================================================================
    
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
    HDRTYPE* hdr =  sopen( fName.c_str(), "r", NULL );
#else
    HDRTYPE* hdr =  sopen( fName.mb_str(), "r", NULL );
#endif
    if (hdr==NULL) {
        errorMsg += "\nBiosig header is empty";
        ReturnData.resize(0);
	sclose(hdr);
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }
    std::cout << "Number of channels: " << hdr->NS << std::endl;
    std::cout << "Data size: " << hdr->data.size[0] << "x" << hdr->data.size[1] << std::endl;
    std::cout << "Sampling rate: " << hdr->SampleRate << std::endl;

    // TODO: Read file data into ReturnData

    sclose(hdr);
    destructHDR(hdr);
}
