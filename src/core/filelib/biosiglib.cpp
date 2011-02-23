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
    size_t blks = sread(NULL, 0, hdr->NRec*hdr->SPR, hdr);

#ifdef _STFDEBUG
    std::cout << "Number of channels: " << hdr->NS << std::endl;
    std::cout << "Number of records per channel: " << hdr->NRec << std::endl;
    std::cout << "Number of samples per record: " << hdr->SPR << std::endl;
    std::cout << "Data size: " << hdr->data.size[0] << "x" << hdr->data.size[1] << std::endl;
    std::cout << "Sampling rate: " << hdr->SampleRate << std::endl;
    std::cout << "Number of events: " << hdr->EVENT.N << std::endl;
    int	res = hdr2ascii(hdr, stdout, 3);
#endif

    int nchannels = hdr->NS;
    for (int nc=0; nc<nchannels; ++nc) {
        int nsections = 0; // TODO: hdr->nsections[nc];

        Channel TempChannel(nsections);
        TempChannel.SetChannelName(""); // TODO: hdr->channelname[nc];
        TempChannel.SetYUnits(""); // TODO: hdr->yunits[nc];
        
        for (int ns=0; ns<nsections; ++ns) {
            if (progress) {
                int progbar =
                    // Channel contribution:
                    (int)(((double)nc/(double)nchannels)*100.0+
                          // Section contribution:
                          (double)ns/(double)nsections*(100.0/nchannels));
#ifndef MODULE_ONLY
                wxString progStr;
                progStr << wxT("Reading channel #") << nc + 1 << wxT(" of ") << nchannels
                        << wxT(", Section #") << ns + 1 << wxT(" of ") << nsections;
                progDlg.Update(progbar, progStr);
#else
                std::cout << "\r";
                std::cout << progbar << "%" << std::flush;
#endif
            }
            Section TempSection(
                                0, // TODO: hdr->nsamplingpoints[nc][ns]
                                "" // TODO: hdr->sectionname[nc][ns]
            );
            // TODO: std::copy(&buffer[0], &buffer[TempSection.size()], TempSection.get_w().begin());
            try {
                TempChannel.InsertSection(TempSection, ns);
            }
            catch (...) {
                ReturnData.resize(0);
                sclose(hdr);
                destructHDR(hdr);
                throw;
            }
        }
        try {
            if ((int)ReturnData.size() < nchannels) {
                ReturnData.resize(nchannels);
            }
            ReturnData.InsertChannel(TempChannel, nc);
        }
        catch (...) {
            ReturnData.resize(0);
            sclose(hdr);
            destructHDR(hdr);
            throw;
        }
    }
    ReturnData.SetXScale(hdr->SampleRate);
    ReturnData.SetComment(""); // TODO: hdr->comment
    ReturnData.SetDate(""); // TODO: hdr->datestring
    ReturnData.SetTime(""); // TODO: hdr->timestring

#ifdef MODULE_ONLY
    if (progress) {
        std::cout << "\r";
        std::cout << "100%" << std::endl;
    }
#endif

    sclose(hdr);
    destructHDR(hdr);
}
