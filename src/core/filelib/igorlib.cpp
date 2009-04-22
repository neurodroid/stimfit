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

// Export Igor binary waves from stimfit
// last revision: 2007-05-07
// CSH, University of Freiburg

// Most of this was shamelessly copied from Wavemetrics' sample code.
// Blame them for bugs.

#include "wx/wxprec.h"
#include "wx/progdlg.h"

/*	The code in this file writes a sample Igor Pro packed experiment file.

	See Igor Pro Tech Note PTN#003 for details.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>					// For offsetof macro.

#include "./../core.h"

#include "./IGORlib.h"

// Headers taken from Wavemetrics' demo files:
#ifdef __cplusplus
extern "C" {
#endif

#include "./igor/IgorBin.h"
#include "./igor/CrossPlatformFileIO.h"

    int WriteVersion5NumericWave(CP_FILE_REF fr, WaveHeader5* whp, const void* data, const char* waveNote, long noteSize);

#ifdef __cplusplus
}
#endif

namespace stf {

wxString IGORError(
        const wxString& msg,
        int nError
);

// Check compatibility before exporting:
bool CheckComp(const Recording& ReturnData);

}

wxString
stf::IGORError(const wxString& msg, int error)
{
    wxString ret;
    ret << wxT("Error # ") << error << wxT(" while writing Igor packed experiment:\n")
    << msg;
    return ret;
}

bool
stf::CheckComp(const Recording& Data) {
    std::size_t oldSize=0;
    if (!Data.get().empty() && !Data[0].get().empty()) {
        oldSize=Data[0][0].size();
    } else {
        return false;
    }
    for (std::size_t n_c=0;n_c<Data.size();++n_c) {
        for (std::size_t n_s=0;n_s<Data[n_c].size();++n_s) {
            if (Data[n_c][n_s].size()!=oldSize) {
                return false;
            }
        }
    }
    return true;
}

bool
stf::exportIGORFile(const wxString& fileBase,const Recording& Data)
{
    // Check compatibility:
    if (!CheckComp(Data)) {
        throw std::runtime_error(
                "File can't be exported:\n"
                "Traces have different sizes"
        );
    }
    wxProgressDialog progDlg( wxT("Igor binary wave export"),
            wxT("Starting file export"), 100, NULL,
            wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );
    // Get unambiguous channel names:
    std::vector<wxString> channel_name(Data.size());
    bool ident=false;
    for (std::size_t n_c=0;n_c<Data.size()-1 && !ident; ++n_c) {
        for (std::size_t n_c2=n_c+1;n_c2<Data.size()&& !ident; ++n_c2) {
            if (Data[n_c].GetChannelName()==Data[n_c2].GetChannelName()) {
                ident=true;
            }
        }
        if (!ident) channel_name[n_c]=Data[n_c].GetChannelName();
    }
    if (ident) {
        for (std::size_t n_c=0;n_c<Data.size()-1; ++n_c) {
            channel_name[n_c] << wxT("Ch") << (int)n_c;
        }
    } else {
        channel_name[Data.size()-1]=Data[Data.size()-1].GetChannelName();
    }

    // Export channels individually:
    for (std::size_t n_c=0;n_c<Data.size();++n_c) {
        unsigned long now;
#ifdef WIN32
        now = 0;			// It would be possible to write a Windows equivalent for the Macintosh GetDateTime function but it is not easy.
#else
        GetDateTime(&now);
#endif

        WaveHeader5 wh;
        memset(&wh, 0, sizeof(wh));
        wh.type = NT_FP64;							// double precision floating point.
		strcpy(wh.bname, channel_name[n_c].char_str());
        strcpy(wh.dataUnits, Data[n_c].GetYUnits().char_str());
        strcpy(wh.dimUnits[0], "ms");
        wh.npnts = (long)(Data[n_c][0].size()*Data[n_c].size());
        wh.nDim[0] = (long)Data[n_c][0].size();
        wh.nDim[1] = (long)Data[n_c].size();
        wh.sfA[0] = Data.GetXScale();
        wh.sfB[0] = 0.0e0;								// Starting from zero.
        wh.modDate = now;

        // Add a wave note:
        std::string waveNote("Wave exported from Stimfit");

        // Create a file:
        int err;
        wxString filePath;
        filePath << fileBase << wxT("_") << channel_name[n_c] << wxT(".ibw");
        if (err = CPCreateFile(filePath.char_str(), 1, 'IGR0', 'IGBW')) {
			throw std::runtime_error( std::string(IGORError(wxT("Error in CPCreateFile()\n"), err).char_str()) );
        }

        // Open the file:
        CP_FILE_REF fr;
        if (err = CPOpenFile(filePath.char_str(), 1, &fr)) {
			throw std::runtime_error( std::string(IGORError(wxT("Error in CPOpenFile()\n"), err).char_str()) );
        }

        // Write the data:
        std::vector<double> cpData(wh.npnts);

        // One unnecessary copy operation due to const-correctness (couldn't const_cast<>)
        Channel TempChannel(Data[n_c]);
        for (std::size_t n_s=0;n_s<Data[n_c].size();++n_s) {
            wxString progStr;
            progStr << wxT("Writing channel #") << (int)n_c + 1 << wxT(" of ") << (int)Data.size()
            << wxT(", Section #") << (int)n_s+1 << wxT(" of ") << (int)Data[n_c].size();
            progDlg.Update(
                    // Channel contribution:
                    (int)(((double)n_c/(double)Data.size())*100.0+
                            // Section contribution:
                            (double)(n_s)/(double)Data[n_c].size()*(100.0/Data.size())),
                            progStr
            );
            // std::copy is faster than explicitly assigning to cpData[c][s][p]
            std::copy( &TempChannel[n_s][0], &TempChannel[n_s][Data[n_c][n_s].size()],
                    &cpData[n_s*wh.nDim[0]] );
        }

        if ( err=WriteVersion5NumericWave( fr, &wh, &cpData[0], waveNote.c_str(),
                (long)strlen(waveNote.c_str()) ) )
        {
            throw std::runtime_error( std::string(IGORError("Error in WriteVersion5NumericWave()\n", err).char_str()) );
        }
        CPCloseFile(fr);
    }
    return true;
}
