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
#include "./../core.h"
#include "wx/wx.h"
#include "wx/progdlg.h"

#if defined(__UNIX__) || defined(__STF__)
#include "./axon/Common/axodefn.h"
#include "./axon/AxAbfFio32/abffiles.h"
#endif

#include "./abflib.h"

namespace stf {

wxString ABFError(const wxString& fName, int nError);

wxString dateToStr(long date);
wxString timeToStr(long time);

}

wxString stf::ABFError(const wxString& fName, int nError) {
    UINT uMaxLen=320;
    std::vector<char> errorMsg(uMaxLen);
    // local copy:
    wxString wxCp = fName;
    ABF_BuildErrorText(nError, wxCp.char_str(),&errorMsg[0], uMaxLen );
    return wxString( &errorMsg[0], wxConvLocal );
}

wxString stf::dateToStr(long date) {
    wxString dateStream;
    ldiv_t year=ldiv(date,(long)10000);
    dateStream << year.quot;
    ldiv_t month=ldiv(year.rem,(long)100);
    dateStream << wxT("/") << month.quot;
    dateStream << wxT("/") << month.rem;
    return dateStream;
}

wxString stf::timeToStr(long time) {
    wxString timeStream;
    ldiv_t hours=ldiv(time,(long)3600);
    timeStream << hours.quot;
    ldiv_t minutes=ldiv(hours.rem,(long)60);
    if (minutes.quot<10)
        timeStream << wxT(":") << wxT('0') << minutes.quot;
    else
        timeStream << wxT(":") << minutes.quot;
    if (minutes.rem<10)
        timeStream << wxT(":") << wxT('0') << minutes.rem;
    else
        timeStream << wxT(":") << minutes.rem;
    return timeStream;
}

void stf::importABFFile(const wxString &fName, Recording &ReturnData) {
    wxProgressDialog progDlg( wxT("Axon binary file import"), wxT("Starting file import"),
        100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );
    int hFile; 
    ABFFileHeader FH;
    UINT uMaxSamples;
    DWORD dwMaxEpi;
    int nError;
    if (!ABF_ReadOpen(fName.char_str(), &hFile, ABF_DATAFILE, &FH,
        &uMaxSamples, &dwMaxEpi, &nError))
    {
        wxString errorMsg(wxT("Exception while calling ABF_ReadOpen():\n"));
        errorMsg+=ABFError(fName,nError);
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }
    /*	if (!ABF_HasData(hFile,&FH)) {
    std::string errorMsg("Exception while calling ABF_ReadOpen():\n"
    "File is empty");
    throw std::runtime_error(errorMsg);
    }
    */	int numberChannels=FH.nADCNumChannels;
    long numberSections=FH.lActualEpisodes;
    if ((DWORD)numberSections>dwMaxEpi) {
        throw std::runtime_error("Error while calling stf::importABFFile():\n"
            "lActualEpisodes>dwMaxEpi");
    }
    for (int nChannel=0;nChannel<numberChannels;++nChannel) {
        Channel TempChannel(numberSections);
        for (DWORD dwEpisode=1;dwEpisode<=(DWORD)numberSections;++dwEpisode) {
            wxString progStr;
            progStr << wxT("Reading channel #") << nChannel + 1 << wxT(" of ") << numberChannels
                << wxT(", Section #") << dwEpisode << wxT(" of ") << numberSections;
            progDlg.Update(
                // Channel contribution:
                (int)(((double)nChannel/(double)numberChannels)*100.0+
                // Section contribution:
                (double)(dwEpisode-1)/(double)numberSections*(100.0/numberChannels)),
                progStr
            );
            unsigned int uNumSamples=0;
            if (!ABF_GetNumSamples(hFile,&FH,dwEpisode,&uNumSamples,&nError)) {
                wxString errorMsg( wxT("Exception while calling ABF_GetNumSamples():\n") );
                errorMsg += ABFError(fName, nError);
                ReturnData.resize(0);
                throw std::runtime_error(std::string(errorMsg.char_str()));
            }
            // Use a vector here because memory allocation can
            // be controlled more easily:
            // request memory:
            std::vector<float> TempSection(uNumSamples);
            unsigned int uNumSamplesW;
            if (!ABF_ReadChannel(hFile,&FH,FH.nADCSamplingSeq[nChannel],dwEpisode,&TempSection[0],
                &uNumSamplesW,&nError))
            {
                wxString errorMsg(wxT("Exception while calling ABF_ReadChannel():\n"));
                errorMsg += ABFError(fName, nError);
                ReturnData.resize(0);
                throw std::runtime_error(std::string(errorMsg.char_str()));
            }
            if (uNumSamples!=uNumSamplesW) {
                throw std::runtime_error("Exception while calling ABF_ReadChannel()");
            }
            wxString label; 
            label << stf::noPath(fName) << wxT(", Section # ") << dwEpisode;
            Section TempSectionT(TempSection.size(),label);
            std::copy(TempSection.begin(),TempSection.end(),&TempSectionT[0]);
            try {
                TempChannel.InsertSection(TempSectionT,dwEpisode-1);
            }
            catch (...) {
                throw;
            }
        }
        try {
            if ((int)ReturnData.size()<numberChannels) {
                ReturnData.resize(numberChannels);
            }
            ReturnData.InsertChannel(TempChannel,nChannel);
        }
        catch (...) {
            ReturnData.resize(0);
            throw;
        }

        wxString channel_name( FH.sADCChannelName[FH.nADCSamplingSeq[nChannel]], wxConvLocal );
        if (channel_name.find(wxT("  "))<channel_name.size()) {
            channel_name.erase(channel_name.begin()+channel_name.find(wxT("  ")),channel_name.end());
        }
        ReturnData[nChannel].SetChannelName(channel_name);

        wxString channel_units( FH.sADCUnits[FH.nADCSamplingSeq[nChannel]], wxConvLocal );
        if (channel_units.find(wxT("  ")) < channel_units.size()) {
            channel_units.erase(channel_units.begin() + channel_units.find(wxT("  ")),channel_units.end());
        }
        ReturnData[nChannel].SetYUnits(channel_units);
    }

    if (!ABF_Close(hFile,&nError)) {
        wxString errorMsg(wxT("Exception in importABFFile():\n"));
        errorMsg += ABFError(fName,nError);
        ReturnData.resize(0);
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }
    // Apparently, the sample interval has to be multiplied by
    // the number of channels for multiplexed data. Thanks to
    // Dominique Engel for noticing.
    ReturnData.SetXScale((double)(FH.fADCSampleInterval/1000.0)*(double)numberChannels);
    wxString comment(wxT("Created with "));
    comment += wxString( FH.sCreatorInfo, wxConvLocal );
    ReturnData.SetComment(comment);
    ReturnData.SetDate(dateToStr(FH.lFileStartDate));
    ReturnData.SetTime(timeToStr(FH.lFileStartTime));
}
