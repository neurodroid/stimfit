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

#if defined(__LINUX__) || defined(__STF__) || defined(__WXMAC__)
#include "./axon/Common/axodefn.h"
#include "./axon/AxAbfFio32/abffiles.h"
#include "./axon2/ProtocolReaderABF2.hpp"
#endif

#include "./abflib.h"


namespace stf {

wxString ABF1Error(const wxString& fName, int nError);

wxString dateToStr(long date);
wxString timeToStr(long time);

}

wxString stf::ABF1Error(const wxString& fName, int nError) {
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

void stf::importABFFile(const wxString &fName, Recording &ReturnData, bool progress) {
    ABF2_FileInfo fileInfo;

    // Open file:
#ifndef _WINDOWS
    FILE* fh = fopen( fName.char_str(), "r" );
	if (!fh) {
        wxString errorMsg(wxT("Exception while calling importABFFile():\nCouldn't open file"));
        fclose(fh);
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }

    // attempt to read first chunk of data:
    int res = fseek( fh, 0, SEEK_SET);
    if (res != 0) {
        wxString errorMsg(wxT("Exception while calling importABFFile():\nCouldn't open file"));
        fclose(fh);
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }
    res = fread( &fileInfo, sizeof( fileInfo ), 1, fh );
    if (res != 1) {
        wxString errorMsg(wxT("Exception while calling importABFFile():\nCouldn't open file"));
        fclose(fh);
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }
    fclose(fh);
#else
    HANDLE hFile = CreateFile(fName, GENERIC_READ, FILE_SHARE_READ, NULL,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
 
    if (hFile == INVALID_HANDLE_VALUE) { 
        wxString errorMsg(wxT("Exception while calling importABFFile():\nCouldn't open file"));
        CloseHandle(hFile);
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }

	// Read one character less than the buffer size to save room for
    // the terminating NULL character.
    DWORD dwBytesRead = 0;

    if( FALSE == ReadFile(hFile, &fileInfo, sizeof( fileInfo ), &dwBytesRead, NULL) ) {
        wxString errorMsg(wxT("Exception while calling importABFFile():\nCouldn't open file"));
        CloseHandle(hFile);
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }

	if (dwBytesRead <= 0) {
        wxString errorMsg(wxT("Exception while calling importABFFile():\nCouldn't open file"));
        CloseHandle(hFile);
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }
    CloseHandle(hFile);
#endif
    
    if (CABF2ProtocolReader::CanOpen( (void*)&fileInfo, sizeof(fileInfo) )) {
        importABF2File( fName, ReturnData, progress );
    } else {
        importABF1File( fName, ReturnData, progress );
    }
}


void stf::importABF2File(const wxString &fName, Recording &ReturnData, bool progress) {
    wxProgressDialog progDlg( wxT("Axon binary file 2.x import"), wxT("Starting file import"),
        100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );
    
    CABF2ProtocolReader abf2;
    if (!abf2.Open( fName )) {
        wxString errorMsg(wxT("Exception while calling importABF2File():\nCouldn't open file"));
        throw std::runtime_error(std::string(errorMsg.char_str()));
        abf2.Close();
    }
#ifdef _STFDEBUG
    else {
        std::cout << "File successfully opened" << std::endl;
    }
#endif
    int nError = 0;
    if (!abf2.Read( &nError )) {
        wxString errorMsg(wxT("Exception while calling importABF2File():\nCouldn't read file"));
        throw std::runtime_error(std::string(errorMsg.char_str()));
        abf2.Close();
    }
            
    const ABF2FileHeader* pFH = abf2.GetFileHeader();
#ifdef _STFDEBUG
    std::cout << "ABF2 file information" << std::endl
              << "File version " <<  pFH->fFileVersionNumber << std::endl
              << "Data format " << pFH->nDataFormat << std::endl
              << "Number of channels " << pFH->nADCNumChannels << std::endl
              << "Number of sweeps " << pFH->lActualEpisodes << std::endl
              << "Sampling points per sweep " << pFH->lNumSamplesPerEpisode << std::endl;
#endif
    
    int numberChannels = pFH->nADCNumChannels;
    long numberSections = pFH->lActualEpisodes;
    int hFile = abf2.GetFileNumber();
    for (int nChannel=0; nChannel < numberChannels; ++nChannel) {
        Channel TempChannel(numberSections);
        for (int nEpisode=1; nEpisode<=numberSections;++nEpisode) {
            if (progress) {
                wxString progStr;
                progStr << wxT("Reading channel #") << nChannel + 1 << wxT(" of ") << numberChannels
                    << wxT(", Section #") << nEpisode << wxT(" of ") << numberSections;
                progDlg.Update(
                        // Channel contribution:
                        (int)(((double)nChannel/(double)numberChannels)*100.0+
                                // Section contribution:
                                (double)(nEpisode-1)/(double)numberSections*(100.0/numberChannels)),
                                progStr
                );
            }
            unsigned int uNumSamples=0;
            if (!ABF2_GetNumSamples(hFile, pFH, nEpisode, &uNumSamples, &nError)) {
                wxString errorMsg( wxT("Exception while calling ABF_GetNumSamples() ") );
                errorMsg += wxT("for episode # "); errorMsg << nEpisode; errorMsg += wxT("\n");
                errorMsg += ABF1Error(fName, nError);
                ReturnData.resize(0);
                ABF_Close(hFile,&nError);
                throw std::runtime_error(std::string(errorMsg.char_str()));
            }
            // Use a vector here because memory allocation can
            // be controlled more easily:
            // request memory:
            std::vector<float> TempSection(uNumSamples, 0.0);
            unsigned int uNumSamplesW;
            if (!ABF2_ReadChannel(hFile, pFH, pFH->nADCSamplingSeq[nChannel],nEpisode,TempSection,
                                  &uNumSamplesW,&nError))
            {
                wxString errorMsg(wxT("Exception while calling ABF_ReadChannel():\n"));
                errorMsg += ABF1Error(fName, nError);
                ReturnData.resize(0);
                ABF_Close(hFile,&nError);
                throw std::runtime_error(std::string(errorMsg.char_str()));
            }
            if (uNumSamples!=uNumSamplesW) {
                ABF_Close(hFile,&nError);
                throw std::runtime_error("Exception while calling ABF_ReadChannel()");
            }
            wxString label;
            label << stf::noPath(fName) << wxT(", Section # ") << nEpisode;
            Section TempSectionT(TempSection.size(),label);
            std::copy(TempSection.begin(),TempSection.end(),&TempSectionT[0]);
            try {
                TempChannel.InsertSection(TempSectionT,nEpisode-1);
            }
            catch (...) {
                ABF_Close(hFile,&nError);
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
            ABF_Close(hFile,&nError);
            throw;
        }

        wxString channel_name( pFH->sADCChannelName[pFH->nADCSamplingSeq[nChannel]], wxConvLocal );
        if (channel_name.find(wxT("  "))<channel_name.size()) {
            channel_name.erase(channel_name.begin()+channel_name.find(wxT("  ")),channel_name.end());
        }
        ReturnData[nChannel].SetChannelName(channel_name);

        wxString channel_units( pFH->sADCUnits[pFH->nADCSamplingSeq[nChannel]], wxConvLocal );
        if (channel_units.find(wxT("  ")) < channel_units.size()) {
            channel_units.erase(channel_units.begin() + channel_units.find(wxT("  ")),channel_units.end());
        }
        ReturnData[nChannel].SetYUnits(channel_units);
    }

    if (!ABF_Close(hFile,&nError)) {
        wxString errorMsg(wxT("Exception in importABFFile():\n"));
        errorMsg += ABF1Error(fName,nError);
        ReturnData.resize(0);
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }
    // Apparently, the sample interval has to be multiplied by
    // the number of channels for multiplexed data. Thanks to
    // Dominique Engel for noticing.
    ReturnData.SetXScale((double)(pFH->fADCSequenceInterval/1000.0)*(double)numberChannels);
    wxString comment(wxT("Created with "));
    comment += wxString( pFH->sCreatorInfo, wxConvLocal );
    ReturnData.SetComment(comment);
    ReturnData.SetDate(dateToStr(pFH->uFileStartDate));
    ReturnData.SetTime(timeToStr(pFH->uFileStartTimeMS));

    
    abf2.Close();
}

void stf::importABF1File(const wxString &fName, Recording &ReturnData, bool progress) {
    wxProgressDialog progDlg( wxT("Axon binary file 1.x import"), wxT("Starting file import"),
        100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );
    int hFile = 0;
    ABFFileHeader FH;
    UINT uMaxSamples = 0;
    DWORD dwMaxEpi = 0;
    int nError = 0;
    if (!ABF_ReadOpen(fName, &hFile, ABF_DATAFILE, &FH,
                      &uMaxSamples, &dwMaxEpi, &nError))
    {
        wxString errorMsg(wxT("Exception while calling ABF_ReadOpen():\n"));
        errorMsg+=ABF1Error(fName,nError);
        ABF_Close(hFile,&nError);
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }
    /*	if (!ABF_HasData(hFile,pFH)) {
    std::string errorMsg("Exception while calling ABF_ReadOpen():\n"
    "File is empty");
    throw std::runtime_error(errorMsg);
    }
    */
    int numberChannels=FH.nADCNumChannels;
    long numberSections=FH.lActualEpisodes;
    if ((DWORD)numberSections>dwMaxEpi) {
        ABF_Close(hFile,&nError);
        throw std::runtime_error("Error while calling stf::importABFFile():\n"
            "lActualEpisodes>dwMaxEpi");
    }
    for (int nChannel=0;nChannel<numberChannels;++nChannel) {
        Channel TempChannel(numberSections);
        for (DWORD dwEpisode=1;dwEpisode<=(DWORD)numberSections;++dwEpisode) {
            if (progress) {
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
            }
            unsigned int uNumSamples=0;
            if (!ABF_GetNumSamples(hFile,&FH,dwEpisode,&uNumSamples,&nError)) {
                wxString errorMsg( wxT("Exception while calling ABF_GetNumSamples():\n") );
                errorMsg += ABF1Error(fName, nError);
                ReturnData.resize(0);
                ABF_Close(hFile,&nError);
                throw std::runtime_error(std::string(errorMsg.char_str()));
            }
            // Use a vector here because memory allocation can
            // be controlled more easily:
            // request memory:
            std::vector<float> TempSection(uNumSamples, 0.0);
            unsigned int uNumSamplesW=0;
            if (!ABF_ReadChannel(hFile, &FH, FH.nADCSamplingSeq[nChannel], dwEpisode, TempSection,
                                 &uNumSamplesW, &nError))
            {
                wxString errorMsg(wxT("Exception while calling ABF_ReadChannel():\n"));
                errorMsg += ABF1Error(fName, nError);
                ReturnData.resize(0);
                ABF_Close(hFile,&nError);
                throw std::runtime_error(std::string(errorMsg.char_str()));
            }
            if (uNumSamples!=uNumSamplesW) {
                ABF_Close(hFile,&nError);
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
                ABF_Close(hFile,&nError);
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
            ABF_Close(hFile,&nError);
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
        errorMsg += ABF1Error(fName,nError);
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
#ifdef _WINDOWS
#pragma optimize ("", on)
#endif
