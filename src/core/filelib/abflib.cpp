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

#ifndef MODULE_ONLY
#include <wx/wx.h>
#include <wx/progdlg.h>
#endif

#if defined(__linux__) || defined(__STF__) || defined(__APPLE__)
#include "./axon/Common/axodefn.h"
#include "./axon/AxAbfFio32/abffiles.h"
#include "./axon2/ProtocolReaderABF2.hpp"
#endif

#include "./abflib.h"


namespace stf {

wxString ABF1Error(const wxString& fName, int nError);

wxString dateToStr(ABFLONG date);
wxString timeToStr(ABFLONG time);

}

wxString stf::ABF1Error(const wxString& fName, int nError) {
    UINT uMaxLen=320;
    std::vector<char> errorMsg(uMaxLen);
    // local copy:
    wxString wxCp = fName;
    ABF_BuildErrorText(nError, wxCp.c_str(),&errorMsg[0], uMaxLen );
    return wxString( &errorMsg[0] );
}

wxString stf::dateToStr(ABFLONG date) {
    std::ostringstream dateStream;
    ldiv_t year=ldiv(date,(ABFLONG)10000);
    dateStream << year.quot;
    ldiv_t month=ldiv(year.rem,(ABFLONG)100);
    dateStream << wxT("/") << month.quot;
    dateStream << wxT("/") << month.rem;
    return dateStream.str();
}

wxString stf::timeToStr(ABFLONG time) {
    std::ostringstream timeStream;
    ldiv_t hours=ldiv(time,(ABFLONG)3600);
    timeStream << hours.quot;
    ldiv_t minutes=ldiv(hours.rem,(ABFLONG)60);
    if (minutes.quot<10)
        timeStream << wxT(":") << wxT('0') << minutes.quot;
    else
        timeStream << wxT(":") << minutes.quot;
    if (minutes.rem<10)
        timeStream << wxT(":") << wxT('0') << minutes.rem;
    else
        timeStream << wxT(":") << minutes.rem;
    return timeStream.str();
}

void stf::importABFFile(const wxString &fName, Recording &ReturnData, bool progress) {
    ABF2_FileInfo fileInfo;

    // Open file:
#ifndef _WINDOWS
    FILE* fh = fopen( fName.c_str(), "r" );
	if (!fh) {
        wxString errorMsg(wxT("Exception while calling importABFFile():\nCouldn't open file"));
        fclose(fh);
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }

    // attempt to read first chunk of data:
    int res = fseek( fh, 0, SEEK_SET);
    if (res != 0) {
        wxString errorMsg(wxT("Exception while calling importABFFile():\nCouldn't open file"));
        fclose(fh);
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }
    res = fread( &fileInfo, sizeof( fileInfo ), 1, fh );
    if (res != 1) {
        wxString errorMsg(wxT("Exception while calling importABFFile():\nCouldn't open file"));
        fclose(fh);
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }
    fclose(fh);
#else
    HANDLE hFile = CreateFile(fName, GENERIC_READ, FILE_SHARE_READ, NULL,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
 
    if (hFile == INVALID_HANDLE_VALUE) { 
        wxString errorMsg(wxT("Exception while calling importABFFile():\nCouldn't open file"));
        CloseHandle(hFile);
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }

	// Read one character less than the buffer size to save room for
    // the terminating NULL character.
    DWORD dwBytesRead = 0;

    if( FALSE == ReadFile(hFile, &fileInfo, sizeof( fileInfo ), &dwBytesRead, NULL) ) {
        wxString errorMsg(wxT("Exception while calling importABFFile():\nCouldn't open file"));
        CloseHandle(hFile);
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }

	if (dwBytesRead <= 0) {
        wxString errorMsg(wxT("Exception while calling importABFFile():\nCouldn't open file"));
        CloseHandle(hFile);
        throw std::runtime_error(std::string(errorMsg.c_str()));
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
#ifndef MODULE_ONLY
    wxProgressDialog progDlg( wxT("Axon binary file 2.x import"), wxT("Starting file import"),
        100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );
#endif
    CABF2ProtocolReader abf2;
    std::wstring wfName;
    wfName.resize(fName.size());
    std::copy(fName.begin(), fName.end(), wfName.begin());
    // for(std::string::size_type i=0; i<fName.size(); ++i) {
    //     wfName[i] = (wchar_t)fName[i];
    // }

    if (!abf2.Open( &wfName[0] )) {
        wxString errorMsg(wxT("Exception while calling importABF2File():\nCouldn't open file"));
        throw std::runtime_error(std::string(errorMsg.c_str()));
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
        throw std::runtime_error(std::string(errorMsg.c_str()));
        abf2.Close();
    }
            
    const ABF2FileHeader* pFH = abf2.GetFileHeader();
#ifdef _STFDEBUG
    std::cout << "ABF2 file information" << std::endl
              << "File version " <<  pFH->fFileVersionNumber << std::endl
              << "Header version " <<  pFH->fHeaderVersionNumber << std::endl
              << "Data format " << pFH->nDataFormat << std::endl
              << "Number of channels " << pFH->nADCNumChannels << std::endl
              << "Number of sweeps " << pFH->lActualEpisodes << std::endl
              << "Sampling points per sweep " << pFH->lNumSamplesPerEpisode << std::endl
              << "File type " << pFH->nOperationMode << std::endl;
#endif
    
    int numberChannels = pFH->nADCNumChannels;
    ABFLONG numberSections = pFH->lActualEpisodes;
    ABFLONG finalSections = numberSections;
    bool gapfree = (pFH->nOperationMode == ABF2_GAPFREEFILE);
    if (gapfree) {
        finalSections = 1;
    }
    int hFile = abf2.GetFileNumber();
    for (int nChannel=0; nChannel < numberChannels; ++nChannel) {
        if (progress) {
#ifndef MODULE_ONLY
            int progbar = (int)(((double)nChannel/(double)numberChannels)*100.0);
            progDlg.Update(progbar, wxT("Memory allocation"));
#endif
        }
        ABFLONG grandsize = pFH->lNumSamplesPerEpisode / numberChannels;
        std::ostringstream label;
        label  
#ifdef MODULE_ONLY
               << fName
#else
               << stf::noPath(fName)
#endif
               << wxT(", gapfree section");
        if (gapfree) {
            grandsize = pFH->lActualAcqLength / numberChannels;
            Vector_double test_size(0);
            ABFLONG maxsize = test_size.max_size()
#ifdef _WINDOWS
                // doesn't seem to return the correct size on Windows.
                /8;
#else
                ;
#endif
            
            if (grandsize <= 0 || grandsize >= maxsize) {
                    
                wxString segstring("Gapfree file is too large for a single section." \
                                   "It will be segmented.\nFile opening may be very slow.");
#ifndef MODULE_ONLY        
                wxMessageBox(segstring,wxT("Information"), wxOK | wxICON_WARNING, NULL);
#else
                std::cout << segstring << std::endl;
#endif
                
                gapfree=false;
                grandsize = pFH->lNumSamplesPerEpisode / numberChannels;
                finalSections=numberSections;
            }
        }
        Channel TempChannel(finalSections, grandsize);
        Section TempSectionGrand(grandsize, label.str());
        for (int nEpisode=1; nEpisode<=numberSections;++nEpisode) {
            if (progress) {
                int progbar =
                    // Channel contribution:
                    (int)(((double)nChannel/(double)numberChannels)*100.0+
                          // Section contribution:
                          (double)(nEpisode-1)/(double)numberSections*(100.0/numberChannels));
#ifndef MODULE_ONLY
                wxString progStr;
                progStr << wxT("Reading channel #") << nChannel + 1 << wxT(" of ") << numberChannels
                    << wxT(", Section #") << nEpisode << wxT(" of ") << numberSections;
                progDlg.Update(progbar, progStr);
#else
                std::cout << "\r";
                std::cout << progbar << "%" << std::flush;
#endif
            }
            
            UINT uNumSamples = 0;
            if (gapfree) {
                if (nEpisode == numberSections) {
                    uNumSamples = grandsize - (nEpisode-1) * pFH->lNumSamplesPerEpisode / numberChannels;
#ifdef _STFDEBUG
                    std::cout << "Last section size " << uNumSamples << std::endl;
#endif
                } else {
                    uNumSamples = pFH->lNumSamplesPerEpisode / numberChannels;
                }
            } else {
                if (!ABF2_GetNumSamples(hFile, pFH, nEpisode, &uNumSamples, &nError)) {
                    std::ostringstream errorMsg;
                    errorMsg << wxT("Exception while calling ABF2_GetNumSamples() ")
                             << wxT("for episode # ")
                             << nEpisode << wxT("\n")
                             << ABF1Error(fName, nError);
                    ReturnData.resize(0);
                    ABF_Close(hFile,&nError);
                    throw std::runtime_error(std::string(errorMsg.str().c_str()));
                }
            }
            // Use a vector here because memory allocation can
            // be controlled more easily:
            // request memory:
            Vector_float TempSection(uNumSamples, 0.0);
            unsigned int uNumSamplesW;
            if (!ABF2_ReadChannel(hFile, pFH, pFH->nADCSamplingSeq[nChannel],nEpisode,TempSection,
                                  &uNumSamplesW,&nError))
            {
                wxString errorMsg(wxT("Exception while calling ABF2_ReadChannel():\n"));
                errorMsg += ABF1Error(fName, nError);
                ReturnData.resize(0);
                ABF_Close(hFile,&nError);
                throw std::runtime_error(std::string(errorMsg.c_str()));
            }
            if (uNumSamples!=uNumSamplesW && !gapfree) {
                ABF_Close(hFile,&nError);
                throw std::runtime_error("Exception while calling ABF2_ReadChannel()");
            }
            if (!gapfree) {
                std::ostringstream label;
                label
#ifdef MODULE_ONLY
                    << fName
#else
                    << stf::noPath(fName)
#endif
                    << wxT(", Section # ") << nEpisode;
                Section TempSectionT(TempSection.size(),label.str());
                std::copy(TempSection.begin(),TempSection.end(),&TempSectionT[0]);
                try {
                    TempChannel.InsertSection(TempSectionT,nEpisode-1);
                }
                catch (...) {
                    ABF_Close(hFile,&nError);
                    throw;
                }
            } else {
                if ((nEpisode-1) * pFH->lNumSamplesPerEpisode / numberChannels + TempSection.size() <= TempSectionGrand.size()) {
                    std::copy(TempSection.begin(),TempSection.end(),
                              &TempSectionGrand[(nEpisode-1) * pFH->lNumSamplesPerEpisode / numberChannels]);
                }
#ifdef _STFDEBUG
                else {
                    std::cout << "Overflow while copying gapfree sections" << std::endl;
                }
#endif
            }
        }
        if (gapfree) {
            try {
                TempChannel.InsertSection(TempSectionGrand,0);
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
        
        if (progress) {
            int progbar = (int)(((double)(nChannel+1)/(double)numberChannels)*100.0);
#ifndef MODULE_ONLY
            progDlg.Update(progbar, wxT("Completing channel reading\n"));
#else
            std::cout << "\r";
            std::cout << progbar << "%" << std::flush;
#endif
        }

        wxString channel_name( pFH->sADCChannelName[pFH->nADCSamplingSeq[nChannel]] );
        if (channel_name.find(wxT("  "))<channel_name.size()) {
            channel_name.erase(channel_name.begin()+channel_name.find(wxT("  ")),channel_name.end());
        }
        ReturnData[nChannel].SetChannelName(channel_name);

        wxString channel_units( pFH->sADCUnits[pFH->nADCSamplingSeq[nChannel]] );
        if (channel_units.find(wxT("  ")) < channel_units.size()) {
            channel_units.erase(channel_units.begin() + channel_units.find(wxT("  ")),channel_units.end());
        }
        ReturnData[nChannel].SetYUnits(channel_units);
    }

    if (!ABF_Close(hFile,&nError)) {
        wxString errorMsg(wxT("Exception in importABFFile():\n"));
        errorMsg += ABF1Error(fName,nError);
        ReturnData.resize(0);
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }
    
    ReturnData.SetXScale((double)(pFH->fADCSequenceInterval/1000.0));
    
    wxString comment(wxT("Created with "));
    comment += wxString( pFH->sCreatorInfo );
    ReturnData.SetComment(comment);
    ReturnData.SetDate(dateToStr(pFH->uFileStartDate));
    ReturnData.SetTime(timeToStr(pFH->uFileStartTimeMS));

    abf2.Close();
#ifdef MODULE_ONLY
    if (progress) {
        std::cout << "\r";
        std::cout << "100%" << std::endl;
    }
#endif
}

void stf::importABF1File(const wxString &fName, Recording &ReturnData, bool progress) {
#ifndef MODULE_ONLY
    wxProgressDialog progDlg( wxT("Axon binary file 1.x import"), wxT("Starting file import"),
                              100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );
#endif
    
    int hFile = 0;
    ABFFileHeader FH;
    UINT uMaxSamples = 0;
    DWORD dwMaxEpi = 0;
    int nError = 0;

    std::wstring wfName;

    for(std::string::size_type i=0; i<fName.size(); ++i) {
        wfName += wchar_t(fName[i]);
    }

    if (!ABF_ReadOpen(wfName.c_str(), &hFile, ABF_DATAFILE, &FH,
                      &uMaxSamples, &dwMaxEpi, &nError))
    {
        wxString errorMsg(wxT("Exception while calling ABF_ReadOpen():\n"));
        errorMsg+=ABF1Error(fName,nError);
        ABF_Close(hFile,&nError);
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }
    /*	if (!ABF_HasData(hFile,pFH)) {
    std::string errorMsg("Exception while calling ABF_ReadOpen():\n"
    "File is empty");
    throw std::runtime_error(errorMsg);
    }
    */
    int numberChannels=FH.nADCNumChannels;
    ABFLONG numberSections=FH.lActualEpisodes;
    if ((DWORD)numberSections>dwMaxEpi) {
        ABF_Close(hFile,&nError);
        throw std::runtime_error("Error while calling stf::importABFFile():\n"
            "lActualEpisodes>dwMaxEpi");
    }
    for (int nChannel=0;nChannel<numberChannels;++nChannel) {
        Channel TempChannel(numberSections);
        for (DWORD dwEpisode=1;dwEpisode<=(DWORD)numberSections;++dwEpisode) {
            if (progress) {
                int progbar = // Channel contribution:
                    (int)(((double)nChannel/(double)numberChannels)*100.0+
                          // Section contribution:
                          (double)(dwEpisode-1)/(double)numberSections*(100.0/numberChannels));
#ifndef MODULE_ONLY
                wxString progStr;
                progStr << wxT("Reading channel #") << nChannel + 1 << wxT(" of ") << numberChannels
                    << wxT(", Section #") << dwEpisode << wxT(" of ") << numberSections;
                progDlg.Update(progbar, progStr);
#else
                std::cout << "\r"; // Remove previous entry
                std::cout << progbar << "%" << std::flush;
#endif
            }
            
            unsigned int uNumSamples=0;
            if (!ABF_GetNumSamples(hFile,&FH,dwEpisode,&uNumSamples,&nError)) {
                wxString errorMsg( wxT("Exception while calling ABF_GetNumSamples():\n") );
                errorMsg += ABF1Error(fName, nError);
                ReturnData.resize(0);
                ABF_Close(hFile,&nError);
                throw std::runtime_error(std::string(errorMsg.c_str()));
            }
            // Use a vector here because memory allocation can
            // be controlled more easily:
            // request memory:
            Vector_float TempSection(uNumSamples, 0.0);
            unsigned int uNumSamplesW=0;
            if (!ABF_ReadChannel(hFile, &FH, FH.nADCSamplingSeq[nChannel], dwEpisode, TempSection,
                                 &uNumSamplesW, &nError))
            {
                wxString errorMsg(wxT("Exception while calling ABF_ReadChannel():\n"));
                errorMsg += ABF1Error(fName, nError);
                ReturnData.resize(0);
                ABF_Close(hFile,&nError);
                throw std::runtime_error(std::string(errorMsg.c_str()));
            }
            if (uNumSamples!=uNumSamplesW) {
                ABF_Close(hFile,&nError);
                throw std::runtime_error("Exception while calling ABF_ReadChannel()");
            }
            std::ostringstream label;
            label
#ifdef MODULE_ONLY
                << fName
#else
                << stf::noPath(fName)
#endif
                << wxT(", Section # ") << dwEpisode;
            Section TempSectionT(TempSection.size(),label.str());
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

        wxString channel_name( FH.sADCChannelName[FH.nADCSamplingSeq[nChannel]] );
        if (channel_name.find(wxT("  "))<channel_name.size()) {
            channel_name.erase(channel_name.begin()+channel_name.find(wxT("  ")),channel_name.end());
        }
        ReturnData[nChannel].SetChannelName(channel_name);

        wxString channel_units( FH.sADCUnits[FH.nADCSamplingSeq[nChannel]] );
        if (channel_units.find(wxT("  ")) < channel_units.size()) {
            channel_units.erase(channel_units.begin() + channel_units.find(wxT("  ")),channel_units.end());
        }
        ReturnData[nChannel].SetYUnits(channel_units);
    }

    if (!ABF_Close(hFile,&nError)) {
        wxString errorMsg(wxT("Exception in importABFFile():\n"));
        errorMsg += ABF1Error(fName,nError);
        ReturnData.resize(0);
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }
    // Apparently, the sample interval has to be multiplied by
    // the number of channels for multiplexed data. Thanks to
    // Dominique Engel for noticing.
    ReturnData.SetXScale((double)(FH.fADCSampleInterval/1000.0)*(double)numberChannels);
    wxString comment(wxT("Created with "));
    comment += wxString( FH.sCreatorInfo );
    ReturnData.SetComment(comment);
    ReturnData.SetDate(dateToStr(FH.lFileStartDate));
    ReturnData.SetTime(timeToStr(FH.lFileStartTime));
#ifdef MODULE_ONLY
    if (progress) {
        std::cout << "\r";
        std::cout << "100%" << std::endl;
    }
#endif
    
}
#ifdef _WINDOWS
#pragma optimize ("", on)
#endif
