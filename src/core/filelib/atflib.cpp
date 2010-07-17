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

#ifndef MODULE_ONLY
#include <wx/wxprec.h>
#include <wx/progdlg.h>
#endif

#include "./atflib.h"

namespace stf {
wxString ATFError(const wxString& fName, int nError);
}

wxString stf::ATFError(const wxString& fName, int nError) {
    int nMaxLen=320;
    std::vector<char> errorMsg(nMaxLen);
    ATF_BuildErrorText(nError, fName.c_str(),&errorMsg[0], nMaxLen );
    return wxString( &errorMsg[0] );
}

bool stf::exportATFFile(const wxString& fName, const Recording& WData) {
    int nColumns=1+(int)WData[0].size() /*time + number of sections*/, nFileNum;
    int nError;
    if (!ATF_OpenFile(fName.c_str(),ATF_WRITEONLY,&nColumns,&nFileNum,&nError)) {
        wxString errorMsg(wxT("Exception while calling ATF_OpenFile():\n"));
        errorMsg += ATFError(fName,nError);
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }
    // Write sections to columns:
    // First column is time:
    for (int n_c=0;n_c<nColumns;++n_c) {
        wxString columnTitle,columnUnits;
        if (n_c==0) {
            columnTitle = wxT("Time");
            columnUnits = WData.GetXUnits();
        } else {
            std::ostringstream titleStr;
            titleStr << wxT("Section[") << n_c-1 << wxT("]");
            columnTitle = titleStr.str();
            columnUnits = WData[0].GetYUnits();
        }
        if (!ATF_SetColumnTitle(nFileNum, columnTitle.c_str(), &nError)) {
            wxString errorMsg(wxT("Exception while calling ATF_SetColumnTitle():\n"));
            errorMsg+=ATFError(fName,nError);
            throw std::runtime_error(std::string(errorMsg.c_str()));
        }
        if (!ATF_SetColumnUnits(nFileNum, columnUnits.c_str(),&nError)) {
            wxString errorMsg(wxT("Exception while calling ATF_SetColumnUnits():\n"));
            errorMsg+=ATFError(fName,nError);
            throw std::runtime_error(std::string(errorMsg.c_str()));
        }
    }
    // Write data line by line:
    std::size_t max_size=WData[0][0].size();
    // check for equal section sizes:
    for (int n_s=1;n_s<(int)WData[0].size();++n_s) {
        if (WData[0][n_s].size() > max_size) {
            max_size=WData[0][n_s].size();
        }
    }
    for (int n_l=0;n_l < (int)max_size; ++n_l) {
        for (int n_c=0;n_c<nColumns;++n_c) {
            if (n_c==0) {
                // Write time:
                double time=n_l*WData.GetXScale();
                if (!ATF_WriteDataRecord1(nFileNum,time,&nError)) {
                    wxString errorMsg(wxT("Exception while calling ATF_WriteDataRecord1():\n"));
                    errorMsg+=ATFError(fName,nError);
                    throw std::runtime_error(std::string(errorMsg.c_str()));
                }
            } else {
                double toWrite = (n_l < (int)WData[0][n_c-1].size()) ?
                        (double)WData[0][n_c-1][n_l] :
                0.0;
                        if (!ATF_WriteDataRecord1(nFileNum,toWrite,&nError)) {
                            wxString errorMsg(wxT("Exception while calling ATF_WriteDataRecord1():\n"));
                            errorMsg+=ATFError(fName,nError);
                            throw std::runtime_error(std::string(errorMsg.c_str()));
                        }
            }
        }
        if (!ATF_WriteEndOfLine(nFileNum,&nError)) {
            wxString errorMsg(wxT("Exception while calling ATF_WriteEndOfLine():\n"));
            errorMsg+=ATFError(fName,nError);
            throw std::runtime_error(std::string(errorMsg.c_str()));
        }
    }
    if (!ATF_CloseFile(nFileNum)) {
        wxString errorMsg(wxT("Exception while calling ATF_CloseFile():\n"));
        errorMsg += wxT("Error while closing ATF file");
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }
    return true;
}

void stf::importATFFile(const wxString &fName, Recording &ReturnData, bool progress) {
#ifndef MODULE_ONLY
    wxProgressDialog progDlg(
            wxT("Axon text file import"),
            wxT("Starting file import"),
            100,
            NULL,
            wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL
                             );
#endif
    int nColumns, nFileNum;
    int nError;
    const int nMaxText=64;

    if (!ATF_OpenFile(fName.c_str(),ATF_READONLY,&nColumns,&nFileNum,&nError)) {
        wxString errorMsg(wxT("Exception while calling ATF_OpenFile():\n"));
        errorMsg+=ATFError(fName,nError);
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }
    // Assume that the first column is time:
    if (nColumns==0) {
        wxString errorMsg(wxT("Error while opening ATF file:\nFile appears to be empty"));
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }
    long sectionSize;
    if (!ATF_CountDataLines(nFileNum,&sectionSize,&nError)) {
        wxString errorMsg(wxT("Exception while calling ATF_CountDataLines():\n"));
        errorMsg+=ATFError(fName,nError);
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }

    // If first column contains time values, determine sampling interval:
    std::vector<char> titleVec(nMaxText);
    if (!ATF_GetColumnTitle(nFileNum,0,&titleVec[0],nMaxText,&nError)) {
        wxString errorMsg(wxT("Exception while calling ATF_GetColumnTitle():\n"));
        errorMsg+=ATFError(fName,nError);
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }
    std::string titleString(titleVec.begin(),titleVec.end());
    int timeInFirstColumn=0;
    if (titleString.find("time")!=std::string::npos ||
            titleString.find("Time")!=std::string::npos ||
            titleString.find("TIME")!=std::string::npos)
    {
        // Read sampling information from first two time values:
        double time[2];
        for (int n_l=0;n_l<2;++n_l) {
            if (!ATF_ReadDataColumn(nFileNum,0,&time[n_l],&nError)) {
                wxString errorMsg(wxT("Exception while calling ATF_ReadDataColumn():\n"));
                errorMsg+=ATFError(fName,nError);
                throw std::runtime_error(std::string(errorMsg.c_str()));
            }
        }
        if (!ATF_RewindFile(nFileNum,&nError)) {
            wxString errorMsg(wxT("Exception while calling ATF_RewindFile():\n"));
            errorMsg+=ATFError(fName,nError);
            throw std::runtime_error(std::string(errorMsg.c_str()));
        }

        ReturnData.SetXScale(time[1]-time[0]);
        timeInFirstColumn=1;
    }
    ReturnData.resize(1);
    Channel TempChannel(nColumns-timeInFirstColumn);
    for (int n_c=timeInFirstColumn;n_c<nColumns;++n_c) {
        if (progress) {
#ifndef MODULE_ONLY
            wxString progStr;
            progStr << wxT("Section #") << n_c+1-timeInFirstColumn << wxT(" of ") << nColumns-timeInFirstColumn;
            progDlg.Update(
                    // Section contribution:
                    (double)100.0*(n_c+1-timeInFirstColumn)/(double)(nColumns-timeInFirstColumn),
                    progStr
                           );
#endif
        }
        std::ostringstream label;
        label
#ifdef MODULE_ONLY
            << fName 
#else
            << stf::noPath(fName) 
#endif
            
            << wxT(", Section # ") << n_c-timeInFirstColumn+1;
        Section TempSection(sectionSize,label.str());
        for (int n_l=0;n_l<sectionSize;++n_l) {
            if (!ATF_ReadDataColumn(nFileNum,n_c,&TempSection[n_l],&nError)) {
                wxString errorMsg(wxT("Exception while calling ATF_ReadDataColumn():\n"));
                errorMsg+=ATFError(fName,nError);
                ReturnData.resize(0);
                throw std::runtime_error(std::string(errorMsg.c_str()));
            }
        }
        if (n_c-timeInFirstColumn==0) {
            std::vector<char> unitsVec(nMaxText);
            if (!ATF_GetColumnUnits(nFileNum,n_c,&unitsVec[0],nMaxText,&nError)) {
                wxString errorMsg(wxT("Exception while calling ATF_GetColumnUnits():\n"));
                errorMsg+=ATFError(fName,nError);
                ReturnData.resize(0);
                throw std::runtime_error(std::string(errorMsg.c_str()));
            }
            ReturnData[0].SetYUnits(wxString( &unitsVec[0] ));
        }
        try {
            TempChannel.InsertSection(TempSection,n_c-timeInFirstColumn);
        }
        catch (...) {
            throw;
        }
        // Rewind file before reading next column:
        if (!ATF_RewindFile(nFileNum,&nError)) {
            wxString errorMsg(wxT("Exception while calling ATF_RewindFile():\n"));
            errorMsg+=ATFError(fName,nError);
            ReturnData.resize(0);
            throw std::runtime_error(std::string(errorMsg.c_str()));
        }
    }
    try {
        ReturnData.InsertChannel(TempChannel,0);
    }
    catch (...) {
        ReturnData.resize(0);
        throw;
    }

    if (!ATF_CloseFile(nFileNum)) {
        wxString errorMsg(wxT("Exception while calling ATF_CloseFile():\n"));
        errorMsg+=wxT("Error while closing ATF file");
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }
}
