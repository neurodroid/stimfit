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

#include "./atflib.h"
#include <iostream>
#include <sstream>

namespace stfio {
    std::string ATFError(const std::string& fName, int nError);
}

std::string stfio::ATFError(const std::string& fName, int nError) {
    int nMaxLen=320;
    std::vector<char> errorMsg(nMaxLen);
    ATF_BuildErrorText(nError, fName.c_str(),&errorMsg[0], nMaxLen );
    return std::string( &errorMsg[0] );
}

bool stfio::exportATFFile(const std::string& fName, const Recording& WData) {
    int nColumns=1+(int)WData[0].size() /*time + number of sections*/, nFileNum;
    int nError;

    if (!ATF_OpenFile(fName.c_str(),ATF_WRITEONLY,&nColumns,&nFileNum,&nError)) {
        std::string errorMsg("Exception while calling ATF_OpenFile():\n");
        errorMsg += ATFError(fName,nError);
        throw std::runtime_error(errorMsg);
    }
    // Write sections to columns:
    // First column is time:
    for (int n_c=0;n_c<nColumns;++n_c) {
        std::string columnTitle,columnUnits;
        if (n_c==0) {
            columnTitle = "Time";
            columnUnits = WData.GetXUnits();
        } else {
            std::ostringstream titleStr;
            titleStr << "Section[" << n_c-1 << "]";
            columnTitle = titleStr.str();
            columnUnits = WData[0].GetYUnits();
        }
        if (!ATF_SetColumnTitle(nFileNum, columnTitle.c_str(), &nError)) {
            std::string errorMsg("Exception while calling ATF_SetColumnTitle():\n");
            errorMsg+=ATFError(fName,nError);
            throw std::runtime_error(errorMsg);
        }
        if (!ATF_SetColumnUnits(nFileNum, columnUnits.c_str(),&nError)) {
            std::string errorMsg("Exception while calling ATF_SetColumnUnits():\n");
            errorMsg+=ATFError(fName,nError);
            throw std::runtime_error(errorMsg);
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
                    std::string errorMsg("Exception while calling ATF_WriteDataRecord1():\n");
                    errorMsg+=ATFError(fName,nError);
                    throw std::runtime_error(errorMsg);
                }
            } else {
                double toWrite = (n_l < (int)WData[0][n_c-1].size()) ?
                        (double)WData[0][n_c-1][n_l] :
                0.0;
                        if (!ATF_WriteDataRecord1(nFileNum,toWrite,&nError)) {
                            std::string errorMsg("Exception while calling ATF_WriteDataRecord1():\n");
                            errorMsg+=ATFError(fName,nError);
                            throw std::runtime_error(errorMsg);
                        }
            }
        }
        if (!ATF_WriteEndOfLine(nFileNum,&nError)) {
            std::string errorMsg("Exception while calling ATF_WriteEndOfLine():\n");
            errorMsg+=ATFError(fName,nError);
            throw std::runtime_error(errorMsg);
        }
    }
    if (!ATF_CloseFile(nFileNum)) {
        std::string errorMsg("Exception while calling ATF_CloseFile():\n");
        errorMsg += "Error while closing ATF file";
        throw std::runtime_error(errorMsg);
    }
    return true;
}

void stfio::importATFFile(const std::string &fName, Recording &ReturnData, ProgressInfo& progDlg) {
    int nColumns, nFileNum;
    int nError;
    const int nMaxText=64;

    if (!ATF_OpenFile(fName.c_str(),ATF_READONLY,&nColumns,&nFileNum,&nError)) {
        std::string errorMsg("Exception while calling ATF_OpenFile():\n");
        errorMsg+=ATFError(fName,nError);
        throw std::runtime_error(errorMsg);
    }
    // Assume that the first column is time:
    if (nColumns==0) {
        std::string errorMsg("Error while opening ATF file:\nFile appears to be empty");
        throw std::runtime_error(errorMsg);
    }
    long sectionSize;
    if (!ATF_CountDataLines(nFileNum,&sectionSize,&nError)) {
        std::string errorMsg("Exception while calling ATF_CountDataLines():\n");
        errorMsg+=ATFError(fName,nError);
        throw std::runtime_error(errorMsg);
    }

    // If first column contains time values, determine sampling interval:
    std::vector<char> titleVec(nMaxText);
    if (!ATF_GetColumnTitle(nFileNum,0,&titleVec[0],nMaxText,&nError)) {
        std::string errorMsg("Exception while calling ATF_GetColumnTitle():\n");
        errorMsg+=ATFError(fName,nError);
        throw std::runtime_error(errorMsg);
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
                std::string errorMsg("Exception while calling ATF_ReadDataColumn():\n");
                errorMsg+=ATFError(fName,nError);
                throw std::runtime_error(errorMsg);
            }
        }
        if (!ATF_RewindFile(nFileNum,&nError)) {
            std::string errorMsg("Exception while calling ATF_RewindFile():\n");
            errorMsg+=ATFError(fName,nError);
            throw std::runtime_error(errorMsg);
        }

        ReturnData.SetXScale(time[1]-time[0]);
        timeInFirstColumn=1;
    }
    ReturnData.resize(1);
    Channel TempChannel(nColumns-timeInFirstColumn);
    for (int n_c=timeInFirstColumn;n_c<nColumns;++n_c) {
        int progbar = (double)100.0*(n_c+1-timeInFirstColumn)/(double)(nColumns-timeInFirstColumn);

        std::ostringstream progStr;
        progStr << "Section #" << n_c+1-timeInFirstColumn << " of " << nColumns-timeInFirstColumn;
        progDlg.Update(progbar, progStr.str());
        std::ostringstream label;
        label
            << fName 
            << ", Section # " << n_c-timeInFirstColumn+1;
        Section TempSection(sectionSize,label.str());
        for (int n_l=0;n_l<sectionSize;++n_l) {
            if (!ATF_ReadDataColumn(nFileNum,n_c,&TempSection[n_l],&nError)) {
                std::string errorMsg("Exception while calling ATF_ReadDataColumn():\n");
                errorMsg+=ATFError(fName,nError);
                ReturnData.resize(0);
                throw std::runtime_error(errorMsg);
            }
        }
        if (n_c-timeInFirstColumn==0) {
            std::vector<char> unitsVec(nMaxText);
            if (!ATF_GetColumnUnits(nFileNum,n_c,&unitsVec[0],nMaxText,&nError)) {
                std::string errorMsg("Exception while calling ATF_GetColumnUnits():\n");
                errorMsg+=ATFError(fName,nError);
                ReturnData.resize(0);
                throw std::runtime_error(errorMsg);
            }
            ReturnData[0].SetYUnits(std::string(&unitsVec[0]));
        }
        try {
            TempChannel.InsertSection(TempSection,n_c-timeInFirstColumn);
        }
        catch (...) {
            throw;
        }
        // Rewind file before reading next column:
        if (!ATF_RewindFile(nFileNum,&nError)) {
            std::string errorMsg("Exception while calling ATF_RewindFile():\n");
            errorMsg+=ATFError(fName,nError);
            ReturnData.resize(0);
            throw std::runtime_error(errorMsg);
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
        std::string errorMsg("Exception while calling ATF_CloseFile():\n");
        errorMsg += "Error while closing ATF file";
        throw std::runtime_error(errorMsg);
    }
#ifdef MODULE_ONLY
    if (progress) {
        std::cout << "\r";
        std::cout << "100%" << std::endl;
    }
#endif
}
