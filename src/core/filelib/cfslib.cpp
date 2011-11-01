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
#include <wx/msgdlg.h>
#endif

#include <iostream>
#include <sstream>

#include "./cfslib.h"
#include "./cfs.h"

namespace stf {

int CFSError(std::string& errorMsg);
std::string CFSReadVar(short fHandle,short varNo,short varKind);

// Resource management of CFS files
// Management of read-only files:
class CFS_IFile {
public:
    explicit CFS_IFile(const std::string& filename);
    ~CFS_IFile();
    short myHandle;
};

// Management of write-only files:
class CFS_OFile  {
public:
    explicit CFS_OFile(
            const std::string& filename,
            const std::string& comment,
            std::size_t nChannels=1
    );
    ~CFS_OFile();
    short myHandle;
};

const int CFSMAXBYTES=64000; // adopted from FPCfs.ips by U Froebe

}

stf::CFS_IFile::CFS_IFile(const std::string& filename) {
    myHandle = OpenCFSFile(filename.c_str(),0,1);
}

stf::CFS_IFile::~CFS_IFile() {
    if (myHandle>0) {
        CloseCFSFile(myHandle);
    }
}

// Management of write-only files:
stf::CFS_OFile::CFS_OFile(const std::string& filename,const std::string& comment,std::size_t nChannels)
{
    TVarDesc *c_DSArray, *c_fileArray;
    c_DSArray=NULL;
    c_fileArray=NULL;
    myHandle=CreateCFSFile(filename.c_str(), comment.c_str(), 512, (short)nChannels,
        c_fileArray, c_DSArray, 0/*number of file vars*/,
        0/*number of section vars*/);
}

stf::CFS_OFile::~CFS_OFile() { CloseCFSFile(myHandle); }

int stf::CFSError(std::string& errorMsg) {
    short pHandle;
    short pFunc;
    short pErr;
    if (!FileError(&pHandle,&pFunc,&pErr)) return 0;
    errorMsg = "Error in stf::";
    switch (pFunc) {
        case  (1): errorMsg += "SetFileChan()"; break;
        case  (2): errorMsg += "SetDSChan()"; break;
        case  (3): errorMsg += "SetWriteData()"; break;
        case  (4): errorMsg += "RemoveDS()"; break;
        case  (5): errorMsg += "SetVarVal()"; break;
        case  (6): errorMsg += "GetGenInfo()"; break;
        case  (7): errorMsg += "GetFileInfo()"; break;
        case  (8): errorMsg += "GetVarDesc()"; break;
        case  (9): errorMsg += "GetVarVal()"; break;
        case (10): errorMsg += "GetFileChan()"; break;
        case (11): errorMsg += "GetDSChan()"; break;
        case (12): errorMsg += "DSFlags()"; break;
        case (13): errorMsg += "OpenCFSFile()"; break;
        case (14): errorMsg += "GetChanData()"; break;
        case (15): errorMsg += "SetComment()"; break;
        case (16): errorMsg += "CommitCFSFile()"; break;
        case (17): errorMsg += "InsertDS()"; break;
        case (18): errorMsg += "CreateCFSFile()"; break;
        case (19): errorMsg += "WriteData()"; break;
        case (20): errorMsg += "ClearDS()"; break;
        case (21): errorMsg += "CloseCFSFile()"; break;
        case (22): errorMsg += "GetDSSize()"; break;
        case (23): errorMsg += "ReadData()"; break;
        case (24): errorMsg += "CFSFileSize()"; break;
        case (25): errorMsg += "AppendDS()"; break;
        default  : errorMsg += ", unknown function"; break;
    }
    errorMsg += ":\n";
    switch (pErr) {
        case  (-1): errorMsg += "No spare file handles."; break;
        case  (-2): errorMsg += "File handle out of range 0-2."; break;
        case  (-3): errorMsg += " File not open for writing."; break;
        case  (-4): errorMsg += "File not open for editing/writing."; break;
        case  (-5): errorMsg += "File not open for editing/reading."; break;
        case  (-6): errorMsg += "File not open."; break;
        case  (-7): errorMsg += "The specified file is not a CFS file."; break;
        case  (-8): errorMsg += "Unable to allocate the memory needed for the filing system data."; break;
        case (-11): errorMsg += "Creation of file on disk failed (writing only)."; break;
        case (-12): errorMsg += "Opening of file on disk failed (reading only)."; break;
        case (-13): errorMsg += "Error reading from data file."; break;
        case (-14): errorMsg += "Error writing to data file."; break;
        case (-15): errorMsg += "Error reading from data section pointer file."; break;
        case (-16): errorMsg += "Error writing to data section pointer file."; break;
        case (-17): errorMsg += "Error seeking disk position."; break;
        case (-18): errorMsg += "Error inserting final data section of the file."; break;
        case (-19): errorMsg += "Error setting the file length."; break;
        case (-20): errorMsg += "Invalid variable description."; break;
        case (-21): errorMsg += "Parameter out of range 0-99."; break;
        case (-22): errorMsg += "Channel number out of range"; break;
        case (-24): errorMsg += "Invalid data section number (not in the range 1 to total number of sections)."; break;
        case (-25): errorMsg += "Invalid variable kind (not 0 for file variable or 1 for DS variable)."; break;
        case (-26): errorMsg += "Invalid variable number."; break;
        case (-27): errorMsg += "Data size specified is out of the correct range."; break;
        case (-30): case (-31): case (-32): case (-33): case (-34): case (-35): case (-36): case (-37): case (-38):
        case (-39): errorMsg += "Wrong CFS version number in file"; break;
        default   : errorMsg += "An unknown error occurred"; break;
    }
    return pErr;
}

std::string stf::CFSReadVar(short fHandle,short varNo,short varKind) {
    std::string errorMsg;
    std::ostringstream outputstream;
    TUnits units;
    char description[1024];
    short varSize=0;
    TDataType varType;
    //Get description of a particular file variable
    //- see manual of CFS file system
    GetVarDesc(fHandle,varNo,varKind,&varSize,&varType,units,description);
    if (CFSError(errorMsg)) throw std::runtime_error(errorMsg);
    //I haven't found a way to directly pass a std::string to GetVarDesc;
    //passing &s_description[0] won't work correctly.
    // Added 11/27/06, CSH: Should be possible with vector<char>
    std::string s_description(description);
    if (s_description != "Spare") {
        switch (varType) {   //Begin switch 'varType'
            case INT1:
            case INT2:
            case INT4: {
                short shortBuffer=0;
                //Read the value of the file variable
                //- see manual of CFS file system
                GetVarVal(fHandle,varNo,varKind, 1,&shortBuffer);
                if (CFSError(errorMsg))	throw std::runtime_error(errorMsg);
                outputstream << s_description << " " << shortBuffer << " " << units;
                break;
                       }
            case WRD1:
            case WRD2: {
                unsigned short ushortBuffer=0;
                GetVarVal(fHandle,varNo,varKind, 1,&ushortBuffer);
                if (CFSError(errorMsg))	throw std::runtime_error(errorMsg);
                outputstream << s_description << " " << ushortBuffer << " " << units;
                break;
                       }
            case RL4:
            case RL8: {
                float floatBuffer=0;
                GetVarVal(fHandle,varNo,varKind, 1,&floatBuffer);
                if (CFSError(errorMsg))	throw std::runtime_error(errorMsg);
                outputstream << s_description << " " << floatBuffer << " " << units;
                break;
                      }
            case LSTR: {
                std::vector<char> vc(varSize+2);
                GetVarVal(fHandle,varNo,varKind, 1, &vc[0]);
                if (CFSError(errorMsg))	throw std::runtime_error(errorMsg);
                std::string s(vc.begin(),vc.end());
                /* std::cout << &vc[0] << std::endl;
                   if (s_description.substr(0,11) == "ScriptBlock") {*/
                outputstream << s_description << " " << s;
                /*} else {
                    outputstream << s_description << " " << s;
                    }*/
                break;
                       }
            default: break;
        }	//End switch 'varType'
    }
    if (s_description.substr(0,11) != "ScriptBlock" ) {
        outputstream << "\n";
    }
    return outputstream.str();
}

bool stf::exportCFSFile(const wxString& fName, const Recording& WData) {
#ifndef MODULE_ONLY
    wxProgressDialog progDlg(
        wxT("CED filing system export"),
        wxT("Starting file export"),
        100,
        NULL,
        wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL
                             );
#endif
    std::string errorMsg;
    if (fName.length()>1024) {
        throw std::runtime_error(
            "Sorry for the inconvenience, but the CFS\n"
            "library is a bit picky with filenames.\n"
            "Please restrict yourself to less than\n"
            "1024 characters.\n"
            );
    }
#if (defined(MODULE_ONLY) || wxCHECK_VERSION(2, 9, 0))
    CFS_OFile CFSFile(std::string(fName.c_str()),std::string(WData.GetComment().c_str()),WData.size());
#else
    CFS_OFile CFSFile(std::string(fName.mb_str()), WData.GetComment(),WData.size());
#endif
    if (CFSFile.myHandle<0) {
        std::string errorMsg;
        CFSError(errorMsg);
        throw std::runtime_error(errorMsg);
    }
    for (std::size_t n_c=0;n_c<WData.size();++n_c) {
        SetFileChan(
            CFSFile.myHandle,
            (short)n_c,
            WData[n_c].GetChannelName().c_str(),
            WData[n_c].GetYUnits().c_str(),
            "ms\0" /* x units */,
            RL4 /* float */,
            EQUALSPACED /* MATRIX */,
            (short)(4*WData.size()) /* bytes between elements */,
            (short)n_c
            );
        if (CFSError(errorMsg))	throw std::runtime_error(errorMsg);
    }

    for (int n_section=0; n_section < (int)WData.GetChannelSize(0); n_section++) {
        int progbar =
            // Section contribution:
            (int)((double)n_section/(double)WData.GetChannelSize(0)*100.0);
#ifndef MODULE_ONLY
        wxString progStr;
        progStr << wxT("Writing section #") << n_section+1 << wxT(" of ") << (int)WData.GetChannelSize(0);
        progDlg.Update(progbar, progStr);
#else
        std::cout << "\r";
        std::cout << progbar << "%" << std::flush;
#endif
        for (std::size_t n_c=0;n_c<WData.size();++n_c) {
            SetDSChan(
                CFSFile.myHandle,
                (short)n_c /* channel */,
                0  /* current section */,
                (CFSLONG)(n_c*4)/*0*/ /* startOffset */,
                (CFSLONG)WData[n_c][n_section].size(),
                1.0 /* yScale */,
                0  /* yOffset */,
                (float)WData.GetXScale(),
                0 /* x offset */
                );
            if (CFSError(errorMsg))	throw std::runtime_error(errorMsg);
        }

        int maxBytes=CFSMAXBYTES/(int)WData.size();
        // Section loop
        int nBlocks, nBlockBytes, nStartByteOffset;
        nBlocks=(int)(((WData[0][n_section].size()*4-1)/maxBytes) + 1);

        for (int b=0; b < nBlocks; b++) {
            // Block loop
            nStartByteOffset=b*maxBytes*(int)WData.size();
            if (b == nBlocks -1)
                nBlockBytes=(int)WData[0][n_section].size()*(int)WData.size()*4 -
                b*maxBytes*(int)WData.size();
            else
                nBlockBytes=maxBytes*(int)WData.size();

            Vector_float faverage_small(nBlockBytes/4);

            for (int n_point=0; n_point < nBlockBytes/4/(int)WData.size(); n_point++) {
                for (std::size_t n_c=0;n_c<WData.size();++n_c) {
                    faverage_small[n_point*WData.size()+n_c]=
                        (float)WData[n_c][n_section][n_point + b*maxBytes/4];
                }
            }
            if (faverage_small.size()==0) {
                std::runtime_error e("array has size zero in exportCFSFile()");
                throw e;
            }
            WriteData(
                CFSFile.myHandle,
                0  /* "0" means current section */,
                nStartByteOffset /* byte offset */,
                (WORD)nBlockBytes,
                &faverage_small[0]
            );
            if (CFSError(errorMsg))	throw std::runtime_error(errorMsg);
        }	//End block loop
        InsertDS(CFSFile.myHandle, 0, noFlags);
        if (CFSError(errorMsg))	throw std::runtime_error(errorMsg);
    }	//End section loop
#ifdef MODULE_ONLY
    std::cout << "\r";
    std::cout << "100%" << std::endl;
#endif
    
    return true;
}

int stf::importCFSFile(const wxString& fName, Recording& ReturnData, bool progress ) {
#ifndef MODULE_ONLY
    wxProgressDialog progDlg( wxT("CED filing system import"), wxT("Starting file import"),
                              100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );
#endif
    
    std::string errorMsg;
    // Open old CFS File (read only) - see manual of CFS file system
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
    CFS_IFile CFSFile(std::string(fName.c_str()));
#else
    CFS_IFile CFSFile(std::string(fName.mb_str()));
#endif
    if (CFSFile.myHandle<0) {
        int err = CFSError(errorMsg);
        if (err==-7) {
            return err;
        }
        errorMsg = std::string("Error while opening file:\n") + errorMsg;
        throw std::runtime_error(errorMsg.c_str());
    }

    //Get general Info of the file - see manual of CFS file system
    TDesc time, date;
    TComment comment;
    GetGenInfo(CFSFile.myHandle, time, date, comment);
    if (CFSError(errorMsg))
        throw std::runtime_error(std::string("Error in GetGenInfo:\n") + errorMsg);
    //Get characteristics of the file - see manual of CFS file system
    short channelsAvail=0, fileVars=0, DSVars=0;
    unsigned short dataSections=0;
    GetFileInfo(CFSFile.myHandle, &channelsAvail, &fileVars, &DSVars, &dataSections);
    if (CFSError(errorMsg))
        throw std::runtime_error(errorMsg);

    //memory allocation
    ReturnData.resize(channelsAvail);

    //Variables to store the Descriptions of a single variable as text
    std::string	file_description,    //File variable
        section_description; //Data section variable

    //1. Read file variables
    for (short n_filevar=0; n_filevar < fileVars; ++n_filevar) {
        //Begin loop: read file variables
        try {
            file_description += CFSReadVar(CFSFile.myHandle,n_filevar,FILEVAR);
        }
        catch (...) {
            throw;
        }
    }//End loop: read file variables

    //2. Data Section variables
    for (short n_sectionvar=0; n_sectionvar < DSVars; ++n_sectionvar)
    {   //Begin loop: read data section variables
        try {
            section_description+=CFSReadVar(CFSFile.myHandle,n_sectionvar,DSVAR);
        }
        catch (...) {
            throw;
        }
    }	//End loop read: data section variables

    //3. Description of scaling factors and offsets
    //can't be read with GetVarVal() since they might change from section
    //to section
    std::string scaling;
    std::vector<CFSLONG> points(dataSections);
    TDataType dataType;
    TCFSKind dataKind;
    short spacing, other;
    float xScale=1.0;
    std::size_t empty_channels=0;
    for (short n_channel=0; n_channel < channelsAvail; ++n_channel) {

        //Get constant information for a particular data channel -
        //see manual of CFS file system.
        std::vector<char> vchannel_name(22),vyUnits(10),vxUnits(10);
        CFSLONG startOffset;
        GetFileChan(CFSFile.myHandle, n_channel, &vchannel_name[0],
            &vyUnits[0], &vxUnits[0], &dataType, &dataKind,
            &spacing, &other);
        if (CFSError(errorMsg))	throw std::runtime_error(errorMsg);
        std::string channel_name(&vchannel_name[0]),
            xUnits(&vxUnits[0]),
            yUnits(&vyUnits[0]);
        //Memory allocation for the current channel
        float yScale, yOffset, xOffset;
        //Begin loop: read scaling and offsets
        //Write the formatted string from 'n_channel' and 'channel_name' to 'buffer'
        std::ostringstream outputstream;
        outputstream << "Channel " << n_channel << " (" << channel_name.c_str() << ")\n";
        scaling += outputstream.str();
        //Get the channel information for a data section or a file
        //- see manual of CFS file system
        GetDSChan(CFSFile.myHandle, n_channel /*first channel*/, 1 /*first section*/, &startOffset,
            &points[0], &yScale, &yOffset,&xScale,&xOffset);
        if (CFSError(errorMsg))	throw std::runtime_error(errorMsg);
        //Write the formatted string from 'yScale' to 'buffer'
        outputstream.clear();
        outputstream << "Yscale=" <<  yScale << "\n";
        scaling += outputstream.str();
        //Write the formatted string from 'xScale' to 'buffer'
        outputstream.clear();
        outputstream << "Xscale=" <<  xScale << "\n";
        scaling += outputstream.str();
        //Write the formatted string from 'yOffset' to 'buffer'
        outputstream.clear();
        outputstream << "YOffset=" <<  yOffset << "\n";
        scaling += outputstream.str();
        //Write the formatted string from 'xOffset' to 'buffer'
        outputstream.clear();
        outputstream << "XOffset=" <<  xOffset << "\n";
        scaling += outputstream.str();

        Channel TempChannel(dataSections);
        TempChannel.SetChannelName(channel_name);
        TempChannel.SetYUnits(yUnits);
        std::size_t empty_sections=0;
        for (int n_section=0; n_section < dataSections; ++n_section) {
            if (progress) {
                int progbar =
                    // Channel contribution:
                    (int)(((double)n_channel/(double)channelsAvail)*100.0+
                          // Section contribution:
                          (double)n_section/(double)dataSections*(100.0/channelsAvail));
#ifndef MODULE_ONLY
                wxString progStr;
                progStr << wxT("Reading channel #") << n_channel + 1 << wxT(" of ") << channelsAvail
                        << wxT(", Section #") << n_section+1 << wxT(" of ") << dataSections;
                progDlg.Update(progbar, progStr);
#else
                std::cout << "\r";
                std::cout << progbar << "%" << std::flush;
#endif
            }
            
            //Begin loop: n_sections
            //Get the channel information for a data section or a file
            //- see manual of CFS file system
            CFSLONG startOffset;
            float yScale, yOffset, xOffset;
            GetDSChan(CFSFile.myHandle,(short)n_channel,(WORD)n_section+1,&startOffset,
                &points[n_section],&yScale,&yOffset,&xScale,&xOffset);
            if (CFSError(errorMsg))	throw std::runtime_error(errorMsg);
            std::ostringstream label;
            label << fName << ", Section # " << n_section+1;
            Section TempSection(
                (int)(points[n_section]),
                label.str()
            );
            //-----------------------------------------------------
            //The following part was modified to read data sections
            //larger than 64 KB as e.g. produced by Igor.
            //Adopted from FPCfs.ipf by U Froebe
            //Sections with a size larger than 64 KB have been made
            //possible by dividing CFS-sections into 'blocks'
            //-----------------------------------------------------
            int nBlocks, //number of blocks
                nBlockBytes; //number of bytes per block

            //Calculation of the number of blocks depending on the data format:
            //RL4 - 4 byte floating point numbers (2 byte int numbers otherwise)
            if (dataType == RL4)
                nBlocks=(int)(((points[n_section]*4-1)/CFSMAXBYTES) + 1);
            else
                nBlocks=(int)(((points[n_section]*2-1)/CFSMAXBYTES) + 1);

            for (int b=0; b < nBlocks; ++b) {
                //Begin loop: storage of blocks
                if (dataType == RL4) {
                    //4 byte data
                    //Read data of the current channel and data section
                    //- see manual of CFS file system
                    //Temporary arrays to store blocks:
                    if (b == nBlocks - 1)
                        nBlockBytes=points[n_section]*4 - b*CFSMAXBYTES;
                    else
                        nBlockBytes=CFSMAXBYTES;
                    Vector_float fTempSection_small(nBlockBytes);
                    GetChanData(CFSFile.myHandle, (short)n_channel, (WORD)n_section+1,
                        b*CFSMAXBYTES/4, (WORD)nBlockBytes/4, &fTempSection_small[0],
                        4*(points[n_section]+1));
                    if (CFSError(errorMsg))	throw std::runtime_error(errorMsg);
                    for (int n=0; n<nBlockBytes/4; ++n) {
                        TempSection[n + b*CFSMAXBYTES/4]=
                            fTempSection_small[n]* yScale +
                            yOffset;
                    }
                } else {
                    //2 byte data
                    //Read data of the current channel and data section
                    //- see manual of CFS file system
                    if (b == nBlocks - 1)
                        nBlockBytes=points[n_section]*2 - b*CFSMAXBYTES;
                    else
                        nBlockBytes=CFSMAXBYTES;
                    std::vector<short> TempSection_small(nBlockBytes);
                    GetChanData(CFSFile.myHandle, (short)n_channel, (WORD)n_section+1,
                        b*CFSMAXBYTES/2, (WORD)nBlockBytes/2, &TempSection_small[0],
                        2*(points[n_section]+1));
                    if (CFSError(errorMsg))	throw std::runtime_error(errorMsg);
                    for (int n=0; n<nBlockBytes/2; ++n) {
                        TempSection[n + b*CFSMAXBYTES/2]=
                            TempSection_small[n]* yScale +
                            yOffset;
                    }
                }
            }	//End loop: storage of blocks
            //-----------------------------------------------------
            //End of the modified part to read data sections larger than
            //64kB (as produced e.g. by Igor)
            //-----------------------------------------------------
            try {
                if (TempSection.size()!=0) {
                    TempChannel.InsertSection(TempSection,n_section-empty_sections);
                } else {
                    empty_sections++;
                    TempChannel.resize(TempChannel.size()-1);
                }
            }
            catch (...) {
                throw;
            }
        }	//End loop: n_section
        try {
            if (TempChannel.size()!=0) {
                ReturnData.InsertChannel(TempChannel,n_channel-empty_channels);
            } else {
                empty_channels++;
                ReturnData.resize(ReturnData.size()-1);
            }
        }
        catch (...) {
            ReturnData.resize(0);
            throw;
        }
    }	//Begin loop: n_channel
    ReturnData.SetXScale(xScale + '\0');
    ReturnData.SetFileDescription(file_description + '\0');
    ReturnData.SetGlobalSectionDescription(section_description + '\0');
    ReturnData.SetScaling(scaling);
    ReturnData.SetTime(time);
    ReturnData.SetDate(date);
    ReturnData.SetComment(comment);
#ifdef MODULE_ONLY
    if (progress) {
        std::cout << "\r";
        std::cout << "100%" << std::endl;
    }
#endif
    return 0;
}
