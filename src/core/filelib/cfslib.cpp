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


#include "wx/wxprec.h"
#include "wx/progdlg.h"
#include <wx/msgdlg.h>

#include "./cfslib.h"
#include "./cfs.h"

namespace stf {

bool CFSError(wxString& errorMsg);
wxString CFSReadVar(short fHandle,short varNo,short varKind);

// Resource management of CFS files
// Management of read-only files:
class CFS_IFile {
public:
    explicit CFS_IFile(const wxString& filename);
    ~CFS_IFile();
    short myHandle;
};

// Management of write-only files:
class CFS_OFile  {
public:
    explicit CFS_OFile(
            const wxString& filename,
            const wxString& comment,
            std::size_t nChannels=1
    );
    ~CFS_OFile();
    short myHandle;
};

const int CFSMAXBYTES=64000; // adopted from FPCfs.ips by U Froebe

}

stf::CFS_IFile::CFS_IFile(const wxString& filename) {
    myHandle=OpenCFSFile(filename.char_str(),0,1);
}

stf::CFS_IFile::~CFS_IFile() {
    if (myHandle>0) {
        CloseCFSFile(myHandle);
    }
}

// Management of write-only files:
stf::CFS_OFile::CFS_OFile(const wxString& filename,const wxString& comment,std::size_t nChannels)
{
    TVarDesc *c_DSArray, *c_fileArray;
    c_DSArray=NULL;
    c_fileArray=NULL;
    myHandle=CreateCFSFile(filename.char_str(), comment.char_str(), 512, (short)nChannels,
        c_fileArray, c_DSArray, 0/*number of file vars*/,
        0/*number of section vars*/);
}

stf::CFS_OFile::~CFS_OFile() { CloseCFSFile(myHandle); }

bool stf::CFSError(wxString& errorMsg) {
    short pHandle;
    short pFunc;
    short pErr;
    if (!FileError(&pHandle,&pFunc,&pErr)) return false;
    errorMsg = wxT("Error in stf::");
    switch (pFunc) {
        case  (1): errorMsg += wxT("SetFileChan()"); break;
        case  (2): errorMsg += wxT("SetDSChan()"); break;
        case  (3): errorMsg += wxT("SetWriteData()"); break;
        case  (4): errorMsg += wxT("RemoveDS()"); break;
        case  (5): errorMsg += wxT("SetVarVal()"); break;
        case  (6): errorMsg += wxT("GetGenInfo()"); break;
        case  (7): errorMsg += wxT("GetFileInfo()"); break;
        case  (8): errorMsg += wxT("GetVarDesc()"); break;
        case  (9): errorMsg += wxT("GetVarVal()"); break;
        case (10): errorMsg += wxT("GetFileChan()"); break;
        case (11): errorMsg += wxT("GetDSChan()"); break;
        case (12): errorMsg += wxT("DSFlags()"); break;
        case (13): errorMsg += wxT("OpenCFSFile()"); break;
        case (14): errorMsg += wxT("GetChanData()"); break;
        case (15): errorMsg += wxT("SetComment()"); break;
        case (16): errorMsg += wxT("CommitCFSFile()"); break;
        case (17): errorMsg += wxT("InsertDS()"); break;
        case (18): errorMsg += wxT("CreateCFSFile()"); break;
        case (19): errorMsg += wxT("WriteData()"); break;
        case (20): errorMsg += wxT("ClearDS()"); break;
        case (21): errorMsg += wxT("CloseCFSFile()"); break;
        case (22): errorMsg += wxT("GetDSSize()"); break;
        case (23): errorMsg += wxT("ReadData()"); break;
        case (24): errorMsg += wxT("CFSFileSize()"); break;
        case (25): errorMsg += wxT("AppendDS()"); break;
        default  : errorMsg += wxT(", unknown function"); break;
    }
    errorMsg += wxT(":\n");
    switch (pErr) {
        case  (-1): errorMsg += wxT("No spare file handles."); break;
        case  (-2): errorMsg += wxT("File handle out of range 0-2."); break;
        case  (-3): errorMsg += wxT(" File not open for writing."); break;
        case  (-4): errorMsg += wxT("File not open for editing/writing."); break;
        case  (-5): errorMsg += wxT("File not open for editing/reading."); break;
        case  (-6): errorMsg += wxT("File not open."); break;
        case  (-7): errorMsg += wxT("The specified file is not a CFS file."); break;
        case  (-8): errorMsg += wxT("Unable to allocate the memory needed for the filing system data."); break;
        case (-11): errorMsg += wxT("Creation of file on disk failed (writing only)."); break;
        case (-12): errorMsg += wxT("Opening of file on disk failed (reading only)."); break;
        case (-13): errorMsg += wxT("Error reading from data file."); break;
        case (-14): errorMsg += wxT("Error writing to data file."); break;
        case (-15): errorMsg += wxT("Error reading from data section pointer file."); break;
        case (-16): errorMsg += wxT("Error writing to data section pointer file."); break;
        case (-17): errorMsg += wxT("Error seeking disk position."); break;
        case (-18): errorMsg += wxT("Error inserting final data section of the file."); break;
        case (-19): errorMsg += wxT("Error setting the file length."); break;
        case (-20): errorMsg += wxT("Invalid variable description."); break;
        case (-21): errorMsg += wxT("Parameter out of range 0-99."); break;
        case (-22): errorMsg += wxT("Channel number out of range"); break;
        case (-24): errorMsg += wxT("Invalid data section number (not in the range 1 to total number of sections)."); break;
        case (-25): errorMsg += wxT("Invalid variable kind (not 0 for file variable or 1 for DS variable)."); break;
        case (-26): errorMsg += wxT("Invalid variable number."); break;
        case (-27): errorMsg += wxT("Data size specified is out of the correct range."); break;
        case (-30): case (-31): case (-32): case (-33): case (-34): case (-35): case (-36): case (-37): case (-38):
        case (-39): errorMsg += wxT("Wrong CFS version number in file"); break;
        default   : errorMsg += wxT("An unknown error occurred"); break;
    }
    return true;
}

wxString stf::CFSReadVar(short fHandle,short varNo,short varKind) {
    wxString errorMsg;
    wxString outputstream;
    TUnits units;
    char description[22];
    short varSize=0;
    TDataType varType;
    //Get description of a particular file variable
    //- see manual of CFS file system
    GetVarDesc(fHandle,varNo,varKind,&varSize,&varType,units,description);
    if (CFSError(errorMsg)) throw std::runtime_error(std::string(errorMsg.char_str()));
    //I haven't found a way to directly pass a std::string to GetVarDesc;
    //passing &s_description[0] won't work correctly.
    // Added 11/27/06, CSH: Should be possible with vector<char>
    wxString s_description(wxString( description, wxConvLocal ));
    if (s_description != wxT("Spare")) {
        switch (varType) {   //Begin switch 'varType'
            case INT1:
            case INT2:
            case INT4: {
                short shortBuffer=0;
                //Read the value of the file variable
                //- see manual of CFS file system
                GetVarVal(fHandle,varNo,varKind, 1,&shortBuffer);
                if (CFSError(errorMsg))	throw std::runtime_error(std::string(errorMsg.char_str()));
                outputstream << s_description << wxT(" ") << shortBuffer << wxT(" ") << wxString( units, wxConvLocal );
                break;
                       }
            case WRD1:
            case WRD2: {
                unsigned short ushortBuffer=0;
                GetVarVal(fHandle,varNo,varKind, 1,&ushortBuffer);
                if (CFSError(errorMsg))	throw std::runtime_error(std::string(errorMsg.char_str()));
                outputstream << s_description << wxT(" ") << ushortBuffer << wxT(" ") << wxString( units, wxConvLocal );
                break;
                       }
            case RL4:
            case RL8: {
                float floatBuffer=0;
                GetVarVal(fHandle,varNo,varKind, 1,&floatBuffer);
                if (CFSError(errorMsg))	throw std::runtime_error(std::string(errorMsg.char_str()));
                outputstream << s_description << wxT(" ") << floatBuffer << wxT(" ") << wxString( units, wxConvLocal );
                break;
                      }
            case LSTR: {
                std::vector<char> vc(varSize+2);
                GetVarVal(fHandle,varNo,varKind, 1, &vc[0]);
                if (CFSError(errorMsg))	throw std::runtime_error(std::string(errorMsg.char_str()));
                std::string s(vc.begin(),vc.begin()+varSize+2);
                if (s_description.substr(0,11) == wxT("ScriptBlock")) {
                    outputstream << wxString( s.c_str(), wxConvLocal );
                } else {
                    outputstream << s_description << wxT(" ") << wxString( s.c_str(), wxConvLocal );
                }
                break;
                       }
            default: break;
        }	//End switch 'varType'
    }
    if (s_description.substr(0,11) != wxT("ScriptBlock") ) {
        outputstream += wxT("\n");
    }
    return outputstream;
}

bool stf::exportCFSFile(const wxString& fName, const Recording& WData) {
    wxProgressDialog progDlg(
        wxT("CED filing system export"),
        wxT("Starting file export"),
        100,
        NULL,
        wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL
        );
    wxString errorMsg;
    if (fName.length()>1024) {
        throw std::runtime_error(
            "Sorry for the inconvenience, but the CFS\n"
            "library is a bit picky with filenames.\n"
            "Please restrict yourself to less than\n"
            "1024 characters.\n"
            );
    }
    CFS_OFile CFSFile(fName,WData.GetComment(),WData.size());
    if (CFSFile.myHandle<0) {
        wxString errorMsg;
        CFSError(errorMsg);
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }
    for (std::size_t n_c=0;n_c<WData.size();++n_c) {
        SetFileChan(
            CFSFile.myHandle,
            (short)n_c,
            WData[n_c].GetChannelName().char_str(),
            WData[n_c].GetYUnits().char_str(),
            "ms\0" /* x units */,
            RL4 /* float */,
            EQUALSPACED /* MATRIX */,
            (short)(4*WData.size()) /* bytes between elements */,
            (short)n_c
            );
        if (CFSError(errorMsg))	throw std::runtime_error(std::string(errorMsg.char_str()));
    }

    for (int n_section=0; n_section < (int)WData.GetChannelSize(0); n_section++) {
        wxString progStr;
        progStr << wxT("Writing section #") << n_section+1 << wxT(" of ") << (int)WData.GetChannelSize(0);
        progDlg.Update(
            // Section contribution:
            (int)((double)n_section/(double)WData.GetChannelSize(0)*100.0),
            progStr
            );

        for (std::size_t n_c=0;n_c<WData.size();++n_c) {
            SetDSChan(
                CFSFile.myHandle,
                (short)n_c /* channel */,
                0  /* current section */,
                (long)(n_c*4)/*0*/ /* startOffset */,
                (long)WData[n_c][n_section].size(),
                1.0 /* yScale */,
                0  /* yOffset */,
                (float)WData.GetXScale(),
                0 /* x offset */
                );
            if (CFSError(errorMsg))	throw std::runtime_error(std::string(errorMsg.char_str()));
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
            if (CFSError(errorMsg))	throw std::runtime_error(std::string(errorMsg.char_str()));
        }	//End block loop
        InsertDS(CFSFile.myHandle, 0, noFlags);
        if (CFSError(errorMsg))	throw std::runtime_error(std::string(errorMsg.char_str()));
    }	//End section loop
    return true;
}

void stf::importCFSFile(const wxString& fName, Recording& ReturnData, bool progress ) {
    wxProgressDialog progDlg( wxT("CED filing system import"), wxT("Starting file import"),
        100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );
    wxString errorMsg;
    // Open old CFS File (read only) - see manual of CFS file system
    CFS_IFile CFSFile(fName);
    if (CFSFile.myHandle<0) {
        wxString errorMsg;
        CFSError(errorMsg);
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }

    //Get general Info of the file - see manual of CFS file system
    TDesc time, date;
    TComment comment;
    GetGenInfo(CFSFile.myHandle, time, date, comment);
    if (CFSError(errorMsg))	throw std::runtime_error(std::string(errorMsg.char_str()));
    //Get characteristics of the file - see manual of CFS file system
    short channelsAvail=0, fileVars=0, DSVars=0;
    unsigned short dataSections=0;
    GetFileInfo(CFSFile.myHandle, &channelsAvail, &fileVars, &DSVars, &dataSections);
    if (CFSError(errorMsg))	throw std::runtime_error(std::string(errorMsg.char_str()));

    //memory allocation
    ReturnData.resize(channelsAvail);

    //Variables to store the Descriptions of a single variable as text
    wxString	file_description,    //File variable
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
    wxString scaling;
    std::vector<long> points(dataSections);
    TDataType dataType;
    TCFSKind dataKind;
    short spacing, other;
    float xScale=1.0;
    std::size_t empty_channels=0;
    for (short n_channel=0; n_channel < channelsAvail; ++n_channel) {

        //Get constant information for a particular data channel -
        //see manual of CFS file system.
        std::vector<char> vchannel_name(22),vyUnits(10),vxUnits(10);
        long startOffset;
        GetFileChan(CFSFile.myHandle, n_channel, &vchannel_name[0],
            &vyUnits[0], &vxUnits[0], &dataType, &dataKind,
            &spacing, &other);
        if (CFSError(errorMsg))	throw std::runtime_error(std::string(errorMsg.char_str()));
        wxString channel_name(wxString( &vchannel_name[0], wxConvLocal )),
            xUnits(wxString( &vxUnits[0], wxConvLocal )),
            yUnits(wxString( &vyUnits[0], wxConvLocal ));
        //Memory allocation for the current channel
        float yScale, yOffset, xOffset;
        //Begin loop: read scaling and offsets
        //Write the formatted string from 'n_channel' and 'channel_name' to 'buffer'
        wxString outputstream;
        outputstream << wxT("Channel ") << n_channel << wxT(" (") << channel_name.c_str() << wxT(")\n");
        scaling += outputstream;
        //Get the channel information for a data section or a file
        //- see manual of CFS file system
        GetDSChan(CFSFile.myHandle, n_channel /*first channel*/, 1 /*first section*/, &startOffset,
            &points[0], &yScale, &yOffset,&xScale,&xOffset);
        if (CFSError(errorMsg))	throw std::runtime_error(std::string(errorMsg.char_str()));
        //Write the formatted string from 'yScale' to 'buffer'
        outputstream.clear();
        outputstream << wxT("Yscale=") <<  yScale << wxT("\n");
        scaling += outputstream;
        //Write the formatted string from 'xScale' to 'buffer'
        outputstream.clear();
        outputstream << wxT("Xscale=") <<  xScale << wxT("\n");
        scaling += outputstream;
        //Write the formatted string from 'yOffset' to 'buffer'
        outputstream.clear();
        outputstream << wxT("YOffset=") <<  yOffset << wxT("\n");
        scaling += outputstream;
        //Write the formatted string from 'xOffset' to 'buffer'
        outputstream.clear();
        outputstream << wxT("XOffset=") <<  xOffset << wxT("\n");
        scaling += outputstream;

        Channel TempChannel(dataSections);
        TempChannel.SetChannelName(channel_name);
        TempChannel.SetYUnits(yUnits);
        std::size_t empty_sections=0;
        for (int n_section=0; n_section < dataSections; ++n_section) {
            wxString progStr;
            if (progress) {
                progStr << wxT("Reading channel #") << n_channel + 1 << wxT(" of ") << channelsAvail
                        << wxT(", Section #") << n_section+1 << wxT(" of ") << dataSections;
                progDlg.Update(
                               // Channel contribution:
                               (int)(((double)n_channel/(double)channelsAvail)*100.0+
                                     // Section contribution:
                                     (double)n_section/(double)dataSections*(100.0/channelsAvail)),
                               progStr
                               );
            }
            //Begin loop: n_sections
            //Get the channel information for a data section or a file
            //- see manual of CFS file system
            long startOffset;
            float yScale, yOffset, xOffset;
            GetDSChan(CFSFile.myHandle,(short)n_channel,(WORD)n_section+1,&startOffset,
                &points[n_section],&yScale,&yOffset,&xScale,&xOffset);
            if (CFSError(errorMsg))	throw std::runtime_error(std::string(errorMsg.char_str()));
            wxString label;
            label << stf::noPath(fName) << wxT(", Section # ") << n_section+1;
            Section TempSection(
                (int)(points[n_section]),
                label
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
                    if (CFSError(errorMsg))	throw std::runtime_error(std::string(errorMsg.char_str()));
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
                    if (CFSError(errorMsg))	throw std::runtime_error(std::string(errorMsg.char_str()));
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
    ReturnData.SetXScale(xScale);
    ReturnData.SetFileDescription(file_description);
    ReturnData.SetGlobalSectionDescription(section_description);
    ReturnData.SetScaling(scaling);
    ReturnData.SetTime( wxString( time, wxConvLocal) );
    ReturnData.SetDate(wxString( date, wxConvLocal) );
    ReturnData.SetComment(wxString( comment, wxConvLocal) );
}
