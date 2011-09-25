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
#include "./../core.h"
#include "./axg/fileUtils.h"
#include "./axg/AxoGraph_ReadWrite.h"
#include "./axg/longdef.h"
#include "./axglib.h"

void stf::importAXGFile(const wxString &fName, Recording &ReturnData, bool progress, wxWindow* parent) {
#ifndef MODULE_ONLY
    wxProgressDialog progDlg( wxT("Axograph binary file import"), wxT("Starting file import"),
                              100, parent, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL | wxPD_CAN_SKIP );
#endif
    std::string errorMsg("Exception while calling AXG_importAXGFile():\n");
    std::string yunits;
    // =====================================================================================================================
    //
    // Open an AxoGraph file and read in the data
    //
    // =====================================================================================================================

    // Open the example file
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
    filehandle dataRefNum = OpenFile( fName.c_str() );
#else
    filehandle dataRefNum = OpenFile( fName.mb_str() );
#endif
    
    if ( dataRefNum == 0 )
    {
        errorMsg += "\n\nError: Could not find file.";
        ReturnData.resize(0);
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }

    // check the AxoGraph header, and get the number of columns to be read
    int fileFormat = 0;
    int result = AG_GetFileFormat( dataRefNum, &fileFormat );
    if ( result )
    {
        errorMsg += "\nError from AG_GetFileFormat - ";
        if ( result == kAG_FormatErr )
            errorMsg += "file is not in AxoGraph format";
        else if ( result == kAG_VersionErr )
            errorMsg += "file is of a more recent version than supported by this code";
        else
            errorMsg += "error";

        ReturnData.resize(0);
        CloseFile( dataRefNum );
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }

    AXGLONG numberOfColumns = 0;
    result = AG_GetNumberOfColumns( dataRefNum, fileFormat, &numberOfColumns );
    if ( result )
    {
        errorMsg += "Error from AG_GetNumberOfColumns";
        ReturnData.resize(0);
        CloseFile( dataRefNum );
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }

    // Sanity check
    if ( numberOfColumns <= 0 )  	// negative columns
    {
        errorMsg += "File format error: number of columns is negative in AxoGraph data file";
        ReturnData.resize(0);
        CloseFile( dataRefNum );
        throw std::runtime_error(std::string(errorMsg.c_str()));
    }

    //	AG_ReadFloatColumn reads column data into a float column structure.
    int numberOfChannels = 0;

    std::vector< Section > section_list;
    std::vector< std::string > channel_names;
    std::vector< std::string > channel_units;
    double xscale = 1.0;
    for ( int columnNumber=0; columnNumber<numberOfColumns; columnNumber++ )
    {
        if (progress && columnNumber != 0) {
            int progbar = (double)columnNumber/(double)numberOfColumns * 100.0;
#ifndef MODULE_ONLY
            wxString progStr;
            progStr << wxT("Section #") << columnNumber << wxT(" of ") << numberOfColumns-1;
            bool skip = false;
            progDlg.Update(progbar, progStr, &skip);
            if (skip) {
                ReturnData.resize(0);
                return;
            }
#else
            std::cout << "\r";
            std::cout << progbar << "%" << std::flush;
#endif
        }

        ColumnData column;
        result = AG_ReadFloatColumn( dataRefNum, fileFormat, columnNumber, &column );

        if ( result )
        {
            errorMsg += "Error from AG_ReadFloatColumn";
            ReturnData.resize(0);
            CloseFile( dataRefNum );
            throw std::runtime_error(std::string(errorMsg.c_str()));
        }
        if ( columnNumber == 0 ) {
            xscale = column.seriesArray.increment * 1.0e3;
        } else {
            section_list.push_back( Section(column.points, column.title ) );
            std::size_t last = section_list.size()-1;

            if (column.points<1) {
                throw std::out_of_range("number of points too small");
            }
            if ((int)column.floatArray.size()<column.points) {
                throw std::out_of_range("floatArray too small in importAXGFile()");
            }
            if (section_list[last].get().size()!=column.floatArray.size()) {
                throw std::out_of_range("section too small in importAXGFile()");
            }

            std::copy(column.floatArray.begin(),column.floatArray.end(),section_list[last].get_w().begin());
            // check whether this is a new channel:
            bool isnew = true;

            // test whether this name has been used before:
            for (std::size_t n_c=0; n_c < channel_names.size(); ++n_c) {
                if ( column.title == channel_names[n_c] || column.title.find("Column")==0 ) {
                    isnew = false;
                    break;
                }
            }
            if (isnew) {
                numberOfChannels++;
                std::string units( column.title );
                std::size_t left = units.find_last_of("(") + 1;
                std::size_t right = units.find_last_of(")");
                yunits = units.substr(left, right-left);
                channel_units.push_back( yunits );
                channel_names.push_back( column.title );
            }
        }
    }
    // Distribute Sections to Channels:
    std::size_t sectionsPerChannel = (numberOfColumns-1) / numberOfChannels;
    for (std::size_t n_c=0; (int)n_c < numberOfChannels; ++n_c) {
        Channel TempChannel(sectionsPerChannel);
        double factor = 1.0;
        if (channel_units[n_c] == "V") {
            channel_units[n_c] = "mV";
            factor = 1.0e3;
        }
        if (channel_units[n_c] == "A") {
            channel_units[n_c] = "pA";
            factor = 1.0e12;
        }
        for (std::size_t n_s=n_c; (int)n_s < numberOfColumns-1; n_s += numberOfChannels) {
            if (factor != 1.0) {
                section_list[n_s].get_w() = stf::vec_scal_mul(section_list[n_s].get(), factor);
            }
            try {
                TempChannel.InsertSection( section_list[n_s], (n_s-n_c)/numberOfChannels );
            }
            catch (...) {
                ReturnData.resize(0);
                CloseFile( dataRefNum );
                throw;
            }
        }
        TempChannel.SetChannelName( channel_names[n_c] );
        TempChannel.SetYUnits( channel_units[n_c] );
        try {
            if ((int)ReturnData.size()<numberOfChannels) {
                ReturnData.resize(numberOfChannels);
            }
            ReturnData.InsertChannel(TempChannel,n_c);
        }
        catch (...) {
            ReturnData.resize(0);
            CloseFile( dataRefNum );
            throw;
        }
    }

    ReturnData.SetXScale( xscale );

    std::string comment = AG_ReadComment(dataRefNum);
    std::string notes = AG_ReadNotes(dataRefNum);
    std::string date = AG_ParseDate(notes);
    std::string time = AG_ParseTime(notes);

    ReturnData.SetComment(comment);
    ReturnData.SetFileDescription(notes);
    ReturnData.SetTime(time);
    ReturnData.SetDate(date);
    
    // Close the import file
    CloseFile( dataRefNum );
#ifdef MODULE_ONLY
    if (progress) {
        std::cout << "\r";
        std::cout << "100%" << std::endl;
    }
#endif
}
