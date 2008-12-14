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
#include "./axg/AxoGraph_ReadWrite.h"
#include "./axg/fileUtils.h"
#include "./axglib.h"

void stf::importAXGFile(const wxString &fName, Recording &ReturnData, bool progress) {
    wxProgressDialog progDlg( wxT("Axograph binary file import"), wxT("Starting file import"),
                              100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );
    wxString errorMsg(wxT("Exception while calling AXG_importAXGFile():\n"));
    wxString yunits;

    // =====================================================================================================================
    //
    // Open an AxoGraph file and read in the data
    //
    // =====================================================================================================================

    // Open the example file
    filehandle dataRefNum = OpenFile( fName.c_str() );
    if ( dataRefNum == 0 )
    {
        errorMsg += wxT("\n\nError: Could not find file.");
        ReturnData.resize(0);
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }

    // check the AxoGraph header, and get the number of columns to be read
    int fileFormat = 0;
    int result = AG_GetFileFormat( dataRefNum, &fileFormat );
    if ( result )
    {
        errorMsg += wxT( "\nError from AG_GetFileFormat - ");
        if ( result == kAG_FormatErr )
            errorMsg += wxT( "file is not in AxoGraph format" );
        else if ( result == kAG_VersionErr )
            errorMsg += wxT( "file is of a more recent version than supported by this code" );
        else
            errorMsg += wxT( "error" );

        ReturnData.resize(0);
        CloseFile( dataRefNum );
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }

    long numberOfColumns = 0;
    result = AG_GetNumberOfColumns( dataRefNum, fileFormat, &numberOfColumns );
    if ( result )
    {
        errorMsg += wxT( "Error from AG_GetNumberOfColumns" );
        ReturnData.resize(0);
        CloseFile( dataRefNum );
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }

    // Sanity check
    if ( numberOfColumns <= 0 )  	// negative columns
    {
        errorMsg += wxT ( "File format error: number of columns is set negative in AxoGraph data file" );
        ReturnData.resize(0);
        CloseFile( dataRefNum );
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }

    //	AG_ReadFloatColumn reads column data into a float column structure.
    int numberOfChannels = 0;

    std::vector< Section > section_list;
    std::vector< wxString > channel_names;
    std::vector< wxString > channel_units;
    double xscale = 1.0;
    for ( int columnNumber=0; columnNumber<numberOfColumns; columnNumber++ )
    {
        if (progress) {
            wxString progStr;
            progStr << wxT(", Section #") << columnNumber+1 << wxT(" of ") << numberOfColumns;
            progDlg.Update( // Section contribution:
                           (double)columnNumber/(double)numberOfColumns * 100.0,
                           progStr );
        }

        ColumnData column;
        result = AG_ReadFloatColumn( dataRefNum, fileFormat, columnNumber, &column );

        if ( result )
        {
            errorMsg += wxT( "Error from AG_ReadFloatColumn" );
            ReturnData.resize(0);
            CloseFile( dataRefNum );
            throw std::runtime_error(std::string(errorMsg.char_str()));
        }
        if ( columnNumber == 0 ) {
            xscale = column.seriesArray.increment * 1.0e3;

		} else {
            section_list.push_back( Section(column.points, column.title) );
            std::size_t last = section_list.size()-1;

            std::copy(&(column.floatArray[0]),&(column.floatArray[column.points]),&(section_list[last].get_w()[0]));
            // check whether this is a new channel:
            bool isnew = true;
            wxString test_name( column.title );
            // test whether this name has been used before:
            for (std::size_t n_c=0; n_c < channel_names.size(); ++n_c) {
                if ( test_name == channel_names[n_c] || test_name.StartsWith( wxT("Column") ) ) {
                    isnew = false;
                    break;
                }
            }
            if (isnew) {
                numberOfChannels++;
                wxString units( column.title );
                std::size_t left = units.find_last_of( wxT("(") ) + 1;
                std::size_t right = units.find_last_of( wxT(")") );
                yunits = units.substr(left, right-left);
                channel_units.push_back( yunits );
                channel_names.push_back( test_name );
            }
        }
        free( column.floatArray );
        free( column.title );
    }
    // Distribute Sections to Channels:
    std::size_t sectionsPerChannel = (numberOfColumns-1) / numberOfChannels;
    for (std::size_t n_c=0; (int)n_c < numberOfChannels; ++n_c) {
        Channel TempChannel(sectionsPerChannel);
        for (std::size_t n_s=n_c; (int)n_s < numberOfColumns-1; n_s += numberOfChannels) {
            if (channel_units[n_c] == wxT("V")) {
                section_list[n_s].get_w() *= 1.0e3;
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
        if (channel_units[n_c] == wxT("V")) {
            channel_units[n_c] = wxT("mV");
        }
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

    // Apparently, the sample interval has to be multiplied by
    // the number of channels for multiplexed data. Thanks to
    // Dominique Engel for noticing and reporting.
    ReturnData.SetXScale( xscale );

    // Close the import file
    CloseFile( dataRefNum );
}
