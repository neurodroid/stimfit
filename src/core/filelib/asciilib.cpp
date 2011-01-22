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

#include <wx/txtstrm.h>

#include "./asciilib.h"

#if 0
wxString stf::NextWord( wxString& str ) {
    int next_space = str.Find( wxT(' ') );
	int next_tab = str.Find( wxT('\t') );
	int next_nline = str.Find( wxT('\n') );
	if ( next_space == wxNOT_FOUND && next_tab == wxNOT_FOUND && next_nline == wxNOT_FOUND )
	    return wxT("\0");
	wxString retStr = wxT("");
	// wxNOT_FOUND is -1!
	if ( next_space != wxNOT_FOUND && ( next_space < next_tab || next_tab == wxNOT_FOUND ) && ( next_space < next_nline || next_nline == wxNOT_FOUND ) ) {
		retStr = str.BeforeFirst( wxT(' ') );
		str = str.AfterFirst( wxT(' ') );
	}
	if ( next_tab != wxNOT_FOUND && ( next_tab < next_space || next_space == wxNOT_FOUND ) && ( next_tab < next_nline || next_nline == wxNOT_FOUND ) ) {
		retStr = str.BeforeFirst( wxT('\t') );
		str = str.AfterFirst( wxT('\t') );
	}
	if ( next_nline != wxNOT_FOUND && ( next_nline < next_tab || next_tab == wxNOT_FOUND ) && ( next_nline < next_space || next_space == wxNOT_FOUND ) ) {
		retStr = str.BeforeFirst( wxT('\n') );
		str = str.AfterFirst( wxT('\n') );
	}
	return retStr;
}
#endif

void stf::importASCIIFile( const wxString& fName, int hLinesToSkip, int nColumns,
        bool firstIsTime, bool toSection, Recording& ReturnRec, bool progress )
{
    wxProgressDialog progDlg( wxT("Importing text file"), wxT("Starting..."), 100,
            NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL | wxPD_CAN_SKIP );

    //Open ASCII file and copy its contents to ReturnData:

    wxFileInputStream fStream( fName );
    wxTextInputStream tStream( fStream );

    // Read header:
    wxString header;
    for(int n_h=0; n_h<hLinesToSkip; n_h++) {
        if ( fStream.Eof() ) {
            ReturnRec.resize(0);
            throw std::runtime_error("Unexpected end of file; aborting file import.");
        }
        header << tStream.ReadLine() << wxT("\n");
    }

    double time[2];
    // Read data:
    // use vectors temporarily to push back values:
    std::vector<Vector_double > tempVec(nColumns-int(firstIsTime));
    std::vector<Vector_double >::iterator it_vd;
    // Reserve memory right here to speed up file reading a little:
    for ( it_vd = tempVec.begin(); it_vd != tempVec.end(); ++it_vd ) {
        // it_vd->reserve( 32768 );
    }
    int n_time=0;
    int nline = 0;
    bool skip = false;
    for (;;) {
        if (fStream.Eof())
            break;
        nline++;
        if ( std::div(nline, 100).rem == 0 ) {
            if ( skip ) {
                ReturnRec.resize(0);
                throw std::runtime_error("File import aborted by user.");
            }
            if (progress) {
                wxString progMsg( wxT("Reading line #") );
                progMsg << nline;
                progDlg.Pulse( progMsg, &skip );
            }
        }
        // calculate sampling rate from first two time values:
        if (firstIsTime) {
            double tempTime = tStream.ReadDouble();
            if (n_time<2) {
                time[n_time++]=tempTime;
            }
        }

        if (fStream.Eof())
            break;
        for (int n_col=0;n_col<nColumns-int(firstIsTime);++n_col) {
            tempVec[n_col].push_back( tStream.ReadDouble() );
        }
    }
    if ( tempVec.empty() ) {
        ReturnRec.resize(0);
        throw std::runtime_error("Empty text file; aborting file import.");
    }
    // insert vectors:
    int n_sec=0, n_ch=0;
    if (toSection) {
        n_sec=nColumns-int(firstIsTime);
        n_ch=1;
    } else {
        n_sec=1;
        n_ch=nColumns-int(firstIsTime);
    }
    ReturnRec.resize(n_ch);
    std::vector<Channel> TempChannel(n_ch,Channel(n_sec));
    for (int n_insert=0;n_insert<nColumns-int(firstIsTime);++n_insert) {
        Section TempSection(tempVec[n_insert].size());
        std::copy(tempVec[n_insert].begin(),tempVec[n_insert].end(),&TempSection[0]);
        try {
            if (toSection) {
                std::ostringstream label;
                label << stf::noPath(fName) << ", Section # " << n_insert+1;
                TempSection.SetSectionDescription(label.str());
                TempChannel[0].InsertSection(TempSection,n_insert);
            } else {
                std::ostringstream label;
                label << fName << ", Section # 1";
                TempSection.SetSectionDescription(label.str());
                TempChannel[n_insert].InsertSection(TempSection,0);
            }
        }
        catch (...) {throw;}
    }
    for (std::size_t n_ch=0;n_ch<TempChannel.size();++n_ch) {
        try {
            ReturnRec.InsertChannel(TempChannel[n_ch],n_ch);
        }
        catch (...) {throw;}
    }

    if (firstIsTime) {
        if (time[1]-time[0] <= 0) {
            ReturnRec.resize(0);
            throw std::runtime_error("Negative sampling interval\n"
                    "Check number of columns");
        }
        ReturnRec.SetXScale(time[1]-time[0]);
    }
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
    ReturnRec.SetFileDescription( header.ToStdString() );
#else
    ReturnRec.SetFileDescription(std::string(header.mb_str()));
#endif    
}

bool stf::exportASCIIFile(const wxString& fName, const Section& Export) {
    wxString out( stf::sectionToString(Export) );
    ofstreamMan ASCIIfile( fName );
    ASCIIfile.myStream.Write( out );
    return true;
}

bool stf::exportASCIIFile(const wxString& fName, const Channel& Export) {
    wxProgressDialog progDlg( wxT("Exporting channel"), wxT("Starting..."), 100,
            NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );

    for (std::size_t n_s=0;n_s<Export.size();++n_s) {
        // create new filename:
        wxString newFName;
        newFName << fName << wxT("_") << (int)n_s << wxT(".txt");
        wxString progStr;
        progStr << wxT("Writing section #") << (int)n_s + 1 << wxT(" of ") << (int)Export.size()
        << wxT("\nto file: ") << newFName;
        progDlg.Update(
                // Section contribution:
                (int)((double)n_s/(double)Export.size()*100.0),
                progStr
        );
        ofstreamMan ASCIIfile( newFName );
        ASCIIfile.myStream.Write( stf::sectionToString(Export[n_s]) );
    }
    return true;
}
