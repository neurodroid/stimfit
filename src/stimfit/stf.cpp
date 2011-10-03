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

/*! \file stf.cpp
 *  \author Christoph Schmidt-Hieber
 *  \date 2011-10-01
 *  \brief General functions for stf
 * 
 * 
 *  Implements some general functions within the stf namespace
 */

#include "stf.h"

stf::Table::Table(std::size_t nRows,std::size_t nCols) :
values(nRows,std::vector<double>(nCols,1.0)),
    empty(nRows,std::deque<bool>(nCols,false)),
    rowLabels(nRows, "\0"),
    colLabels(nCols, "\0")
    {}

stf::Table::Table(const std::map< std::string, double >& map)
: values(map.size(),std::vector<double>(1,1.0)), empty(map.size(),std::deque<bool>(1,false)),
rowLabels(map.size(), "\0"), colLabels(1, "Results")
{
    std::map< std::string, double >::const_iterator cit;
    sst_it it1 = rowLabels.begin();
    std::vector< std::vector<double> >::iterator it2 = values.begin();
    for (cit = map.begin();
         cit != map.end() && it1 != rowLabels.end() && it2 != values.end();
         cit++)
    {
        (*it1) = cit->first;
        it2->at(0) = cit->second;
        it1++;
        it2++;
    }
}

double stf::Table::at(std::size_t row,std::size_t col) const {
    try {
        return values.at(row).at(col);
    }
    catch (...) {
        throw;
    }
}

double& stf::Table::at(std::size_t row,std::size_t col) {
    try {
        return values.at(row).at(col);
    }
    catch (...) {
        throw;
    }
}

bool stf::Table::IsEmpty(std::size_t row,std::size_t col) const {
    try {
        return empty.at(row).at(col);
    }
    catch (...) {
        throw;
    }
}

void stf::Table::SetEmpty(std::size_t row,std::size_t col,bool value) {
    try {
        empty.at(row).at(col)=value;
    }
    catch (...) {
        throw;
    }
}

void stf::Table::SetRowLabel(std::size_t row,const std::string& label) {
    try {
        rowLabels.at(row)=label;
    }
    catch (...) {
        throw;
    }
}

void stf::Table::SetColLabel(std::size_t col,const std::string& label) {
    try {
        colLabels.at(col)=label;
    }
    catch (...) {
        throw;
    }
}

const std::string& stf::Table::GetRowLabel(std::size_t row) const {
    try {
        return rowLabels.at(row);
    }
    catch (...) {
        throw;
    }
}

const std::string& stf::Table::GetColLabel(std::size_t col) const {
    try {
        return colLabels.at(col);
    }
    catch (...) {
        throw;
    }
}

void stf::Table::AppendRows(std::size_t nRows_) {
    std::size_t oldRows=nRows();
    rowLabels.resize(oldRows+nRows_);
    values.resize(oldRows+nRows_);
    empty.resize(oldRows+nRows_);
    for (std::size_t nRow = 0; nRow < oldRows + nRows_; ++nRow) {
        values[nRow].resize(nCols());
        empty[nRow].resize(nCols());
    }
}

#if 0
wxString stf::sectionToString(const Section& section) {
    wxString retString;
    retString << (int)section.size() << wxT("\n");
    for (int n=0;n<(int)section.size();++n) {
        retString << section.GetXScale()*n << wxT("\t") << section[n] << wxT("\n");
    }
    return retString;
}

wxString stf::CreatePreview(const wxString& fName) {
    ifstreamMan ASCIIfile( fName );
    // Stop reading if we are either at the end or at line 100:
    wxString preview;
	ASCIIfile.myStream.ReadAll( &preview );
    return preview;
}
#endif

stf::wxProgressInfo::wxProgressInfo(const std::string& title, const std::string& message, int maximum, bool verbose)
    : ProgressInfo(title, message, maximum, verbose),
      pd(stf::std2wx(title), stf::std2wx(message), maximum, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL )
{
    
}

bool stf::wxProgressInfo::Update(int value, const std::string& newmsg, bool* skip) {
    return pd.Update(value, stf::std2wx(newmsg), skip);
}

std::string stf::wx2std(const wxString& wxs) {
    return std::string(wxs.mb_str());
}
wxString stf::std2wx(const std::string& sst) {
    return wxString(sst.c_str(), wxConvUTF8);
}

stf::SectionAttributes::SectionAttributes() :
    eventList(),pyMarkers(),isFitted(false),
    isIntegrated(false),fitFunc(NULL),bestFitP(0),quad_p(0),storeFitBeg(0),storeFitEnd(0),
    storeIntBeg(0),storeIntEnd(0),bestFit(0,0)
{}

stf::SectionPointer::SectionPointer(Section* pSec, const stf::SectionAttributes& sa) :
    pSection(pSec), sec_attr(sa)
{}
