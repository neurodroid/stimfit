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
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
    return wxs.ToStdString();
#else
    return std::string(wxs.mb_str());
#endif
}

wxString stf::std2wx(const std::string& sst) {
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
    /* Problems with direct constructor; copying each character for the time being.
       return wxString(sst); */
    wxString wxs;
    std::string::const_iterator it;
    for (it = sst.begin(); it != sst.end(); ++it) {
        if (*it < 0)
            wxs += ' ';
        else
            wxs += (char)*it;
    }
    return wxs;
#else
    return wxString(sst.c_str(), wxConvUTF8);
#endif
}

stf::SectionAttributes::SectionAttributes() :
    eventList(),pyMarkers(),isFitted(false),
    isIntegrated(false),fitFunc(NULL),bestFitP(0),quad_p(0),storeFitBeg(0),storeFitEnd(0),
    storeIntBeg(0),storeIntEnd(0),bestFit(0,0)
{}

stf::SectionPointer::SectionPointer(Section* pSec, const stf::SectionAttributes& sa) :
    pSection(pSec), sec_attr(sa)
{}

stf::Event::Event(std::size_t start, std::size_t peak, std::size_t size, wxCheckBox* cb) :
    eventStartIndex(start), eventPeakIndex(peak), eventSize(size), checkBox(cb)
{
    checkBox->Show(true);
    checkBox->SetValue(true);
}

stf::Event::~Event()
{
}
