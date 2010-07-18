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
    #include <wx/wx.h>
#endif

#include "./channel.h"

Channel::Channel(void) 
: name(wxT("\0")), yunits( wxT("\0") ),
SectionArray(0), zoom(500,0.1,false) {}

Channel::Channel(const Section& c_Section) 
: name(wxT("\0")), yunits( wxT("\0") ),
SectionArray(1, c_Section), zoom(500,0.1,false) {}

Channel::Channel(const std::vector<Section>& SectionList) 
: name(wxT("\0")), yunits( wxT("\0") ),
SectionArray(SectionList), zoom(500,0.1,false) {}

Channel::Channel(std::size_t c_n_sections, std::size_t section_size) 	
: name(wxT("\0")), yunits( wxT("\0") ),
SectionArray(c_n_sections, Section(section_size)), zoom(500,0.1,false) {
}


Channel::~Channel(void) {
}

void Channel::InsertSection(const Section& c_Section, std::size_t pos) {
    try {
        if (SectionArray.at(pos).size() != c_Section.size()) {
            SectionArray.at(pos).resize(c_Section.size());
        }
        SectionArray.at(pos) = c_Section;
    }
    catch (...) {
        throw;
    }
}

const Section& Channel::at(std::size_t at_) const {
    try {
        return SectionArray.at(at_);
    }
    catch (...) {
        // Forward all exceptions, can't deal with them here:
        throw;
    }
}

Section& Channel::at(std::size_t at_) {
    try {
        return SectionArray.at(at_);
    }
    catch (...) {
        // Forward all exceptions, can't deal with them here:
        throw;
    }
}
