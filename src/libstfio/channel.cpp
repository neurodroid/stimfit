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

#include "./channel.h"

Channel::Channel(void) 
: name("\0"), yunits( "\0" ),
SectionArray(0) {}

Channel::Channel(const Section& c_Section) 
: name("\0"), yunits( "\0" ),
SectionArray(1, c_Section) {}

Channel::Channel(const std::vector<Section>& SectionList) 
: name("\0"), yunits( "\0" ),
SectionArray(SectionList) {}

Channel::Channel(std::size_t c_n_sections, std::size_t section_size) 	
: name("\0"), yunits( "\0" ),
SectionArray(c_n_sections, Section(section_size)) {}

Channel::~Channel(void) {}

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
