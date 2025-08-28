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

#include "./stfio.h"
#include "./section.h"

// Definitions------------------------------------------------------------
// Default constructor definition
// For reasons why to use member initializer lists instead of assignments
// within the constructor, see [1]248 and [2]28

Section::Section(void)
    : section_description(), x_scale(1.0), data(0), AnnotationsList()
{}

Section::Section( const Vector_double& valA, const std::string& label )
    : section_description(label), x_scale(1.0), data(valA), AnnotationsList()
{}

Section::Section(std::size_t size, const std::string& label)
    : section_description(label), x_scale(1.0), data(size), AnnotationsList()
{}

Section::~Section(void) {
}


double Section::at(std::size_t at_) const {
    if (at_>=data.size()) {
        std::out_of_range e("subscript out of range in class Section");
        throw (e);
    }
    return data[at_];
}

double& Section::at(std::size_t at_) {
    if (at_>=data.size()) {
        std::out_of_range e("subscript out of range in class Section");
        throw (e);
    }
    return data[at_];
}

void Section::SetXScale( double value ) {
    if ( x_scale >= 0 )
        x_scale=value;
    else
        throw std::runtime_error( "Attempt to set x-scale <= 0" );
}

void Section::AddAnnotation(int position, Annotation annotation)
{
    if (position == -1) {
        this->AnnotationsList.push_back(annotation);
    }
    else{
        this->AnnotationsList.insert(AnnotationsList.begin() + position, annotation);
    }
}

void Section::RemoveAnnotation(size_t index)
{
    if (index < this->AnnotationsList.size()){
        this->AnnotationsList.erase(this->AnnotationsList.begin() + index);
    }
}

void Section::EraseAllAnnotations()
{
    this->AnnotationsList.clear();
}

std::vector<Annotation> Section::GetAnnotationList()
{
    return this->AnnotationsList;
}

std::size_t Section::GetSectionSize()
{
    return data.size();
}