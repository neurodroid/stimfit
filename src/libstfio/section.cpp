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

#include "./section.h"

// Definitions------------------------------------------------------------
// Default constructor definition
// For reasons why to use member initializer lists instead of assignments
// within the constructor, see [1]248 and [2]28

Section::Section(void)
    : section_description(), x_scale(1.0), data(0)
    , eventList(),pyMarkers(),isFitted(false),
      isIntegrated(false),fitFunc(NULL),bestFitP(0),quad_p(0),storeFitBeg(0),storeFitEnd(0),
      storeIntBeg(0),storeIntEnd(0),bestFit(0,0)
{}

Section::Section( const Vector_double& valA, const std::string& label )
    : section_description(label), x_scale(1.0), data(valA)
    , eventList(),pyMarkers(),isFitted(false),
      isIntegrated(false),fitFunc(NULL),bestFitP(0),quad_p(0),storeFitBeg(0),storeFitEnd(0),
      storeIntBeg(0),storeIntEnd(0),bestFit(0,0)
{}

Section::Section(std::size_t size, const std::string& label)
    : section_description(label), x_scale(1.0), data(size)
    , eventList(),pyMarkers(),isFitted(false),
      isIntegrated(false),fitFunc(NULL),bestFitP(0),quad_p(0),storeFitBeg(0),storeFitEnd(0),
      storeIntBeg(0),storeIntEnd(0),bestFit(0,0)
{}

Section::~Section(void) {
}


double Section::at(std::size_t at_) const {
	if (at_<0||at_>=data.size()) {
		std::out_of_range e("subscript out of range in class Section");
		throw (e);
	}
	return data[at_];
}

double& Section::at(std::size_t at_) {
	if (at_<0||at_>=data.size()) {
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

void Section::SetIsIntegrated(bool value, std::size_t begin, std::size_t end, const Vector_double& quad_p_) {
    if (value==false) {
        isIntegrated=value;
        return;
    }
    if (end<=begin) {
        throw std::out_of_range("integration limits out of range in Section::SetIsIntegrated");
    }
    int n_intervals=std::div((int)end-(int)begin,2).quot;
    if ((int)quad_p_.size() != n_intervals*3) {
        throw std::out_of_range("Wrong number of parameters for quadratic equations in Section::SetIsIntegrated");
    }
    quad_p = quad_p_;
    isIntegrated=value;
    storeIntBeg=begin;
    storeIntEnd=end;
}

const stfio::Event& Section::GetEvent(std::size_t n_e) const {
    try {
        return eventList.at(n_e);
    }
    catch (const std::out_of_range& e) {
        throw e;
    }
}

const stfio::PyMarker& Section::GetPyMarker(std::size_t n_e) const {
    try {
        return pyMarkers.at(n_e);
    }
    catch (const std::out_of_range& e) {
        throw e;
    }
}

void Section::SetIsFitted( const Vector_double& bestFitP_, stfio::storedFunc* fitFunc_,
        double chisqr, std::size_t fitBeg, std::size_t fitEnd )
{
    if ( !fitFunc_ ) {
        throw std::runtime_error("Function pointer is zero in Section::SetIsFitted");
    }
    if ( fitFunc_->pInfo.size() != bestFitP_.size() ) {
        throw std::runtime_error("Number of best-fit parameters doesn't match number\n \
                                 of function parameters in Section::SetIsFitted");
    }
    fitFunc = fitFunc_;
    if ( bestFitP.size() != bestFitP_.size() )
        bestFitP.resize(bestFitP_.size()); 
    bestFitP = bestFitP_;
    bestFit = fitFunc->output( bestFitP, fitFunc->pInfo, chisqr );
    storeFitBeg = fitBeg;
    storeFitEnd = fitEnd;
    isFitted = true;
}

void Section::DeleteFit() {
    fitFunc = NULL;
    bestFitP.resize( 0 );
    bestFit = stfio::Table( 0, 0 );
    isFitted = false;
}
