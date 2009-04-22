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

#ifndef _SONLIB_H
#define _SONLIB_H

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include "../../core/stimfit_core.h"
#include "Son.H"
#pragma once

namespace stimfit {
	namespace SON {
		//Open a CFS file and store its contents to a Recording object:
		std::string SONError(short errorCode);
		COREDLL_API
		void importSONFile(const wxString& fName, Recording& ReturnData);
	}
}

#endif
