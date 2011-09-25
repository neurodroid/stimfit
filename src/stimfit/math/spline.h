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

/*! \file spline.h
 *  \author John Burkardt, Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Cubic spline interpolation.
 * 
 * 
 *  Based on algorithms by John Burkardt:
 *  http://www.scs.fsu.edu/~burkardt/index.html
 */

#ifndef _SPLINE_H
#define _SPLINE_H

namespace stf {
	Vector_double d3_np_fs(Vector_double& a, const Vector_double& b);
    void dvec_bracket3 (const Vector_double& t, double tval, int& left );
    Vector_double spline_cubic_set ( const Vector_double& t, const Vector_double& y, 
		int ibcbeg, double ybcbeg, int ibcend, double ybcend );
	double spline_cubic_val (const Vector_double& t, double tval, const Vector_double& y, 
		const Vector_double& ypp, double& ypval, double& yppval );
}

#endif
