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

/*! \file zoom.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares the Zoom struct.
 */

#ifndef _ZOOM_H
#define _ZOOM_H

//! Handles y-scaling of traces
class YZoom {
public:
    //! Default constructor
    YZoom()
    : startPosY(500),  yZoom(0.1), isLogScaleY(false)
    {}
    //! Constructor
    /*! \param spy1 The y offset in pixels. 
     *  \param yz1 The y-scaling. 
     *  \param lsy Currently unused.
     */
    YZoom(int spy1, double yz1, bool lsy=false)
    : startPosY(spy1), yZoom(yz1), isLogScaleY(lsy)
    {}
    int startPosY; /*!< The y offset in pixels. */
    double yZoom; /*!< The y-scaling. */
    bool isLogScaleY; /*!< Currently unused. */
    
    YZoom operator*( double factor );
};

//! Handles x-scaling of traces
class XZoom {
public:
    //! Default constructor
    XZoom()
    : startPosX(0), xZoom(0.1), isLogScaleX(false)
    {}
    //! Constructor
    /*! \param spx The x offset in pixels. 
     *  \param xz The x-scaling. 
     *  \param lsx Currently unused.
     */
    XZoom( int spx, double xz, bool lsx=false )
    : startPosX(spx), xZoom(xz), isLogScaleX(lsx)
    {}
    int startPosX; /*!< The x offset in pixels. */
    double xZoom; /*!< The x-scaling. */
    bool isLogScaleX; /*!< Currently unused. */
   
    XZoom operator*( double factor );
};

#endif
