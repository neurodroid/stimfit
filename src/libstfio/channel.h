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

/*! \file channel.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares the Channel class.
 */

#ifndef _CHANNEL_H
#define _CHANNEL_H

/*! \addtogroup stfgen
 *  @{
 */

#include "./section.h"

//! A Channel contains several data \link #Section Sections \endlink representing observations of the same physical quantity.
class StfDll Channel {
public:

    //ctor/dtor---------------------------------------------------
    //! Default constructor
    explicit Channel(void);

    //! Constructor
    /*! \param c_Section A single section from which to construct the channel
     */
    explicit Channel(const Section& c_Section); 

    //! Constructor
    /*! \param SectionList A vector of Sections from which to construct the channel
     */
    explicit Channel(const std::vector<Section>& SectionList); 

    //! Constructor
    /*! Setting the number of sections at construction time will avoid unnecessary 
     *  memory re-allocations.
     *  \param c_n_sections The number of sections.
     *  \param section_size Initial section size. Will serve additional
     *         re-alocations if known at construction time.
     */
    explicit Channel(std::size_t c_n_sections, std::size_t section_size = 0);
    
    //! Destructor
    ~Channel();

    //operators---------------------------------------------------

    //! Unchecked access to a section (read and write)
    /*! Use at() for range-checked access.
     *  \param at_ The section index.
     *  \return The section at index at_.
     */
    Section& operator[](std::size_t at_) { return SectionArray[at_]; }

    //! Unchecked access to a section (read-only)
    /*! Use at() for range-checked access.
     *  \param at_ The section index.
     *  \return The section at index at_.
     */
    const Section& operator[](std::size_t at_) const { return SectionArray[at_]; }

    //member access: read-----------------------------------------

    //! Retrieves the channel name
    /*! \return The channel name.
     */
    const std::string& GetChannelName() const { return name; }

    //! Retrieves the y units string.
    /*! \return The y units string.
     */
    const std::string& GetYUnits( ) const { return yunits; }

    //! Retrieves the size of the section array.
    /*! \return The size of the section array.
     */
    size_t size() const { return SectionArray.size(); }

    //! Range-checked access to a section (read-only).
    /*! Will throw std::out_of_range if out of range.
     *  \param at_ The index of the section.
     *  \return The section at index at_.
     */
    const Section& at(std::size_t at_) const;

    //! Range-checked access to a section (read and write).
    /*! Will throw std::out_of_range if out of range.
     *  \param at_ The index of the section.
     *  \return The section at index at_.
     */
    Section& at(std::size_t at_);

    //! Low-level access to the section array (read-only).
    /*! \return The vector containing the sections.
     */
    const std::vector< Section >& get() const { return SectionArray; }


    //! Low-level access to the section array (read and write).
    /*! \return The vector containing the sections.
     */
    std::vector< Section >& get() { return SectionArray; }
    
    //! Returns the current zoom settings for this channel (read-only).
    /*! \return The current zoom settings.
     */
    const YZoom& GetYZoom() { return zoom; }

    //! Returns the current zoom settings for this channel (read & write).
    /*! \return The current zoom settings.
     */
    YZoom& GetYZoomW() { return zoom; }

    //member access: write----------------------------------------

    //! Sets the channel name
    /*! \param value The channel name.
     */
    void SetChannelName(const std::string& value) { name = value; }
    
    //! Sets the y units string
    /*! \param value The new y units string.
     */
    void SetYUnits( const std::string& value ) { yunits = value; }

    //misc--------------------------------------------------------
    
    //! Inserts a section at the given position, overwriting anything that's currently stored at that position
    /*! Meant to be used after constructing with Channel(const unsigned int& c_n_sections}.
     *  The section array size has to be larger than pos because it won't be resized.
     *  Will throw std::out_of_range if out of range.
     *  \param c_Section The section to be inserted.
     *  \param pos The position at which to insert the section.
     */
    void InsertSection(const Section& c_Section, std::size_t pos);

    //! Resize the section array.
    /*! \param newSize The new number of sections.
     */
    void resize(std::size_t newSize) { SectionArray.resize(newSize); }

    //! Reserve memory for a number of sections.
    /*! This will avoid unnecessary memory re-allocations.
     *  \param resSize The number of sections to reserve memory for.
     */
    void reserve(std::size_t resSize) { SectionArray.reserve(resSize); }

private:
    //private members---------------------------------------------
    
    std::string name, yunits;

    // An array of sections
    std::vector< Section > SectionArray;
    // The zoom settings
    YZoom zoom;

};

/*@}*/

#endif

