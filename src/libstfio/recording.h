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

/*! \file recording.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares the Recording class.
 */

#ifndef _RECORDING_H
#define _RECORDING_H

/*! \defgroup stfgen Generic stimfit classes and functions
 *  @{
 */

// #include "./channel.h"
// #include "./section.h"
// #include "./stfio.h"

class Channel;
class Section;

//! Represents the data within a file.
/*! Contains an array of channels that can be accessed either via at() (range-checked,
 *  will throw an exception if out of range) or the []-operator (range unchecked). Moreover
 *  all the metadata such as time, date, samling rate and comments are stored here.
 */
class StfDll Recording {
 public:

    //ctor/dtor-------------------------------------------------------
    //! Default constuctor
    explicit Recording();

    //! Constructor
    /*! \param c_Channel The Channel from which to construct a new Recording.
     */
    explicit Recording(const Channel& c_Channel); 

    //! Constructor
    /*! \param ChannelList A vector of channels from which to construct a new Recording.
     */
    explicit Recording(const std::vector<Channel>& ChannelList); 

    //! Constructor
    /*! Setting the number of channels and sections at construction time will avoid unnecessary 
     *  memory re-allocations.
     *  \param c_n_channels The number of channels.
     *  \param c_n_sections The number of sections.
     *  \param c_n_points The number of sampling points per section.
     */
    explicit Recording( std::size_t c_n_channels, std::size_t c_n_sections = 0, std::size_t c_n_points = 0 );

    //! Destructor
    virtual ~Recording();

    //member access functions: read-----------------------------------
    
    //! Retrieves the number of sections in a channel.
    /*! \param n_channel The index of the channel (range-checked).
     *  \return The number of sections in n_channel.
     */
    std::size_t GetChannelSize(std::size_t n_channel) const;

    //! Retrieves the channels (read-only).
    /*! \return A vector containing the channels.
     */
    const std::vector<Channel>& get() const { return ChannelArray; }
    
    //! Retrieves the channels (read and write).
    /*! \return A vector containing the channels.
     */
    std::vector<Channel>& get() { return ChannelArray; }
    
    //! Retrieves the file description.
    /*! \return The file description.
     */
    const std::string& GetFileDescription() const { return file_description; }

    //! Retrieves the common section description.
    /*! \return The common section description.
     */
    const std::string& GetGlobalSectionDescription() const { return global_section_description; }

    //! Retrieves the scaling as a string.
    /*! \return A string containing the description.
     */
    const std::string& GetScaling() const { return scaling; }

    //! Retrieves the time of recording as a string.
    /*! \return A string containing the time of recording.
     */
    const std::string& GetTime() const { return time; }

    //! Retrieves the date of recording as a string.
    /*! \return A string containing the date of recording.
     */
    const std::string& GetDate() const { return date; }

    //! Retrieves a comment string.
    /*! \return A string containing a comment.
     */
    const std::string& GetComment() const { return comment; }
    
    //! Retrieves the x units.
    /*! \return The x units. Currently hard-coded to be "ms".
     */
    const std::string& GetXUnits() const { return xunits; }

    //! Retrieves the size of the channel array.
    /*! \return The size of the channel array.
     */
    std::size_t size() const { return ChannelArray.size(); }
    
    //! Retrieves the x scaling (sampling interval).
    /*! \return The x scaling.
     */
    double GetXScale() const { return dt; }
    
    //! Retrieves the sampling rate ( 1 / x-scale )
    /*! \return The sampling rate.
     */
    double GetSR() const { return 1.0/dt; }

    //! Range-checked access to a channel (read-only).
    /*! Will throw std::out_of_range if out of range.
     *  \param n_c The index of the channel.
     *  \return The channel at index n_c.
     */
    const Channel& at(std::size_t n_c) const;

    //! Range-checked access to a channel (read and write).
    /*! Will throw std::out_of_range if out of range.
     *  \param n_c The index of the channel.
     *  \return The channel at index n_c.
     */
    Channel& at(std::size_t n_c);

    //member access functions: write---------------------------------

    //! Sets the file description.
    /*! \param value The file description.
     */
    void SetFileDescription(const std::string& value) { file_description=value; }

    //! Sets the common section description.
    /*! \param value The common section description.
     */
    void SetGlobalSectionDescription(const std::string& value) {
        global_section_description=value;
    }

    //! Sets the scaling as a string.
    /*! \param value A string containing the description.
     */
    void SetScaling(const std::string& value) { scaling=value; }
 
    //! Sets the time of recording as a string.
    /*! \param value A string containing the time of recording.
     */
    void SetTime(const std::string& value) { time=value; }

    //! Sets the date of recording as a string.
    /*! \param value A string containing the date of recording.
     */
    void SetDate(const std::string& value) { date=value; }

    //! Sets a comment string.
    /*! \param value A string containing a comment.
     */
    void SetComment(const std::string& value) { comment=value; }
    
    //! Sets the y units for a channel.
    /*! \param n_channel The channel index for which to set the units.
     *  \param value A string containing the y units.
     */
    void SetGlobalYUnits(std::size_t n_channel, const std::string& value);

    //! Sets the x units.
    /*! \param value A string containing the x units.
     */
    void SetXUnits(const std::string& value) { xunits=value; }

    //! Sets the x scaling.
    /*! Note that setting the global x-scale will set it for all sections
     *  \param value The x scaling.
     */
    void SetXScale(double value);
    
    //misc-----------------------------------------------------------

    //! Resize the Recording to a new number of channels.
    /*! Resizes both the channel and the global y units arrays.
     *  \param c_n_channels The new number of channels.
     */
    virtual void resize(std::size_t c_n_channels);

    //! Insert a Channel at a given position.
    /*! Will throw std::out_of_range if range check fails.
     *  \param c_Channel The Channel to be inserted.
     *  \param pos The position at which to insert the channel (0-based).
     */
    virtual void InsertChannel(Channel& c_Channel, std::size_t pos);

    //! Copy descriptive attributes from another Recording to this Recording.
    /*! This will copy the file and global section decription, the scaling, time, date, 
     *  comment and global y units strings and the x-scale.
     *  \param c_Recording The Recording from which to copy the attributes.
     */
    void CopyAttributes(const Recording& c_Recording);

    //! Calculates an average of several traces.
    /*! \param AverageReturn The average will be returned in this variable by passing 
     *         a reference. AverageReturn has to have the correct size upon entering 
     *         this function already, it won't be resized.
     *  \param SigReturn The standard deviation will be returned in this variable by 
     *         passing a reference if isSig == true. SigReturn has to have the correct 
     *         size upon entering this function already, it won't be resized.
     *  \param channel The index of the channel to be used.
     *  \param section_index A vector containing the indices of the sections to be
     *         used for the average.
     *  \param isSig Set to true if the standard deviation should be calculated as well.
     *  \param shift A vector indicating by how many data points each section should be
     *         shifted before averaging.
     */
    void MakeAverage( Section& AverageReturn, Section& SigReturn, std::size_t channel,
                      const std::vector<std::size_t>& section_index, bool isSig,
                      const std::vector<int>& shift) const;

    //! Add a Recording at the end of this Recording.
    /*! \param toAdd The Recording to be added.
     */
    void AddRec(const Recording& toAdd);
    
    //operators------------------------------------------------------

    //! Unchecked channel access (read and write)
    /*! Use at() for range-checked access.
     *  \param at The channel index.
     */
    Channel& operator[](std::size_t at) { return ChannelArray[at]; }

    //! Unchecked channel access (read-only)
    /*! Use at() for range-checked access.
     *  \param at The channel index.
     */
    const Channel& operator[](std::size_t at) const { return ChannelArray[at]; }

 private:
    std::vector<Channel> ChannelArray;
    std::string global_section_description, scaling;

    /* public: */
    
    double dt;
    std::string file_description, time, date, comment, xunits;

    void init();

};

/*@}*/

#endif
