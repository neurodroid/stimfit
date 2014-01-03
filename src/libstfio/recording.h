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

#include "./channel.h"
// #include "./section.h"
// #include "./stfio.h"

class Section;

//! Represents the data within a file.
/*! Contains an array of channels that can be accessed either via at() (range-checked,
 *  will throw an exception if out of range) or the []-operator (range unchecked). Moreover
 *  all the metadata such as time, date, samling rate and comments are stored here.
 */
class StfioDll Recording {
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
    const std::string& GetTime();

    //! Retrieves the date of recording as a string.
    /*! \return A string containing the date of recording.
     */
    const std::string& GetDate();

    //! Retrieves the date of recording as a string.
    /*! \return A string containing the date of recording.
     */
    struct tm GetDateTime() const { return datetime; };

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

    //! Retrieves the index of the current channel.
    /*! \return The index of the current channel.
     */
    std::size_t GetCurChIndex() const { return cc; }

    //! Retrieves the index of the second channel.
    /*! \return The index of the second channel.
     */
    std::size_t GetSecChIndex() const { return sc; }

    //! Retrieves the index of the current section.
    /*! \return The index of the current section.
     */
    std::size_t GetCurSecIndex() const { return cs; }
    
    //! Retrieves the indices of the selected sections (read-only).
    /*! \return A vector containing the indices of the selected sections.
     */
    const std::vector<std::size_t>& GetSelectedSections() const { return selectedSections; } 

    //! Retrieves the indices of the selected sections (read and write).
    /*! \return A vector containing the indices of the selected sections.
     */
    std::vector<std::size_t>& GetSelectedSectionsW() { return selectedSections; } 

    //! Retrieves the stored baseline values of the selected sections (read-only).
    /*! \return A vector containing the stored baseline values of the selected sections.
     */
    const Vector_double& GetSelectBase() const { return selectBase; } 

    //! Retrieves the stored baseline values of the selected sections (read and write).
    /*! \return A vector containing the stored baseline values of the selected sections.
     */
    Vector_double& GetSelectBaseW() { return selectBase; }

    //! Retrieves the currently accessed section in the active channel (read-only)
    /*! \return The currently accessed section in the active channel.
     */
    const Section& cursec() const { return (*this)[cc][cs]; }

    //! Retrieves the currently accessed section in the active channel (read and write)
    /*! \return The currently accessed section in the active channel.
     */
    Section& cursec() { return (*this)[cc][cs]; }

    //! Retrieves the currently accessed section in the second (reference) channel (read-only)
    /*! \return The currently accessed section in the second (reference) channel.
     */
    const Section& secsec() const { return (*this)[sc][cs]; }

    //! Retrieves the active channel (read-only)
    /*! \return The active channel.
     */
    const Channel& curch() const { return (*this)[cc]; }

    //! Retrieves active channel (read and write)
    /*! \return The active channel.
     */
    Channel& curch() { return (*this)[cc]; }

    //! Retrieves the second (reference) channel (read-only)
    /*! \return The second (reference) channel.
     */
    const Channel& secch() const { return (*this)[sc]; }

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
     *  \return 0 in case of success, non-zero in case of failure
     */
    int SetTime(const std::string& value);
    int SetTime(int hour, int minute, int sec);

    //! Sets the date of recording as a string.
    /*! \param value A string containing the date of recording.
     *  \return 0 in case of success, non-zero in case of failure
     */
    int SetDate(const std::string& value);
    int SetDate(int year, int month, int mday);

    //! Sets the date and time of recording as struct tm
    /*! \param value  has type struct tm
     */
    void SetDateTime(const struct tm &value) { memcpy(&datetime, &value, sizeof(struct tm)); }
    void SetDateTime(int year, int month, int mday, int hour, int minute, int sec) ;

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

    //! Sets the index of the current channel.
    /*! \param value The index of the current channel.
     */
    void SetCurChIndex(std::size_t value);

    //! Sets the index of the second channel.
    /*! \param value The index of the second channel.
     */
    void SetSecChIndex(std::size_t value);

    //! Sets the index of the current section.
    /*! \param value The index of the current section.
     */
    void SetCurSecIndex(std::size_t value);

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

    //! Selects a section
    /*! \param sectionToSelect The index of the section to be selected.
     *  \param base_start Start index for baseline
     *  \param base_end End index for baseline
     */
    void SelectTrace(std::size_t sectionToSelect, std::size_t base_start, std::size_t base_end);

    //! Unselects a section if it was selected before
    /*! \param sectionToUnselect The index of the section to be unselected.
     *  \return true if the section was previously selected, false otherwise.
     */
    bool UnselectTrace(std::size_t sectionToUnselect);
    
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

    // only neeed for GetData() and GetTime(): should be replaced by alternative interface.
    __attribute__ ((deprecated)) std::string time0, date; 

    /* public: */
    
    double dt;
    std::string file_description, comment, xunits;
    struct tm datetime;


    // currently accessed channel:
    std::size_t cc;
    // second channel:
    std::size_t sc;
    // currently accessed section:
    std::size_t cs;

    // Indices of the selected sections
    std::vector<std::size_t> selectedSections;
    // Base line value for each selected trace
    Vector_double selectBase;
    
    void init();

};

/*@}*/

#endif
