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

/*! \file section.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares the Section class.
 */

/*------------------------------------------------------------------------
 *  References:
 *  [1] Stroustrup, B.: The C++ Programming Language. 3rd ed. 1997
 *  [2] Meyers, S.: Effective C++. 3rd ed. 2005
--------------------------------------------------------------------------*/
// only compile once, even if included more often:
#ifndef _SECTION_H
#define _SECTION_H

/*! \addtogroup stfgen
 *  @{
 */

#include "./stimdefs.h"

//! Represents a continuously sampled sweep of data points
class StfDll Section {
public:
    // Construction/Destruction-----------------------------------------------
    //! Default constructor.
    explicit Section();

    //! Constructor
    /*! \param valA A vector of values that will make up the section.
     *  \param label An optional section label string.
     */
    explicit Section(
            const Vector_double& valA,
            const std::string& label="\0"
    );

    //! Yet another constructor
    /*! \param size Number of data points.
     *  \param label An optional section label string.
     */
    explicit Section(
            std::size_t size,
            const std::string& label="\0"
    );

    //! Destructor
    ~Section();

    // Operators--------------------------------------------------------------
    //! Unchecked access. Returns a non-const reference.
    /*! \param at Data point index.
     *  \return Copy of the data point with index at.
     */
    double& operator[](std::size_t at) { return data[at]; }

    //! Unchecked access. Returns a copy.
    /*! \param at Data point index.
     *  \return Reference to the data point with index at.
     */
    double operator[](std::size_t at) const { return data[at]; }

    // Public member functions------------------------------------------------

    //! Range-checked access. Returns a copy.
    /*! Throws std::out_of_range if out of range.
     *  \param at_ Data point index.
     *  \return Copy of the data point at index at_
     */
    double at(std::size_t at_) const;

    //! Range-checked access. Returns a non-const reference.
    /*! Throws std::out_of_range if out of range.
     *  \param at_ Data point index.
     *  \return Reference to the data point at index at_
     */
    double& at(std::size_t at_);

    //! Low-level access to the valarray (read-only).
    /*! An explicit function is used instead of implicit type conversion
     *  to access the valarray.
     *  \return The valarray containing the data points.
     */
    const Vector_double& get() const { return data; }

    //! Low-level access to the valarray (read and write).
    /*! An explicit function is used instead of implicit type conversion
     *  to access the valarray.
     *  \return The valarray containing the data points.
     */
    Vector_double& get_w() { return data; }

    //! Resize the Section to a new number of data points; deletes all previously stored data when gcc is used.
    /*! Note that in the gcc implementation of std::vector, resizing will
     *  delete all the original data. This is different from std::vector::resize().
     *  \param new_size The new number of data points.
     */
    void resize(std::size_t new_size) { data.resize(new_size); }

    //! Retrieve the number of data points.
    /*! \return The number of data points.
     */
    size_t size() const { return data.size(); }

    //! Sets the x scaling.
    /*! \param value The x scaling.
     */
    void SetXScale(double value);

    //! Retrieves the x scaling.
    /*! \return The x scaling.
     */
    double GetXScale() const { return x_scale; }
    
    //! Retrieves a section description.
    /*! \return A string describing this section.
     */
    const std::string& GetSectionDescription() const { return section_description; }

    //! Sets a section description.
    /*! \param value A string describing this section.
     */
    void SetSectionDescription(const std::string& value) { section_description=value; }

#ifndef MODULE_ONLY
    //! Retrieves a waveform of the evaluated best-fit function (read-only)
    /*! \return A valarray containing the evaluated best-fit function.
     */
    const stf::storedFunc* GetFitFunc() const { return fitFunc; }
    
    //! Indicates whether a fit has been performed on this section.
    /*! \return true if a fit has been performed, false otherwise.
     */
    bool IsFitted() const { return isFitted; }

    //! Deletes the current fit, sete isFitted to false;
    void DeleteFit();
    
    //! Sets the best-fit parameters when a fit has been performed on this section.
    /*! \param bestFitP_ The best-fit parameters
        \param fitFunc_ The function used for fitting
        \param chisqr The sum of squared errors
        \param fitBeg Sampling point index where the fit starts
        \param fitEnd Sampling point index where the fit ends
     */
    void SetIsFitted( const Vector_double& bestFitP_, stf::storedFunc* fitFunc_,
            double chisqr, std::size_t fitBeg, std::size_t fitEnd );

    //! Retrieves the parameters of quadratic functions, each going through three consecutive data points
    /*! \return Parameters of quadratic functions of the form 
     *          a0*x^2 + b0*x + c0, a1*x^2 + b1*x + c1, ..., an*x^2 + bn*x + cn 
     *          Each quadratic function goes through three consecutive data points.
     */
    const Vector_double& GetQuadP() const { return quad_p; }

    //! Retrieves the best-fit parameters for the most recently performed fit.
    /*! \return A std::vector of best-fit parameters.
     */
    const Vector_double& GetBestFitP() const { return bestFitP; }

    //! Indicates whether an integral has been calculated in this section.
    /*! \return true if an integral has been calculated, false otherwise.
     */
    bool IsIntegrated() const { return isIntegrated; }

    //! Determines whether an integral has been calculated in this section.
    /*! \return true if an integral has been calculated, false otherwise.
     */
    void SetIsIntegrated(bool value=true, std::size_t begin=0, std::size_t end=0);

    //! Retrieves the position of a stored fit start cursor.
    /*! Note that cursors are usually managed in Recording. However, the fit
     *  cursors are stored here so that a previous fit can be restored when this
     *  section is activated again.
     *  \return Fit start cursor position from a previous fit.
     */
    std::size_t GetStoreFitBeg() const { return storeFitBeg; }

    //! Retrieves the position of a stored fit end cursor.
    /*! Note that cursors are usually managed in Recording. However, the fit
     *  cursors are stored here so that a previous fit can be restored when this
     *  section is activated again.
     *  \return Fit end cursor position from a previous fit.
     */
    std::size_t GetStoreFitEnd() const { return storeFitEnd; }

    //! Retrieves the position of a stored integral start cursor.
    /*! Note that cursors are usually managed in Recording. However, the integral
     *  cursors are stored here so that a previous integral can be restored when this
     *  section is activated again.
     *  \return Integral start cursor position from a previous integral calculation.
     */
    std::size_t GetStoreIntBeg() const { return storeIntBeg; }

    //! Retrieves the position of a stored integral end cursor.
    /*! Note that cursors are usually managed in Recording. However, the integral
     *  cursors are stored here so that a previous integral can be restored when this
     *  section is activated again.
     *  \return Integral end cursor position from a previous integral calculation.
     */
    std::size_t GetStoreIntEnd() const { return storeIntEnd; }

    //! Stores the position of an integral start cursor.
    /*! Note that cursors are usually managed in Recording. However, the integral
     *  cursors are stored here so that a previous integral can be restored when this
     *  section is activated again.
     *  \param value Integral start cursor position to be stored.
     */
    void SetStoreIntBeg(std::size_t value) { storeIntBeg=value; }

    //! Stores the position of an integral end cursor.
    /*! Note that cursors are usually managed in Recording. However, the integral
     *  cursors are stored here so that a previous integral can be restored when this
     *  section is activated again.
     *  \param value Integral end cursor position to be stored.
     */
    void SetStoreIntEnd(std::size_t value) { storeIntEnd=value; }
    
    //! Retrieves imformation about the last fit that has been performed (read-only).
    /*! \return A table with information about the last fit.
     */
    const stf::Table& GetBestFit() const { return bestFit; }
    
    //! Retrieves information about an event.
    /*! \param n_e The index of the event.
     *  \return An Event object containing information about an event.
     */
    const stf::Event& GetEvent(std::size_t n_e) const;

    //! Creates a new event.
    /*! \param event Information about the event.
     */
    void CreateEvent(const stf::Event& event) { eventList.push_back( event ); }

    //! Checks whether any events have been created yet.
    /*! \return true if there are any events, false otherwise.
     */
    bool HasEvents() const { return !eventList.empty(); }

    //! Retrieves a list with information about events (read-only).
    /*! \return A vector with event information.
     */
    const std::vector<stf::Event>& GetEvents() const { return eventList; }

    //! Retrieves a list with information about events (read and write).
    /*! \return A vector with event information.
     */
    std::vector<stf::Event>& GetEventsW() { return eventList; }

    //! Erases all events.
    void EraseEvents() { eventList.clear(); }

    //! Retrieves the position of a marker.
    /*! \param n_e The index of the marker.
     *  \return The marker (a pair of x,y coordinates) 
     */
    const stf::PyMarker& GetPyMarker(std::size_t n_e) const;

    //! Sets a new marker.
    /*! \param marker The new marker.
     */
    void SetPyMarker(const stf::PyMarker& marker) { pyMarkers.push_back( marker ); }

    //! Checks whether any marker has been set yet.
    /*! \return true if there are any markers, false otherwise.
     */
    bool HasPyMarkers() const { return !pyMarkers.empty(); }

    //! Retrieves a list with information about markers (read-only).
    /*! \return A vector with markers.
     */
    const std::vector<stf::PyMarker>& GetPyMarkers() const { return pyMarkers; }

    //! Erases all events.
    void ErasePyMarkers() { pyMarkers.clear(); }
#endif

 private:
    //Private members-------------------------------------------------------

    // A description that is specific to this section:
    std::string section_description;

    // The sampling interval:
    double x_scale;

    // The data:
    Vector_double data;

#ifndef MODULE_ONLY
    std::vector<stf::Event> eventList;
    std::vector<stf::PyMarker> pyMarkers;
    bool isFitted,isIntegrated;
    stf::storedFunc *fitFunc;
    Vector_double bestFitP;
    Vector_double quad_p;
    std::size_t storeFitBeg;
    std::size_t storeFitEnd;
    std::size_t storeIntBeg;
    std::size_t storeIntEnd;
    stf::Table bestFit;
#endif
};

/*@}*/

#endif
