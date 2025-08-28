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
#include "../stimfit/Annotation.h"
/*! \addtogroup stfgen
 *  @{
 */

//! Represents a continuously sampled sweep of data points
class StfioDll Section {
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
    /*! Throws std::out_of_range if+#include "../stimfit/Annotation.h" out of range.
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
    
    // Annotation operations //
    // Add Annotation at sample number 'position'
    void AddAnnotation(int position, Annotation annotation);
    // Change position of Annotation[index] to new sample number 'position'
    void MoveAnnotation(size_t index, int new_position);
    // Remove Annotation[index] from list of Annotations
    void RemoveAnnotation(size_t index);
    //Erase all annotation in the list of Annotations
    void EraseAllAnnotations();
    //Get list of all annotation in this section.
    std::vector<Annotation> GetAnnotationList();


 private:
    //Private members-------------------------------------------------------

    // A description that is specific to this section:
    std::string section_description;

    // The sampling interval:
    double x_scale;

    // The data:
    Vector_double data;

    // list of annotations in this section
    std::vector<Annotation> AnnotationsList;
};

/*@}*/

#endif




