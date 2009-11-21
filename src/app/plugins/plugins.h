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

/*! \file plugins.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-20
 *  \brief User-defined extension plugins.
 */

#ifndef _PLUGINS_H
#define _PLUGINS_H

// common definitions needed here:
#include "./../../core/stimdefs.h"
#include "./../../core/core.h"

namespace stf {

/*! \addtogroup stfgen
 *  @{
 */

    //! Creates 100 resampled data sets from the selected sections in \e data.
    /*! First, creates a vector of 100 repetitions of the set of indices; then,
     *  performs a random permutation on these indices; finally, the randomly
     *  permutated super-set is cut into 100 pieces of equal length, and the
     *  newly created 100 sets of indices is used for creating averages. See
     *  Roth & Hausser (2001) for a detailed explanation.
     *  \param data Data to be resampled.
     *  \param input Currently unused.
     *  \param results A map with results (e.g. "Amplitude", 10)
     *  \return 100 resampled data sets.
     */
    Recording bootstrap(
            const Recording& data,
            const Vector_double& input,
            std::map< wxString, double >& results
    );

    //! Creates an I-V relationship by averaging every <em>n</em>-th, <em>n</em>+1-th, ... trace.
    /*! \param data Original dataset for the IV.
     *  \param input A vector of parameters, where \n
     *         \e input[0] is the number of pulses for the IV, \n
     *         \e input[1] is the absolute holding (reference) voltage, \n
     *         \e input[2] is the amplitude step size of the voltage pulses, and \n
     *         \e input[3] is the first section to use for the IV.
     *  \param results A map with results (e.g. "Amplitude", 10)
     *  \return The averages of every <em>n</em>-th, <em>n</em>+1-th, ... trace.
     */
    Recording analyze_iv(
            const Recording& data,
            const Vector_double& input,
            std::map< wxString, double >& results
    );
    
    //! Perform a random permutation on the indices in \e input
    /*! \param input A vector of indices.
     *  \param B Number of resampled index vectors to be generated.
     *  \return A super-vector, containing \e B sub-vectors of randomly
     *          permutated indices. Each sub-vector has the same size as
     *          \e input.
     */
    std::vector<std::vector<std::size_t> >
    randomPermutation(const std::vector<std::size_t>& input, std::size_t B);

    //! Calculates the average of a vector of valarrays.
    /*! \param set A vector of valarrays to be averaged.
     *  \return The average of all valarrays in \e set.
     */
    Vector_double
    average(const std::vector<Vector_double >& set);

    //! Get the plugins from the application.
    /*! \return A vector of plugins registered by the user.
     */
    std::vector< Plugin > GetPluginLib();

    /*@}*/

}

#endif
