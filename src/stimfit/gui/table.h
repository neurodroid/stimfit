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

/*! \file table.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares wxStfTable. Derived from wxGridTableBase.
 */

#ifndef _TABLE_H
#define _TABLE_H

#include "../stf.h"

/*! \addtogroup wxstf
 *  @{
 */

//! Adapts stf::Table to be used by wxStfGrid
class wxStfTable : public wxGridTableBase {
public:
    //! Constructor
    /*! \param table_ The associated stf::Table
     */
    wxStfTable(const stf::Table& table_) : table(table_) {}

    //! Get the number of rows.
    /*! \return The number of rows.
     */
    virtual int GetNumberRows() {return (int)table.nRows()+1;}
    
    //! Get the number of columns.
    /*! \return The number of columns.
     */
    virtual int GetNumberCols() {return (int)table.nCols()+1;}
    
    //! Check whether a cell is empty.
    /*! \param row The row number of the cell.
     *  \param col The column number of the cell.
     *  \return true if the cell is empty, false otherwise.
     */
    virtual bool IsEmptyCell(int row,int col);

    //! Retrieve a cell entry.
    /*! \param row The row number of the cell.
     *  \param col The column number of the cell.
     *  \return The cell entry as a string.
     */
    virtual wxString GetValue( int row, int col );

    //! Set a cell entry.
    /*! \param row The row number of the cell.
     *  \param col The column number of the cell.
     *  \param value The new cell entry.
     */
    virtual void SetValue( int row, int col, const wxString& value );

    //! Retrieve values from selected cells.
    /*! \param selection The selected cells.
     *  \return The selection as a single string.
     */
    wxString GetSelection(const wxGridCellCoordsArray& selection);

private:
    stf::Table table;
};

/*@}*/

#endif

