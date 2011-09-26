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

// table.cpp
// Table entries for spreadsheets in wxStfGrid.
// 2007-12-27, Christoph Schmidt-Hieber, University of Freiburg

#include "wx/wxprec.h"

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif
#include "wx/grid.h"

#include "./table.h"

bool wxStfTable::IsEmptyCell( int row, int col ) {
	try {
		if (row==0 && col>=1) {
			return table.GetColLabel(col-1) == "\0";
		} else if (col==0 && row>=1) {
			return table.GetRowLabel(row-1) == "\0";
		} else if (col!=0 && row!=0) {
            return table.IsEmpty(row-1,col-1); 
		} else {
			return true;
		}
	}
	catch (const std::out_of_range&) {
		return true;
	}
}

wxString wxStfTable::GetValue( int row, int col ) {
	try {
		if (row==0 && col>=1) {
                    return stf::std2wx(table.GetColLabel(col-1));
		} else if (col==0 && row>=1) {
                    return stf::std2wx(table.GetRowLabel(row-1));
		} else if (col!=0 && row!=0) {
            if (table.IsEmpty(row-1,col-1))
                return wxT("\0");
			wxString strVal; 
			strVal << table.at(row-1,col-1);
			return strVal;
		} else {
			return wxT("\0");
		}
	}
	catch (const std::out_of_range&) {
		return wxT("\0");
	}
}

void wxStfTable::SetValue( int row, int col, const wxString& value ) {
	try {
		if (row==0 && col>=1) {
                    return table.SetColLabel(col-1, stf::wx2std(value));
		} else if (col==0 && row>=1) {
                    return table.SetRowLabel(row-1, stf::wx2std(value));
		} else if (col!=0 && row!=0) {
                    wxString strVal; 
                    strVal << value;
                    double in=0.0;
                    strVal.ToDouble(&in);
                    table.at(row-1,col-1)=in;
		} else {
                    return;
		}
	}
	catch (const std::out_of_range&) {
            return;
	}
}

wxString wxStfTable::GetSelection(const wxGridCellCoordsArray& selection) {
	wxString ret(wxT("\0"));
	for (std::size_t n_sel=0;n_sel<selection.size();++n_sel) {
		try {
			ret+=
				GetValue(
					selection[n_sel].GetRow(),
					selection[n_sel].GetCol()
				)+ wxT("\t");
		}
		catch (const std::out_of_range&) {
			return wxT("\0");
		}
	}
	return ret;
}
