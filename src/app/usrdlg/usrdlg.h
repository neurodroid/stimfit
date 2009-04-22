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

/*! \file usrdlg.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-20
 *  \brief Declares wxStfUsrDlg.
 */

#ifndef _USRDLG_H
#define _USRDLG_H

/*! \addtogroup wxstf
 *  @{
 */

#ifdef _MSC_VER
#pragma warning( disable : 4251 )  // Disable warning messages
#endif

#include <vector>
#include <string>
#include <sstream>

#include "./../../core/stimdefs.h"

//! A user-defined dialog for entering floating-point numbers.
class wxStfUsrDlg : public wxDialog 
{
    DECLARE_EVENT_TABLE()

private:
    stf::UserInput input;
    std::vector<double> retVec;
    wxStdDialogButtonSizer* m_sdbSizer;
    std::vector<wxTextCtrl*> m_textCtrlArray;
    std::vector<wxStaticText*> m_staticTextArray;

    //! Only called when a modal dialog is closed with the OK button.
    /*! \return true if all dialog entries could be read successfully
     */
    bool OnOK();

public:
    //! Constructor
    /*! \param parent Pointer to parent window.
     *  \param input_ A stf::UserInput struct.
     *  \param id Window id.
     *  \param pos Initial position.
     *  \param size Initial size.
     *  \param style Dialog style.
     */
    wxStfUsrDlg(
            wxWindow* parent,
            const stf::UserInput& input_,
            int id = wxID_ANY,
            wxPoint pos = wxDefaultPosition,
            wxSize size = wxDefaultSize,
            int style = wxCAPTION
    );
    
    //! Get the user entries.
    /*! \return The user entries as a vector of doubles.
     */
    std::vector<double> readInput() const {return retVec;}
    
    //! Called upon ending a modal dialog.
    /*! \param retCode The dialog button id that ended the dialog
     *         (e.g. wxID_OK)
     */
    virtual void EndModal(int retCode);
};

/*! @} */

#endif
