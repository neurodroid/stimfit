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

/*! \file stfcheckbox.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares wxStfCheckBox. Derived from wxCheckBox.
 */

#ifndef _STFCHECKBOX_H
#define _STFCHECKBOX_H

/*! \addtogroup wxstf
 *  @{
 */

//! A checkbox used to select or unselect detected events.
/*! Toggles the stf::Event status, and forwards keyboard input to the graph.*/
class wxStfCheckBox : public wxCheckBox {
DECLARE_CLASS(wxStfCheckBox)
public:
    //! Constructor
    /*! \param parent Pointer to the parent window.
     *  \param id Window id.
     *  \param label The checkbox label.
     *  \param pEvent The event attached to this checkbox.
     *  \param pos Initial checkbox position.
     *  \param size Initial checkbox size.
     *  \param style Checkbox style.
     *  \param validator Checkbox validator.
     *  \param name Name of this grid.
     */
    wxStfCheckBox(
        wxWindow *parent,
        wxWindowID id,
        const wxString& label,
        stf::Event* pEvent,
        const wxPoint& pos = wxDefaultPosition,
        const wxSize& size = wxDefaultSize,
        long style = 0,
        const wxValidator& validator = wxDefaultValidator,
        const wxString& name = wxCheckBoxNameStr
    );

    //! Resets the pointer to the attached event
    /*! \param pEvent The pointer to the new event
     */
    void ResetEvent( stf::Event* pEvent ) { m_pEvent = pEvent; SetValue( !pEvent->GetDiscard() );}

private:
    void OnKeyDown(wxKeyEvent& event);
    void OnStfClicked(wxMouseEvent& event);

    stf::Event* m_pEvent;

DECLARE_EVENT_TABLE()
};

/*@}*/

#endif
