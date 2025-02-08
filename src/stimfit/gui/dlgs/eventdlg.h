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

/*! \file eventdlg.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-18
 *  \brief Declares wxStfEventDlg.
 */

#ifndef _EVENTDLG_H
#define _EVENTDLG_H

/*! \addtogroup wxstf
 *  @{
 */

#include <vector>

class SectionPointer;

//! Dialog for event-detection settings.
class wxStfEventDlg : public wxDialog 
{
    DECLARE_EVENT_TABLE()

private:
    double m_threshold;
    stf::extraction_mode m_mode;
    bool isExtract;
    int m_minDistance;
    int m_template;
    wxStdDialogButtonSizer* m_sdbSizer;
    wxTextCtrl *m_textCtrlThr, *m_textCtrlDist;
    wxStaticBoxSizer* m_radioBox;
    wxComboBox* m_comboBoxTemplates;

    void OnClements( wxCommandEvent & event );
    void OnJonas( wxCommandEvent & event );
    void OnPernia( wxCommandEvent & event );

    //! Only called when a modal dialog is closed with the OK button.
    /*! \return true if all dialog entries could be read successfully
     */
    bool OnOK();

public:
    //! Constructor
    /*! \param parent Pointer to parent window.
     *  \param templateSections A vector of pointers to sections that contain fits
     *         which might be used as a template.
     *  \param isExtract true if events are to be detected for later extraction
     *         (rather than just plotting the detection criterion or 
     *  \param id Window id.
     *  \param title Dialog title.
     *  \param pos Initial position.
     *  \param size Initial size.
     *  \param style Dialog style.
     */
    wxStfEventDlg(
            wxWindow* parent,
            const std::vector<stf::SectionPointer>& templateSections,
            bool isExtract,
            int id = wxID_ANY,
            wxString title = wxT("Event detection settings"),
            wxPoint pos = wxDefaultPosition,
            wxSize size = wxDefaultSize,
            int style = wxCAPTION
    );

    //! Get the event detection threshold.
    /*! \return The event detection threshold.
     */
    double GetThreshold() const {return m_threshold;}
    
    //! Indicates the selected extraction algorithm
    /*! \return The extraction algorithm.
     */
    stf::extraction_mode GetMode() const {return m_mode;}
    
    //! Get the minimal distance between events.
    /*! \return The minimal distance between events in units of sampling points.
     */
    int GetMinDistance() const {return m_minDistance;}
    
    //! Get the selected template.
    /*! \return The index of the template fit to be used for event detection.
     */
    int GetTemplate() const {return m_template;}
    
    //! Called upon ending a modal dialog.
    /*! \param retCode The dialog button id that ended the dialog
     *         (e.g. wxID_OK)
     */
    virtual void EndModal(int retCode);
};

/* @} */

#endif
