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

/*! \file fitseldlg.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares wxStfFitSelDlg.
 */

#ifndef _FITSELDLG_H
#define _FITSELDLG_H

/*! \addtogroup wxstf
 *  @{
 */

#include <vector>
#include <valarray>
#include "wx/listctrl.h"
#include "./../../core/stimdefs.h"

//! Non-linear regression settings dialog
class wxStfFitSelDlg : public wxDialog 
{
    DECLARE_EVENT_TABLE()

private:
    int m_fselect;
    std::valarray<double> init_p;
    std::valarray<double> opts;
    bool noInput;

    void SetPars();
    void SetOpts();
    void InitOptions(wxFlexGridSizer* optionsGrid);
    void Update_fselect();
    void read_init_p();
    void read_opts();
    static const int MAXPAR=20;

    void OnListItemSelected( wxListEvent& event );
    void OnButtonClick( wxCommandEvent& event );

    wxStdDialogButtonSizer* m_sdbSizer;
    wxListCtrl* m_listCtrl;
    wxTextCtrl *m_textCtrlMu,*m_textCtrlJTE,*m_textCtrlDP,*m_textCtrlE2,
    *m_textCtrlMaxiter, *m_textCtrlMaxpasses;
    std::vector< wxStaticText* > paramDescArray;
    std::vector< wxTextCtrl* > paramEntryArray;

    //! Only called when a modal dialog is closed with the OK button.
    /*! \return true if all dialog entries could be read successfully
     */
    bool OnOK();

public:
    //! Constructor
    /*! \param parent Pointer to parent window.
     *  \param id Window id.
     *  \param title Dialog title.
     *  \param pos Initial position.
     *  \param size Initial size.
     *  \param style Dialog style.
     */
    wxStfFitSelDlg(
            wxWindow* parent,
            int id = wxID_ANY,
            wxString title = wxT("Non-linear regression"),
            wxPoint pos = wxDefaultPosition,
            wxSize size = wxDefaultSize,
            int style = wxCAPTION
    );
    
    //! Called upon ending a modal dialog.
    /*! \param retCode The dialog button id that ended the dialog
     *         (e.g. wxID_OK)
     */
    virtual void EndModal(int retCode);

    //! Get the selected fit function.
    /*! \return The index of the selected fit function.
     */
    int GetFSelect() const {return m_fselect;}
    
    //! Get the initial parameters.
    /*! \return A valarray containing the initial parameter set to start the fit.
     */
    std::valarray<double> GetInitP() const {return init_p;}
    
    //! Get options for the algorithm.
    /*! \return A valarray containing the initial parameters for the algorithm.
     */
    std::valarray<double> GetOpts() const {return opts;}
    
    //! Determines whether user-defined initial parameters are allowed.
    /*! \param noInput_ Set to true if the user may set the initial parameters, false otherwise.
     *         Needed for batch analysis.
     */
    void SetNoInput(bool noInput_) {noInput=noInput_;}
};

/* @} */

#endif

