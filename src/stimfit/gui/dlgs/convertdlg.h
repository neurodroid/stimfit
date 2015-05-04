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

/*! \file convertdlg.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2015-05-04
 *  \brief Batch conversion of files
 */

#ifndef _CONVERTDLG_H
#define _CONVERTDLG_H

/*! \addtogroup wxstf
 *  @{
 */

#include <wx/dirctrl.h>
#include <wx/filename.h>
#include <vector>
#include "./../../stf.h"

//! Dialog for batch conversion of files.from cfs to atf.
class wxStfConvertDlg : public wxDialog 
{
    DECLARE_EVENT_TABLE()

private:
    wxGenericDirCtrl *mySrcDirCtrl, *myDestDirCtrl; 
    wxString srcDir,destDir;
    wxString srcFilter;
    wxCheckBox* myCheckBoxSubdirs;

    stfio::filetype srcFilterExt, destFilterExt;
    wxArrayString srcFileNames;

    bool ReadPath(const wxString& path);

    void OnComboBoxSrcExt(wxCommandEvent& event);
    void OnComboBoxDestExt(wxCommandEvent& event);

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
    wxStfConvertDlg( wxWindow* parent, int id = wxID_ANY, wxString title = wxT("Convert file series"),
            wxPoint pos = wxDefaultPosition, wxSize size = wxDefaultSize, int style = wxCAPTION );

    //! Get the source directory.
    /*! \return The source directory.
     */
    wxString GetSrcDir() const {return srcDir;}

    //! Get the destination directory.
    /*! \return The destination directory.
     */
    wxString GetDestDir() const {return destDir;}

    //! Get the source extension filter.
    /*! \return The source extension filter.
     */
    wxString GetSrcFilter() const {return srcFilter;}

    //! Get the source extension as stfio::filetype.
    /*! \return The source extension as stfio::filetype.
     */
    stfio::filetype GetSrcFileExt() const {return srcFilterExt;}

    //! Get the destination extension as stfio::filetype.
    /*! \return The destination extension as stfio::filetype.
     */
    stfio::filetype GetDestFileExt() const {return destFilterExt;}

    //! Get the list of file names.
    /*! \return A vector with source file names.
     */
    wxArrayString GetSrcFileNames() const {return srcFileNames;}
    
    //! Called upon ending a modal dialog.
    /*! \param retCode The dialog button id that ended the dialog
     *         (e.g. wxID_OK)
     */
    virtual void EndModal(int retCode);

};

/* @} */

#endif
