#ifndef STIMFIT_READERDLG_H
#define STIMFIT_READERDLG_H

#include <wx/dialog.h>
#include <vector>
#include "../../../libstfio/stfio.h"

class wxChoice;
class wxCheckBox;

class wxStfReaderDlg : public wxDialog {
public:
    wxStfReaderDlg(wxWindow* parent, const wxString& filename, const stfio::ReaderInfo& info,
                   stfio::fileReader preference);
    stfio::fileReader GetReader() const;
    bool RememberChoice() const;

private:
    std::vector<stfio::fileReader> readers;
    wxChoice* choice;
    wxCheckBox* remember;
};

#endif
