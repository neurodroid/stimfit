#include <wx/wxprec.h>

#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#include "./usrdlg.h"

BEGIN_EVENT_TABLE( wxStfUsrDlg, wxDialog )
END_EVENT_TABLE()

wxStfUsrDlg::wxStfUsrDlg(
		wxWindow* parent,
		const stf::UserInput& input_,
		int id,
		wxPoint pos,
		wxSize size,
		int style
                
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
) : wxDialog( parent, id, input_.title, pos, size, style ),
#else
    ) : wxDialog( parent, id, wxString(input_.title.c_str(), wxConvUTF8), pos, size, style ),
#endif
    input(input_),
    retVec(input_.labels.size()),
    m_textCtrlArray(input_.labels.size()),
    m_staticTextArray(input_.labels.size())
{
	wxFlexGridSizer* gSizer;
	gSizer = new wxFlexGridSizer( (int)input.labels.size(), 2, 0, 0 );

	for (std::size_t nRow=0;nRow<input.labels.size();++nRow) {
		m_staticTextArray[nRow]=
			new wxStaticText(
				this,
				wxID_ANY,
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
				input.labels[nRow],
#else
				wxString(input.labels[nRow].c_str(), wxConvUTF8),
#endif
				wxDefaultPosition,
				wxDefaultSize,
				wxTE_LEFT
			);
		gSizer->Add( m_staticTextArray[nRow], 0, wxALIGN_CENTER_VERTICAL | wxALL, 2 );

		wxString defLabel;
		defLabel << input.defaults[nRow];
		m_textCtrlArray[nRow]=
			new wxTextCtrl(
				this,
				wxID_ANY,
				defLabel,
				wxDefaultPosition,
				wxSize(64,20),
				wxTE_RIGHT
			);
		gSizer->Add( m_textCtrlArray[nRow], 0, wxALIGN_CENTER_VERTICAL | wxALL, 2 );
	}
	wxSizer* topSizer;
	topSizer=new wxBoxSizer(wxVERTICAL);
	topSizer->Add(gSizer,0,wxALIGN_CENTER,5);

	m_sdbSizer = new wxStdDialogButtonSizer();
	m_sdbSizer->AddButton( new wxButton( this, wxID_OK ) );
	m_sdbSizer->AddButton( new wxButton( this, wxID_CANCEL ) );
	m_sdbSizer->Realize();
	topSizer->Add( m_sdbSizer, 0, wxALIGN_CENTER, 5 );

	topSizer->SetSizeHints(this);
	this->SetSizer( topSizer );

	this->Layout();
}

void wxStfUsrDlg::EndModal(int retCode) {
	// similar to overriding OnOK in MFC (I hope...)
	if (retCode==wxID_OK) {
		if (!OnOK()) {
			wxLogMessage(wxT("Check your entries"));
			return;
		}
	}
	wxDialog::EndModal(retCode);
}

bool wxStfUsrDlg::OnOK() {
	try {
		for (std::size_t n=0;n<retVec.size();++n) {
			// Read entry to string:
			wxString entry;
			entry << m_textCtrlArray.at(n)->GetValue();
			entry.ToDouble( &retVec[n] );
		}
	}
	catch (const std::out_of_range&) {
		return false;
	}
	return true;
}
