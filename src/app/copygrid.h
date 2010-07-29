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

/*! \file copygrid.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares wxStfGrid. Derived from wxGrid to allow copying to clipboard.
 */

#ifndef _COPYGRID_H
#define _COPYGRID_H

/*! \addtogroup wxstf
 *  @{
 */

//! Derived from wxGrid. Allows to copy cells to the clipboard.
class wxStfGrid : public wxGrid {
    DECLARE_CLASS(wxStfGrid)
public:
    //! Constructor
    /*! \param parent Pointer to the parent window.
     *  \param id Window id.
     *  \param pos Initial window position.
     *  \param size Initial window size.
     *  \param style Grid style.
     *  \param name Name of this grid.
     */
    wxStfGrid(
            wxWindow* parent, 
            wxWindowID id, 
            const wxPoint& pos = wxDefaultPosition, 
            const wxSize& size = wxDefaultSize, 
            long style = wxWANTS_CHARS, 
            const wxString& name = wxGridNameStr
    );
    
    //! Get the selection.
    /*! \return The selected cells as a string.
     */
    wxString GetSelection() const {return selection;}

    // Get the context menu.
    /*! \return A pointer to the context menu.
     */
    wxMenu* get_labelMenu() {return m_labelContext.get();}
    
    //! Updates the context menu.
    void ViewResults();

private:
    wxString selection;
    void Copy(wxCommandEvent& event);
    void OnRClick(wxGridEvent& event);
    void OnLabelRClick(wxGridEvent& event);
    void OnKeyDown(wxKeyEvent& event);
    void ViewCrosshair(wxCommandEvent& event);
    void ViewBaseline(wxCommandEvent& event);
    void ViewBaseSD(wxCommandEvent& event);
    void ViewThreshold(wxCommandEvent& event);
    void ViewPeakzero(wxCommandEvent& event);
    void ViewPeakbase(wxCommandEvent& event);
    void ViewPeakthreshold(wxCommandEvent& event);
    void ViewRT2080(wxCommandEvent& event);
    void ViewT50(wxCommandEvent& event);
    void ViewRD(wxCommandEvent& event);
    void ViewSloperise(wxCommandEvent& event);
    void ViewSlopedecay(wxCommandEvent& event);
    void ViewLatency(wxCommandEvent& event);
#ifdef WITH_PSLOPE
    void ViewPSlope(wxCommandEvent& event);
#endif
    void ViewCursors(wxCommandEvent& event);
    void SetCheckmark(const wxString& RegEntry, int id);

    boost::shared_ptr<wxMenu> m_context;
    boost::shared_ptr<wxMenu> m_labelContext;
    DECLARE_EVENT_TABLE()
};

/*@}*/

#endif
