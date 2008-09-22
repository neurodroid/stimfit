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

/*! \file printout.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares wxStfPrintout. Derived from wxPrintout to handle printing of traces.
 */

#ifndef _PRINTOUT_H
#define _PRINTOUT_H

/*! \addtogroup wxstf
 *  @{
 */

//! Handles printing of traces.
class wxStfPrintout : public wxPrintout
{
public:
    //! Constructor
    /*! \param title Printout title
     */
    wxStfPrintout(const wxChar *title = _T("Printout"));

    //! Called by the framework when a page should be printed.
    /*! \param page The page number to be printed
     *  \return false will cancel the print job.
     */
    bool OnPrintPage(int page);
    
    //! Checks whether a page exists.
    /*! \param page The page number to be checked.
     *  \return True if \e page exists.
     */
    bool HasPage(int page);
    
    //! Called by the framework at the start of document printing.
    /*! \param startPage Page from which to start printing.
     *  \param endPage Page at which to end printing.
     *  \return false cancels the print job.
     */
    bool OnBeginDocument(int startPage, int endPage);

    //! Retrieve information about the pages that will be printed.
    /*! \param minPage On return, a pointer to the minimal page number allowed.
     *  \param maxPage On return, a pointer to the maximal page number allowed.
     *  \param selPageFrom On return, a pointer to the minimal selected page number allowed.
     *  \param selPageTo On return, a pointer to the maximal selected page number allowed.
     */
    void GetPageInfo(int *minPage, int *maxPage, int *selPageFrom, int *selPageTo);

    //! Prints the first (and only) page.
    void DrawPageOne();

private:
    void PrintHeader(wxDC* pDC,double scale);
    bool store_noGimmicks;

    wxStfDoc* Doc() const {return wxGetApp().GetActiveDoc();}

};

/*@}*/

#endif
