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

/*! \file view.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares wxStfView.
 */

#ifndef _VIEW_H
#define _VIEW_H

/*! \addtogroup wxstf
 *  @{
 */

#include <wx/docview.h>
class wxStfDoc;
class wxStfGraph;

//! The view class, derived from wxView.
/*! It is used to model the viewing and editing component of the file-based data.
 *  It is part of the document/view framework supported by wxWidgets. Note that
 *  this class does almost nothing in stimfit. Its only purpose is to adhere to
 *  the doc/view paradigm. All of the actual drawing happens in wxStfGraph.
 */
class wxStfView : public wxView
{
public:
    //! Constructor
    wxStfView();
    //! Destructor
    ~wxStfView() {}

    //! Override default view creation
    /*! Creates a child frame and a graph, sets the window title.
     *  \param doc Pointer to the attached document.
     *  \param flags View creation flags.
     *  \return true upon successful view creation, false otherwise.
     */
    virtual bool OnCreate(wxDocument *doc, long flags);

    //! The drawing function (note that actual drawing happens in wxStfGraph::OnDraw())
    /*! \param dc Pointer to the device context.
     *  \sa wxStfGraph::OnDraw()
     */
    virtual void OnDraw(wxDC *dc);

    //! Override default updating behaviour
    /*! Called when the graph should be refreshed.
     *  \param sender Pointer to the view that requested the update.
     *  \param hint Unused.
     */
    virtual void OnUpdate(wxView *sender, wxObject *hint = (wxObject *) NULL);

    //! Override default file closing behaviour
    /*! \param deleteWindow true if the child frame should be deleted.
     *  \return true if file closing was successful.
     */
    virtual bool OnClose(bool deleteWindow = true);

    //! Retrieve the attached graph
    /*! \return A pointer to the attached graph.
     */
    wxStfGraph* GetGraph() { return graph; }

    //! Retrieve the attached document
    /*! \return A pointer to the attached document.
     */
    wxStfDoc* Doc();

    //! Retrieve the attached document
    /*! \return A pointer to the attached document.
     */
    wxStfDoc* DocC() const;

protected:
    //! Called when the view is activated; dialogs and menus are then updated. 
    /*! \param activate true if this view is being activated.
     *  \param activeView Pointer to the view that is now active.
     *  \param deactiveView Pointer to the view that has just been deactivated.
     */
    virtual void OnActivateView(bool activate, wxView *activeView, wxView *deactiveView);

private:
    DECLARE_DYNAMIC_CLASS(wxStfView)
    DECLARE_EVENT_TABLE()

    wxStfGraph *graph;
    wxStfChildFrame *childFrame;
    
};

/*@}*/

#endif

