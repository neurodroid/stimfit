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

/*! \file stfpython.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Python components.
 */
#if 0
#ifndef _STFPYTHON_H
#define _STFPYTHON_H

#include <wx/thread.h>
#include <Python.h>
#include <wx/wxPython/wxPython.h>

/*! \defgroup stfpython Stimfit's Python components (experimental)
 *  @{
 */

class CallOnExit {
public:
    ~CallOnExit() { wxTheApp->OnExit(); }
};

//! 
class wxStfPython {
public:
    wxStfPython() : entry(true), init(true) { }
    ~wxStfPython() { Leave(); }
    
    bool Run();
    bool Init();
    
private:
    bool Leave();
    bool entry, init;
};

//! 
class wxStfPythonThread : public wxThread {
public:
    wxStfPythonThread(wxWindow* parent) : wxThread(), globals(0), pPyWnd(0), pParent(parent), isInitialized(false) {  }
    ~wxStfPythonThread() { }

    // thread execution starts here
    virtual void *Entry();

    // called when the thread exits - whether it terminates normally or is
    // stopped with Delete() (but not when it is Kill()ed!)
    virtual void OnExit();
    
    wxWindow * GetPyWnd() { return pPyWnd; }
    
    bool IsInitialized() const { return isInitialized; }
    
private:
    PyObject * globals;
    wxWindow * pPyWnd;
    wxWindow* pParent;
    wxPyBlock_t blocked;
    bool isInitialized;
};

/*@}*/

#endif
#endif
