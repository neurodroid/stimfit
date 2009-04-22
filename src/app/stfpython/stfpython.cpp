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
#if 0
#include <iostream>
#include "wx/init.h"
#include "../app.h"
#include "./stfpython.h"

bool wxStfPython::Init() {
    std::cout << "Loading stimfit " << _STFMAJOR << "." << _STFMINOR << "." << _STFSUB << "... ";
    int argc = 0;
    wxChar **argv = NULL;

    // library initialization
    if ( !wxEntryStart(argc, argv) )
    {
        entry = false;
        init = false;
        return false;
    }

    // app initialization
    if ( !wxGetApp().CallOnInit() )
    {
        // don't call OnExit() if OnInit() failed
        init = false;
        return false;
    }
    return true;
}

bool wxStfPython::Run() { 
    return wxGetApp().OnRun();
}

bool wxStfPython::Leave() {
    if (entry) wxGetApp().OnExit();
    if (init)  wxEntryCleanup();
    return true;
}
#if 0
const char* python_code2 = "\
import sys\n\
sys.path.append('.')\n\
import wx\n\
from wx.py import shell, version\n\
import stf\n\
\n\
class MyPanel(wx.Panel):\n\
    def __init__(self, parent):\n\
        wx.Panel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER)\n\
\n\
        intro = 'stimfit 0.8.3 (Mar 22 2008, University of Freiburg)'\n\
        pycrust = shell.Shell(self, -1, introText=intro)\n\
        pycrust.push('import stf')\n\
        pycrust.push('stf.attach_app(wx.GetApp())')\n\
#        pycrust = wx.TextCtrl(self, -1, intro)\n\
\n\
        sizer = wx.BoxSizer(wx.VERTICAL)\n\
        sizer.Add(pycrust, 1, wx.EXPAND|wx.BOTTOM|wx.LEFT|wx.RIGHT, 10)\n\
\n\
        self.SetSizer(sizer)\n\
\n\
\n\
def sendApp( pStfApp ):\n\
    stf.attach_app( pStfApp )\n\
\n\
def makeWindow(parent):\n\
    win = MyPanel(parent)\n\
    return win\n\
";
#endif
void * wxStfPythonThread::Entry() {
#if 0
    // More complex embedded situations will require passing C++ objects to
    // Python and/or returning objects from Python to be used in C++.  This
    // sample shows one way to do it.  NOTE: The above code could just have
    // easily come from a file, or the whole thing could be in the Python
    // module that is imported and manipulated directly in this C++ code.  See
    // the Python API for more details.

    wxWindow *window = NULL;
    PyObject *result, *result2;

    // As always, first grab the GIL
    blocked = wxPyBeginBlockThreads();

    // Now make a dictionary to serve as the global namespace when the code is
    // executed.  Put a reference to the builtins module in it.  (Yes, the
    // names are supposed to be different, I don't know why...)
    globals = PyDict_New();
    PyObject* builtins = PyImport_ImportModule("__builtin__");
    PyDict_SetItemString(globals, "__builtins__", builtins);
    Py_DECREF(builtins);

    // Execute the code to make the makeWindow function
    result = PyRun_String(python_code2, Py_file_input, globals, globals);
    // Was there an exception?
    if (! result) {
        PyErr_Print();
        wxPyEndBlockThreads(blocked);
        return NULL;
    }
    Py_DECREF(result);

    // Now there should be an object named 'makeWindow' in the dictionary that
    // we can grab a pointer to:
    PyObject* func = PyDict_GetItemString(globals, "makeWindow");
    if (!PyCallable_Check(func)) {
        PyErr_Print();
        wxGetApp().ErrorMsg(wxT("Couldn't create python shell"));
        wxPyEndBlockThreads(blocked);
        return NULL;
    }

    // Now build an argument tuple and call the Python function.  Notice the
    // use of another wxPython API to take a wxWindows object and build a
    // wxPython object that wraps it.
    PyObject* arg = wxPyMake_wxObject(pParent, false);
    wxASSERT(arg != NULL);
    PyObject* tuple = PyTuple_New(1);
    PyTuple_SET_ITEM(tuple, 0, arg);
    result = PyEval_CallObject(func, tuple);
    Py_DECREF(tuple);

    // Was there an exception?
    if (! result) {
        PyErr_Print();
        wxGetApp().ErrorMsg(wxT("Couldn't create python shell"));
        wxPyEndBlockThreads(blocked);
        return NULL;
    }
    else {
        // Otherwise, get the returned window out of Python-land and
        // into C++-ville...
        if (!wxPyConvertSwigPtr(result, (void**)&window, _T("wxWindow"))) {
            PyErr_Print();
            wxGetApp().ErrorMsg(wxT("Returned object was not a wxWindow!"));
            Py_DECREF(tuple);
            wxPyEndBlockThreads(blocked);
            return NULL;
        }
        Py_DECREF(result);
    }

    // Attach the application to the stf module:
    // Get a pointer to the application:
    PyObject* func2 = PyDict_GetItemString(globals, "sendApp");
    if ( func2 == NULL ) {
        PyErr_Print();
        wxGetApp().ErrorMsg(wxT("Couldn't find sendApp; aborting python now"));
        Py_DECREF(globals);
        wxPyEndBlockThreads(blocked);
        return NULL;
    }
    if (!PyCallable_Check(func2)) {
        PyErr_Print();
        wxGetApp().ErrorMsg(wxT("Can't call sendApp; aborting python now"));
        Py_DECREF(globals);
        wxPyEndBlockThreads(blocked);
        return NULL;
    }
    // Now build an argument tuple and call the Python function.  Notice the
    // use of another wxPython API to take a wxWindows object and build a
    // wxPython object that wraps it.
    PyObject* arg2 = wxPyMake_wxObject(&wxGetApp(), false);
    wxASSERT(arg2 != NULL);
    PyObject* tuple2 = PyTuple_New(1);
    PyTuple_SET_ITEM(tuple2, 0, arg2);
    result2 = PyEval_CallObject(func2, tuple);
    Py_DECREF(tuple2);

    // Was there an exception?
    if (! result2) {
        PyErr_Print();
        wxGetApp().ErrorMsg(wxT("sendApp didn't succeed; aborting python now"));
        Py_DECREF(globals);
        wxPyEndBlockThreads(blocked);
        return NULL;
    }
    Py_DECREF(result2);
    
    isInitialized = true;
    pPyWnd = window;
#endif
    return (void*) pPyWnd;
}

void wxStfPythonThread::OnExit() {     
    std::cerr << "Prematurely ended python thread";

    // Release the python objects we still have
    Py_DECREF(globals);

    // Finally, after all Python stuff is done, release the GIL
    wxPyEndBlockThreads(blocked);
}
#endif