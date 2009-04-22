#ifdef WITH_PYTHON

// For compilers that support precompilation, includes "wx/wx.h".
#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include <Python.h>
#include <wx/wxPython/wxPython.h>

#include "./app.h"
#include "./parentframe.h"

#ifdef __WXMAC__
// Code to find executable path. Retrieved from:
// http://www.wxwidgets.org/docs/technote/install.htm

#include <ApplicationServices/ApplicationServices.h>

#if wxCHECK_VERSION(2, 9, 0)
#include <wx/osx/carbon/private.h>
#else
#include <wx/mac/carbon/private.h>
#endif
#include <wx/filename.h>

wxString GetExecutablePath()
{
    static bool found = false;
    static wxString path;

    ProcessInfoRec processinfo;
    ProcessSerialNumber procno ;
    FSSpec fsSpec;
    
    procno.highLongOfPSN = NULL ;
    procno.lowLongOfPSN = kCurrentProcess ;
    processinfo.processInfoLength = sizeof(ProcessInfoRec);
    processinfo.processName = NULL;
    processinfo.processAppSpec = &fsSpec;
    
    GetProcessInformation( &procno , &processinfo ) ;
    path = wxMacFSSpec2MacFilename(&fsSpec);
    found = true;
    return path;
}
#endif

bool wxStfApp::Init_wxPython()
{
    // Initialize Python
    Py_Initialize();
    PyEval_InitThreads();

#ifdef __WXMAC__
    // Add the cwd to the present path:
    wxString app_path = wxFileName( GetExecutablePath() ).GetPath();
    wxString cwd;
    cwd << wxT("import os\n");
    cwd << wxT("cwd=\"") << app_path << wxT("/../Frameworks\"\n");
    cwd << wxT("import sys\n");
    cwd << wxT("sys.path.append(cwd)\n");
    cwd << wxT("cwd=\"") << app_path << wxT("/../Frameworks/stimfit\"\n");
    cwd << wxT("sys.path.append(cwd)\n");
    cwd << wxT("print cwd\n");
    int cwd_result = PyRun_SimpleString(cwd.utf8_str());
    if (cwd_result!=0) {
        PyErr_Print();
        ErrorMsg( wxT("Couldn't modify Python path") );
        Py_Finalize();
        return false;
    }
#endif
    
    // Load the wxPython core API.  Imports the wx._core_ module and sets a
    // local pointer to a function table located there.  The pointer is used
    // internally by the rest of the API functions.
    
    // Specify version of the wx module to be imported
    PyObject* wxversion = PyImport_ImportModule("wxversion");
    if (wxversion==NULL) {
        PyErr_Print();
        ErrorMsg( wxT("Couldn't import wxversion") );
        Py_Finalize();
        return false;
    }
    PyObject* wxselect = PyObject_GetAttrString(wxversion, "select");
    Py_DECREF(wxversion);
    if (!PyCallable_Check(wxselect)) {
        PyErr_Print();
        ErrorMsg( wxT("Couldn't select correct version of wx") );
        Py_Finalize();
        return false;
    }
#if wxCHECK_VERSION(2, 9, 0)
    PyObject* ver_string = Py_BuildValue("ss","2.9.0.0","");
#else
    PyObject* ver_string = Py_BuildValue("ss","2.8","");
#endif
    PyObject* result = PyEval_CallObject(wxselect, ver_string);
    Py_DECREF(ver_string);
    if (result == NULL) {
        PyErr_Print();
        ErrorMsg( wxT("Couldn't call wxversion.select") );
        Py_Finalize();
        return false;
    }

    long iresult = PyInt_AsLong(result);
    Py_DECREF(result);
    if (iresult == 0) {
        PyErr_Print();
        ErrorMsg( wxT("Couldn't select correct version of wx") );
        Py_Finalize();
        return false;
    }
    
    if ( ! wxPyCoreAPI_IMPORT() ) {
        PyErr_Print();
        wxString errormsg;
        errormsg << wxT("Couldn't load wxPython core API.\n");
#ifdef _WINDOWS
        errormsg << wxT("You need to set the current working directory\n");
        errormsg << wxT("to the program directory so that all shared\n");
        errormsg << wxT("libraries can be found.\n");
        errormsg << wxT("If you used a shortcut, right-click on it,\n");
        errormsg << wxT("choose \"Properties\", select the \"Shortcut\"\n");
        errormsg << wxT("tab, then set \"Run in...\" to the stimfit program\n");
        errormsg << wxT("directory (typically C:\\Program Files\\Stimfit).\n");
        errormsg << wxT("If you started stimfit from the command line, you\n");
        errormsg << wxT("have to set the current working directory using the\n");
        errormsg << wxT("\"/d\" option (e.g. /d=C:\\Program Files\\Stimfit)\n");
#endif
        ErrorMsg( errormsg );
        Py_Finalize();
        return false;
    }        
    
    // Save the current Python thread state and release the
    // Global Interpreter Lock.
    m_mainTState = wxPyBeginAllowThreads();

    return true;
}

bool wxStfApp::Exit_wxPython()
{
    wxPyEndAllowThreads(m_mainTState);
    Py_Finalize();
    return true;
}

void wxStfParentFrame::RedirectStdio()
{
    // This is a helpful little tidbit to help debugging and such.  It
    // redirects Python's stdout and stderr to a window that will popup
    // only on demand when something is printed, like a traceback.
    const char* python_redirect = "\
import sys\n\
import wx\n\
output = wx.PyOnDemandOutputWindow()\n\
sys.stdin = sys.stderr = output\n\
";
    wxPyBlock_t blocked = wxPyBeginBlockThreads();
    PyRun_SimpleString(python_redirect);
    wxPyEndBlockThreads(blocked);
}

wxWindow* wxStfParentFrame::DoPythonStuff(wxWindow* parent)
{
    // More complex embedded situations will require passing C++ objects to
    // Python and/or returning objects from Python to be used in C++.  This
    // sample shows one way to do it.  NOTE: The above code could just have
    // easily come from a file, or the whole thing could be in the Python
    // module that is imported and manipulated directly in this C++ code.  See
    // the Python API for more details.

    wxWindow *window = NULL;
    PyObject *result;

    // As always, first grab the GIL
    wxPyBlock_t blocked = wxPyBeginBlockThreads();

    // Now make a dictionary to serve as the global namespace when the code is
    // executed.  Put a reference to the builtins module in it.  (Yes, the
    // names are supposed to be different, I don't know why...)
    PyObject* globals = PyDict_New();
    PyObject* builtins = PyImport_ImportModule("__builtin__");
    PyDict_SetItemString(globals, "__builtins__", builtins);
    Py_DECREF(builtins);

    // Execute the code to make the makeWindow function
    result = PyRun_String(python_code2.char_str(), Py_file_input, globals, globals);
    // Was there an exception?
    if (! result) {
        PyErr_Print();
        wxGetApp().ErrorMsg(wxT("Couldn't initialize python shell"));
        wxPyEndBlockThreads(blocked);
        return NULL;
    }
    Py_DECREF(result);

    // Now there should be an object named 'makeWindow' in the dictionary that
    // we can grab a pointer to:
    PyObject* func = PyDict_GetItemString(globals, "makeWindow");
    if (!PyCallable_Check(func)) {
        PyErr_Print();
        wxGetApp().ErrorMsg(wxT("Couldn't initialize window for the python shell"));
        wxPyEndBlockThreads(blocked);
        return NULL;
    }

    // Now build an argument tuple and call the Python function.  Notice the
    // use of another wxPython API to take a wxWindows object and build a
    // wxPython object that wraps it.
    PyObject* arg = wxPyMake_wxObject(parent, false);
    wxASSERT(arg != NULL);
    PyObject* tuple = PyTuple_New(1);
    PyTuple_SET_ITEM(tuple, 0, arg);
    result = PyEval_CallObject(func, tuple);
    Py_DECREF(tuple);

    // Was there an exception?
    if (! result) {
        PyErr_Print();
        wxGetApp().ErrorMsg(wxT("Couldn't create window for the python shell"));
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

    // Release the python objects we still have
    Py_DECREF(globals);

    // Finally, after all Python stuff is done, release the GIL
    wxPyEndBlockThreads(blocked);

    return window;
}

#endif
