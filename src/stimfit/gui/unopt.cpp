#ifdef WITH_PYTHON

// For compilers that support precompilation, includes "wx/wx.h".
#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif // WX_PRECOMP

#ifdef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE_WAS_DEF
#undef _POSIX_C_SOURCE
#endif
#ifdef _XOPEN_SOURCE
#define _XOPEN_SOURCE_WAS_DEF
#undef _XOPEN_SOURCE
#endif
#include <Python.h>
#ifdef _POSIX_C_SOURCE_WAS_DEF
  #ifndef _POSIX_C_SOURCE
    #define _POSIX_C_SOURCE
  #endif
#endif
#ifdef _XOPEN_SOURCE_WAS_DEF
  #ifndef _XOPEN_SOURCE
    #define _XOPEN_SOURCE
  #endif
#endif

#if defined(__WXMAC__) || defined(__WXGTK__)
  #pragma GCC diagnostic ignored "-Wwrite-strings"
#endif
#if PY_MAJOR_VERSION >= 3
#include <sip.h>
#include <wxPython/wxpy_api.h>
#define PyString_Check PyUnicode_Check
#define PyString_AsString PyBytes_AsString
#define PyString_FromString PyUnicode_FromString
#else
#include <wx/wxPython/wxPython.h>
#endif
// revert to previous behaviour
#if defined(__WXMAC__) || defined(__WXGTK__)
  #pragma GCC diagnostic warning "-Wwrite-strings"
#endif

//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "./app.h"
#include "./doc.h"
#include "./parentframe.h"

int stf::Extension::n_extensions = 0;

#if defined(__WXMAC__) || defined (__WXGTK__)
#include <wx/stdpaths.h>

wxString GetExecutablePath()
{
    return  wxStandardPaths::Get( ).GetExecutablePath();
}
#endif // __WXMAC__ || __WXGTK__

#ifdef _WINDOWS
#include <winreg.h>
#include <wx/filename.h>

wxString GetExecutablePath() {

    HKEY keyHandle;

    if( RegOpenKeyEx( HKEY_CURRENT_USER, wxT("Software\\Stimfit 0.14"), 0, 
                      KEY_QUERY_VALUE, &keyHandle) == ERROR_SUCCESS)
    {
        DWORD BufferSize = 8192;
        DWORD cbData = BufferSize;
        		
        wxCharTypeBuffer<wxChar> data( BufferSize );
        DWORD dwRet = RegQueryValueEx( keyHandle, TEXT("InstallLocation"),
                                       NULL, NULL, (LPBYTE) data.data(), &cbData );
        while( dwRet == ERROR_MORE_DATA )
        {
            // Get a buffer that is big enough.

            BufferSize += 4096;
            data.extend( BufferSize );
            cbData = BufferSize;

            dwRet = RegQueryValueEx( keyHandle, TEXT("InstallLocation"),
                                     NULL, NULL, (LPBYTE) data.data(), &cbData );
        }
        if( dwRet == ERROR_SUCCESS ) {
            RegCloseKey(keyHandle);
            return wxString( data );
        } else {
            // wxGetApp().ErrorMsg( wxT("Couldn't read registry key for Stimfit") );
            return wxT("");
        }
    } else {
        // wxGetApp().ErrorMsg( wxT("Couldn't open registry key for Stimfit") );
        return wxT("");
    }
}
#endif // _WINDOWS

#if PY_MAJOR_VERSION >= 3
PyObject*  wxPyMake_wxObject(wxObject* source, bool setThisOwn) {
    bool checkEvtHandler = true;
    PyObject* target = NULL;
    bool      isEvtHandler = false;
    bool      isSizer = false;

    if (source) {
        // If it's derived from wxEvtHandler then there may
        // already be a pointer to a Python object that we can use
        // in the OOR data.
        if (checkEvtHandler && wxIsKindOf(source, wxEvtHandler)) {
            isEvtHandler = true;
            wxEvtHandler* eh = (wxEvtHandler*)source;
            wxPyClientData* data = (wxPyClientData*)eh->GetClientObject();
            if (data) {
                target = data->GetData();
            }
        }

        // Also check for wxSizer
        if (!target && wxIsKindOf(source, wxSizer)) {
            isSizer = true;
            wxSizer* sz = (wxSizer*)source;
            wxPyClientData* data = (wxPyClientData*)sz->GetClientObject();
            if (data) {
                target = data->GetData();
            }
        }
        if (! target) {
            // Otherwise make it the old fashioned way by making a new shadow
            // object and putting this pointer in it.  Look up the class
            // heirarchy until we find a class name that is located in the
            // python module.
            const wxClassInfo* info   = source->GetClassInfo();
            wxString           name   = info->GetClassName();
	    wxString           childname = name.Clone();
            if (info) {
                target = wxPyConstructObject((void*)source, name.c_str(), setThisOwn);
		while (target == NULL) {
		    info = info->GetBaseClass1();
		    name = info->GetClassName();
		    if (name == childname)
                        break;
		    childname = name.Clone();
		    target = wxPyConstructObject((void*)source, name.c_str(), setThisOwn);
		}
                if (target && isEvtHandler)
                    ((wxEvtHandler*)source)->SetClientObject(new wxPyClientData(target));
                if (target && isSizer)
                    ((wxSizer*)source)->SetClientObject(new wxPyClientData(target));
            } else {
                wxString msg(wxT("wxPython class not found for "));
                msg += source->GetClassInfo()->GetClassName();
                PyErr_SetString(PyExc_NameError, msg.mbc_str());
                target = NULL;
            }
        }
    } else {  // source was NULL so return None.
        Py_INCREF(Py_None); target = Py_None;
    }
    return target;
}
#endif

bool wxStfApp::Init_wxPython()
{
    // Initialize the Python interpreter
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    PyEval_InitThreads();

    wxString cwd;
#ifdef __WXMAC__
    // Add the cwd to the present path:
    wxString app_path = wxFileName( GetExecutablePath() ).GetPath();
    cwd << wxT("import os\n");
    cwd << wxT("cwd=\"") << app_path << wxT("/../Frameworks\"\n");
    cwd << wxT("import sys\n");
    cwd << wxT("sys.path.append(cwd)\n");
    cwd << wxT("cwd=\"") << app_path << wxT("/../Frameworks/stimfit\"\n");
    cwd << wxT("sys.path.append(cwd)\n");
    // cwd << wxT("cwd=\"") << app_path << wxT("/../Frameworks/numpy\"\n");
    // cwd << wxT("sys.path.insert(0,cwd)\n");
#ifdef _STFDEBUG
    cwd << wxT("print(sys.path)\n");
    cwd << wxT("import numpy\n");
    cwd << wxT("print(numpy.version.version)\n");
#endif // _STFDEBUG
#endif // __WXMAC__
    
#ifdef __WXGTK__
    // Add the cwd to the present path:
    wxString app_path = wxFileName( GetExecutablePath() ).GetPath();
    cwd << wxT("import os\n");
    cwd << wxT("cwd=\"") << app_path << wxT("/../lib/stimfit\"\n");
    cwd << wxT("import sys\n");
    cwd << wxT("sys.path.append(cwd)\n");
#ifdef _STFDEBUG
    cwd << wxT("print(sys.path)\n");
    cwd << wxT("import numpy\n");
    cwd << wxT("print(numpy.version.version)\n");
#endif // _STFDEBUG
#endif // __WXGTK__

#ifdef _WINDOWS
    // Add the cwd to the present path:
    wxString app_path = GetExecutablePath().BeforeFirst( wxUniChar('\0') );
	cwd << wxT("cwd = \"") << app_path 
		<< wxT("\\wx-3.0-msw\"\nimport sys\nsys.path.insert(0,cwd)\n");
	cwd << wxT("cwd = \"") << app_path 
		<< wxT("\\stf-site-packages\"\nsys.path.insert(0,cwd)\n");
	cwd << wxT("cwd = \"") << app_path
		<< wxT("\"\nsys.path.insert(0,cwd)\n");
#endif

    int cwd_result = PyRun_SimpleString(cwd.utf8_str());
    if (cwd_result!=0) {
        PyErr_Print();
        ErrorMsg( wxT("Couldn't modify Python path") );
        Py_Finalize();
        return false;
    }

    // Load the wxPython core API.  Imports the wx._core_ module and sets a
    // local pointer to a function table located there.  The pointer is used
    // internally by the rest of the API functions.
    
#if PY_MAJOR_VERSION < 3
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
#if wxCHECK_VERSION(3, 1, 0)
    PyObject* ver_string = Py_BuildValue("ss","3.1","");
#elif wxCHECK_VERSION(3, 0, 0)
    PyObject* ver_string = Py_BuildValue("ss","3.0","");
#elif wxCHECK_VERSION(2, 9, 0)
    PyObject* ver_string = Py_BuildValue("ss","2.9","");
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
#endif // < python 3
    
#if 0 // wxversion.select doesn't return an error code, but raises an exception
    long iresult = PyInt_AsLong(result);
    Py_DECREF(result);
    if (iresult == 0) {
        PyErr_Print();
        ErrorMsg( wxT("Couldn't select correct version of wx") );
        Py_Finalize();
        return false;
    }
#endif
#if PY_MAJOR_VERSION >= 3
    if (wxPyGetAPIPtr()==NULL) {
#else
    if ( ! wxPyCoreAPI_IMPORT() ) {
#endif
        PyErr_Print();
        wxString errormsg;
        errormsg << wxT("Couldn't load wxPython core API.\n");

#ifdef _WINDOWS
        errormsg << wxT("Try the following steps:\n");
        errormsg << wxT("1.\tUninstall a previous Stimfit installation\n");
        errormsg << wxT("\t(Control Panel->Software)\n");
        errormsg << wxT("2.\tUninstall a previous wxPython installation\n");
        errormsg << wxT("\t(Control Panel->Software)\n");
        errormsg << wxT("3.\tUninstall a previous Python 2.5 installation\n");
        errormsg << wxT("\t(Control Panel->Software)\n");
        errormsg << wxT("4.\tSet the current working directory\n");
        errormsg << wxT("\tto the program directory so that all shared\n");
        errormsg << wxT("\tlibraries can be found.\n");
        errormsg << wxT("\tIf you used a shortcut, right-click on it,\n");
        errormsg << wxT("\tchoose \"Properties\", select the \"Shortcut\"\n");
        errormsg << wxT("\ttab, then set \"Run in...\" to the stimfit program\n");
        errormsg << wxT("\tdirectory (typically C:\\Program Files\\Stimfit).\n");
        errormsg << wxT("\tIf you started stimfit from the command line, you\n");
        errormsg << wxT("\thave to set the current working directory using the\n");
        errormsg << wxT("\t\"/d\" option (e.g. /d=C:\\Program Files\\Stimfit)\n");
#endif // _WINDOWS


        ErrorMsg( errormsg );
        Py_Finalize();
#if PY_MAJOR_VERSION < 3
        Py_DECREF(result);
#endif
        return false;
    }        
    
    // Save the current Python thread state and release the
    // Global Interpreter Lock.
    m_mainTState = wxPyBeginAllowThreads();

#ifdef IPYTHON
    // Set a dummy sys.argv for IPython
    wxPyBlock_t blocked = wxPyBeginBlockThreads();
    char* argv = (char *)"\0";
    PySys_SetArgv(1, &argv);
    wxPyEndBlockThreads(blocked);
#endif

    return true;
}

void wxStfApp::ImportPython(const wxString &modulelocation) {
        
    // Get path and filename from modulelocation 
    wxString python_path = wxFileName(modulelocation).GetPath();
    wxString python_file = wxFileName(modulelocation).GetName();

    // Grab the Global Interpreter Lock.
    wxPyBlock_t blocked = wxPyBeginBlockThreads();

    wxString python_import;
#ifdef IPYTHON
    // the ip object is created to access the interactive IPython session
    python_import << wxT("import IPython.ipapi\n");
    python_import << wxT("ip = IPython.ipapi.get()\n");
    python_import << wxT("import sys\n");
    python_import << wxT("sys.path.append(\"") << python_path << wxT("\")\n");
#if (PY_VERSION_HEX < 0x03000000)
    python_import << wxT("if not sys.modules.has_key(\"") << python_file << wxT("\"):");
#else
    python_import << wxT("if '") << python_file << wxT("' not in sys.modules:");
#endif
    python_import << wxT("ip.ex(\"import ") << python_file << wxT("\")\n");
    python_import << wxT("else:") << wxT("ip.ex(\"reload(") << python_file << wxT(")") << wxT("\")\n");
    python_import << wxT("sys.path.remove(\"") << python_path << wxT("\")\n");

#else
    // Python code to import a module with PyCrust 
    python_import << wxT("import sys\n");
    python_import << wxT("sys.path.append(\"") << python_path << wxT("\")\n");
#if (PY_VERSION_HEX < 0x03000000)
    python_import << wxT("if not sys.modules.has_key(\"") << python_file << wxT("\"):");
#else
    python_import << wxT("if '") << python_file << wxT("' not in sys.modules:");
#endif
    python_import << wxT("import ") << python_file << wxT("\n");
    python_import << wxT("else:") << wxT("reload(") << python_file << wxT(")") << wxT("\n");
    python_import << wxT("sys.path.remove(\"") << python_path << wxT("\")\n");
    python_import << wxT("del sys\n");

#endif

#if ((wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY)) && !defined(__WXMAC__))
    PyRun_SimpleString(python_import);
#else
    PyRun_SimpleString(python_import.char_str());

    // Release the Global Interpreter Lock
    wxPyEndBlockThreads(blocked);
#endif

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

    wxString python_redirect;
    python_redirect = wxT("import sys, wx\n");
    python_redirect << wxT("output = wx.PyOnDemandOutputWindow()\n");
    python_redirect << wxT("sys.stdin = sys.stderr = output\n");
    python_redirect << wxT("del sys, wx\n");

    wxPyBlock_t blocked = wxPyBeginBlockThreads();
#if ((wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY)) && !defined(__WXMAC__))
    PyRun_SimpleString(python_redirect);
#else
    PyRun_SimpleString(python_redirect.char_str());
#endif
    
    wxPyEndBlockThreads(blocked);
}

new_wxwindow wxStfParentFrame::MakePythonWindow(const std::string& windowFunc, const std::string& mgr_name, const std::string& caption, bool show,
                                                bool full, bool isfloat, int width, int height, double mpl_width, double mpl_height) {
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

    RedirectStdio();

    // Now make a dictionary to serve as the global namespace when the code is
    // executed.  Put a reference to the builtins module in it.  (Yes, the
    // names are supposed to be different, I don't know why...)
    PyObject* globals = PyDict_New();
#if PY_MAJOR_VERSION >= 3
    PyObject* builtins = PyImport_ImportModule("builtins");
#else
    PyObject* builtins = PyImport_ImportModule("__builtin__");
#endif
    PyDict_SetItemString(globals, "__builtins__", builtins);
    Py_DECREF(builtins);

    // Execute the code to make the makeWindow function
    result = PyRun_String(python_code2.c_str(), Py_file_input, globals, globals);
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
    PyObject* func = NULL;
    func = PyDict_GetItemString(globals, windowFunc.c_str());
    if (!PyCallable_Check(func)) {
        PyErr_Print();
        wxGetApp().ErrorMsg(wxT("Couldn't initialize window for the python shell"));
        wxPyEndBlockThreads(blocked);
        return NULL;
    }

    // Now build an argument tuple and call the Python function.  Notice the
    // use of another wxPython API to take a wxWindows object and build a
    // wxPython object that wraps it.
    PyObject* arg = wxPyMake_wxObject(this, false);
    wxASSERT(arg != NULL);
    PyObject* py_mpl_width = PyFloat_FromDouble(mpl_width);
    wxASSERT(py_mpl_width != NULL);
    PyObject* py_mpl_height = PyFloat_FromDouble(mpl_height);
    wxASSERT(py_mpl_height != NULL);
    PyObject* figsize = PyTuple_New(2);
    PyTuple_SET_ITEM(figsize, 0, py_mpl_width);
    PyTuple_SET_ITEM(figsize, 1, py_mpl_height);
    PyObject* argtuple = PyTuple_New(2);
    PyTuple_SET_ITEM(argtuple, 0, arg);
    PyTuple_SET_ITEM(argtuple, 1, figsize);
    result = PyEval_CallObject(func, argtuple);
    Py_DECREF(argtuple);
    Py_DECREF(py_mpl_width);
    Py_DECREF(py_mpl_height);
    Py_DECREF(figsize);

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
#if PY_MAJOR_VERSION >= 3
        if (!wxPyConvertWrappedPtr(result, (void**)&window, _T("wxWindow"))) {
#else
        if (!wxPyConvertSwigPtr(result, (void**)&window, _T("wxWindow"))) {
#endif
            PyErr_Print();
            wxGetApp().ErrorMsg(wxT("Returned object was not a wxWindow!"));
            wxPyEndBlockThreads(blocked);
            return NULL;
        }
        Py_DECREF(result);
    }

    // Release the python objects we still have
    Py_DECREF(globals);

    // Finally, after all Python stuff is done, release the GIL
    wxPyEndBlockThreads(blocked);
    
    if (!full) {
        if (isfloat) {
            m_mgr.AddPane( window, wxAuiPaneInfo().Name(stf::std2wx(mgr_name)).
                           CloseButton(true).
                           Show(show).Caption(stf::std2wx(caption)).Float().
                           BestSize(width, height) );
        } else {
            m_mgr.AddPane( window, wxAuiPaneInfo().Name(stf::std2wx(mgr_name)).
                           CloseButton(true).
                           Show(show).Caption(stf::std2wx(caption)).Dockable(true).Bottom().
                           BestSize(width, height) );
        }
    } else {
        m_mgr.AddPane( window, wxAuiPaneInfo().Name(stf::std2wx(mgr_name)).
                       Floatable(false).CaptionVisible(false).
                       BestSize(GetClientSize().GetWidth(),GetClientSize().GetHeight()).Fixed() );
    }
    m_mgr.Update();
    
    return new_wxwindow(window, result);
}

std::vector<stf::Extension> wxStfApp::LoadExtensions() {
    std::vector< stf::Extension > extList;

    // As always, first grab the GIL
    wxPyBlock_t blocked = wxPyBeginBlockThreads();

    // import extensions.py:
    PyObject* pModule = PyImport_ImportModule("extensions");
    if (!pModule) {
        PyErr_Print();
#ifdef _STFDEBUG
        wxGetApp().ErrorMsg(wxT("Couldn't load extensions.py"));
#endif
        wxPyEndBlockThreads(blocked);
        return extList;
    }

    PyObject* pExtList = PyObject_GetAttrString(pModule, "extensionList");
    if (!pExtList) {
        PyErr_Print();
        wxGetApp().ErrorMsg(wxT("Couldn't find extensionList in extensions.py"));
        wxPyEndBlockThreads(blocked);
        Py_DECREF(pModule);
        return extList;
    }

    if (!PyList_Check(pExtList)) {
        PyErr_Print();
        wxGetApp().ErrorMsg(wxT("extensionList is not a Python list in extensions.py"));
        wxPyEndBlockThreads(blocked);
        Py_DECREF(pExtList);
        Py_DECREF(pModule);
        return extList;
    }

    // retrieve values from list:
    for (int i=0; i<PyList_Size(pExtList); ++i) {
        PyObject* pExt = PyList_GetItem(pExtList, i);
        if (!pExt) {
            PyErr_Print();
            wxString missingStr;
            missingStr << wxT("Could not retrieve item #") << i
                       << wxT(" in extensionList");
            wxGetApp().ErrorMsg(missingStr);
        } else {
            if (!PyObject_HasAttrString(pExt, "menuEntryString") ||
                !PyObject_HasAttrString(pExt, "pyFunc") ||
                !PyObject_HasAttrString(pExt, "description") ||
                !PyObject_HasAttrString(pExt, "requiresFile"))
            {
                wxString attrStr;
                attrStr << wxT("Item #") << i
                        << wxT(" in extensionList misses an attribute");
                wxGetApp().ErrorMsg(attrStr);
            } else {
                PyObject* pMenuEntry = PyObject_GetAttrString(pExt, "menuEntryString");
                PyObject* pPyFunc = PyObject_GetAttrString(pExt, "pyFunc");
                PyObject* pDescription = PyObject_GetAttrString(pExt, "description");
                PyObject* pRequiresFile = PyObject_GetAttrString(pExt, "requiresFile");
                if (!pMenuEntry || !pPyFunc || !pDescription || !pRequiresFile ||
                    !PyString_Check(pMenuEntry) || !PyFunction_Check(pPyFunc) ||
                    !PyCallable_Check(pPyFunc) ||
                    !PyString_Check(pDescription) || !PyBool_Check(pRequiresFile))
                {
                    wxString typeStr;
                    typeStr << wxT("One of the attributes in item #") << i
                            << wxT(" of extensionList misses an attribute");
                    wxGetApp().ErrorMsg(typeStr);

                } else {
                    std::string menuEntry(PyString_AsString(pMenuEntry));
                    std::string description(PyString_AsString(pDescription));
                    bool requiresFile = (pRequiresFile==Py_True);
                    extList.push_back(stf::Extension(menuEntry, (void*)pPyFunc, description, requiresFile));
                }
                Py_XDECREF(pMenuEntry);
                Py_XDECREF(pPyFunc);
                Py_XDECREF(pDescription);
                Py_XDECREF(pRequiresFile);
            }
        }
    }
    
    Py_DECREF(pExtList);
    Py_DECREF(pModule);

    // Finally, after all Python stuff is done, release the GIL
    wxPyEndBlockThreads(blocked);

    return extList;
}

void wxStfApp::OnUserdef(wxCommandEvent& event) {
    int id = event.GetId()-ID_USERDEF;

    if (id >= (int)GetExtensionLib().size() || id<0) {
        wxString msg(wxT("Couldn't find extension function"));
        ErrorMsg( msg );
        return;
    }

    // As always, first grab the GIL
    wxPyBlock_t blocked = wxPyBeginBlockThreads();

    // retrieve function
    PyObject* pPyFunc = (PyObject*)(GetExtensionLib()[id].pyFunc);
    // retrieve function name
    wxString FuncName = stf::std2wx(GetExtensionLib()[id].menuEntry);
    if (!pPyFunc || !PyCallable_Check(pPyFunc)) {
        wxString msg(FuncName << wxT(" Couldn't call extension function "));
        ErrorMsg( msg );
        wxPyEndBlockThreads(blocked);
        return;
    }

    // call function
    PyObject* res = PyObject_CallObject(pPyFunc, NULL);
    if (!res) {
        PyErr_Print();
        wxString msg(FuncName << wxT(" call failed"));
        ErrorMsg( msg );
        wxPyEndBlockThreads(blocked);
        return;
    }

    if (res==Py_False) {
        wxString msg(FuncName << wxT(" returned False"));
        ErrorMsg( msg );
    }
    
    Py_XDECREF(res);

    // Finally, after all Python stuff is done, release the GIL
    wxPyEndBlockThreads(blocked);
    
}

bool wxStfDoc::LoadTDMS(const std::string& filename, Recording& ReturnData) {
    // Grab the Global Interpreter Lock.
    wxPyBlock_t blocked = wxPyBeginBlockThreads();

    PyObject* stf_mod = PyImport_ImportModule("tdms");
    if (!stf_mod) {
        PyErr_Print();
#ifdef _STFDEBUG
        wxGetApp().ErrorMsg(wxT("Couldn't load tdms.py"));
#endif
        wxPyEndBlockThreads(blocked);
        return false;
    }

    PyObject* py_fn = PyString_FromString(filename.c_str());
    PyObject* stf_tdms_f = PyObject_GetAttrString(stf_mod, "tdms_open");
    PyObject* stf_tdms_res = NULL;

    if (PyCallable_Check(stf_tdms_f)) {
        PyObject* stf_tdms_args = PyTuple_Pack(1, py_fn);
        stf_tdms_res = PyObject_CallObject(stf_tdms_f, stf_tdms_args);
        PyErr_Print();
        Py_DECREF(stf_mod);
        Py_DECREF(py_fn);
        Py_DECREF(stf_tdms_args);
    } else {
        Py_DECREF(stf_mod);
        Py_DECREF(py_fn);

        return false;
    }

    if (stf_tdms_res == Py_None) {
        wxGetApp().ErrorMsg( wxT("nptdms module unavailable. Cannot read tdms files."));
        Py_DECREF(stf_tdms_res);
        return false;
    }

    if (!PyTuple_Check(stf_tdms_res)) {
        wxGetApp().ErrorMsg(wxT("Return value of tdms_open is not a tuple. Aborting now."));
        Py_DECREF(stf_tdms_res);
        return false;
    }

    if (PyTuple_Size(stf_tdms_res) != 2) {
        wxGetApp().ErrorMsg( wxT("Return value of tdms_open is not a 2-tuple. Aborting now."));
        Py_DECREF(stf_tdms_res);
        return false;
    }

    PyObject* data_list = PyTuple_GetItem(stf_tdms_res, 0);
    PyObject* py_dt = PyTuple_GetItem(stf_tdms_res, 1);
    double dt = PyFloat_AsDouble(py_dt);
    // Py_DECREF(py_dt);

    Py_ssize_t nchannels = PyList_Size(data_list);
    ReturnData.resize(nchannels);
    int nchannels_nonempty = 0;
    for (int nc=0; nc<nchannels; ++nc) {
        PyObject* section_list = PyList_GetItem(data_list, nc);
        Py_ssize_t nsections = PyList_Size(section_list);
        if (nsections != 0) {
            Channel ch(nsections);
            for (int ns=0; ns<nsections; ++ns) {
                PyObject* np_array = PyList_GetItem(section_list, ns);
                npy_intp* arr_shape = PyArray_DIMS(np_array);
                int nsamples = arr_shape[0];
                Section sec(nsamples);
                double* data = (double*)PyArray_DATA(np_array);
                std::copy(&data[0], &data[nsamples], &sec.get_w()[0]);
                ch.InsertSection(sec, ns);
                // Py_DECREF(np_array);
            }
            ReturnData.InsertChannel(ch, nc);
            nchannels_nonempty++;
        }
        // Py_DECREF(section_list);
    }
    // Py_DECREF(data_list);
    // Py_DECREF(stf_tdms_res);
    ReturnData.resize(nchannels_nonempty);
    ReturnData.SetXScale(dt);
    wxPyEndBlockThreads(blocked);

    return true;
}

#endif // WITH_PYTHON
