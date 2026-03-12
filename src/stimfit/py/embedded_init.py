#=========================================================================
# embedded_init.py
# 2009.12.31
# 
# This file loads both Numpy and stf modules into the current namespace.
# Additionally, it loads the custom initialization script (stf_init.py)
#
# 2010.06.12
# Major stf classes were added (Recording, Channel, Section)
#
# It is used by embedded_stf.py and embedded_ipython.py 
# Please, do not modify this file unless you know what you are doing
#
#=========================================================================

import importlib
import os
import sys

import numpy as np
import wx


_stf_module = None
_stf_import_error = None


def _load_stf_module():
    """Load the stf module once and cache the result."""
    global _stf_module
    global _stf_import_error

    if _stf_module is not None:
        return _stf_module

    if _stf_import_error is not None:
        return None

    try:
        _stf_module = importlib.import_module("stf")
    except ImportError as exc:
        _stf_import_error = exc
        print("Stimfit embedded shell: unable to import 'stf': %s" % (exc,), file=sys.stderr)
        return None

    return _stf_module


class _LazyStfProxy:
    """Proxy that imports stf on first attribute access."""

    def __getattr__(self, name):
        module = _load_stf_module()
        if module is None:
            raise AttributeError("stf module is unavailable")
        return getattr(module, name)


# Embedded shell startup should stay resilient: avoid eager SWIG loading there.
if os.environ.get("STF_EMBEDDED_SHELL") == "1":
    stf = _LazyStfProxy()
else:
    stf = _load_stf_module()
    if stf is not None:
        from stf import *

from os.path import basename

try:
    from stf_init import *
except ImportError:
    # let the user know  stf_init does not work!
    pass
except SyntaxError:
    pass
else:
    pass

def intro_msg():
    """ this is the starting message of the embedded Python shell.
    Contains the current Stimfit version, together with the NumPy
    and wxPython version.
    """
    # access current versions of wxWidgets and NumPy
    from wx import version as wx_version
    from numpy.version import version as numpy_version


    version_s = 'NumPy %s, wxPython %s' % (numpy_version, wx_version())

    # Do not force importing stf during embedded shell startup, as this can
    # trigger duplicate wx RTTI registration in mixed build/install layouts.
    if _stf_module is not None:
        intro = '%s, using %s' % (_stf_module.get_versionstring(), version_s)
    else:
        intro = 'Stimfit Python shell, using %s' % (version_s,)

    return intro
