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

import numpy as np
import stf
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
    intro = '%s, using %s' % (stf.get_versionstring(), version_s)

    return intro
