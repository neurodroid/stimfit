# -*- coding: utf-8 -*-
'''
Python module to read common electrophysiology file formats.
'''

from .stfio import *
from . import stfio_plot as plot
try:
    from . import stfio_neo as neo
except ImportError:
    pass
