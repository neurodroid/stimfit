#!/usr/bin/env python

# Relatively generic setup.py that should be easily tailorable to
# other python modules.  It gets most of the parameters from the
# packaged module itself, so this file shouldn't have to be changed
# much.

import os,sys,distutils
from distutils.core import setup

# patch distutils if it can't cope with the "classifiers" or
# "download_url" keywords
if sys.version < '2.2.3':
    from distutils.dist import DistributionMetadata
    DistributionMetadata.classifiers = None
    DistributionMetadata.download_url = None

# ignore a warning about large int constants in Python 2.3.*
if sys.version >= '2.3' and sys.version < '2.4':
    import warnings
    warnings.filterwarnings('ignore', "hex/oct constants", FutureWarning)

# python looks in current directory first, so we'll be sure to get our
# freshly unpacked version here.
import pyemf

module=pyemf

module_list=[] # list extra modules here
module_list.append(module.__name__)

# grab the first paragraph out of the module's docstring to use as the
# long description.
long_description = ''
lines = module.__doc__.splitlines()
for firstline in range(len(lines)):
    # skip leading blank lines
    if len(lines[firstline])==0 or lines[firstline].isspace():
        pass
    else:
        break
for lastline in range(firstline,len(lines)):
    # stop when we reach a blank line
    if len(lines[lastline])==0 or lines[lastline].isspace():
        break
long_description = os.linesep.join(lines[firstline:lastline])
#print long_description

#print "Version = %s" % str(module.__version__)

setup(name = module.__name__,
      version = str(module.__version__),
      description = module.__description__,
      long_description = long_description,
      keywords = module.__keywords__,
      license = module.__license__,
      author = module.__author__,
      author_email = module.__author_email__,
      url = module.__url__,
      download_url = module.__download_url__,
      platforms='any',
      py_modules = module_list,
      
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Topic :: Multimedia :: Graphics',
                   ]
      )

