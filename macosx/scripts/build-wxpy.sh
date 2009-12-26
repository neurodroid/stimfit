#! /bin/bash

/usr/bin/python setup.py build_ext --inplace WXPORT=osx_cocoa WX_CONFIG=~/wxbin/bin/wx-config
# /usr/bin/python setup.py build_ext --inplace --force  ARCH=x86_64 WXPORT=osx_cocoa WX_CONFIG=~/wxbin/bin/wx-config
