#! /bin/bash

../../configure --enable-python --enable-debug --with-wx-config=/Users/cs/wxWidgets/bld/wx-config CXXFLAGS='-I/opt/local/include -I/Users/cs/wxWidgets/wxPython/include' LDFLAGS='-headerpad_max_install_names -L/opt/local/lib -L/Users/cs/wxWidgets/bld/lib'
