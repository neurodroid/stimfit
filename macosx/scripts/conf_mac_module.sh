#! /bin/bash

arch_flags=""
py_dir="/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/"
../../configure --enable-module --disable-dependency-tracking --prefix=${py_dir}/stfio CPPFLAGS="-DH5_USE_16_API" CFLAGS="" CXXFLAGS="-I/opt/local/include" LDFLAGS="-headerpad_max_install_names -L/opt/local/lib -L/usr/lib" CC=/usr/bin/clang CXX=/usr/bin/clang PYTHON_SITE_PKG=${py_dir}
