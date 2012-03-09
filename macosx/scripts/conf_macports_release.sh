#! /bin/bash

../../configure --with-wx-config=/opt/local/bin/wx-config --disable-dependency-tracking --prefix=/Users/cs/stimfit/build/release/stimfit.app/Contents --disable-shave CPPFLAGS="-DH5_USE_16_API" CFLAGS="" CXXFLAGS="-I/opt/local/include" LDFLAGS="-headerpad_max_install_names -L/opt/local/lib -L/usr/lib" PYTHON=/opt/local/bin/python2.7 CC=/usr/bin/llvm-gcc-4.2 CXX=/usr/bin/llvm-g++-4.2
