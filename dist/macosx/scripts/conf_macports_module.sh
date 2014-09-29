#! /bin/bash

prefix="/opt/local"
PYVER=3.4

MYCC=/usr/bin/clang
MYCXX=/usr/bin/clang++
MYLD=ld

config_args="--disable-dependency-tracking \
             --enable-module \
             --with-biosig"

../../configure ${config_args} CC="${MYCC} -I${prefix}/include" CXX="${MYCXX} -I${prefix}/include" LD="${MYLD}" LDFLAGS="-headerpad_max_install_names -L${prefix}/lib" PYTHON="${prefix}/bin/python${PYVER}"
