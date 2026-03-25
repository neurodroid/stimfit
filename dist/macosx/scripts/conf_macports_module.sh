#! /bin/bash

prefix="/opt/local"
PYVER=2.7

MYCC=/usr/bin/clang
MYCXX=/usr/bin/clang++
MYLD=ld

cmake_args="-S ../.. \
            -B build-macports-module \
            -DSTF_BUILD_MODULE=ON \
            -DSTF_WITH_BIOSIG=ON \
            -DSTF_BIOSIG_PROVIDER=SUBMODULE \
            -DCMAKE_C_COMPILER=${MYCC} \
            -DCMAKE_CXX_COMPILER=${MYCXX} \
            -DCMAKE_LINKER=${MYLD} \
            -DCMAKE_C_FLAGS=-I${prefix}/include \
            -DCMAKE_CXX_FLAGS=-I${prefix}/include \
            -DCMAKE_EXE_LINKER_FLAGS=-headerpad_max_install_names\ -L${prefix}/lib"

cmake ${cmake_args} -DPython3_EXECUTABLE="${prefix}/bin/python${PYVER}"
