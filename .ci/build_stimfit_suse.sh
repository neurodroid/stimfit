#!/usr/bin/env bash

sudo zypper install -y cmake gcc-c++ make \
       python3-devel \
       python3-numpy-devel \
       wxWidgets-devel \
       swig hdf5-devel blas-devel lapack-devel \
       fftw3-devel python3-wheel \
       python3-setuptools python3-pip\
       python3-sip-devel python3-wxPython

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
(
    cd $SCRIPT_DIR
    mkdir _opensuse 
    cd _opensuse
    # cmake ../..
    /usr/bin/cmake ../../  -DCMAKE_INSTALL_PREFIX:PATH=/usr \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
    make -j$(nproc)
)
