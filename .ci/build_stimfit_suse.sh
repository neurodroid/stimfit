#!/usr/bin/env bash

set -x
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# These are required for building python module.
sudo zypper install -y cmake gcc-c++ make \
         zlib-devel \
         python3-devel \
         python3-numpy-devel \
         swig hdf5-devel blas-devel lapack-devel \
         fftw3-devel python3-wheel \
         python3-setuptools python3-pip\
         python3-sip-devel

# These are optional and are required for the GUI.
sudo zypper install -y wxWidgets-devel python3-wxPython  || echo "GUI won't build"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
(
    cd $SCRIPT_DIR
    mkdir -p _opensuse && cd _opensuse
    cmake ../..  -DCMAKE_INSTALL_PREFIX:PATH=/usr -DCMAKE_BUILD_TYPE=RelWithDebInfo
    make -j$(nproc)
    sudo make install
)

python3 -c "import stfio; print(dir(stfio))"
