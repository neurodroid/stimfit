#!/usr/bin/env bash

set -x
set -e

sudo apt install -y \
    cmake \
    python3-dev \
    python3-numpy \
    python3-setuptools \
    python3-pip \
    python3-wheel \
    libhdf5-dev \
    swig \
    python3-wxgtk4.0 \
    libwxgtk3.0-gtk3-dev \
    liblapack-dev \
    wx-common \
    fftw3-dev \
    cmake \
    python3-sip-dev


python3 -m pip install -U \
    -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04 \
    wxPython 

# Missing libraries after wxPython install.
# https://stackoverflow.com/a/59277031/1805129 
sudo apt install -y \
    curl libsdl2-mixer-2.0-0 libsdl2-image-2.0-0 libsdl2-2.0-0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
(
    cd $SCRIPT_DIR
    mkdir _ubuntu && cd _ubuntu
    cmake ../../  -DCMAKE_INSTALL_PREFIX:PATH=/usr \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
    make -j$(nproc)
    sudo make install
)

python3 -c "import stfio; print(stfio.__version__)"
