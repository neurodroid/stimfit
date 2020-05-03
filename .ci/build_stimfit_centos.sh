#!/usr/bin/env bash

set -e
set -x

sudo yum -y install dnf
sudo dnf -y install epel-release 
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --set-enabled PowerTools
sudo dnf -y install cmake gcc-c++ make \
       python3-devel \
       python3-numpy \
       wxWidgets-devel \
       swig \
       hdf5-devel \
       blas-devel lapack-devel \
       fftw3-devel python3-wheel \
       python3-setuptools python3-pip\
       sip python3-sip-devel \
       wxGTK3-devel


# blas-devel lapack-devel python3-sip-devel hdf5-devel \

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
(
    cd $SCRIPT_DIR
    mkdir -p _centos && cd _centos
    cmake ../../  -DCMAKE_INSTALL_PREFIX:PATH=/usr -DCMAKE_BUILD_TYPE=RelWithDebInfo
    make -j$(nproc)
)
