#!/usr/bin/env bash

set -e
set -x

sudo yum -y install dnf

sudo dnf -y install epel-release 
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --set-enabled PowerTools || echo "Not available on centos7"

sudo dnf -y install cmake3 || sudo dnf -y install cmake 

if [ -f /usr/bin/cmake3 ]; then
    if [ -f /usr/bin/cmake ]; then 
        rm -f /usr/bin/cmake 
    fi
    ln -s /usr/bin/cmake3 /usr/bin/cmake 
fi

sudo dnf -y install gcc-c++ make \
       python3-devel \
       python3-numpy \
       wxWidgets-devel \
       swig \
       hdf5-devel \
       blas-devel lapack-devel \
       fftw3-devel python3-wheel \
       python3-setuptools python3-pip

sudo dnf -y install boost-devel
sudo dnf -y install sip python3-sip-devel || echo "Sip not found"
sudo dnf -y install wxGTK3-devel || echo "wxgtk-devel not found"


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
(
    cd $SCRIPT_DIR
    mkdir -p _centos && cd _centos
    cmake ../../  -DCMAKE_INSTALL_PREFIX:PATH=/usr -DCMAKE_BUILD_TYPE=RelWithDebInfo
    make -j$(nproc)
    sudo make install
)

python3 -c "import stfio; print(dir(stfio))"
