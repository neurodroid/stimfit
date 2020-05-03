#!/usr/bin/env bash
set -e
set -x

PYTHON=$(which python3)

brew install wxpython
brew install wxwidgets
brew install cmake  || echo "already installed?"
brew install lapack  || echo "Failed to install lapack"
brew install blas || echo "Failed to install blas"
btew install fftw || echo "failed to install fftw"

mkdir _osx && cd _osx
cmake ..
make -j$(nproc)
make install 

$(PYTHON) -c "import stfio"
