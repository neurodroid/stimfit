#!/usr/bin/env bash
set -e
set -x

PYTHON=$(which python3)

brew install wxpython
brew install wxwidgets
brew install cmake 
brew install lapack blas fftw

mkdir _osx && cd _osx
cmake ..
make -j$(nproc)
make install 

$(PYTHON) -c "import stfio"
