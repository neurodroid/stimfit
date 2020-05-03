#!/usr/bin/env bash
set -e
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PYTHON=$(which python3)

brew install wxpython
brew install wxwidgets
brew install swig
brew install cmake  || brew update cmake
brew install lapack  || brew update lapack
brew install fftw || brew update fftw

(
    cd $SCRIPT_DIR
    mkdir -p _osx && cd _osx
    cmake ../..
    make -j$(nproc)
    make install 
)

$(PYTHON) -c "import stfio"
