#!/usr/bin/env bash
set -e
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PATH=/usr/local/bin:$PATH
export PATH

PYTHON=$(which python3)

brew install wxpython
brew install wxwidgets
brew install swig
brew install numpy || brew upgrade numpy
brew install cmake  || brew upgrade cmake
brew install lapack  || brew upgrade lapack
brew install fftw || brew upgrade fftw

$PYTHON -m pip install numpy --user

(
    cd $SCRIPT_DIR
    mkdir -p _osx && cd _osx
    cmake ../..
    make -j$(nproc)
    make install 
)

$(PYTHON) -c "import stfio;print(dir(stfio))"
