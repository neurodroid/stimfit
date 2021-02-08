#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
(
    cd $SCRIPT_DIR/..    # got to top level directory
    mkdir _build
    cd _build
    cmake ..
    make -j$(nproc)
    make install
)
