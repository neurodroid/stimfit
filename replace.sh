#! /bin/bash

find ./ -type f  \( -name "*.*" \) -exec sed -i 's/'linux'/'linux'/' {} \;
