#! /bin/bash

WX_CONFIG=/Users/cs/wxWidgets/bld/wx-config
WXPY_DIR=/Users/cs/wxWidgets/wxPython

make stimfit.app

sudo cp ./src/stfswig/.libs/libstf.0.dylib /usr/local/lib
sudo cp `${WX_CONFIG} --exec-prefix`/lib/libwx*.dylib /usr/local/lib
dylibbundler -of -b -x ./stimfit.app/Contents/MacOS/stimfit -d ./stimfit.app/Contents/libs/
sudo rm /usr/local/lib/libstf.0.dylib

ln -sf ./../../libs/libstf.0.dylib ./stimfit.app/Contents/Frameworks/stimfit/_stf.so

cp -v ../../src/stfswig/*.py ./stimfit.app/Contents/Frameworks/stimfit/

if test -n "$1"; then
  if [ $1 = '1' ]; then
    cp -v ${WXPY_DIR}/wx/*.so ./stimfit.app/Contents/Frameworks/wx-2.8-mac-unicode/wx
    find ./stimfit.app/Contents/Frameworks/wx-2.8-mac-unicode/wx -name "*.so" -exec dylibbundler -of -b -x '{}' -d ./stimfit.app/Contents/libs/ \;
  fi
fi

sudo rm /usr/local/lib/libwx*.dylib
