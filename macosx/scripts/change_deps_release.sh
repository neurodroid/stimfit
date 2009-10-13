#! /bin/bash

WX_CONFIG=/Users/cs/wxbin/bin/wx-config
WXPY_DIR=/Users/cs/wxPython-2.9
WXPY_VER=wx-2.9.0-osx_cocoa-unicode

make stimfit.app

sudo cp ./src/stfswig/.libs/libstf.0.dylib /usr/local/lib
sudo rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx*.dylib /usr/local/lib/
dylibbundler -of -b -x ./stimfit.app/Contents/MacOS/stimfit -d ./stimfit.app/Contents/libs/
sudo rm /usr/local/lib/libstf.0.dylib

ln -sf ./../../libs/libstf.0.dylib ./stimfit.app/Contents/Frameworks/stimfit/_stf.so

cp -v ../../src/stfswig/*.py ./stimfit.app/Contents/Frameworks/stimfit/

if test -n "$1"; then
  if [ $1 = '1' ]; then
    rm -rf ./stimfit.app/Contents/Frameworks/numpy*
    cp -R /System/Library/Frameworks/Python.framework//Versions/2.6/Extras/lib/python/numpy* ./stimfit.app/Contents/Frameworks/
    cp -v ${WXPY_DIR}/wx/*.so ./stimfit.app/Contents/Frameworks/${WXPY_VER}/wx
    find ./stimfit.app/Contents/Frameworks/${WXPY_VER}/wx -name "*.so" -exec dylibbundler -of -b -x '{}' -d ./stimfit.app/Contents/libs/ \;
  fi
fi

sudo rm /usr/local/lib/libwx*.dylib
rm ./stimfit.app/Contents/libs/libwx*
rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx_baseu_net-* ./stimfit.app/Contents/libs/
rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx_baseu-* ./stimfit.app/Contents/libs/
rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx_osx_cocoau_adv* ./stimfit.app/Contents/libs/
rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx_osx_cocoau_aui* ./stimfit.app/Contents/libs/
rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx_osx_cocoau_core* ./stimfit.app/Contents/libs/
