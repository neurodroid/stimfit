#! /bin/bash

WX_CONFIG=/Users/cs/wxbin/bin/wx-config
# WXPY_DIR=/Users/cs/wxPython-src-2.9.1.1/wxPython
# WXPY_VER=wx-2.9.1-osx_cocoa
# WXPY_INSTALL_DIR=/Users/cs/wxPython-2.9/dummy-install/lib/python2.5/site-packages

mkdir -p stimfit.app
mkdir -p stimfit.app/Contents
mkdir -p stimfit.app/Contents/Frameworks
mkdir -p stimfit.app/Contents/MacOS
mkdir -p stimfit.app/Contents/libs

chown -R cs:staff stimfit.app

make stimfit.app
cp -v .libs/stimfit  ./stimfit.app/Contents/MacOS/stimfit
chmod +x ./stimfit.app/Contents/MacOS/stimfit
mkdir -p ./stimfit.app/Contents/Frameworks/stimfit

# if test -n "$1"; then
#   if [ $1 = '1' ]; then
#     rm -rf ./stimfit.app/Contents/Frameworks/numpy*
#     cp -R /System/Library/Frameworks/Python.framework//Versions/2.6/Extras/lib/python/numpy* ./stimfit.app/Contents/Frameworks/
#     rm -rf ./stimfit.app/Contents/Frameworks/wx*
#     cp -R ${WXPY_INSTALL_DIR}/wx* ./stimfit.app/Contents/Frameworks/
#     mkdir -p ./stimfit.app/Contents/Frameworks/${WXPY_VER}
#     mkdir -p ./stimfit.app/Contents/Frameworks/${WXPY_VER}/wx
#     rsync -l ${WXPY_INSTALL_DIR}/${WXPY_VER}/wx/*.so ./stimfit.app/Contents/Frameworks/${WXPY_VER}/wx
#     find ./stimfit.app -name "*.so" -exec dylibbundler -of -b -x '{}' -d ./stimfit.app/Contents/libs/ \;
#     find ./stimfit.app -name "*.pyc" -exec rm '{}' \;
#   fi
# fi

##
# rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx*.dylib ./stimfit.app/Contents/libs/
mkdir -p ./stimfit.app/Contents/lib/stimfit
cp -v ./src/stfswig/.libs/libstf.dylib ./stimfit.app/Contents/lib/stimfit/libstf.dylib
cp -v ./src/app/.libs/libstimfit.dylib ./stimfit.app/Contents/lib/stimfit/libstimfit.dylib
rm -fv ./stimfit.app/Contents/Frameworks/stimfit/_stf.so
rm -fv ./stimfit.app/Contents/libs/libstf.dylib
rm -fv ./stimfit.app/Contents/libs/libstimfit.dylib
# cp -v ./src/app/.libs/libstimfit.dylib ./stimfit.app/Contents/libs/libstimfit.dylib
dylibbundler -of -b -x ./stimfit.app/Contents/MacOS/stimfit -d ./stimfit.app/Contents/libs/
CURDIR=`pwd`
cd stimfit.app/Contents/Frameworks/stimfit
ln -sf ../../libs/libstf.dylib _stf.so
cd ${CURDIR}

if test -n "$1"; then
  if [ $1 = '1' ]; then
    find ./stimfit.app  -name "*.dylib" -exec dylibbundler -of -b -x '{}' -d ./stimfit.app/Contents/libs/ \;
    find ./stimfit.app  -name "*.so" -exec dylibbundler -of -b -x '{}' -d ./stimfit.app/Contents/libs/ \;
  fi
fi
rm -rfv ./stimfit.app/Contents/lib


cp -v ../../src/stfswig/*.py ./stimfit.app/Contents/Frameworks/stimfit/

# # rm ./stimfit.app/Contents/libs/libwx*
# # rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx_baseu_net-* ./stimfit.app/Contents/libs/
# # rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx_baseu-* ./stimfit.app/Contents/libs/
# # rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx_osx_cocoau_adv* ./stimfit.app/Contents/libs/
# # rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx_osx_cocoau_aui* ./stimfit.app/Contents/libs/
# # rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx_osx_cocoau_core* ./stimfit.app/Contents/libs/
