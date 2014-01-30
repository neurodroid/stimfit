#! /bin/bash

WX_CONFIG=/opt/local/bin/wx-config
WXPY_DIR=/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages

mkdir -p stimfit.app
mkdir -p stimfit.app/Contents
mkdir -p stimfit.app/Contents/Frameworks
mkdir -p stimfit.app/Contents/MacOS
mkdir -p stimfit.app/Contents/libs

cp -v -r ${WXPY_DIR}/wx* ./stimfit.app/Contents/Frameworks

chown -R ${USER}:staff stimfit.app

make stimfit.app
cp -v .libs/stimfit  ./stimfit.app/Contents/MacOS/stimfit
chmod +x ./stimfit.app/Contents/MacOS/stimfit
mkdir -p ./stimfit.app/Contents/Frameworks/stimfit
mkdir -p ./stimfit.app/Contents/lib/stimfit
cp -v ./src/stimfit/py/.libs/libpystf.dylib ./stimfit.app/Contents/lib/stimfit/libpystf.dylib
cp -v ./src/stimfit/.libs/libstimfit.dylib ./stimfit.app/Contents/lib/stimfit/libstimfit.dylib
cp -v ./src/libstfio/.libs/libstfio.dylib ./stimfit.app/Contents/lib/stimfit/libstfio.dylib
rm -fv ./stimfit.app/Contents/Frameworks/stimfit/_stf.so
rm -fv ./stimfit.app/Contents/libs/libpystf.dylib
rm -fv ./stimfit.app/Contents/libs/libstimfit.dylib
# cp -v ./src/app/.libs/libstimfit.dylib ./stimfit.app/Contents/libs/libstimfit.dylib
dylibbundler -of -b -x ./stimfit.app/Contents/MacOS/stimfit -d ./stimfit.app/Contents/libs/
CURDIR=`pwd`
cd stimfit.app/Contents/Frameworks/stimfit
ln -sf ../../libs/libpystf.dylib _stf.so
cd ${CURDIR}

if test -n "$1"; then
  if [ $1 = '1' ]; then
    find ./stimfit.app  -name "*.dylib" -exec dylibbundler -of -b -x '{}' -d ./stimfit.app/Contents/libs/ \;
    find ./stimfit.app  -name "*.so" -exec dylibbundler -of -b -x '{}' -d ./stimfit.app/Contents/libs/ \;
  fi
fi
rm -rfv ./stimfit.app/Contents/lib


cp -v ../../src/stimfit/py/*.py ./stimfit.app/Contents/Frameworks/stimfit/
cp -v ../../src/pystfio/*.py ./stimfit.app/Contents/Frameworks/stimfit/

if test -n "$1"; then
  if [ $1 = '1' ]; then
    rsync -av ./stimfit.app /Applications/
  fi
fi
