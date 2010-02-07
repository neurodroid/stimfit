#! /bin/bash

sudo chown -R cs:staff stimfit.app

WX_CONFIG=/Users/cs/wxbin/bin/wx-config
WXPY_DIR=/Users/cs/wxPython-2.9
WXPY_VER=wx-2.9.0-osx_cocoa-unicode
WXPY_INSTALL_DIR=/Users/cs/wxPython-2.9/dummy-install/lib/python2.6/site-packages
make stimfit.app
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
rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx*.dylib ./stimfit.app/Contents/libs/
sudo cp -v ./src/stfswig/.libs/libstf.0.dylib /usr/local/lib/libstf.0.dylib
rm -f ./stimfit.app/Contents/Frameworks/stimfit/_stf.so
cp -v ./src/stfswig/.libs/libstf.0.dylib ./stimfit.app/Contents/Frameworks/stimfit/_stf.so
# ln -sf ./stimfit.app/Contents/Frameworks/stimfit/_stf.so ./stimfit.app/Contents/libs/libstf.0.dylib
rm -f ./stimfit.app/Contents/libs/libstf.0.dylib
dylibbundler -of -b -x ./stimfit.app/Contents/MacOS/stimfit -d ./stimfit.app/Contents/libs/
##

# sudo mv ./stimfit.app/Contents/libs/libstf.0.dylib /usr/local/lib/_stf.so
# ln -sf /usr/local/lib/_stf.so ./stimfit.app/Contents/libs/libstf.0.dylib

##
find ./stimfit.app  -name "*.dylib" -exec dylibbundler -of -b -x '{}' -d ./stimfit.app/Contents/libs/ \;
find ./stimfit.app  -name "*.so" -exec dylibbundler -of -b -x '{}' -d ./stimfit.app/Contents/libs/ \;
sudo rm /usr/local/lib/*stf*
##

# cp -v ../../src/stfswig/*.py ./stimfit.app/Contents/Frameworks/stimfit/


# # sudo rm /usr/local/lib/libwx*.dylib
# # sudo chown -R root:admin stimfit.app 

# # rm ./stimfit.app/Contents/libs/libwx*
# # rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx_baseu_net-* ./stimfit.app/Contents/libs/
# # rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx_baseu-* ./stimfit.app/Contents/libs/
# # rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx_osx_cocoau_adv* ./stimfit.app/Contents/libs/
# # rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx_osx_cocoau_aui* ./stimfit.app/Contents/libs/
# # rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx_osx_cocoau_core* ./stimfit.app/Contents/libs/
