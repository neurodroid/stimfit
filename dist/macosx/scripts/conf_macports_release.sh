#! /bin/bash

prefix="/opt/local"
# WX_CONF="${prefix}/Library/Frameworks/wxWidgets.framework/Versions/wxWidgets/3.1/bin/wx-config"
WX_CONF="/Users/cs/Phoenix/build/wxbld/wx-config"

MYCC=`${WX_CONF} --cc`
MYCFLAGS=`${WX_CONF} --cflags`
MYCFLAGS="${MYCFLAGS} -I${prefix}/include"
MYCXX=`${WX_CONF} --cxx`
MYCXXFLAGS=`${WX_CONF} --cxxflags`
MYCXXFLAGS="${MYCXXFLAGS} -I${prefix}/include -I/Users/cs/Phoenix/wx/include/wxPython"
MYLD=`${WX_CONF} --ld`
MYLDFLAGS=`${WX_CONF} --libs core base aui`
MYLDFLAGS="${MYLDFLAGS} -headerpad_max_install_names -L${prefix}/lib"
config_args="--with-wx-config=${WX_CONF} \
             --disable-dependency-tracking \
             --with-biosiglite"
#              --with-lapack-lib=/opt/local/lib/libatlas.a \
#              --enable-debug \

../../configure ${config_args} CC="${MYCC}" CFLAGS="${MYCFLAGS}" CXX="${MYCXX}" CXXFLAGS="${MYCXXFLAGS}" LD="${MYLD}" LDFLAGS="${MYLDFLAGS}" PYTHON="${prefix}/bin/python3.8" PYTHONPATH="/Users/cs/Phoenix"
