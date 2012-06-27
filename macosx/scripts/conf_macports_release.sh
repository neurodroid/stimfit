#! /bin/bash

prefix="/opt/local"

MYCC=`${prefix}/bin/wx-config --cc`
MYCXX=`${prefix}/bin/wx-config --cxx`
MYLD=`${prefix}/bin/wx-config --ld`

config_args="--with-wx-config=${prefix}/bin/wx-config \
                    --disable-dependency-tracking \
                    --disable-shave"
                    
../../configure ${config_args} CC="${MYCC} -I${prefix}/include" CXX="${MYCXX} -I${prefix}/include" LD="${MYLD}" LDFLAGS="-headerpad_max_install_names -L${prefix}/lib"
