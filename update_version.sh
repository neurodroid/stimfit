#! /bin/bash

#! /bin/bash
GSED=`which gsed`
if [ "${GSED}" = "" ]
then
    GSED=`which sed`
fi

CMD="find ./ -type f  \( -name \"stfconf.h\" -o -name \"index.xml\" -o -name \"configure.ac\" -o -name \"conf.py\" -o -name \"installer.nsi\" -o -name \"Home.html\" -o -name \"stimfit.plist.in\" -o -name \"mkdeb.sh\" -o -name \"update-deb.sh\" -o -name \"mkquick.sh\" -o -name \"mkimage.sh\" -o -name \"stimfit.1\" -o -name \"Doxyfile\" -o -name insert_checksums.sh -o -name setup.py -o -path \"*/py-stfio*/meta.yaml\" \) -exec $GSED -i 's/'$1'/'$2'/' {} \;"
echo ${CMD}
${CMD}
