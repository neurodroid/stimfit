#! /bin/bash

echo find ./ -type f  \( -name "stfconf.h" -o -name "index.xml" -o -name "configure.in" -o -name "conf.py" -o -name "installer.nsi" -o -name "Home.html" -o -name "stimfit.plist.in" -o -name "mkdeb.sh" -o -name "mkimage.sh" -o -name "stimfit.1" \) -exec sed -i 's/'$1'/'$2'/' {} \;
find ./ -type f  \( -name "stfconf.h" -o -name "index.xml" -o -name "configure.in" -o -name "conf.py" -o -name "installer.nsi" -o -name "Home.html" -o -name "stimfit.plist.in" -o -name "mkdeb.sh" -o -name "mkquick.sh" -o -name "mkimage.sh" -o -name "stimfit.1" \) -exec sed -i 's/'$1'/'$2'/' {} \;
