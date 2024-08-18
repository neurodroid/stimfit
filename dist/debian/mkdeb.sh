#! /bin/bash

VERSION=0.16.4

make dist
mkdir -p ../deb/
rm -rf ../deb/*
cp -v stimfit-${VERSION}.tar.gz ../deb/
cp -v stimfit-${VERSION}.tar.gz ../deb/stimfit_${VERSION}.orig.tar.gz
cd ../deb/
tar -xzf stimfit_${VERSION}.orig.tar.gz
cd stimfit-${VERSION}
cp -rv ../../stimfit/dist/debian ./
# debuild -S -sa
debuild -i -us -uc -S
debuild -i -us -uc -b
# sudo pbuilder build --basetgz /var/cache/pbuilder/base.tgz ../*.dsc
