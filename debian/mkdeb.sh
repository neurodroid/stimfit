#! /bin/bash

VERSION=0.10.14

make dist
mkdir -p ../deb/
rm -rf ../deb/*
cp -v stimfit-${VERSION}.tar.gz ../deb/
cp -v stimfit-${VERSION}.tar.gz ../deb/stimfit_${VERSION}.orig.tar.gz
cd ../deb/
tar -xzf stimfit_${VERSION}.orig.tar.gz
cd stimfit-${VERSION}
mkdir -p debian
cp -rv ../../../debian/* ./debian
debuild -S -sa
sudo pbuilder build ../*.dsc
