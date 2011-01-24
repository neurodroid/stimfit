#! /bin/bash

make dist
mkdir -p ../deb/
cp -v stimfit-0.10.6.tar.gz ../deb/
cp -v stimfit-0.10.6.tar.gz ../deb/stimfit_0.10.5.orig.tar.gz
cd ../deb/
tar -xzf stimfit_0.10.6.orig.tar.gz
cd stimfit-0.10.6
mkdir -p debian
cp -rv ../../../dist/debian/* ./debian
debuild -S -sa
sudo pbuilder build ../*.dsc
