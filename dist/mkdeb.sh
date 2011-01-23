#! /bin/bash

make dist
cp stimfit-0.10.5.tar.gz ../deb/
cp stimfit-0.10.5.tar.gz ../deb/stimfit_0.10.5.orig.tar.gz
cd ../deb/
tar -xzf stimfit_0.10.5.orig.tar.gz
cd stimfit-0.10.5
debuild -S
sudo pbuilder build ../*.dsc
