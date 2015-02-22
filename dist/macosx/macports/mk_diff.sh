#! /bin/sh

CURDIR=`pwd`
cd ~/macports
svn update
cd $CURDIR

declare -a arr=("python/py-stfio" "science/stimfit")

for TARGET in "${arr[@]}"
do
    mkdir -p tmp/a
    mkdir -p tmp/b
    cp ~/macports/dports/$TARGET/Portfile ./tmp/a/
    gsed -i '2s/.*/# $Id$/' ./tmp/a/Portfile
    cp $TARGET/Portfile ./tmp/b/
    cd tmp
    PORT=`echo $TARGET | cut -d'/' -f 2`
    diff -ur a b > ../Portfile-$PORT.diff
    cd ..
    rm -r tmp
done
