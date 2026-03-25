#! /bin/sh

CURDIR=`pwd`
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-master}"
cd ~/macports/dports
git pull origin "$UPSTREAM_BRANCH"
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
