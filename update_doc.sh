#! /bin/sh

git checkout gh-pages
git merge master
cd doc/sphinx
make -f Makefile.sphinx html
cd ../..
rsync -av ./doc/sphinx/.build/html/* ./
git add .
git commit -m "Update documentation"     
git push origin gh-pages
git checkout master
