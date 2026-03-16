#! /bin/sh

DOC_SOURCE_BRANCH="${DOC_SOURCE_BRANCH:-master}"

git checkout gh-pages
git merge "$DOC_SOURCE_BRANCH"
cd doc/sphinx
make -f Makefile.sphinx html
cd ../..
rsync -av ./doc/sphinx/.build/html/* ./
git add .
git commit -m "Update documentation"     
git push origin gh-pages
git checkout "$DOC_SOURCE_BRANCH"
