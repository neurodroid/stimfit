#! /bin/bash
REMOTE=https://christsc\@gmx.de:Xe9Vr2Jf2Xd9\@stimfit.googlecode.com/hg/stimfit/
echo hg addremove
hg addremove
echo hg ci
hg ci
echo hg push $REMOTE
hg push $REMOTE
