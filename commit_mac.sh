#! /bin/bash
REMOTE=https://christsc\@gmx.de\@stimfit.googlecode.com/hg/
echo hg addremove
hg addremove
echo hg ci
hg ci
echo sudo hg push $REMOTE
sudo hg push $REMOTE
