#! /bin/bash
REMOTE=https://christsc\@gmx.de\@stimfit.googlecode.com/hg/stimfit/
echo hg addremove
hg addremove
echo hg ci
hg ci
echo hg push $REMOTE
hg push $REMOTE
