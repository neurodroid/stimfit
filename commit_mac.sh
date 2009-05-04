#! /bin/bash
REMOTE=https://christoph.schmidt-hieber:Xe9Vr2Jf2Xd9@stimfit.googlecode.com/hg
echo hg addremove
hg addremove
echo hg ci
hg ci
echo hg --debug push $REMOTE
hg --debug push $REMOTE
