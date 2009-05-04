#! /bin/bash
REMOTE=https://christoph.schmidthieber@stimfit.googlecode.com/hg
echo hg addremove
hg addremove
echo hg ci
hg ci
echo hg --debug push $REMOTE
hg --debug push $REMOTE
