#! /bin/bash

REMOTE=https://christoph.schmidthieber@stimfit.googlecode.com/hg

# make sure we have the latest revision before commiting:
echo hg -v pull -u $REMOTE
hg -v pull -u $REMOTE

echo hg -v addremove
hg -v addremove

echo hg -v ci
hg -v ci

echo hg -v push $REMOTE
hg -v push $REMOTE
