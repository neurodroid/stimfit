#! /bin/bash

REMOTE=https://christoph.schmidthieber@stimfit.googlecode.com/hg

# make sure we have the latest revision before commiting:
echo hg pull -u $REMOTE
hg pull -u $REMOTE

echo hg addremove
hg addremove

echo hg ci
hg ci

echo hg --debug push $REMOTE
hg --debug push $REMOTE
