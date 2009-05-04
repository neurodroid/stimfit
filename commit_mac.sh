#! /bin/bash
REMOTE=http://code.google.com/p/stimfit/source/checkout/
echo hg addremove
hg addremove
echo hg ci
hg ci
echo sudo hg push $REMOTE
sudo hg push $REMOTE
