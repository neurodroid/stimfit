#! /bin/bash
echo hg add
hg addremove
echo hg ci
hg ci
echo sudo hg push /media/dendrite/Christoph/stimfit
sudo hg push /media/dendrite/Christoph/stimfit
