#! /bin/sh

rsync -av ./.build/html/ $1@schmidt-hieber.de:/kunden/homepages/32/d34288459/htdocs/StimfitJ/doc/sphinx
