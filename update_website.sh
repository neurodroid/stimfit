#! /bin/bash

rm -rf /media/webftp/StimfitJ/doc/sphinx/*
rsync -rtuvz ./doc/sphinx/.build/html/ /media/webftp/StimfitJ/doc/sphinx/
