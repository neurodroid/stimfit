#! /bin/sh

result=`python -c "import sys, string; ver = string.split(sys.version)[0]; print (ver >= '2.5.0' and ver < '2.6.0')"`

if [ $result="True" ]
then
    exit 0
else
    exit 1
fi