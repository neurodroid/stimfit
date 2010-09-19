#! /bin/bash

hdiutil create stimfit-{$1}.dmg -srcfolder ./stimfit.app -ov -format UDBZ
