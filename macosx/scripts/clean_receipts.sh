#! /bin/sh

ls /Library/Receipts/astimfit* > /dev/null
if [ $? -eq 0 ]; then
    sudo rm /Library/Receipts/stimfit*
fi
 