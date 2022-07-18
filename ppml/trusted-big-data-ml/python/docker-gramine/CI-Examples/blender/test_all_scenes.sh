#!/bin/sh

set -e

if test -n "$SGX"
then
    GRAMINE=gramine-sgx
else
    GRAMINE=gramine-direct
fi

for i in `ls "$DATA_DIR"/scenes/`;
do
    rm -f "$DATA_DIR"/images/"$i"0001.png
    $GRAMINE "../blender" -b /data/scenes/$i -t 4 -F PNG -o /data/images/$i -f 1
    # TODO add a better test, probably some diff with a precomputed image
    [ -f "$DATA_DIR"/images/"$i"0001.png ]
done
