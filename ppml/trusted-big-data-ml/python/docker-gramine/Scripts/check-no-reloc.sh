#!/bin/sh

set -e
test $# -eq 2 || exit 2

relocs=$(LC_ALL=C readelf -r "$1")
case "$relocs" in
*'There are no relocations in this file.'*)
    cp -a "$1" "$2"
    ;;
*)
    printf %s\\n "$relocs"
    exit 1
    ;;
esac
