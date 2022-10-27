#!/bin/bash
set -e

BLUE='\033[1;34m'
NC='\033[0m'

if [[ $1 == "musl" ]]; then
    echo "*** Build and run musl-libc dcap demo ***"
    CC=occlum-gcc
    LD=occlum-ld
    LIBPATH="/opt/occlum/toolchains/dcap_lib/musl"
else
    echo "*** Build and run glibc dcap demo ***"
    CC=gcc
    LD=ld
    LIBPATH="/opt/occlum/toolchains/dcap_lib/glibc"
fi

INCPATH="/opt/occlum/toolchains/dcap_lib/inc"

CC=$CC LD=$LD LIBPATH=$LIBPATH make -C c_app clean
CC=$CC LD=$LD LIBPATH=$LIBPATH INCPATH=$INCPATH make -C c_app
