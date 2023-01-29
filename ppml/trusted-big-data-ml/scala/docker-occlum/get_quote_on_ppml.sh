#!/bin/bash
set -e

BLUE='\033[1;34m'
NC='\033[0m'

echo "*** Build and run glibc dcap demo ***"
CC=gcc
LD=ld
LIBPATH="/opt/occlum/toolchains/dcap_lib/glibc"
INCPATH="/opt/occlum/toolchains/dcap_lib/inc"

CC=$CC LD=$LD LIBPATH=$LIBPATH make -C c_app clean
CC=$CC LD=$LD LIBPATH=$LIBPATH INCPATH=$INCPATH make -C c_app
