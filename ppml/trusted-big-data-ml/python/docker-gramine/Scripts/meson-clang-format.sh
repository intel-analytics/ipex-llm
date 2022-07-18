#!/bin/sh

set -e

if test x"$1" = x-d
then
    # debug mode: just print, and developer is responsible for running in source root
    FINDACTION="-print"
else
    test -n "$MESON_SOURCE_ROOT"
    cd "$MESON_SOURCE_ROOT"
    FINDACTION="-exec clang-format -i {} +"
fi

find Pal LibOS tools \
    -path common/src/crypto/mbedtls -prune -o \
    -path Pal/src/host/Linux-SGX/tools/common/cJSON.c -prune -o \
    -path Pal/src/host/Linux-SGX/tools/common/cJSON.h -prune -o \
    -path Pal/src/host/Linux-SGX/tools/common/cJSON-\*/cJSON.c -prune -o \
    -path Pal/src/host/Linux-SGX/tools/common/cJSON-\*/cJSON.h -prune -o \
    -path LibOS/shim/test/ltp -prune -o \
    -path LibOS/glibc\* -prune -o \
    \( -name \*.c -o -name \*.h \) \
    $FINDACTION
