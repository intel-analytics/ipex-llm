#!/bin/sh

CFLAGS='-O2 -DMBEDTLS_CONFIG_FILE=\"mbedtls/config-pal.h\"'

# TODO: We disable the stack protector because the default configuration doesn't work inside
# Gramine. Instead, we should pass the right stack protector options from Meson.
CFLAGS="$CFLAGS -fno-stack-protector"

# Generate position-independent code even for a static library, so that it can be used in PAL and
# LibOS
CFLAGS="$CFLAGS -fPIC"

# Don't assume existence of builtins (currently Clang emits references to `bcmp`)
CFLAGS="$CFLAGS -fno-builtin"

export CFLAGS
exec "$(dirname "$0")"/compile.sh "$@"
