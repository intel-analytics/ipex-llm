#!/bin/sh

# Usage:
#
#    install_musl_tests destdir musl_a musl_b musl_c...
#
# Installs `musl_*` binaries to a target directory, removing the `musl_` prefix.

set -e

DEST_DIR="$1"
shift 1

mkdir -p ${DEST_DIR}

for file in "$@"
do
    bn=`basename "${file}"`
    cp "${file}" "${DEST_DIR}/${bn#musl_}"
done
