#!/bin/sh

set -e

log() {
    echo "musl: $*"
}

CPU_FAMILY="$1"
CURRENT_SOURCE_DIR="$2"
CURRENT_BUILD_DIR="$3"
PRIVATE_DIR="$4"
PREFIX="$5"
LIBDIR="$6"
DEBUG="$7"
shift 7

CFLAGS="\
    -I$(realpath "$CURRENT_SOURCE_DIR")/../../LibOS/shim/include \
    -I$(realpath "$CURRENT_SOURCE_DIR")/../../LibOS/shim/include/arch/$CPU_FAMILY \
"
export CFLAGS

BUILD_LOG=$(realpath "$CURRENT_BUILD_DIR/musl-build.log")
rm -f "$BUILD_LOG"

log "see $BUILD_LOG for full build log"

log "preparing sources..."

rm -rf "$PRIVATE_DIR"
cp -ar "$CURRENT_SOURCE_DIR" "$PRIVATE_DIR"

for patch in "$CURRENT_SOURCE_DIR"/*.patch
do
    patch --quiet -p1 --directory "$PRIVATE_DIR" <"$patch"
done

BUILDDIR="$PRIVATE_DIR"/build

if [ "$DEBUG" = "1" ]
then
    MUSL_CONFIGURE_FLAGS="--enable-debug --disable-optimize"
else
    MUSL_CONFIGURE_FLAGS="--enable-optimize"
fi

mkdir -p "$BUILDDIR"

(
    cd "$BUILDDIR"

    log "running configure..."
    ../configure \
        --prefix="$PREFIX" \
        --libdir="$PREFIX"/"$LIBDIR"/gramine/runtime/musl \
        --disable-static \
        $MUSL_CONFIGURE_FLAGS \
        >>"$BUILD_LOG" 2>&1

    log "running make..."
    make -j"$(nproc)" >>"$BUILD_LOG" 2>&1
)

for output in "$@"
do
    cp -aP "$BUILDDIR/$output" "$CURRENT_BUILD_DIR"/
done

log "done"
