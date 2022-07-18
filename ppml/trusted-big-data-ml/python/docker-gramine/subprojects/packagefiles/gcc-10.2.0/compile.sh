#!/bin/sh

set -e

log() {
    echo "gcc (libgomp): $*"
}

CPU_FAMILY="$1"
CURRENT_SOURCE_DIR="$2"
CURRENT_BUILD_DIR="$3"
PRIVATE_DIR="$4"
PREFIX="$5"
LIBDIR="$6"
shift 6

# We explicitly unset CC, CXX, AS environment variables for the case we're compiling the rest of the
# project with clang. This is because in GCC build, "defining certain environment variables such as
# CC can interfere with the functioning of make" (https://gcc.gnu.org/install/build.html). Indeed,
# defining CC=gcc or CC=clang leads to errors during GCC build.
unset CC CXX AS

BUILD_LOG=$(realpath "$CURRENT_BUILD_DIR/gcc-build.log")
rm -f "$BUILD_LOG"

log "see $BUILD_LOG for full build log"

log "preparing sources..."

rm -rf "$PRIVATE_DIR"
cp -ar "$CURRENT_SOURCE_DIR" "$PRIVATE_DIR"

for patch in "$CURRENT_SOURCE_DIR"/*.patch
do
    patch --quiet -p1 --directory "$PRIVATE_DIR" <"$patch"
done

# Copy `gramine_entry_api.h` into (patched) GCC sources. We cannot pass the path using CPPFLAGS,
# because it will not get passed to the libgomp build.
cp "$CURRENT_SOURCE_DIR/../../LibOS/shim/include/arch/$CPU_FAMILY/gramine_entry_api.h" \
    "$PRIVATE_DIR/libgomp/config/linux/x86/"

BUILDDIR="$PRIVATE_DIR"/build
mkdir -p "$BUILDDIR"

(
    cd "$BUILDDIR"

    # We pass `--disable-bootstrap` to disable the 3-stage build process. That means GCC (as well as
    # `libgomp`, which is what we actually care about) will be built using the system compiler, not
    # the bootstrapped GCC, but the whole process will be much faster.
    log "running configure..."
    ../configure \
        --prefix="$PREFIX" \
        --libdir="$PREFIX"/"$LIBDIR"/gramine/runtime/glibc \
        --enable-languages=c \
        --disable-multilib \
        --disable-bootstrap \
        >>"$BUILD_LOG" 2>&1

    log "running make..."
    make -j"$(nproc)" >>"$BUILD_LOG" 2>&1
)

for output in "$@"
do
    cp -aP "$BUILDDIR/$output" "$CURRENT_BUILD_DIR"/
done

log "done"
