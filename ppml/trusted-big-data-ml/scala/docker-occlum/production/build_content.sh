#!/bin/bash
set -e

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd )"
export DEP_LIBS_DIR="${script_dir}/dep_libs"
export INITRA_DIR="${script_dir}/init_ra"
export RATLS_DIR="${script_dir}/../../ra_tls"

function build_ra_tls()
{
    rm -rf ${DEP_LIBS_DIR} && mkdir ${DEP_LIBS_DIR}
    pushd ${RATLS_DIR}
    bash ./download_and_prepare.sh
    bash ./build_and_install.sh musl
    bash ./build_occlum_instance.sh musl

    cp ./grpc-src/examples/cpp/ratls/build/libgrpc_ratls_client.so ${DEP_LIBS_DIR}/
    cp ./grpc-src/examples/cpp/ratls/build/libhw_grpc_proto.so ${DEP_LIBS_DIR}/

    popd
}


function build_init_ra()
{
    pushd ${INITRA_DIR}
    sed -i "s#localhost:50051#${ATTESTATION_SERVER_IP}:${ATTESTATION_SERVER_PORT}#g" src/main.rs
    occlum-cargo clean
    occlum-cargo build --release
    popd
}

arg=$1
case "$arg" in
    build_init_ra)
        build_init_ra
        ;;
    build_ra_tls)
        build_ra_tls
        ;;
esac
