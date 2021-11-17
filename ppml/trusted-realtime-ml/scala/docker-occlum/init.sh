#!/bin/bash

set -x

sgx_mem_size=$SGX_MEM_SIZE

make SGX=1 OCCLUMDIR=/graphene THIS_DIR=/ppml/trusted-realtime-ml O_SGX_SIZE=$sgx_mem_size
