#!/bin/bash

set -x

sgx_mem_size=$SGX_MEM_SIZE

make SGX=1 GRAPHENEDIR=/graphene THIS_DIR=/ppml/trusted-big-data-ml G_SGX_SIZE=$sgx_mem_size clean
