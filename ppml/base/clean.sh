#!/bin/bash

set -x

sgx_mem_size=$SGX_MEM_SIZE

make SGX=1 THIS_DIR=/ppml G_SGX_SIZE=$sgx_mem_size clean
