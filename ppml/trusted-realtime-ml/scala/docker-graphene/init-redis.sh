#!/bin/bash

sgx_mem_size=$SGX_MEM_SIZE

make clean
make SGX=1 -j`nproc` GRAPHENEDIR=/graphene G_SGX_SIZE=$sgx_mem_size
