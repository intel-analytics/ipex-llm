#!/bin/bash
# 1. install oneccl for intel mpi
# can skip this step if oneccl/oneapi is already installed on your machine
# report to https://github.com/oneapi-src/oneCCL if any issue
git clone https://github.com/oneapi-src/oneCCL.git
cd oneCCL
mkdir build
cd build
cmake ..
make -j install
mkdir -p /opt/intel/oneccl
mv ./_install/env /opt/intel/oneccl
# 2. install torch and ipex
pip install torch==2.1.0
pip install intel_extension_for_pytorch==2.1.0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
# install torchccl (oneccl binding for pytorch)
pip install https://intel-extension-for-pytorch.s3.amazonaws.com/torch_ccl/cpu/oneccl_bind_pt-2.1.0%2Bcpu-cp39-cp39-linux_x86_64.whl
# 3. install deepspeed
pip install deepspeed==0.11.1
# 4. exclude intel deepspeed extension, which is only for XPU
pip uninstall intel-extension-for-deepspeed
# 5. install ipex-llm
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
