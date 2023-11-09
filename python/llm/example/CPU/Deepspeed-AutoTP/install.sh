#!/bin/bash
# install torch
pip install torch==2.1.0
# install torchccl
pip install https://intel-extension-for-pytorch.s3.amazonaws.com/torch_ccl/cpu/oneccl_bind_pt-2.1.0%2Bcpu-cp39-cp39-linux_x86_64.whl
# install deepspeed
pip install deepspeed==0.11.1
# exclude intel deepspeed extension, which is only for XPU
pip uninstall intel-extension-for-deepspeed
