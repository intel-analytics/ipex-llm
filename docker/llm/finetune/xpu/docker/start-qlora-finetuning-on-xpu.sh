#!/bin/bash
set -x
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
source /opt/intel/oneapi/setvars.sh
export ACCELERATE_USE_IPEX=true
export ACCELERATE_USE_XPU=true
python qlora_finetune_xpu.py
