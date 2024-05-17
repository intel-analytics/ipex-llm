#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

## Validated BKC for Qwen1.5-14B-Chat on 2 ARC with
## Ubuntu 22.04.4, kernel 6.5.0-27-generic, level-zero 1.14.0, NEO(compute runtime) 24.09.28717.12

export MASTER_ADDR=127.0.0.1
export FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so:${LD_PRELOAD}
basekit_root=/opt/intel/oneapi
source $basekit_root/setvars.sh --force
source $basekit_root/ccl/latest/env/vars.sh --force

NUM_GPUS=2 # number of used GPU
export USE_XETLA=OFF
if grep -q "Core" /proc/cpuinfo; then
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
fi
export TORCH_LLM_ALLREDUCE=0 # Different from PVC

mpirun -np $NUM_GPUS --prepend-rank \
    python deepspeed_autotp.py --repo-id-or-model-path 'Qwen/Qwen1.5-14B-Chat' --low-bit 'sym_int4'
