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

export ZE_AFFINITY_MASK="0,1" # specify the used GPU
NUM_GPUS=2 # number of used GPU
export MASTER_ADDR=127.0.0.1
export FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so:${LD_PRELOAD}
basekit_root=/opt/intel/oneapi
source $basekit_root/setvars.sh --force
# source $basekit_root/ccl/latest/env/vars.sh --force   deprecate oneccl_bind_pt and use internal oneccl for better performance
source /opt/intel/1ccl-wks/setvars.sh

export OMP_NUM_THREADS=$((56/$NUM_GPUS))
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=1
export BIGDL_IMPORT_IPEX=0
mpirun -np $NUM_GPUS --prepend-rank \
    python deepspeed_autotp.py --repo-id-or-model-path 'meta-llama/Llama-2-70b-chat-hf' --low-bit 'sym_int4'
