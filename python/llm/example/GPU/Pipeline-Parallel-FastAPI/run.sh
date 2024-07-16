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

export no_proxy=localhost
export FI_PROVIDER=tcp
export OMP_NUM_THREADS=32

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
basekit_root=/opt/intel/oneapi
source $basekit_root/setvars.sh --force
source $basekit_root/ccl/latest/env/vars.sh --force

export USE_XETLA=OFF
if [[ $KERNEL_VERSION != *"6.5"* ]]; then
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
fi
export TORCH_LLM_ALLREDUCE=0
export IPEX_LLM_QUANTIZE_KV_CACHE=1
export IPEX_LLM_LAST_LM_HEAD=1
export IPEX_LLM_LOW_MEM=1

export MODEL_PATH=YOUR_MODEL_PATH
export NUM_GPUS=2
export ZE_AFFINITY_MASK=0,1
export LOW_BIT="fp8"
# max requests = max_num_reqs * rank_num
export MAX_NUM_SEQS="4"
export MAX_PREFILLED_SEQS=0

if [[ $NUM_GPUS -eq 1 ]]; then
    export ZE_AFFINITY_MASK=0
    python serving.py --repo-id-or-model-path $MODEL_PATH --low-bit $LOW_BIT
else
    CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node $NUM_GPUS pipeline_serving.py --repo-id-or-model-path $MODEL_PATH --low-bit $LOW_BIT --max-num-seqs $MAX_NUM_SEQS --max-prefilled-seqs $MAX_PREFILLED_SEQS
fi
