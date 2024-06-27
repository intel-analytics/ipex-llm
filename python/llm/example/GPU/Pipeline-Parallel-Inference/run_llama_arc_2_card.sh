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

source /opt/intel/oneapi/setvars.sh
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=9090
export FI_PROVIDER=tcp
export USE_XETLA=OFF
export OMP_NUM_THREADS=6
export IPEX_LLM_QUANTIZE_KV_CACHE=1
if [[ $KERNEL_VERSION != *"6.5"* ]]; then
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
fi
export TORCH_LLM_ALLREDUCE=0

NUM_GPUS=2 # number of used GPU

# To run Llama-2-7b-chat-hf
CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node $NUM_GPUS \
    generate.py --repo-id-or-model-path 'meta-llama/Llama-2-7b-chat-hf' --gpu-num $NUM_GPUS --low-bit 'sym_int4'

# # To run Llama-2-13b-chat-hf
# CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node $NUM_GPUS \
#     generate.py --repo-id-or-model-path 'meta-llama/Llama-2-13b-chat-hf' --gpu-num $NUM_GPUS --low-bit 'sym_int4'

# # To run Meta-Llama-3-8B-Instruct
# CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node $NUM_GPUS \
#     generate.py --repo-id-or-model-path 'meta-llama/Meta-Llama-3-8B-Instruct' --gpu-num $NUM_GPUS --low-bit 'sym_int4'
