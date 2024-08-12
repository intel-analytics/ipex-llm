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

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29503
export FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=0
basekit_root=/opt/intel/oneapi
source $basekit_root/setvars.sh --force
source $basekit_root/ccl/latest/env/vars.sh --force

NUM_GPUS=2 # number of used GPU
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0 # Different from PVC
export DS_SKIP_CUDA_CHECK=1

mpirun -n $NUM_GPUS \
          python -u ./alpaca_qlora_finetuning.py \
          --base_model "meta-llama/Llama-2-13b-hf" \
          --data_path "yahma/alpaca-cleaned" \
          --output_dir "./ipex-llm-qlora-alpaca" \
          --gradient_checkpointing True \
          --micro_batch_size 2 \
          --batch_size 32 \
          --deepspeed ./deepspeed_zero3.json
