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

basekit_root=/opt/intel/oneapi
source $basekit_root/setvars.sh --force
source $basekit_root/ccl/latest/env/vars.sh --force

NUM_GPUS=2 # number of used GPU
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0 # Different from PVC

mpirun -n $NUM_GPUS \
    python ./alpaca_lora_finetuning.py \
       --base_model "THUDM/chatglm3-6b" \
       --data_path "yahma/alpaca-cleaned" \
       --output_dir "./ipex-llm-lora-alpaca" \
       --gradient_checkpointing True \
       --lora_target_modules "['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h']" \
       --micro_batch_size 1 \
       --batch_size 2 \
       --save_checkpoint False \
       --deepspeed_zero3 True > lora_deepspeed_zero3_finetune_chatglm3_6b.log

