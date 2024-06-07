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

# You could also specify `--base_model` to the local path of the huggingface model checkpoint folder and `--data_path` to the local path of the dataset JSON file
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=56
export FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets

mpirun -n 2 \
       python -u ./alpaca_qlora_finetuning.py \
       --base_model "THUDM/chatglm3-6b" \
       --data_path "yahma/alpaca-cleaned" \
       --output_dir "./ipex-llm-lora-alpaca" \
       --gradient_checkpointing True \
       --micro_batch_size 1 \
       --batch_size 128 \
       --deepspeed ./deepspeed_zero2.json \
       --saved_low_bit_model ./llama-2-7b-hf > training.log
