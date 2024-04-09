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

# save Llama-2-70b-hf model with ipex-llm low-bit optimization first
python save_low_bit_70b_model.py --output_path "./llama-2-70b-hf-nf4"

export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=56
export FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets

mpirun -n 8 \
       python -u ./alpaca_qlora_finetuning.py \
       --base_model "meta-llama/Llama-2-70b-hf" \
       --data_path "yahma/alpaca-cleaned" \
       --output_dir "./ipex-llm-qlora-alpaca" \
       --gradient_checkpointing True \
       --micro_batch_size 8 \
       --batch_size 128 \
       --deepspeed ./deepspeed_zero2.json \
       --saved_low_bit_model ./llama-2-70b-hf-nf4 > training.log

