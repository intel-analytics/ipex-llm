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
export OMP_NUM_THREADS=56
export FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=ofi

mpirun -n 2 \
       python -u ./alpaca_qalora_finetuning.py \
       --base_model "meta-llama/Llama-2-7b-hf" \
       --data_path "yahma/alpaca-cleaned" \
       --output_dir "./ipex-llm-qlora-alpaca" \
       --learning_rate 9e-5 \
       --micro_batch_size 8 \
       --batch_size 128 \
       --lora_r 8 \
       --lora_alpha 16 \
       --lora_dropout 0.05 \
       --val_set_size 2000 > training.log
