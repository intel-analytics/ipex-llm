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
python ./alpaca_qlora_finetuning.py \
    --base_model "mistralai/Mixtral-8x7B-v0.1" \
    --data_path "yahma/alpaca-cleaned" \
    --output_dir "./bigdl-qlora-alpaca" \
    --micro_batch_size 1 \
    --batch_size 8 \
    --num_epochs 1 \
    --gradient_checkpointing True \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --cutoff_len 256 \
    --learning_rate 0.0002 \
    --lora_target_modules "['k_proj', 'q_proj', 'o_proj', 'v_proj', 'w1', 'w2', 'w3', 'gate']"
