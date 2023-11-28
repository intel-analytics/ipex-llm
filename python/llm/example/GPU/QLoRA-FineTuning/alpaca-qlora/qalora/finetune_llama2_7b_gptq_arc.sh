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
source /opt/intel/oneapi/setvars.sh
export PYTHONPATH=/home/arda/yina/BigDL/python/llm/src

python ./qalora.py \
    --model_path "/mnt/disk1/models/Llama-2-7B-GPTQ" \
    --dataset "alpaca-clean" \
    --output_dir "./bigdl-qalora-alpaca" \
    --learning_rate 0.0002 \
    --bf16 False \
    --gradient_accumulation_steps 4 \
    --optim "adamw_torch" \
    # --do_eval True \
    # --do_train False \
    # --lora_r 64 \
    # --lora_alpha 16 \
    # --lora_dropout 0.0 \
    # --batch_size 32 \
    # --micro_batch_size 2 \
    
    