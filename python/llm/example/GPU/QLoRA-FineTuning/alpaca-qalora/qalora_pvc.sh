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
source /home/wangruonan/intel/oneapi/setvars.sh --force
# export ONEAPI_DEVICE_SELECTOR=level_zero:6
# sycl-ls
export PYTHONPATH=/home/wangruonan/yina/BigDL/python/llm/src

python ./alpaca_qalora_finetuning.py \
    --base_model "/home/wangruonan/models/Llama-2-13b-chat-hf" \
    --data_path "/home/wangruonan/yina/alpaca-cleaned" \
    --output_dir "./bigdl-qalora-alpaca-2" \
    --learning_rate 9e-5 \
    --micro_batch_size 8 \
    --batch_size 128 \
    --gradient_checkpointing False \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --val_set_size 2000