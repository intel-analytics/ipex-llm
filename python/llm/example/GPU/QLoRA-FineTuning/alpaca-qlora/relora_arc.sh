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
source /home/arda/intel/oneapi/setvars.sh
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 ENABLE_SDP_FUSION=1

export PYTHONPATH=/home/arda/yina/BigDL/python/llm/src

python ./alpaca_qlora_finetuning.py \
    --base_model "/mnt/disk1/models/Llama-2-7b-chat-hf" \
    --data_path "/mnt/disk1/data/alpaca-cleaned" \
    --output_dir "./bigdl-relora-alpaca" \
    --relora_steps 300 \
    --relora_warmup_steps 10