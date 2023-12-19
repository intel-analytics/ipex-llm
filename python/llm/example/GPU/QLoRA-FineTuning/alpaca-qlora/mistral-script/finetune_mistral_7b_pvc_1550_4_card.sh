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
export OMP_NUM_THREADS=7  # adjust this to 1/4 of total physical cores
export FI_PROVIDER=tcp

mpirun -n 8 \
       python -u ./alpaca_qlora_finetuning.py \
       --base_model "mistralai/Mistral-7B-v0.1" \
       --data_path "yahma/alpaca-cleaned" \
       --output_dir "./bigdl-qlora-alpaca" \
       --lora_target_modules "['k_proj', 'q_proj', 'o_proj', 'v_proj']" > training.log
