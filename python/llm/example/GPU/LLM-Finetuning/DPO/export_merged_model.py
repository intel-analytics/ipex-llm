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
import os

import torch
from transformers import AutoTokenizer
import argparse

current_dir = os.path.dirname(os.path.realpath(__file__))
common_util_path = os.path.join(current_dir, '..')
import sys
sys.path.append(common_util_path)
from common.utils import merge_adapter

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Merge the adapter into the original model for Mistral model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="teknium/OpenHermes-2.5-Mistral-7B",
                        help='The huggingface repo id the Mistral (e.g. `teknium/OpenHermes-2.5-Mistral-7B`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--adapter_path', type=str,)
    parser.add_argument('--output_path', type=str,)

    args = parser.parse_args()
    base_model = model_path = args.repo_id_or_model_path
    adapter_path = args.adapter_path
    output_path = args.output_path
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    merge_adapter(base_model, tokenizer, adapter_path, output_path)
    print(f'Finish to merge the adapter into the original model and you could find the merged model in {output_path}.')
