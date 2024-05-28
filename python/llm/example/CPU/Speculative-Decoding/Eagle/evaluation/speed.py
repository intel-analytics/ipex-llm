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
# This file is adapted from
# 
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/evaluation/speed.py
#
# Copyright 2024 SafeAI Lab (SAIL). 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import json
from transformers import AutoTokenizer
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Application to calculate and compare speeds between base and optimized llm models.')
    parser.add_argument("--base-model-path", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="1")
    parser.add_argument("--jsonl-file", type=str, required=True, help='Enter path of model result file')

    args = parser.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(args.base_model_path)

    data = []
    with open(args.jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    
    total_time=0
    total_token=0
    speeds=[]
    for datapoint in data:
        qid=datapoint["question_id"]
        answer=datapoint["choices"][0]['turns']
        tokens = 0
        for i in answer:
            tokens += (len(tokenizer(i).input_ids) - 1)
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds.append(tokens / times)
        total_time+=times
        total_token+=tokens

    print(f'total time: {total_time} s')
    print('total token',total_token)
    print(f'speed: {np.array(speeds).mean()} tps')

