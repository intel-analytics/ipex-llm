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

import torch
from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer
import argparse
import time
import numpy as np


torch.nn.Linear.reset_parameters = lambda x: None
seed=42
torch.manual_seed(seed)
np.random.seed(seed)

import json
from itertools import zip_longest

prompt_list = [
    'Hello, my name is',
    'The president of the United States is',
    'The capital of France is',
    'The future of AI is',
    'Where is China?'
]
 
# grouped = list(zip_longest(*(iter(prompt_list),) * 4, fillvalue=None))
# new_prompt_list = [list(group) for group in grouped]
new_prompt_list = [prompt_list[:i+2] for i in range(len(prompt_list))]

# new_prompt_list = [[
#     'Hello, my name is',
#     'The president of the United States is',
#     # 'The capital of France is',
#     # 'The future of AI is',
#     # 'Where is China?',
#     # 'Hello, my name is',
# ]]
if __name__ == '__main__':
    model_path = "/mnt/disk1/models/Llama-2-7b-chat-hf/"
    n_predict = 128

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 optimize_model=True,
                                                #  torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True, 
                                                 load_in_low_bit="sym_int4",
                                                #  speculative=False,
                                                 trust_remote_code=True,
                                                 use_cache=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding=True)
    # if tokenizer.pad_token is None:
    # tokenizer.pad_token = tokenizer.eos_token

    # actual_in_len = []
    # for prompt in prompt_list:
    #     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    #     actual_in_len.append(input_ids.size(1))

    # print('actual_in_len: ', actual_in_len)

    with torch.inference_mode():
        for prompts in new_prompt_list:
        # prompts = prompt_list
            while prompts[-1] is None:
                prompts = prompts[:-1]
            inputs = tokenizer(prompts, return_tensors='pt', padding=True)
            input_ids = inputs.input_ids.to(model.device)
            # print(input_ids.shape)
            attention_mask = inputs.attention_mask.to(model.device)
            # position_ids = attention_mask.long().cumsum(-1) - 1

            # input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
            actual_in_len = input_ids.shape[1]
            # print("actual input_ids length:" + str(actual_in_len))

            output = model.generate(input_ids,
                                    max_new_tokens=n_predict,
                                    min_new_tokens=n_predict,
                                    attention_mask=attention_mask,
                                    # position_ids=position_ids,
                                    do_sample=False)

            for i in range(input_ids.size(0)):
                output_str = tokenizer.decode(output[i].int(), skip_special_tokens=False)
                print(i,": ", output_str)