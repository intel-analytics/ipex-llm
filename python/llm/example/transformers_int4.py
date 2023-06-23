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
import os
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer

if __name__ == '__main__':
    model_path = 'decapoda-research/llama-7b-hf'

    # load_in_4bit=True in bigdl.llm.transformers will convert
    # the relevant layers in the model into int4 format
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    input_str = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

    with torch.inference_mode():
        input_ids = tokenizer.encode(input_str, return_tensors="pt")
        output = model.generate(input_ids, do_sample=False, max_new_tokens=32)
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_str)
