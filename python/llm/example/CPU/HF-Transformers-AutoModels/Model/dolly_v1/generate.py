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
import time
import argparse
import numpy as np

from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/databricks/dolly-v1-6b#generate-text
DOLLY_V1_PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Dolly v1 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="databricks/dolly-v1-6b",
                        help='The huggingface repo id for the Dolly v1 model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_4bit=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Generate predicted tokens
    with torch.inference_mode():
        prompt = DOLLY_V1_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        end_key_token_id=tokenizer.encode("### End")[0]
        st = time.time()
        # enabling `use_cache=True` allows the model to utilize the previous
        # key/values attentions to speed up decoding;
        # to obtain optimal performance with IPEX-LLM INT4 optimizations,
        # it is important to set use_cache=True for Dolly v1 models
        output = model.generate(input_ids,
                                use_cache=True,
                                max_new_tokens=args.n_predict,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=end_key_token_id)
        end = time.time()
        end_token_position = None
        end_token_positions = np.where(output[0] == end_key_token_id)[0]
        if len(end_token_positions) > 0:
            end_token_position = end_token_positions[0]
        output_str = tokenizer.decode(output[0][:end_token_position], skip_special_tokens=False)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)
