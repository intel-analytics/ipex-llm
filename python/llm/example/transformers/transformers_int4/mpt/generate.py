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

from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer INT4 example for MPT model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="mosaicml/mpt-7b-chat",
                        help='The huggingface repo id for the large language model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="<human>What is AI?<bot>",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # load model in 4 bit
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 load_in_4bit=True)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    
    # Generate predict tokens
    with torch.inference_mode():
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
        st = time.time()
        output = model.generate(input_ids,
                                do_sample=False,
                                use_cache=True,
                                max_new_tokens=args.n_predict)
        end = time.time()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print(f'Prompt:\n{args.prompt}')
        print(f'Output:\n{output_str}')
