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

import argparse
import time
import torch
from ipex_llm import optimize_model
from transformers import AutoTokenizer

from model import MambaLMHeadModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Mamba model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="state-spaces/mamba-1.4b",
                        help='The huggingface repo id for the Mamba model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--tokenizer-repo-id-or-path', type=str, default="EleutherAI/gpt-neox-20b",
                        help='The huggingface repo id for the Mamba tokenizer to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    tokenizer_path = args.tokenizer_repo_id_or_path

    # Load model
    model = MambaLMHeadModel.from_pretrained(model_path)

    # With only one line to enable IPEX-LLM optimization on model
    model = optimize_model(model, low_bit='asym_int4', modules_to_not_convert=["dt_proj", "x_proj", "out_proj"])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Generate predicted tokens
    with torch.inference_mode():
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
        st = time.time()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        end = time.time()
        output_str = tokenizer.decode(output[0])
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Output', '-'*20)
        print(output_str)
