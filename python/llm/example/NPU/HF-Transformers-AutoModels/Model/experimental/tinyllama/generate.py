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

from ipex_llm.transformers.npu_model import AutoModelForCausalLM
from transformers import AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for npu model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="D:\llm-models\TinyLlama-1.1B-Chat-v1.0",
                        help='The huggingface repo id for the tinyllama model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="Once upon a time, there is a little girl named Lily who lives in a small village.",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                 npu_backend="openvino")

    with torch.inference_mode():
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
        print("finish to load")
        print('input length:', len(input_ids[0]))
        st = time.time()
        output = model.generate(input_ids, do_sample=False, max_new_tokens=args.n_predict)
        end = time.time()
        print(f'Inference time: {end-st} s')
        output_str = tokenizer.decode(output[0], skip_special_tokens=False)
        print('-'*20, 'Output', '-'*20)
        print(output_str)

    print('-'*80)
    print('done')
