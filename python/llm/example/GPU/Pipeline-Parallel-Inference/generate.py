
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

from ipex_llm.transformers import AutoModel, AutoModelForCausalLM, init_pipeline_parallel
from transformers import AutoTokenizer

init_pipeline_parallel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-13b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    parser.add_argument('--low-bit', type=str, default='sym_int4', help='The quantization type the model will convert to.')
    parser.add_argument('--gpu-num', type=int, default=2, help='GPU number to use')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    low_bit = args.low_bit

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     load_in_low_bit=low_bit,
                                                     optimize_model=True,
                                                     trust_remote_code=True,
                                                     use_cache=True,
                                                     torch_dtype=torch.float16,
                                                     pipeline_parallel_stages=args.gpu_num)
    except:
        model = AutoModel.from_pretrained(model_path,
                                          load_in_low_bit=low_bit,
                                          optimize_model=True,
                                          trust_remote_code=True,
                                          use_cache=True,
                                          pipeline_parallel_stages=args.gpu_num)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    local_rank = torch.distributed.get_rank()

    # Generate predicted tokens
    with torch.inference_mode():
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(f'xpu:{local_rank}')
        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)

        # start inference
        st = time.time()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        torch.xpu.synchronize()
        end = time.time()
        output = output.cpu()
        if local_rank == args.gpu_num - 1:
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f'Inference time: {end-st} s')
            print(f"First token cost {model.first_token_time:.4f} s and rest tokens cost average {model.rest_cost_mean:.4f} s")
            print('-'*20, 'Prompt', '-'*20)
            print(args.prompt)
            print('-'*20, 'Output', '-'*20)
            print(output_str)

