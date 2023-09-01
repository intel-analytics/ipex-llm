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
from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity
import psutil
import os
benchmark_util_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
import sys
sys.path.append(benchmark_util_path)
from benchmark_util import BenchmarkWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer INT4 example')
    parser.add_argument('--repo-id', type=str, required=True,
                        help='The huggingface repo id for the large language model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--local-model-hub', type=str,
                        help='the local caching hub dir.')
    parser.add_argument('--prompt-len', type=int, default=32,
                        help='prompt length to infer')
    parser.add_argument('--infer-times', type=int, default=3,
                        help='inference times for tests')
    parser.add_argument('--max-tokens', type=int, default=32,
                        help='max tokens to generate.')
    args = parser.parse_args()

    max_tokens = args.max_tokens
    repo_id = args.repo_id
    if args.local_model_hub:
        repo_model_name = repo_id.split("/")[1]
        model_path = args.local_model_hub + "/" + repo_model_name
    else:
        model_path = repo_id
    infer_times = args.infer_times

    st = time.time()
    # TODO: auto detect using config.json
    if repo_id in ['THUDM/chatglm-6b', 'THUDM/chatglm2-6b']:
        model = AutoModel.from_pretrained(model_path, load_in_4bit=True, trust_remote_code=True, torch_dtype='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True)
        end = time.time()
        print(">> loading of model costs {}s".format(end-st))
        tokenizer = LlamaTokenizer.from_pretrained(model_path)

    end = time.time()
    print(">> loading and conversion of model costs {}s".format(end-st))

    model = BenchmarkWrapper(model)

    input_str = open(f"prompt/{args.prompt_len}.txt", 'r').read()

    with torch.inference_mode():
        for i in range(infer_times):
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = input_ids[:, :args.prompt_len]
            print("input length is : ", input_ids.shape[1])
            st = time.perf_counter()
            output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=max_tokens)
            end = time.perf_counter()
            print("model generate cost: " + str(end - st))
            output = tokenizer.batch_decode(output_ids)
            true_input = tokenizer.batch_decode(input_ids)
            print(true_input[0] + output[0])
