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
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import argparse
import time
import numpy as np


torch.nn.Linear.reset_parameters = lambda x: None
seed=42
torch.manual_seed(seed)
np.random.seed(seed)

ZIYA_PROMPT_FORMAT = "<human>: \n{prompt}\n<bot>: \n"
prompt = "写一段快速排序"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Mistral model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="IDEA-CCNL/Ziya-Coding-34B-v1.0",
                        help='The huggingface repo id for the Ziya (e.g. `IDEA-CCNL/Ziya-Coding-34B-v1.0`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default=prompt,
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=128,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model in optimized bf16 here.
    # Set `speculative=True`` to enable speculative decoding,
    # it only works when load_in_low_bit="fp16" on Intel GPU or load_in_low_bit="bf16" on latest Intel Xeon CPU
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 optimize_model=True,
                                                 torch_dtype=torch.bfloat16,
                                                 load_in_low_bit="bf16",
                                                 speculative=True,
                                                 trust_remote_code=True,
                                                 use_cache=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with torch.inference_mode():
        prompt = ZIYA_PROMPT_FORMAT.format(prompt=args.prompt)
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs.input_ids.to(model.device)
        actual_in_len = input_ids.shape[1]
        print("actual input_ids length:" + str(actual_in_len))
        attention_mask = inputs.attention_mask.to(model.device)

        # warmup
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                attention_mask=attention_mask,
                                do_sample=False)
        output_str = tokenizer.decode(output[0])

        # speculative decoding
        st = time.perf_counter()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                attention_mask=attention_mask,
                                do_sample=False)
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        end = time.perf_counter()

        print(f"E2E Generation time {(end - st):.4f}s")
        print(output_str)

        # When the IPEX_CPU optimized models recive short prompts(length < 256)
        # it will use normal generate() and has not these attr
        from ipex_llm.transformers.convert import get_enable_ipex
        _enable_ipex = get_enable_ipex()
        if not _enable_ipex or actual_in_len >= 256:
            print(f"Tokens generated {model.n_token_generated}")
            print(f"First token latency {model.first_token_time:.4f}s")
