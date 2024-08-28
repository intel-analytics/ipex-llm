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

import os
import torch
import time
import argparse

from ipex_llm.transformers.npu_model import AutoModelForCausalLM
from transformers import AutoTokenizer

from transformers.utils import logging

logger = logging.get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Tokens using `generate()` API for npu model"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="The huggingface repo id for the Qwen2 model to be downloaded"
        ", or the path to the huggingface checkpoint folder",
    )
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument("--n-predict", type=int, default=32, help="Max tokens to predict")
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--disable-transpose-value-cache", action="store_true", default=False)
    parser.add_argument("--intra-pp", type=int, default=2)
    parser.add_argument("--inter-pp", type=int, default=1)

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
        load_in_low_bit="sym_int4",
        optimize_model=True,
        max_output_len=args.max_output_len,
        max_prompt_len=args.max_prompt_len,
        intra_pp=args.intra_pp,
        inter_pp=args.inter_pp,
        transpose_value_cache=not args.disable_transpose_value_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("-" * 80)
    print("done")
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)
    with torch.inference_mode():
        print("finish to load")
        for i in range(3):
            _input_ids = tokenizer([text], return_tensors="pt").input_ids
            print("input length:", len(_input_ids[0]))
            st = time.time()
            output = model.generate(
                _input_ids, num_beams=1, do_sample=False, max_new_tokens=args.n_predict
            )
            end = time.time()
            print(f"Inference time: {end-st} s")
            input_str = tokenizer.decode(_input_ids[0], skip_special_tokens=False)
            print("-" * 20, "Input", "-" * 20)
            print(input_str)
            output_str = tokenizer.decode(output[0], skip_special_tokens=False)
            print("-" * 20, "Output", "-" * 20)
            print(output_str)

    print("-" * 80)
    print("done")
    print("success shut down")
