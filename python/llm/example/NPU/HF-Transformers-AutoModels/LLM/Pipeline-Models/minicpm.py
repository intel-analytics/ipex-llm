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
from transformers import AutoTokenizer, TextStreamer
from transformers.utils import logging
import os

logger = logging.get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Tokens using `generate()` API for npu model"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="openbmb/MiniCPM-1B-sft-bf16", # or "openbmb/MiniCPM-2B-sft-bf16"
        help="The huggingface repo id for the MiniCPM model to be downloaded"
        ", or the path to the huggingface checkpoint folder",
    )
    parser.add_argument("--lowbit-path", type=str,
        default="",
        help="The path to the lowbit model folder, leave blank if you do not want to save. \
             If path not exists, lowbit model will be saved there. \
             Else, lowbit model will be loaded.",
    )
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument("--n-predict", type=int, default=32, help="Max tokens to predict")
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--disable-transpose-value-cache", action="store_true", default=False)
    parser.add_argument("--disable-streaming", action="store_true", default=False)

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    if not args.lowbit_path or not os.path.exists(args.lowbit_path):
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     optimize_model=True,
                                                     pipeline=True,
                                                     max_context_len=args.max_context_len,
                                                     max_prompt_len=args.max_prompt_len,
                                                     torch_dtype=torch.float16,
                                                     attn_implementation="eager",
                                                     transpose_value_cache=not args.disable_transpose_value_cache,
                                                     trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.load_low_bit(
            args.lowbit_path,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            max_context_len=args.max_context_len,
            max_prompt_len=args.max_prompt_len,
            pipeline=True,
            transpose_value_cache=not args.disable_transpose_value_cache,
            trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if args.lowbit_path and not os.path.exists(args.lowbit_path):
        model.save_low_bit(args.lowbit_path)

    if args.disable_streaming:
        streamer = None
    else:
        streamer = TextStreamer(tokenizer=tokenizer, skip_special_tokens=True)

    print("-" * 80)
    print("done")
    with torch.inference_mode():
        print("finish to load")
        for i in range(5):
            prompt = "<用户>{}<AI>".format(args.prompt)
            _input_ids = tokenizer.encode(prompt, return_tensors="pt")
            print("input length:", len(_input_ids[0]))
            st = time.time()
            output = model.generate(
                _input_ids, max_new_tokens=args.n_predict, do_print=True, streamer=streamer
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
